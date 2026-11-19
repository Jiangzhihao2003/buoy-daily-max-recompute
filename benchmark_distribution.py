"""Controlled distribution and synthetic size tests for daily-maximum updates.

Run ``python benchmark_distribution.py --check-only`` for correctness checks.
Without that option, JSON on stdout includes a short warmed timing experiment.
All cache construction, workload generation, and oracle checks occur outside the
timed region. Each timed batch includes 24 assignments, affected-block discovery,
summary maintenance, and returning the daily maximum.
"""

import argparse
import json
import os
import platform
import random
import time
from statistics import median, quantiles

from recompute_daily_max import (
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    PEAK_MINUTE,
    QUARTER_HOUR_MINUTES,
    QUARTERS_PER_HOUR,
    build_block_maxima,
    refresh_maxima,
    temperatures_for_day,
)


CORRECTIONS_PER_BATCH = 24
ROUNDS = 15
PAIRS_PER_ROUND = 128
WARMUP_PAIRS = 32
ORDER_SEED = 117
SECOND_PEAK_MINUTE = 17 * MINUTES_PER_HOUR + 27


def synthetic_readings(readings_per_minute):
    """Preserve minute observations and add explicitly artificial sub-samples."""
    original = temperatures_for_day()
    readings = []
    for minute, temperature in enumerate(original):
        readings.append(temperature)
        ordinary_temperature = temperature
        if minute in (PEAK_MINUTE, SECOND_PEAK_MINUTE):
            ordinary_temperature = original[(minute + 1) % MINUTES_PER_DAY]
        for sample in range(1, readings_per_minute):
            perturbation = (minute + sample) % 3 - 1
            readings.append(min(244, ordinary_temperature + perturbation))
    if len(readings) != MINUTES_PER_DAY * readings_per_minute:
        raise ValueError("Synthetic series has an unexpected length.")
    return readings


def controlled_cases():
    six_hours = (10, 14, 18, 22, 2, 6)
    two_hours = (10, 22)
    eight_hours = (10, 13, 16, 19, 22, 1, 4, 7)
    return [
        {
            "scenario_id": "fixed_h6_q6",
            "comparison": "fixed_affected_hours",
            "minutes": [hour * 60 + offset for hour in six_hours for offset in range(4)],
            "expected_h": 6,
            "expected_q": 6,
        },
        {
            "scenario_id": "fixed_h6_q24",
            "comparison": "fixed_affected_hours",
            "minutes": [hour * 60 + offset for hour in six_hours for offset in (0, 15, 30, 45)],
            "expected_h": 6,
            "expected_q": 24,
        },
        {
            "scenario_id": "fixed_q8_h2",
            "comparison": "fixed_affected_quarter_hours",
            "minutes": [
                hour * 60 + quarter * 15 + offset
                for hour in two_hours
                for quarter in range(4)
                for offset in range(3)
            ],
            "expected_h": 2,
            "expected_q": 8,
        },
        {
            "scenario_id": "fixed_q8_h8",
            "comparison": "fixed_affected_quarter_hours",
            "minutes": [hour * 60 + offset for hour in eight_hours for offset in range(3)],
            "expected_h": 8,
            "expected_q": 8,
        },
    ]


def all_cases():
    cases = []
    for case in controlled_cases():
        cases.append({**case, "group": "controlled", "readings_per_minute": 1})
    patterns = {
        "concentrated": (list(range(PEAK_MINUTE, PEAK_MINUTE + 24)), 1, 2),
        "spread_out": (list(range(0, MINUTES_PER_DAY, MINUTES_PER_HOUR)), 24, 24),
    }
    for density in (1, 4, 16):
        for name, (minutes, hours, quarters) in patterns.items():
            cases.append({
                "scenario_id": f"size_{density}_{name}",
                "group": "size_sensitivity",
                "comparison": name,
                "readings_per_minute": density,
                "minutes": minutes,
                "expected_h": hours,
                "expected_q": quarters,
            })
    return cases


def correction_batches(readings, minutes, density):
    minutes = sorted(minutes)
    if len(minutes) != CORRECTIONS_PER_BATCH or len(set(minutes)) != len(minutes):
        raise ValueError("Each batch must correct 24 distinct minute observations.")
    if any(minute < 0 or minute >= MINUTES_PER_DAY for minute in minutes):
        raise ValueError("A correction is outside the simulated day.")
    forward = []
    reverse = []
    for minute in minutes:
        index = minute * density
        old_value = readings[index]
        forward.append((index, 210 if minute == PEAK_MINUTE else old_value - 3))
        reverse.append((index, old_value))
    return forward, reverse


def methods_for(readings, density):
    def full_scan(values, updates):
        return max(values)

    def single_level(block_length):
        maxima = build_block_maxima(readings, block_length)

        def recompute(values, updates):
            affected = {index // block_length for index, _ in updates}
            refresh_maxima(values, maxima, affected, block_length)
            return max(maxima)

        return recompute

    quarter_length = QUARTER_HOUR_MINUTES * density
    quarter_maxima = build_block_maxima(readings, quarter_length)
    hour_maxima = build_block_maxima(quarter_maxima, QUARTERS_PER_HOUR)

    def two_levels(values, updates):
        quarters = {index // quarter_length for index, _ in updates}
        hours = {quarter // QUARTERS_PER_HOUR for quarter in quarters}
        refresh_maxima(values, quarter_maxima, quarters, quarter_length)
        refresh_maxima(quarter_maxima, hour_maxima, hours, QUARTERS_PER_HOUR)
        return max(hour_maxima)

    methods = {
        "full_scan": full_scan,
        "hourly": single_level(MINUTES_PER_HOUR * density),
        "quarter_hourly": single_level(quarter_length),
        "two_levels": two_levels,
    }
    return {name: (readings.copy(), recompute) for name, recompute in methods.items()}


def apply_batch(values, recompute, updates):
    for index, temperature in updates:
        values[index] = temperature
    return recompute(values, updates)


def logical_work(readings, forward, density):
    hour_length = MINUTES_PER_HOUR * density
    quarter_length = QUARTER_HOUR_MINUTES * density
    quarters = {index // quarter_length for index, _ in forward}
    hours = {index // hour_length for index, _ in forward}
    h, q = len(hours), len(quarters)
    return {
        "full_scan": {
            "distinct_raw_reads": len(readings),
            "quarter_summary_reads": 0,
            "hour_summary_reads": 0,
            "max_comparisons": len(readings) - 1,
            "cached_maximum_values": 0,
        },
        "hourly": {
            "distinct_raw_reads": h * hour_length,
            "quarter_summary_reads": 0,
            "hour_summary_reads": 24,
            "max_comparisons": h * (hour_length - 1) + 23,
            "cached_maximum_values": 24,
        },
        "quarter_hourly": {
            "distinct_raw_reads": q * quarter_length,
            "quarter_summary_reads": 96,
            "hour_summary_reads": 0,
            "max_comparisons": q * (quarter_length - 1) + 95,
            "cached_maximum_values": 96,
        },
        "two_levels": {
            "distinct_raw_reads": q * quarter_length,
            "quarter_summary_reads": 4 * h,
            "hour_summary_reads": 24,
            "max_comparisons": q * (quarter_length - 1) + 3 * h + 23,
            "cached_maximum_values": 120,
        },
    }


def verify_methods(methods, readings, forward, reverse):
    oracle = readings.copy()
    expected = []
    batches = (forward, reverse, forward, reverse)
    for updates in batches:
        for index, temperature in updates:
            oracle[index] = temperature
        expected.append(max(oracle))
    if expected != [245, 260, 245, 260]:
        raise ValueError("Synthetic forward/reverse peak values changed unexpectedly.")
    for name, (values, recompute) in methods.items():
        observed = [apply_batch(values, recompute, updates) for updates in batches]
        if observed != expected or values != readings:
            raise ValueError(f"{name} disagrees with the full-scan oracle.")
    return {
        "all_methods_match_exact_oracle": True,
        "checked_batches_per_method": len(batches),
        "oracle_maximum_tenths_c": expected,
        "forward_maximum_c": expected[0] / 10,
        "reverse_maximum_c": expected[1] / 10,
        "all_methods_return_to_original_raw_values": True,
    }


def time_round(method, forward, reverse, pairs):
    values, recompute = method
    started = time.perf_counter_ns()
    for _ in range(pairs):
        apply_batch(values, recompute, forward)
        apply_batch(values, recompute, reverse)
    return (time.perf_counter_ns() - started) / (2 * pairs) / 1_000


def measure_case(case, random_source, rounds, pairs, warmup_pairs, check_only):
    density = case["readings_per_minute"]
    readings = synthetic_readings(density)
    forward, reverse = correction_batches(readings, case["minutes"], density)
    h = len({minute // MINUTES_PER_HOUR for minute in case["minutes"]})
    q = len({minute // QUARTER_HOUR_MINUTES for minute in case["minutes"]})
    if (h, q) != (case["expected_h"], case["expected_q"]):
        raise ValueError("Controlled workload does not have the specified H and Q.")
    methods = methods_for(readings, density)
    correctness = verify_methods(methods, readings, forward, reverse)
    summaries = None
    if not check_only:
        for method in methods.values():
            time_round(method, forward, reverse, warmup_pairs)
        samples = {name: [] for name in methods}
        order = list(methods)
        for _ in range(rounds):
            random_source.shuffle(order)
            for name in order:
                samples[name].append(time_round(methods[name], forward, reverse, pairs))
        summaries = {}
        for name, times in samples.items():
            q1, _, q3 = quantiles(times, n=4, method="inclusive")
            summaries[name] = {
                "median": median(times), "q1": q1, "q3": q3,
                "iqr": q3 - q1, "samples": times,
            }
    return {
        "scenario_id": case["scenario_id"],
        "group": case["group"],
        "comparison": case["comparison"],
        "readings_per_minute": density,
        "readings": len(readings),
        "corrections_per_batch": len(forward),
        "correction_fraction": len(forward) / len(readings),
        "affected_hours": h,
        "affected_quarter_hours": q,
        "correction_minutes": sorted(case["minutes"]),
        "correction_record_indices": [index for index, _ in forward],
        "correction_values_tenths_c": [temperature for _, temperature in forward],
        "logical_work": logical_work(readings, forward, density),
        "correctness": correctness,
        "microseconds_per_batch": summaries,
    }


def run(rounds=ROUNDS, pairs=PAIRS_PER_ROUND, warmup_pairs=WARMUP_PAIRS, check_only=False):
    if rounds < 2 or pairs < 1 or warmup_pairs < 1:
        raise ValueError("Use at least two rounds and positive batch-pair counts.")
    cases = all_cases()
    random_source = random.Random(ORDER_SEED)
    execution_order = cases.copy()
    random_source.shuffle(execution_order)
    measured = {
        case["scenario_id"]: measure_case(
            case, random_source, rounds, pairs, warmup_pairs, check_only
        )
        for case in execution_order
    }
    return {
        "schema_version": 1,
        "experiment": "controlled_distribution_and_synthetic_size_sensitivity",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": os.environ.get("BUOY_BENCH_CPU", platform.processor()),
        },
        "timing_performed": not check_only,
        "rounds_per_method_and_case": 0 if check_only else rounds,
        "batches_per_round": 2 * pairs,
        "warmup_batches": 2 * warmup_pairs,
        "execution_order_seed": ORDER_SEED,
        "case_execution_order": [case["scenario_id"] for case in execution_order],
        "measurement_scope": {
            "container": "Python list of integer tenths of a degree Celsius",
            "timed": [
                "24 value assignments", "affected-block identification",
                "summary refresh", "returning the daily maximum",
            ],
            "excluded": [
                "synthetic data generation", "cache initialization",
                "oracle verification", "file I/O",
            ],
            "initialization_raw_reads_per_cache": "1440 * readings_per_minute",
            "logical_counts_exclude": ["24 update writes", "index and set operations"],
            "logical_reads_are_disk_io_measurements": False,
            "sample_unit": "mean microseconds per batch within one timed round",
        },
        "synthetic_size_limitations": [
            "All series span the same 1440 minutes and are synthetic.",
            "The first sub-sample retains each original minute observation.",
            "Additional sub-samples use the minute temperature plus deterministic +/-0.1 C perturbations.",
            "At the two manually injected peaks, additional samples use the next minute's ordinary temperature.",
            "Additional sub-samples are capped at 24.4 C; only the original samples contain the manual peaks.",
            "Each batch corrects sub-sample zero in 24 distinct minutes.",
            "Increasing density with 24 fixed corrections also lowers the corrected fraction.",
            "This tests synthetic within-block size, not a causal effect of real sensor sampling frequency.",
            "Controlled comparisons change explicit timestamp layouts and do not cover every layout with the same H or Q.",
        ],
        "cases": [measured[case["scenario_id"]] for case in cases],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Skip all performance timing.")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--pairs-per-round", type=int, default=PAIRS_PER_ROUND)
    parser.add_argument("--warmup-pairs", type=int, default=WARMUP_PAIRS)
    args = parser.parse_args()
    result = run(args.rounds, args.pairs_per_round, args.warmup_pairs, args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
