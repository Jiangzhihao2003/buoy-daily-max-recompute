"""Measure the cost of applying one batch of corrections and updating the daily maximum."""

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
ROUNDS = 25
PAIRS_PER_ROUND = 512
WARMUP_PAIRS = 64
ORDER_SEED = 91


def batch_for_hours(readings, hour_count):
    """Spread 24 corrections across a controlled number of distinct hours."""
    per_hour, remainder = divmod(CORRECTIONS_PER_BATCH, hour_count)
    peak_hour = PEAK_MINUTE // MINUTES_PER_HOUR
    corrected_minutes = []
    for index in range(hour_count):
        hour = (peak_hour + index * 24 // hour_count) % 24
        start = hour * MINUTES_PER_HOUR
        corrected_minutes.extend(range(start, start + per_hour + (index < remainder)))
    corrected_minutes.sort()
    forward = [(minute, 210 if minute == PEAK_MINUTE else readings[minute] - 3)
               for minute in corrected_minutes]
    reverse = [(minute, readings[minute]) for minute in corrected_minutes]
    return forward, reverse


def methods_for(readings):
    def full_scan(values, updates):
        return max(values)

    def single_level(block_length):
        maxima = build_block_maxima(readings, block_length)

        def recompute(values, updates):
            affected = {minute // block_length for minute, _ in updates}
            refresh_maxima(values, maxima, affected, block_length)
            return max(maxima)

        return recompute

    quarter_maxima = build_block_maxima(readings, QUARTER_HOUR_MINUTES)
    hour_maxima = build_block_maxima(quarter_maxima, QUARTERS_PER_HOUR)

    def two_levels(values, updates):
        quarters = {minute // QUARTER_HOUR_MINUTES for minute, _ in updates}
        hours = {quarter // QUARTERS_PER_HOUR for quarter in quarters}
        refresh_maxima(values, quarter_maxima, quarters, QUARTER_HOUR_MINUTES)
        refresh_maxima(quarter_maxima, hour_maxima, hours, QUARTERS_PER_HOUR)
        return max(hour_maxima)

    methods = {
        "full_scan": full_scan,
        "hourly": single_level(MINUTES_PER_HOUR),
        "quarter_hourly": single_level(QUARTER_HOUR_MINUTES),
        "two_levels": two_levels,
    }
    return {name: (readings.copy(), recompute) for name, recompute in methods.items()}


def apply_batch(values, recompute, updates):
    for minute, temperature in updates:
        values[minute] = temperature
    return recompute(values, updates)


def time_round(method, forward, reverse, pairs):
    values, recompute = method
    started = time.perf_counter_ns()
    for _ in range(pairs):
        apply_batch(values, recompute, forward)
        apply_batch(values, recompute, reverse)
    elapsed = time.perf_counter_ns() - started
    return elapsed / (2 * pairs) / 1_000


def measure_case(readings, hour_count, random_source):
    forward, reverse = batch_for_hours(readings, hour_count)
    if len(forward) != CORRECTIONS_PER_BATCH or len({minute for minute, _ in forward}) != len(forward):
        raise ValueError("A batch must contain 24 distinct corrections.")
    methods = methods_for(readings)
    corrected = readings.copy()
    for minute, temperature in forward:
        corrected[minute] = temperature
    forward_maximum, reverse_maximum = max(corrected), max(readings)
    for name, (values, recompute) in methods.items():
        if (apply_batch(values, recompute, forward), apply_batch(values, recompute, reverse)) != (
            forward_maximum, reverse_maximum
        ):
            raise ValueError(f"{name} disagrees with the full scan for {hour_count} affected hours.")
        time_round((values, recompute), forward, reverse, WARMUP_PAIRS)

    samples = {name: [] for name in methods}
    order = list(methods)
    for _ in range(ROUNDS):
        random_source.shuffle(order)
        for name in order:
            samples[name].append(time_round(methods[name], forward, reverse, PAIRS_PER_ROUND))

    summaries = {}
    for name, times in samples.items():
        q1, _, q3 = quantiles(times, n=4, method="inclusive")
        summaries[name] = {"median": median(times), "q1": q1, "q3": q3, "samples": times}

    return {
        "affected_hours": hour_count,
        "affected_quarter_hours": len({minute // QUARTER_HOUR_MINUTES for minute, _ in forward}),
        "correction_minutes": [minute for minute, _ in forward],
        "daily_maximum_after_corrections_c": forward_maximum / 10,
        "microseconds_per_batch": summaries,
        "two_level_faster_rounds": {
            name: sum(t < alternate for t, alternate in zip(samples["two_levels"], times))
            for name, times in samples.items() if name != "two_levels"
        },
    }


def main():
    readings = temperatures_for_day()
    random_source = random.Random(ORDER_SEED)
    hours = list(range(1, 25))
    random_source.shuffle(hours)
    cases = [measure_case(readings, count, random_source) for count in hours]
    return {
        "environment": {
            "os": platform.platform(), "python": platform.python_version(),
            "cpu": os.environ.get("BUOY_BENCH_CPU", platform.processor()),
        },
        "readings": MINUTES_PER_DAY,
        "corrections_per_batch": CORRECTIONS_PER_BATCH,
        "rounds_per_method_and_case": ROUNDS,
        "batches_per_round": 2 * PAIRS_PER_ROUND,
        "warmup_batches": 2 * WARMUP_PAIRS,
        "execution_order_seed": ORDER_SEED,
        "cases": sorted(cases, key=lambda case: case["affected_hours"]),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
