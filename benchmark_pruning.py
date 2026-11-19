"""Check and benchmark safe pruning of daily-maximum recomputation.

Run ``python -B benchmark_pruning.py --check-only`` for correctness checks.
Run without that flag for a small benchmark. JSON is written to stdout.
Temperatures use integer tenths of a degree Celsius. Each timed batch changes
24 distinct samples, followed by a batch restoring those same samples.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
import platform
import random
from statistics import median, quantiles
import time


DAY_SIZE = 1440
QUARTER_SIZE = 15
QUARTERS_PER_HOUR = 4
CASE_HOURS = (2, 5, 8, 10, 15, 20)
METHOD_NAMES = ("full_scan", "flat15", "two_level", "stop_upward", "safe_scan")


@dataclass
class Stats:
    scan_raw_reads: int = 0
    corrected_old_reads: int = 0
    quarter_scans: int = 0
    hours_recomputed: int = 0
    daily_scans: int = 0
    quarter_maxima_changed: int = 0
    hour_maxima_changed: int = 0

    def record(self):
        result = asdict(self)
        result["raw_reads"] = self.scan_raw_reads + self.corrected_old_reads
        return result


def maxima(values, block_size):
    return [max(values[start:start + block_size])
            for start in range(0, len(values), block_size)]


def max_and_count(values, start, length):
    """Read each sample once, including when several samples share the max."""
    maximum = values[start]
    count = 1
    for index in range(start + 1, start + length):
        value = values[index]
        if value > maximum:
            maximum, count = value, 1
        elif value == maximum:
            count += 1
    return maximum, count


class FullScan:
    def __init__(self, values):
        self.values = values.copy()

    def update(self, updates, stats=None):
        for index, value in updates:
            self.values[index] = value
        if stats is not None:
            stats.scan_raw_reads += len(self.values)
            stats.daily_scans += 1
        return max(self.values)


class Flat15:
    def __init__(self, values):
        self.values = values.copy()
        self.quarters = maxima(values, QUARTER_SIZE)

    def update(self, updates, stats=None):
        affected = set()
        for index, value in updates:
            self.values[index] = value
            affected.add(index // QUARTER_SIZE)
        for quarter in affected:
            start = quarter * QUARTER_SIZE
            previous = self.quarters[quarter]
            self.quarters[quarter] = max(self.values[start:start + QUARTER_SIZE])
            if stats is not None:
                stats.scan_raw_reads += QUARTER_SIZE
                stats.quarter_scans += 1
                stats.quarter_maxima_changed += self.quarters[quarter] != previous
        if stats is not None:
            stats.daily_scans += 1
        return max(self.quarters)


class Hierarchy:
    def __init__(self, values, stop_upward=False, safe_scan=False):
        self.values = values.copy()
        self.quarters = maxima(values, QUARTER_SIZE)
        self.hours = maxima(self.quarters, QUARTERS_PER_HOUR)
        self.daily = max(self.hours)
        self.stop_upward = stop_upward
        self.safe_scan = safe_scan
        self.counts = []
        if safe_scan:
            self.counts = [max_and_count(values, start, QUARTER_SIZE)[1]
                           for start in range(0, len(values), QUARTER_SIZE)]

    def safe_quarter(self, quarter, changes, stats):
        previous = self.quarters[quarter]
        remaining = self.counts[quarter] - sum(old == previous for old, _ in changes)
        new_maximum = max(new for _, new in changes)
        if new_maximum > previous:
            return new_maximum, sum(new == new_maximum for _, new in changes)
        if remaining or new_maximum == previous:
            return previous, remaining + sum(new == previous for _, new in changes)
        if stats is not None:
            stats.scan_raw_reads += QUARTER_SIZE
            stats.quarter_scans += 1
        return max_and_count(self.values, quarter * QUARTER_SIZE, QUARTER_SIZE)

    def update(self, updates, stats=None):
        affected = set()
        changes_by_quarter = {}
        for index, value in updates:
            quarter = index // QUARTER_SIZE
            if self.safe_scan:
                changes_by_quarter.setdefault(quarter, []).append((self.values[index], value))
            self.values[index] = value
            affected.add(quarter)
        if self.safe_scan and stats is not None:
            stats.corrected_old_reads += len(updates)

        changed_quarters = set()
        for quarter in affected:
            previous = self.quarters[quarter]
            if self.safe_scan:
                new_maximum, count = self.safe_quarter(quarter, changes_by_quarter[quarter], stats)
                self.counts[quarter] = count
            else:
                start = quarter * QUARTER_SIZE
                new_maximum = max(self.values[start:start + QUARTER_SIZE])
                if stats is not None:
                    stats.scan_raw_reads += QUARTER_SIZE
                    stats.quarter_scans += 1
            self.quarters[quarter] = new_maximum
            if new_maximum != previous:
                changed_quarters.add(quarter)

        propagation_source = changed_quarters if self.stop_upward else affected
        affected_hours = {quarter // QUARTERS_PER_HOUR for quarter in propagation_source}
        hour_changed = False
        for hour in affected_hours:
            start = hour * QUARTERS_PER_HOUR
            previous = self.hours[hour]
            self.hours[hour] = max(self.quarters[start:start + QUARTERS_PER_HOUR])
            changed = self.hours[hour] != previous
            hour_changed = hour_changed or changed
            if stats is not None:
                stats.hours_recomputed += 1
                stats.hour_maxima_changed += changed
        if hour_changed or not self.stop_upward:
            self.daily = max(self.hours)
            if stats is not None:
                stats.daily_scans += 1
        if stats is not None:
            stats.quarter_maxima_changed += len(changed_quarters)
        return self.daily


class PlainHierarchy(Hierarchy):
    """Recompute every affected ancestor without tracking changed summary values."""

    def update(self, updates, stats=None):
        if stats is not None:
            old_quarters = self.quarters.copy()
            old_hours = self.hours.copy()
        for index, value in updates:
            self.values[index] = value
        quarters = {index // QUARTER_SIZE for index, _ in updates}
        hours = {quarter // QUARTERS_PER_HOUR for quarter in quarters}
        for quarter in quarters:
            start = quarter * QUARTER_SIZE
            self.quarters[quarter] = max(self.values[start:start + QUARTER_SIZE])
        for hour in hours:
            start = hour * QUARTERS_PER_HOUR
            self.hours[hour] = max(self.quarters[start:start + QUARTERS_PER_HOUR])
        self.daily = max(self.hours)
        if stats is not None:
            stats.scan_raw_reads = len(quarters) * QUARTER_SIZE
            stats.quarter_scans = len(quarters)
            stats.hours_recomputed = len(hours)
            stats.daily_scans = 1
            stats.quarter_maxima_changed = sum(self.quarters[q] != old_quarters[q] for q in quarters)
            stats.hour_maxima_changed = sum(self.hours[h] != old_hours[h] for h in hours)
        return self.daily


def build_methods(values):
    return {
        "full_scan": FullScan(values),
        "flat15": Flat15(values),
        "two_level": PlainHierarchy(values),
        "stop_upward": Hierarchy(values, stop_upward=True),
        "safe_scan": Hierarchy(values, stop_upward=True, safe_scan=True),
    }


def base_day():
    source = random.Random(42)
    values = [source.randint(180, 220) for _ in range(DAY_SIZE)]
    for start in range(0, DAY_SIZE, QUARTER_SIZE):
        values[start] = 240
    values[60] = 260
    values[17 * 60 + 27] = 245
    return values


def labelled_cases():
    positions = [hour * 60 + offset for hour in CASE_HOURS for offset in range(1, 5)]
    result = []
    for name in ("nonmax_changes", "one_tie_survives", "all_ties_removed",
                 "new_max_established", "mixed_down_and_up", "daily_max_declines"):
        values = base_day()
        if name in ("one_tie_survives", "all_ties_removed", "mixed_down_and_up"):
            for hour in CASE_HOURS:
                start = hour * 60
                values[start + 1] = 240
                if name != "one_tie_survives":
                    values[start] = 200
                    values[start + 2] = 240
        if name == "daily_max_declines":
            values[60] = 230
            values[10 * 60 + 1] = 260
        updates = [(index, values[index] - 3) for index in positions]
        if name in ("one_tie_survives", "all_ties_removed", "mixed_down_and_up"):
            updates = [(index, 190 if index % 60 in (1, 2) else new)
                       for index, new in updates]
        if name == "new_max_established":
            updates = [(index, 270 if index % 60 == 1 else new) for index, new in updates]
        if name == "mixed_down_and_up":
            updates = [(index, 270 if index % 60 == 2 else new) for index, new in updates]
        if name == "daily_max_declines":
            updates = [(index, 210 if index == 601 else new) for index, new in updates]
        result.append((name, values, updates))
    values = base_day()
    result.append(("all_day_nonmax", values,
                   [(hour * 60 + 1, values[hour * 60 + 1] - 3) for hour in range(24)]))
    return result


def validate_updates(updates):
    indices = [index for index, _ in updates]
    if len(indices) != len(set(indices)) or not all(0 <= index < DAY_SIZE for index in indices):
        raise ValueError("Each batch must use distinct valid sample indices.")


def verify_state(name, method, expected):
    if method.values != expected:
        raise AssertionError(f"{name}: raw values differ")
    if isinstance(method, (Flat15, Hierarchy)):
        if method.quarters != maxima(expected, QUARTER_SIZE):
            raise AssertionError(f"{name}: stale quarter maxima")
    if isinstance(method, Hierarchy):
        if method.hours != maxima(method.quarters, QUARTERS_PER_HOUR):
            raise AssertionError(f"{name}: stale hour maxima")
        if method.daily != max(expected):
            raise AssertionError(f"{name}: stale daily maximum")
        if method.safe_scan:
            counts = [max_and_count(expected, start, QUARTER_SIZE)[1]
                      for start in range(0, DAY_SIZE, QUARTER_SIZE)]
            if method.counts != counts:
                raise AssertionError(f"{name}: stale maximum multiplicity")


def correctness(cases, random_batches):
    checks = 0
    for case_name, values, updates in cases:
        validate_updates(updates)
        if len(updates) != 24:
            raise AssertionError("Labelled cases must have exactly 24 changes")
        reverse = [(index, values[index]) for index, _ in updates]
        for batch in (updates, reverse):
            expected = values.copy() if batch is updates else corrected.copy()
            methods = build_methods(expected)
            for update in batch:
                expected[update[0]] = update[1]
                oracle = max(expected)
                for name, method in methods.items():
                    if method.update([update]) != oracle:
                        raise AssertionError(f"{case_name}/{name}: prefix oracle mismatch")
                    checks += 1
            for name, method in methods.items():
                verify_state(name, method, expected)
            if batch is updates:
                corrected = expected.copy()

    source = random.Random(726)
    expected = base_day()
    methods = build_methods(expected)
    # A small discrete value set deliberately creates many tied maxima.
    possible_values = (180, 190, 200, 210, 220, 240, 240, 240, 260, 270)
    for batch_index in range(random_batches):
        updates = [(index, source.choice(possible_values))
                   for index in source.sample(range(DAY_SIZE), 24)]
        validate_updates(updates)
        for index, value in updates:
            expected[index] = value
        oracle = max(expected)
        for name, method in methods.items():
            if method.update(updates) != oracle:
                raise AssertionError(f"random batch {batch_index}/{name}: oracle mismatch")
            verify_state(name, method, expected)
            checks += 1
    return {"passed": True, "labelled_cases": len(cases), "prefix_updates_per_case": 48,
            "random_sequential_batches": random_batches, "random_batch_size": 24,
            "method_oracle_checks": checks, "maximum_multiplicity_verified": True}


def time_pairs(method, forward, reverse, pairs):
    started = time.perf_counter_ns()
    for _ in range(pairs):
        method.update(forward)
        method.update(reverse)
    return (time.perf_counter_ns() - started) / (2 * pairs) / 1000


def measure_case(case, rounds, pairs, warmup_pairs, order_source, skip_timing):
    name, values, forward = case
    reverse = [(index, values[index]) for index, _ in forward]
    expected = values.copy()
    for index, value in forward:
        expected[index] = value
    methods = build_methods(values)
    records = {}
    for method_name, method in methods.items():
        forward_stats, reverse_stats = Stats(), Stats()
        if method.update(forward, forward_stats) != max(expected):
            raise AssertionError(f"{name}/{method_name}: batch oracle mismatch")
        verify_state(method_name, method, expected)
        if method.update(reverse, reverse_stats) != max(values):
            raise AssertionError(f"{name}/{method_name}: restoration oracle mismatch")
        verify_state(method_name, method, values)
        records[method_name] = {"forward_stats": forward_stats.record(),
                                "restore_stats": reverse_stats.record()}
    if not skip_timing:
        for method in methods.values():
            time_pairs(method, forward, reverse, warmup_pairs)
        samples = {method_name: [] for method_name in methods}
        order = list(methods)
        for _ in range(rounds):
            order_source.shuffle(order)
            for method_name in order:
                samples[method_name].append(time_pairs(methods[method_name], forward, reverse, pairs))
        for method_name, times in samples.items():
            first, _, third = quantiles(times, n=4, method="inclusive")
            records[method_name]["timing"] = {
                "median_us": median(times), "q1_us": first, "q3_us": third, "samples_us": times}
        for method_name, method in methods.items():
            verify_state(method_name, method, values)
    return {"name": name, "changes_per_batch": len(forward),
            "affected_hours": len({index // 60 for index, _ in forward}),
            "affected_quarters": len({index // QUARTER_SIZE for index, _ in forward}),
            "initial_values_tenths_c": values,
            "forward_changes_tenths_c": forward,
            "initial_daily_c": max(values) / 10, "forward_daily_c": max(expected) / 10,
            "restore_daily_c": max(values) / 10, "methods": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--random-batches", type=int, default=120)
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 1 or args.random_batches < 1:
        parser.error("rounds >= 2, pairs >= 1 and random-batches >= 1 are required")
    rounds, pairs = (2, 2) if args.smoke else (args.rounds, args.pairs)
    warmup_pairs = 2 if args.smoke else 32
    random_batches = 12 if args.smoke else args.random_batches
    cases = labelled_cases()
    checked = correctness(cases, random_batches)
    order_source = random.Random(901)
    order = cases.copy()
    order_source.shuffle(order)
    measured = [measure_case(case, rounds, pairs, warmup_pairs, order_source, args.check_only)
                for case in order]
    position = {name: index for index, (name, _, _) in enumerate(cases)}
    measured.sort(key=lambda case: position[case["name"]])
    result = {
        "environment": {"os": platform.platform(), "python": platform.python_version(),
                        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())},
        "protocol": {
            "data": "synthetic one-day integer temperatures, one sample per minute",
            "samples": DAY_SIZE, "quarter_samples": QUARTER_SIZE,
            "rounds": rounds, "batches_per_round": pairs * 2,
            "warmup_batches": warmup_pairs * 2, "order_seed": 901,
            "timing_run": not args.check_only, "smoke_run": args.smoke,
            "time_definition": "mean per batch across forward corrections and restoration; "
                               "cache creation, correctness checks and diagnostic counters excluded",
            "raw_reads_definition": "scan_raw_reads + corrected_old_reads; raw writes and "
                                    "summary-array accesses excluded; a scan reads every cell once",
            "safe_scan_state": "maximum and multiplicity per quarter, plus hour and daily maxima",
            "safety": "batch indices are unique; fallback scan when all old maxima are replaced "
                      "and every corrected new value is below the old maximum",
            "limits": "hot-memory synthetic Python benchmark; extra multiplicity storage, "
                      "grouping and branches may outweigh eliminated scans",
        },
        "correctness": checked, "cases": measured,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
