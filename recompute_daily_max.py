"""Compare full and block-based recomputation after buoy readings are corrected."""

import json
import random


MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR
QUARTER_HOUR_MINUTES = 15
QUARTERS_PER_HOUR = MINUTES_PER_HOUR // QUARTER_HOUR_MINUTES
BLOCK_LENGTHS = (MINUTES_PER_HOUR, QUARTER_HOUR_MINUTES)
SEED = 42
PEAK_MINUTE = 10 * MINUTES_PER_HOUR


def temperatures_for_day() -> list[int]:
    """Use integer tenths of a degree Celsius for exact comparisons."""
    random_source = random.Random(SEED)
    readings = [200 + random_source.randint(-20, 20) for _ in range(MINUTES_PER_DAY)]
    readings[PEAK_MINUTE] = 260
    readings[17 * 60 + 27] = 245
    return readings


def read_block(values: list[int], block_index: int, block_length: int) -> list[int]:
    start = block_index * block_length
    return values[start : start + block_length]


def build_block_maxima(values: list[int], block_length: int) -> list[int]:
    return [
        max(read_block(values, block_index, block_length))
        for block_index in range(len(values) // block_length)
    ]


def refresh_maxima(
    values: list[int], maxima: list[int], affected_blocks: set[int], block_length: int
) -> None:
    for block_index in affected_blocks:
        block = read_block(values, block_index, block_length)
        maxima[block_index] = max(block)


def recompute_blocks(
    readings: list[int],
    cached_maxima: list[int],
    corrected_minutes: list[int],
    block_length: int,
) -> dict:
    affected_blocks = {minute // block_length for minute in corrected_minutes}
    updated_maxima = cached_maxima.copy()
    refresh_maxima(readings, updated_maxima, affected_blocks, block_length)
    raw_records_read = len(affected_blocks) * block_length

    return {
        "daily_maximum_c": max(updated_maxima) / 10,
        "blocks_recomputed": len(affected_blocks),
        "raw_records_read": raw_records_read,
        "block_maxima_combined": len(updated_maxima),
        "max_comparisons": raw_records_read - len(affected_blocks) + len(updated_maxima) - 1,
    }


def recompute_two_levels(
    readings: list[int],
    cached_quarter_maxima: list[int],
    cached_hour_maxima: list[int],
    corrected_minutes: list[int],
) -> dict:
    affected_quarters = {minute // QUARTER_HOUR_MINUTES for minute in corrected_minutes}
    affected_hours = {quarter // QUARTERS_PER_HOUR for quarter in affected_quarters}
    quarter_maxima = cached_quarter_maxima.copy()
    hour_maxima = cached_hour_maxima.copy()
    refresh_maxima(readings, quarter_maxima, affected_quarters, QUARTER_HOUR_MINUTES)
    refresh_maxima(quarter_maxima, hour_maxima, affected_hours, QUARTERS_PER_HOUR)
    raw_records_read = len(affected_quarters) * QUARTER_HOUR_MINUTES
    quarter_maxima_combined = len(affected_hours) * QUARTERS_PER_HOUR
    max_comparisons = (
        raw_records_read - len(affected_quarters)
        + quarter_maxima_combined - len(affected_hours)
        + len(hour_maxima) - 1
    )

    return {
        "daily_maximum_c": max(hour_maxima) / 10,
        "quarter_blocks_recomputed": len(affected_quarters),
        "hour_maxima_recomputed": len(affected_hours),
        "raw_records_read": raw_records_read,
        "quarter_maxima_combined": quarter_maxima_combined,
        "hour_maxima_combined": len(hour_maxima),
        "max_comparisons": max_comparisons,
    }


def run() -> dict:
    readings = temperatures_for_day()
    cached_maxima = {
        length: build_block_maxima(readings, length) for length in BLOCK_LENGTHS
    }
    cached_quarter_maxima = cached_maxima[QUARTER_HOUR_MINUTES]
    hourly_maxima_from_quarters = build_block_maxima(cached_quarter_maxima, QUARTERS_PER_HOUR)
    scenarios = {
        "concentrated": list(range(PEAK_MINUTE, PEAK_MINUTE + 24)),
        "spread_out": list(range(0, MINUTES_PER_DAY, MINUTES_PER_HOUR)),
    }
    results = {}
    for scenario, corrected_minutes in scenarios.items():
        corrected_readings = readings.copy()
        for minute in corrected_minutes:
            corrected_readings[minute] = 210 if minute == PEAK_MINUTE else readings[minute] - 3

        full_scan = {
            "daily_maximum_c": max(corrected_readings) / 10,
            "raw_records_read": len(corrected_readings),
            "max_comparisons": len(corrected_readings) - 1,
        }
        results[scenario] = {
            "corrections": len(corrected_minutes),
            "full_scan": full_scan,
            "block_recomputation": {
                str(length): recompute_blocks(
                    corrected_readings, cached_maxima[length], corrected_minutes, length
                )
                for length in BLOCK_LENGTHS
            },
            "two_levels": recompute_two_levels(
                corrected_readings, cached_quarter_maxima, hourly_maxima_from_quarters,
                corrected_minutes,
            ),
        }

    return {
        "data": "one simulated buoy reporting once per minute for one day",
        "seed": SEED,
        "readings_per_day": len(readings),
        "daily_maximum_before_corrections_c": max(readings) / 10,
        "one_time_raw_records_read_to_build_each_cache": len(readings),
        "cached_maxima_by_method": {
            "60_minute_blocks": len(cached_maxima[MINUTES_PER_HOUR]),
            "15_minute_blocks": len(cached_quarter_maxima),
            "two_levels": len(cached_quarter_maxima) + len(hourly_maxima_from_quarters),
        },
        "scenarios": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
