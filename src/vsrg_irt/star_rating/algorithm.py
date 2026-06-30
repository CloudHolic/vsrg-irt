"""osu! lazer osu!mania Star Rating Calculator (standalone Python port)"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class HitObject:
    start_time: float   # ms
    end_time: float     # ms (Same as start_time in single note.)
    column: int


def parse_osu(path: str, clock_rate: float = 1.0) -> tuple[int, list[HitObject]]:
    """
    Parse an .osu file, and returns (total_columns, hit_objects).
    clock_rate: Playback rate. (DT=1.5, HT=0.75, NM=1.0)
    """
    total_columns = 4
    objects: list[HitObject] = []

    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    section = ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("["):
            section = line
            continue

        if section == "[Difficulty]" and line.startswith("CircleSize:"):
            total_columns = int(float(line.split(":")[1]))

        if section == "[HitObjects]":
            parts = line.split(",")
            if len(parts) < 5:
                continue

            x = int(parts[0])
            raw_start = float(parts[2])
            hit_type = int(parts[3])

            column = int(x * total_columns / 512)
            column = max(0, min(total_columns - 1, column))

            start_time = raw_start / clock_rate

            # 128 = hold note (LN)
            if hit_type & 128:
                end_raw = float(parts[5].split(":")[0])
                end_time = end_raw / clock_rate
            else:
                end_time = start_time

            objects.append(HitObject(start_time, end_time, column))

    objects.sort(key=lambda o: round(o.start_time))
    return total_columns, objects


@dataclass
class ManiaDifficultyHitObject:
    start_time: float
    end_time: float
    delta_time: float
    column: int
    column_strain_time: float
    previous_hit_objects: list[Optional[ManiaDifficultyHitObject]]
    index: int


def build_difficulty_objects(
    hit_objects: list[HitObject],
    total_columns: int,
) -> list[ManiaDifficultyHitObject]:
    if len(hit_objects) < 2:
        return []

    per_column: list[list[ManiaDifficultyHitObject]] = [[] for _ in range(total_columns)]
    objects: list[ManiaDifficultyHitObject] = []

    # Treat the first note as dummy.
    for i in range(1, len(hit_objects)):
        curr = hit_objects[i]
        prev = hit_objects[i - 1]

        delta_time = curr.start_time - prev.start_time
        col = curr.column

        prev_in_col = per_column[col][-1] if per_column[col] else None
        column_strain_time = (
            curr.start_time - prev_in_col.start_time
            if prev_in_col is not None
            else curr.start_time
        )

        if objects:
            prev_note = objects[-1]
            prev_ho = list(prev_note.previous_hit_objects)
            prev_ho[prev_note.column] = prev_note
        else:
            prev_ho = [None] * total_columns

        obj = ManiaDifficultyHitObject(
            start_time=curr.start_time,
            end_time=curr.end_time,
            delta_time=delta_time,
            column=col,
            column_strain_time=column_strain_time,
            previous_hit_objects=list(prev_ho),
            index=i - 1,  # 0-based
        )

        objects.append(obj)
        per_column[col].append(obj)

    return objects


# Evaluators
def _definitely_bigger(a: float, b: float, tolerance: float = 1.0) -> bool:
    return a - b > tolerance


def individual_strain_of(current: ManiaDifficultyHitObject) -> float:
    start_time = current.start_time
    end_time = current.end_time

    hold_factor = 1.0
    for prev in current.previous_hit_objects:
        if prev is None:
            continue
        if (_definitely_bigger(prev.end_time, end_time) and
                _definitely_bigger(start_time, prev.start_time)):
            hold_factor = 1.25
            break

    return 2.0 * hold_factor


def _logistic(x: float, multiplier: float, midpoint_offset: float) -> float:
    return 1.0 / (1.0 + math.exp(-multiplier * (x - midpoint_offset)))


RELEASE_THRESHOLD = 30.0


def overall_strain_of(current: ManiaDifficultyHitObject) -> float:
    start_time = current.start_time
    end_time = current.end_time

    is_overlapping = False
    closest_end_time = abs(end_time - start_time)
    hold_factor = 1.0
    hold_addition = 0.0

    for prev in current.previous_hit_objects:
        if prev is None:
            continue

        is_overlapping |= (
            _definitely_bigger(prev.end_time, start_time) and
            _definitely_bigger(end_time, prev.end_time) and
            _definitely_bigger(start_time, prev.start_time)
        )

        if (_definitely_bigger(prev.end_time, end_time) and
                _definitely_bigger(start_time, prev.start_time)):
            hold_factor = 1.25

        closest_end_time = min(closest_end_time, abs(end_time - prev.end_time))

    if is_overlapping:
        hold_addition = _logistic(
            x=closest_end_time,
            multiplier=0.27,
            midpoint_offset=RELEASE_THRESHOLD,
        )

    return (1.0 + hold_addition) * hold_factor


# Strain skill
INDIVIDUAL_DECAY_BASE = 0.125
OVERALL_DECAY_BASE = 0.30
SKILL_MULTIPLIER = 1.0
STRAIN_DECAY_BASE = 1.0   # StrainDecaySkill.StrainDecayBase
SECTION_LENGTH = 400      # ms
DECAY_WEIGHT = 0.9
DIFFICULTY_MULTIPLIER = 0.018


def _apply_decay(value: float, delta_time: float, decay_base: float) -> float:
    return value * math.pow(decay_base, delta_time / 1000.0)


def calculate_difficulty_value(
    objects: list[ManiaDifficultyHitObject],
    total_columns: int,
) -> float:
    if not objects:
        return 0.0

    individual_strains = [0.0] * total_columns
    highest_individual_strain = 0.0
    overall_strain = 1.0
    current_strain = 0.0  # StrainDecaySkill.CurrentStrain

    strain_peaks: list[float] = []
    current_section_peak = 0.0
    current_section_end = math.ceil(objects[0].start_time / SECTION_LENGTH) * SECTION_LENGTH

    def calculate_initial_strain(offset: float, current: ManiaDifficultyHitObject) -> float:
        """Strain.CalculateInitialStrain"""
        prev_start = objects[current.index - 1].start_time if current.index > 0 else 0.0
        dt = offset - prev_start
        return (
            _apply_decay(highest_individual_strain, dt, INDIVIDUAL_DECAY_BASE) +
            _apply_decay(overall_strain, dt, OVERALL_DECAY_BASE)
        )

    for obj in objects:
        while obj.start_time > current_section_end:
            strain_peaks.append(current_section_peak)
            current_section_peak = calculate_initial_strain(current_section_end, obj)
            current_section_end += SECTION_LENGTH

        # StrainDecaySkill.StrainValueAt
        current_strain *= math.pow(STRAIN_DECAY_BASE, obj.delta_time / 1000.0)

        # Strain.StrainValueOf
        nonlocal_col = obj.column

        individual_strains[nonlocal_col] = _apply_decay(
            individual_strains[nonlocal_col],
            obj.column_strain_time,
            INDIVIDUAL_DECAY_BASE,
        )
        individual_strains[nonlocal_col] += individual_strain_of(obj)

        # chord: Remain maximum value for multiple inputs(delta_time <= 1ms).
        if obj.delta_time <= 1:
            highest_individual_strain = max(
                highest_individual_strain,
                individual_strains[nonlocal_col],
            )
        else:
            highest_individual_strain = individual_strains[nonlocal_col]

        overall_strain = _apply_decay(overall_strain, obj.delta_time, OVERALL_DECAY_BASE)
        overall_strain += overall_strain_of(obj)

        strain_value_of = highest_individual_strain + overall_strain - current_strain
        current_strain += strain_value_of * SKILL_MULTIPLIER

        current_section_peak = max(current_strain, current_section_peak)

    # Save the last section peak.
    strain_peaks.append(current_section_peak)

    # DifficultyValue(): Add weights by decay_weight from the highest section peaks.
    peaks = sorted((p for p in strain_peaks if p > 0), reverse=True)
    difficulty = 0.0
    weight = 1.0
    for peak in peaks:
        difficulty += peak * weight
        weight *= DECAY_WEIGHT

    return difficulty * DIFFICULTY_MULTIPLIER


def calculate_star_rating(path: str, clock_rate: float = 1.0) -> float:
    total_columns, hit_objects = parse_osu(path, clock_rate)
    diff_objects = build_difficulty_objects(hit_objects, total_columns)
    return calculate_difficulty_value(diff_objects, total_columns)


def calculate_star_rating_from_objects(
    hit_objects: list[HitObject],
    total_columns: int,
) -> float:
    diff_objects = build_difficulty_objects(hit_objects, total_columns)
    return calculate_difficulty_value(diff_objects, total_columns)
