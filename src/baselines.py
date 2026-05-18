"""Baseline test-selection policies for validation comparisons."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Mapping

from .test_library import RnDTest


FIXED_SEQUENCE = ("T5", "T1", "T4", "T2", "T3", "T6")


def random_selection(available_tests: Sequence[str], rng: random.Random) -> str:
    """Select a random available test."""

    if not available_tests:
        raise ValueError("No available tests")
    return rng.choice(list(available_tests))


def cheapest_first(available_tests: Sequence[str], tests: Mapping[str, RnDTest]) -> str:
    """Select the cheapest available test, breaking ties by test id."""

    if not available_tests:
        raise ValueError("No available tests")
    return min(available_tests, key=lambda test_id: (tests[test_id].cost, test_id))


def fixed_sequence(
    available_tests: Sequence[str],
    sequence: Sequence[str] = FIXED_SEQUENCE,
) -> str:
    """Select the first still-available test from a fixed diagnostic sequence."""

    if not available_tests:
        raise ValueError("No available tests")
    available = set(available_tests)
    for test_id in sequence:
        if test_id in available:
            return test_id
    return sorted(available_tests)[0]


def grid_all_tests(available_tests: Sequence[str]) -> str:
    """Select tests in stable id order for an exhaustive all-tests baseline."""

    if not available_tests:
        raise ValueError("No available tests")
    return sorted(available_tests)[0]

