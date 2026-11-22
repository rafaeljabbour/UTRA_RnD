"""Shared test data."""


def finished_metrics(**kw):
    m = {
        "state": "finished",
        "elapsed_s": 120.0,
        "time_limit_s": 480.0,
        "within_time_limit": True,
        "distance_ft": 14.0,
        "steps": 14,
        "unique_cells": 10,
        "speed_steps_per_min": 7.0,
        "turns": 5,
        "drove": True,
        "turned": True,
        "lz_reached_s": 30.0,
        "collision_tracking": True,
        "minor_collisions": 0,
        "major_collisions": 0,
        "assists": 0,
        "target_b": "B2",
        "b_coverage": {"B1": 0.0, "B2": 0.8, "B3": 0.0, "B4": 0.0},
    }
    m.update(kw)
    return m
