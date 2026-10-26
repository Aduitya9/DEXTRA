import math

import pytest

from src.utils.aqi_calculator import (
    calculate_aqi_from_pollutants,
    calculate_sub_index,
    get_aqi_category,
)


@pytest.mark.parametrize("conc, expected", [
    (0.0, 0.0),
    (30.0, 50.0),
    (60.0, 100.0),
    (90.0, 200.0),
    (120.0, 300.0),
    (250.0, 400.0),
    (500.0, 500.0),
])
def test_pm25_bracket_boundaries(conc, expected):
    assert calculate_sub_index(conc, "PM2.5") == pytest.approx(expected)


def test_concentration_above_top_bracket_is_capped_at_500():
    assert calculate_sub_index(900.0, "PM2.5") == 500.0


def test_value_in_gap_between_brackets_is_not_nan():
    # 30.05 sits between the 30.0 upper and 30.1 lower PM2.5 breakpoints
    sub = calculate_sub_index(30.05, "PM2.5")
    assert not math.isnan(sub)
    assert 50.0 < sub < 51.0


@pytest.mark.parametrize("conc, pollutant", [
    (-1.0, "PM2.5"),
    (float("nan"), "PM2.5"),
    (10.0, "UNKNOWN"),
])
def test_invalid_inputs_return_nan(conc, pollutant):
    assert math.isnan(calculate_sub_index(conc, pollutant))


def test_aqi_is_max_sub_index_with_three_pollutants_including_pm():
    aqi = calculate_aqi_from_pollutants({"PM2.5": 100.0, "NO2": 50.0, "SO2": 10.0})
    assert aqi == pytest.approx(233.78, abs=0.01)


@pytest.mark.parametrize("pollutants", [
    {"PM2.5": 100.0},                                   # one pollutant
    {"PM2.5": 100.0, "NO2": 50.0},                      # two pollutants
    {"PM2.5": 100.0, "NO2": float("nan"), "SO2": 10.0}, # NaN does not count
    {"NO2": 50.0, "SO2": 10.0, "O3": 30.0},             # three, but no PM
])
def test_cpcb_minimum_pollutant_rule(pollutants):
    assert math.isnan(calculate_aqi_from_pollutants(pollutants))


@pytest.mark.parametrize("aqi, category", [
    (50, 0), (51, 1), (100, 1), (101, 2), (200, 2),
    (201, 3), (300, 3), (301, 4), (400, 4), (401, 5), (500, 5),
])
def test_category_boundaries(aqi, category):
    assert get_aqi_category(aqi) == category


def test_nan_aqi_has_no_category():
    assert get_aqi_category(float("nan")) == -1
