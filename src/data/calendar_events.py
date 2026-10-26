"""
Calendar event flags for India-specific pollution drivers: Diwali fireworks and
the Punjab/Haryana crop-residue (stubble) burning season.
"""

import pandas as pd

# Lakshmi Puja dates (Drik Panchang, New Delhi). Diwali is lunar, so it moves by
# weeks between years and must never be approximated by a fixed day of year.
# 2024 is listed as 31 Oct or 1 Nov depending on region; 31 Oct was the main
# fireworks night in North India, and the +/-3 day window covers both.
DIWALI_DATES = {
    2019: '2019-10-27',
    2020: '2020-11-14',
    2021: '2021-11-04',
    2022: '2022-10-24',
    2023: '2023-11-12',
    2024: '2024-10-31',
    2025: '2025-10-20',
}
DIWALI_WINDOW_DAYS = 3

# Stubble-burning season, inclusive (month, day) bounds
STUBBLE_SEASON_START = (10, 15)
STUBBLE_SEASON_END = (11, 30)


def diwali_table():
    return pd.DataFrame(
        {'year': list(DIWALI_DATES), 'diwali_date': pd.to_datetime(list(DIWALI_DATES.values()))}
    )


def is_diwali(dates, window_days=DIWALI_WINDOW_DAYS):
    """True within +/- window_days of that year's Diwali. Unlisted years raise rather than silently read False."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    missing = sorted(set(dates.dt.year) - set(DIWALI_DATES))
    if missing:
        raise ValueError(f"No Diwali date recorded for {missing}; add them to DIWALI_DATES")
    diwali = pd.to_datetime(dates.dt.year.map(DIWALI_DATES))
    return (dates - diwali).abs() <= pd.Timedelta(days=window_days)


def is_stubble_season(dates):
    dates = pd.to_datetime(pd.Series(dates))
    month_day = dates.dt.month * 100 + dates.dt.day
    start = STUBBLE_SEASON_START[0] * 100 + STUBBLE_SEASON_START[1]
    end = STUBBLE_SEASON_END[0] * 100 + STUBBLE_SEASON_END[1]
    return month_day.between(start, end)
