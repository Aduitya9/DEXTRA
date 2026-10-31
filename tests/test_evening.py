import numpy as np
import pandas as pd
import pytest

from src.data import evening
from src.utils.aqi_calculator import calculate_sub_index


def _station(pm25, pm10=None, day='2024-11-01'):
    """One station-day of hourly readings: pm25 and pm10 are 24 values each, or callables of the hour."""
    hours = pd.date_range(day, periods=24, freq='h')
    pm10 = pm10 if pm10 is not None else [2 * v for v in pm25]
    return pd.DataFrame({'timestamp': hours, 'site_id': 1, 'station': 's',
                         'PM2.5': np.asarray(pm25, dtype=float), 'PM10': np.asarray(pm10, dtype=float)})


def test_each_window_reads_its_own_hours():
    pm25 = [10.0] * 12 + [40.0] * 6 + [100.0] * 3 + [130.0] * 3      # morning, afternoon, evening
    out = evening.station_evening(_station(pm25)).iloc[0]
    assert out['pm25_morning'] == pytest.approx(10.0)
    assert out['pm25_evening'] == pytest.approx(115.0)               # 18:00-23:00
    assert out['pm25_late'] == pytest.approx(130.0)                  # 21:00-23:00


def test_a_window_with_too_few_hours_gives_no_value():
    pm25 = [50.0] * 24
    pm25[18:] = [80.0, 80.0, np.nan, np.nan, np.nan, np.nan]          # two of six evening hours
    out = evening.station_evening(_station(pm25)).iloc[0]
    assert np.isnan(out['pm25_evening']) and np.isnan(out['pm25_late'])
    assert out['pm25_morning'] == pytest.approx(50.0)


def test_a_station_day_whose_pm_contradicts_itself_loses_both():
    # PM2.5 above PM10 is impossible: the daily table blanks that station-day, so must this.
    out = evening.station_evening(_station([90.0] * 24, pm10=[40.0] * 24)).iloc[0]
    assert np.isnan(out['pm25_evening']) and np.isnan(out['pm10_evening'])


def test_stations_combine_by_their_median_and_the_rise_is_evening_minus_morning():
    frames = [evening.station_evening(_station([v] * 12 + [v + 60] * 12)) for v in (20.0, 30.0, 400.0)]
    city = evening.city_evening(frames, 'Patna').iloc[0]
    assert city['city'] == 'Patna'
    assert city['pm25_evening'] == pytest.approx(90.0)               # median of 80, 90, 460
    assert city['pm25_rise'] == pytest.approx(60.0)


def test_the_evening_index_is_the_larger_pm_sub_index():
    index = evening.evening_index([150.0, np.nan, np.nan], [100.0, 300.0, np.nan])
    assert index[0] == pytest.approx(max(calculate_sub_index(150.0, 'PM2.5'), calculate_sub_index(100.0, 'PM10')))
    assert index[1] == pytest.approx(calculate_sub_index(300.0, 'PM10'))
    assert np.isnan(index[2])


def test_era5_evening_is_read_in_indian_standard_time():
    # 18:00-23:00 IST is 12:30-17:30 UTC; mark exactly those hours.
    utc = pd.date_range('2024-11-01 00:30', periods=48, freq='h')
    ist_hour = (utc + pd.Timedelta('5h30min')).hour
    in_evening = (ist_hour >= 18) & (ist_hour <= 23)
    surface = pd.DataFrame({'city': 'Patna', 'time_utc': utc,
                            'blh': np.where(in_evening, 100.0, 900.0), 'wind_speed': np.where(in_evening, 1.0, 5.0)})
    out = evening.era5_evening(surface).set_index('date')
    assert out.loc[pd.Timestamp('2024-11-01'), 'blh_evening'] == pytest.approx(100.0)
    assert out.loc[pd.Timestamp('2024-11-01'), 'wind_evening'] == pytest.approx(1.0)


def test_era5_hours_missing_from_the_archive_are_filled_from_the_patch():
    utc = pd.date_range('2024-03-01 12:30', periods=6, freq='h')
    surface = pd.DataFrame({'city': 'Patna', 'time_utc': utc, 'blh': np.nan, 'wind_speed': 2.0})
    patch = pd.DataFrame({'city': 'Patna', 'time_utc': utc, 'blh': 250.0})
    out = evening.era5_evening(surface, patch).iloc[0]
    assert out['blh_evening'] == pytest.approx(250.0)


def test_the_table_has_every_calendar_day_and_every_feature():
    city = evening.city_evening([evening.station_evening(_station([50.0] * 24))], 'Patna')
    weather = pd.DataFrame({'city': ['Patna'], 'date': [pd.Timestamp('2024-11-02')],
                            'blh_evening': [300.0], 'wind_evening': [1.5]})
    table = evening.build_table([city], weather, '2024-10-31', '2024-11-03')
    assert table.columns.tolist() == ['city', 'date'] + evening.FEATURES
    assert table.date.dt.strftime('%m-%d').tolist() == ['10-31', '11-01', '11-02', '11-03']
    assert table.loc[table.date == '2024-11-01', 'pm25_evening'].item() == pytest.approx(50.0)
    assert table.loc[table.date == '2024-11-02', 'blh_evening'].item() == pytest.approx(300.0)
