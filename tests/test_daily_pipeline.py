import numpy as np
import pandas as pd
import pytest

from src.config import HAZARDOUS_AQI_THRESHOLD, TRAIN_END_DATE, VAL_END_DATE
from src.data import assemble, daily_cpcb, era5_features, fire_features


def _hourly(date='2021-01-15', **values):
    """One station-day of hourly readings; any pollutant given as a scalar is constant."""
    index = pd.date_range(f'{date} 00:00', periods=24, freq='h')
    frame = pd.DataFrame({'timestamp': index, 'site_id': 5247, 'station': 't_t_nagar'})
    for name, value in values.items():
        frame[name] = value if hasattr(value, '__len__') else [value] * 24
    return frame


# --------------------------------------------------------------------------- CPCB daily rules


def test_24h_pollutants_use_the_mean():
    hourly = _hourly(**{'PM2.5': [10.0] * 12 + [30.0] * 12})
    daily = daily_cpcb.station_daily(hourly)
    assert daily['PM2.5'].iloc[0] == pytest.approx(20.0)


def test_co_and_o3_use_the_highest_8_hour_mean_not_the_daily_mean():
    # Calm all day except an 8-hour block; values stay inside each instrument's range.
    co = [0.5] * 8 + [6.0] * 8 + [0.5] * 8
    o3 = [10.0] * 8 + [120.0] * 8 + [10.0] * 8
    daily = daily_cpcb.station_daily(_hourly(CO=co, O3=o3))
    assert daily['CO'].iloc[0] == pytest.approx(6.0)
    assert daily['O3'].iloc[0] == pytest.approx(120.0)
    assert np.mean(co) == pytest.approx(2.333, abs=1e-3)   # what a plain daily mean would have given


def test_a_day_with_fewer_than_16_valid_hours_has_no_value():
    sparse = [20.0] * 15 + [np.nan] * 9
    assert pd.isna(daily_cpcb.station_daily(_hourly(**{'PM2.5': sparse}))['PM2.5'].iloc[0])

    just_enough = [20.0] * 16 + [np.nan] * 8
    assert daily_cpcb.station_daily(_hourly(**{'PM2.5': just_enough}))['PM2.5'].iloc[0] == 20.0


def test_readings_outside_instrument_range_are_discarded_before_averaging():
    with_fault = [20.0] * 23 + [5000.0]        # 5000 ug/m3 is a sensor fault, not air
    daily = daily_cpcb.station_daily(_hourly(**{'PM2.5': with_fault}))
    assert daily['PM2.5'].iloc[0] == pytest.approx(20.0)


def test_negative_readings_are_discarded():
    daily = daily_cpcb.station_daily(_hourly(**{'PM2.5': [-5.0] * 4 + [20.0] * 20}))
    assert daily['PM2.5'].iloc[0] == pytest.approx(20.0)


# --------------------------------------------------------------------------- city median


def test_city_value_is_the_median_across_stations_and_counts_them():
    stations = [daily_cpcb.station_daily(_hourly(**{'PM2.5': v})) for v in (10.0, 20.0, 90.0)]
    city = daily_cpcb.city_daily(stations, 'Bhopal')
    assert city['PM2.5'].iloc[0] == pytest.approx(20.0)   # median, so the outlier station cannot drag it
    assert city['n_stations'].iloc[0] == 3
    assert city['city'].iloc[0] == 'Bhopal'


def test_station_with_no_valid_day_is_not_counted():
    good = daily_cpcb.station_daily(_hourly(**{'PM2.5': 20.0}))
    dead = daily_cpcb.station_daily(_hourly(**{'PM2.5': [np.nan] * 24}))
    city = daily_cpcb.city_daily([good, dead], 'Guwahati')
    assert city['n_stations'].iloc[0] == 1
    assert city['PM2.5'].iloc[0] == pytest.approx(20.0)


def test_aqi_follows_cpcb_rules_on_daily_values():
    station = daily_cpcb.station_daily(_hourly(**{'PM2.5': 100.0, 'PM10': 180.0, 'NO2': 50.0}))
    city = daily_cpcb.add_aqi(daily_cpcb.city_daily([station], 'Delhi'))
    assert city['AQI'].iloc[0] == pytest.approx(233.78, abs=0.01)   # driven by PM2.5
    assert city['AQI_category'].iloc[0] == 3                        # Poor

    too_few = daily_cpcb.add_aqi(daily_cpcb.city_daily(
        [daily_cpcb.station_daily(_hourly(**{'PM2.5': 100.0}))], 'Delhi'))
    assert pd.isna(too_few['AQI'].iloc[0])


def test_calendar_reindex_makes_missing_days_explicit():
    station = daily_cpcb.station_daily(_hourly(date='2021-01-03', **{'PM2.5': 20.0}))
    city = daily_cpcb.city_daily([station], 'Patna')
    full = daily_cpcb.reindex_to_calendar(city, '2021-01-01', '2021-01-05')
    assert len(full) == 5
    assert full['n_stations'].tolist() == [0, 0, 1, 0, 0]


# --------------------------------------------------------------------------- ERA5 features


def test_wind_direction_uses_a_circular_mean():
    # Arithmetic mean of 350 and 10 is 180 -- the opposite direction.
    assert era5_features.circular_mean_degrees([350.0, 10.0]) == pytest.approx(0.0, abs=1e-6)
    assert era5_features.circular_mean_degrees([80.0, 100.0]) == pytest.approx(90.0, abs=1e-6)
    assert np.isnan(era5_features.circular_mean_degrees([np.nan, np.nan]))


def test_utc_hours_are_grouped_into_indian_calendar_days():
    # 19:00 UTC is 00:30 IST the next day, so it belongs to the following day.
    frame = pd.DataFrame({'time_utc': pd.to_datetime(['2024-01-01 18:00', '2024-01-01 19:00'])})
    dates = era5_features.to_ist(frame)['date']
    assert dates.iloc[0] == pd.Timestamp('2024-01-01')
    assert dates.iloc[1] == pd.Timestamp('2024-01-02')


def test_surface_daily_aggregates_each_variable_appropriately():
    # Three UTC days so the middle IST day is a complete 24 hours.
    index = pd.date_range('2024-01-01 00:00', periods=72, freq='h')
    surface = pd.DataFrame({
        'city': 'Delhi', 'time_utc': index,
        'blh': [100.0] * 36 + [900.0] * 36,
        'temp2m': 20.0, 'precip': 0.5, 'wind_speed': 1.0, 'wind_dir': 90.0,
    })
    surface.loc[surface.time_utc == pd.Timestamp('2024-01-02 12:00'), 'wind_speed'] = 20.0
    daily = era5_features.surface_daily(surface)
    middle = daily[daily.date == pd.Timestamp('2024-01-02')].iloc[0]

    assert middle['hours'] == 24                              # a full IST day
    assert middle['blh_min'] == pytest.approx(100.0)          # minimum kept too
    assert middle['precip'] == pytest.approx(12.0)            # summed: 0.5 x 24
    # mean wind, not max: one windy hour must not make a calm day look ventilated
    assert middle['wind_speed'] < 2.0


def test_blh_anomaly_is_measured_against_the_citys_own_normal():
    dates = pd.date_range('2024-01-01', periods=40, freq='D')
    daily = pd.DataFrame({'city': 'Delhi', 'date': dates, 'blh': [500.0] * 39 + [100.0]})
    out = era5_features.add_blh_anomaly(daily, window_days=30)
    assert out['blh_anomaly'].iloc[-1] < -300    # the stagnant day stands out
    assert out['blh_anomaly'].iloc[20] == pytest.approx(0.0, abs=1e-9)


def test_inversion_index_uses_each_citys_own_level_and_keeps_the_strongest_morning():
    times = pd.to_datetime(['2024-01-01 00:00', '2024-01-01 03:00'])
    pressure = pd.DataFrame({'city': 'Bengaluru', 'time_utc': times,
                             'temp_850hPa': [18.0, 16.0], 'temp_925hPa': [99.0, 99.0]})
    surface = pd.DataFrame({'city': 'Bengaluru', 'time_utc': times, 'temp2m': [12.0, 14.0]})
    out = era5_features.inversion_index(pressure, surface, {'Bengaluru': 850})
    assert out['inv_idx'].iloc[0] == pytest.approx(6.0)    # max(18-12, 16-14), not the 925 column
    assert out['inversion_level_hpa'].iloc[0] == 850


# --------------------------------------------------------------------------- fire features


def test_only_confident_vegetation_fires_are_counted():
    fires = pd.DataFrame({'confidence': [90, 50, 90], 'type': [0, 0, 2]})
    kept = fire_features.filter_detections(fires)
    assert len(kept) == 1 and kept.index.tolist() == [0]


def test_fires_are_counted_in_the_right_ring_around_each_city():
    # Amritsar is ~400 km from Delhi: inside the 500 km ring, outside the 100 km one.
    fires = pd.DataFrame({
        'latitude': [31.63, 28.65], 'longitude': [74.87, 77.25],
        'acq_date': pd.to_datetime(['2024-11-05', '2024-11-05']), 'acq_time': [530, 530],
        'confidence': [90, 90], 'type': [0, 0], 'frp': [100.0, 7.0],
    })
    daily = fire_features.city_fire_daily(fires, {'Delhi': {'lat': 28.6139, 'lon': 77.2090}})
    row = daily.iloc[0]
    assert row['fire_count_100km'] == 1       # only the nearby one
    assert row['fire_count_500km'] == 2       # both
    assert row['frp_500km'] == pytest.approx(107.0)


def test_detection_times_are_converted_to_indian_days():
    # 20:00 UTC on the 4th is 01:30 IST on the 5th.
    fires = pd.DataFrame({'latitude': [28.65], 'longitude': [77.25],
                          'acq_date': pd.to_datetime(['2024-11-04']), 'acq_time': [2000],
                          'confidence': [90], 'type': [0], 'frp': [5.0]})
    daily = fire_features.city_fire_daily(fires, {'Delhi': {'lat': 28.6139, 'lon': 77.2090}})
    assert daily['date'].iloc[0] == pd.Timestamp('2024-11-05')


def test_days_without_detections_are_zero_not_missing():
    fires = pd.DataFrame({'latitude': [28.65], 'longitude': [77.25],
                          'acq_date': pd.to_datetime(['2024-11-05']), 'acq_time': [530],
                          'confidence': [90], 'type': [0], 'frp': [5.0]})
    cities = {'Delhi': {'lat': 28.6139, 'lon': 77.2090}}
    full = fire_features.fill_calendar(fire_features.city_fire_daily(fires, cities),
                                       '2024-11-04', '2024-11-06', cities)
    assert full['fire_count_500km'].tolist() == [0, 1, 0]
    assert full['fire_count_500km'].notna().all()


# --------------------------------------------------------------------------- assembly


def _daily_frame(cities, start='2019-01-01', end='2025-12-31', aqi=120.0):
    rows = []
    for city in cities:
        dates = pd.date_range(start, end, freq='D')
        rows.append(pd.DataFrame({'city': city, 'date': dates, 'AQI': aqi,
                                  'SO2': 10.0, 'NO2': 40.0, 'n_stations': 3}))
    return pd.concat(rows, ignore_index=True)


def test_splits_never_share_a_city_or_a_date_range():
    daily = _daily_frame(['Delhi', 'Kanpur'])
    daily['split'] = assemble.assign_split(daily)
    assert set(daily.loc[daily.city == 'Kanpur', 'split']) == {'test_unseen'}

    delhi = daily[daily.city == 'Delhi']
    assert delhi.loc[delhi.split == 'train', 'date'].max() <= pd.Timestamp(TRAIN_END_DATE)
    assert delhi.loc[delhi.split == 'val', 'date'].min() > pd.Timestamp(TRAIN_END_DATE)
    assert delhi.loc[delhi.split == 'val', 'date'].max() <= pd.Timestamp(VAL_END_DATE)
    assert delhi.loc[delhi.split == 'test_seen', 'date'].min() > pd.Timestamp(VAL_END_DATE)
    assert 'unassigned' not in set(daily['split'])


def test_unseen_city_static_features_see_only_the_support_window():
    daily = _daily_frame(['Delhi', 'Kanpur'])
    # Kanpur's SO2 changes sharply after its first 30 days; the feature must not notice.
    late = (daily.city == 'Kanpur') & (daily.date >= daily.date.min() + pd.Timedelta(days=30))
    daily.loc[late, 'SO2'] = 500.0

    static = assemble.city_static_features(daily, support_days=30).set_index('city')
    assert static.loc['Kanpur', 'static_window_days'] == 30
    assert static.loc['Kanpur', 'industrial_index'] == pytest.approx(0.25)   # 10/40, the early value
    assert static.loc['Delhi', 'static_window_days'] > 1800                  # full training period


def test_scalers_are_fitted_on_training_rows_only():
    daily = _daily_frame(['Delhi'])
    daily['PM2.5'] = 50.0
    daily['split'] = assemble.assign_split(daily)
    daily.loc[daily.split == 'test_seen', 'PM2.5'] = 900.0   # an extreme the scaler must not see

    params = assemble.fit_scalers(daily[daily.split == 'train'])
    assert params['pollutants_minmax_per_city']['Delhi']['PM2.5']['max'] == pytest.approx(50.0)

    scaled = assemble.apply_scalers(daily, params)
    assert scaled.loc[scaled.split == 'test_seen', 'PM2.5'].max() > 1.0   # out of range, not clipped away


def test_unseen_city_falls_back_to_the_pooled_training_range():
    daily = _daily_frame(['Delhi', 'Kanpur'])
    daily['PM2.5'] = 50.0
    daily['split'] = assemble.assign_split(daily)
    params = assemble.fit_scalers(daily[daily.split == 'train'])
    assert 'Kanpur' not in params['pollutants_minmax_per_city']

    scaled = assemble.apply_scalers(daily, params)
    assert scaled.loc[scaled.city == 'Kanpur', 'PM2.5'].notna().all()


def test_episodes_are_contiguous_runs_above_the_hazardous_threshold():
    dates = pd.date_range('2024-11-01', periods=10, freq='D')
    aqi = [100, 250, 260, 240, 100, 100, 300, 100, 100, 100]
    daily = pd.DataFrame({'city': 'Delhi', 'date': dates, 'AQI': aqi})
    episodes = assemble.extreme_episodes(daily, threshold=HAZARDOUS_AQI_THRESHOLD, min_days=2)
    assert len(episodes) == 1                       # the single 300 day is too short
    row = episodes.iloc[0]
    assert row['duration_days'] == 3 and row['peak_aqi'] == 260
    assert row['start_date'] == pd.Timestamp('2024-11-02')


def test_calendar_features_cover_the_india_specific_events():
    daily = pd.DataFrame({'city': 'Delhi', 'date': pd.to_datetime(['2023-11-12', '2023-06-15'])})
    out = assemble.add_calendar_features(daily)
    assert out['is_diwali'].tolist() == [1, 0]
    assert out['is_stubble_season'].tolist() == [1, 0]
    assert out['day_of_year'].iloc[0] == 316
