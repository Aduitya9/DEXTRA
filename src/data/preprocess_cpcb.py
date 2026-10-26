"""
CPCB Raw Air Quality Data Preprocessing Pipeline
Cleans and standardizes raw CPCB data for the 15 target Indian cities.
Handles station aggregation (median), outlier clipping, and NAQI recomputation.
"""

import os
from pathlib import Path
from src.config import TARGET_CITIES, POLLUTANTS, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.aqi_calculator import calculate_aqi_from_pollutants, get_aqi_category

PHYSICAL_LIMITS = {
    'PM2.5': (0.0, 999.0),
    'PM10': (0.0, 1500.0),
    'NO2': (0.0, 600.0),
    'SO2': (0.0, 1000.0),
    'CO': (0.0, 50.0),
    'O3': (0.0, 500.0),
    'NH3': (0.0, 1000.0),
}

def clean_cpcb_dataframe(df):
    """
    Standard cleaning routine applied to any CPCB pandas DataFrame.
    Expects columns: 'Date', 'City', and pollutant columns (PM2.5, PM10, etc.)
    """
    import pandas as pd
    import numpy as np

    # Standardize column names
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)
    
    # Filter target cities
    city_names = set(TARGET_CITIES.keys())
    if 'City' in df.columns:
        df = df[df['City'].isin(city_names)].copy()
    elif 'city' in df.columns:
        df = df[df['city'].isin(city_names)].copy()
        df = df.rename(columns={'city': 'City'})
    
    # Ensure Date is parsed
    date_col = 'Date' if 'Date' in df.columns else 'date'
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Outlier clipping based on physical sensor limits
    for poll, (low, high) in PHYSICAL_LIMITS.items():
        if poll in df.columns:
            df[poll] = pd.to_numeric(df[poll], errors='coerce')
            df.loc[(df[poll] < low) | (df[poll] > high), poll] = np.nan
    
    # Aggregate across stations for each city and date using median
    avail_pollutants = [p for p in POLLUTANTS if p in df.columns]
    agg_dict = {p: 'median' for p in avail_pollutants}
    
    city_daily = df.groupby(['City', 'Date']).agg(agg_dict).reset_index()
    
    # Fill small consecutive gaps (<= 3 days) per city using ffill/bfill
    cleaned_rows = []
    for city, group in city_daily.groupby('City'):
        group = group.set_index('Date').sort_index()
        # Reindex to continuous daily calendar
        full_idx = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')
        group = group.reindex(full_idx)
        group['City'] = city
        for p in avail_pollutants:
            group[p] = group[p].ffill(limit=3).bfill(limit=1)
        cleaned_rows.append(group.reset_index().rename(columns={'index': 'Date'}))
    
    cleaned_df = pd.concat(cleaned_rows, ignore_index=True)
    
    # Recompute CPCB AQI and AQI Category
    def compute_row_aqi(row):
        p_dict = {p: row[p] for p in avail_pollutants if not np.isnan(row[p])}
        return calculate_aqi_from_pollutants(p_dict)
    
    print("Recomputing CPCB NAQI for all city-day records...")
    cleaned_df['AQI_calculated'] = cleaned_df.apply(compute_row_aqi, axis=1)
    cleaned_df['AQI_category'] = cleaned_df['AQI_calculated'].apply(get_aqi_category)
    
    return cleaned_df

def process_cpcb_pipeline(raw_file_path=None):
    import pandas as pd
    
    if raw_file_path is None:
        cpcb_dir = RAW_DATA_DIR / "cpcb"
        csv_files = list(cpcb_dir.glob("*.csv"))
        if not csv_files:
            print(f"No raw CPCB CSV files found in {cpcb_dir}.")
            print("Please place the downloaded Kaggle/CPCB CSV in data/raw/cpcb/.")
            return None
        raw_file_path = csv_files[0]
    
    print(f"Loading raw CPCB data from: {raw_file_path}...")
    df = pd.read_csv(raw_file_path, low_memory=False)
    cleaned_df = clean_cpcb_dataframe(df)
    
    out_dir = PROCESSED_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = out_dir / "city_daily_pollutants.parquet"
    out_csv = out_dir / "city_daily_pollutants.csv"
    
    try:
        cleaned_df.to_parquet(out_parquet, index=False)
        print(f"Saved processed dataset to: {out_parquet}")
    except Exception:
        cleaned_df.to_csv(out_csv, index=False)
        print(f"Saved processed dataset to: {out_csv}")
        
    return cleaned_df

if __name__ == '__main__':
    process_cpcb_pipeline()
