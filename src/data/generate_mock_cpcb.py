"""
Synthetic / Mock CPCB Dataset Generator for Pipeline Verification
Generates realistic multi-city air quality data with seasonal patterns,
winter inversions, and extreme Diwali/stubble burning spikes to test the end-to-end pipeline.
"""

import os
import numpy as np
from pathlib import Path
from src.config import TARGET_CITIES, RAW_DATA_DIR

def generate_mock_cpcb_data(start_date="2021-01-01", end_date="2024-12-31", output_path=None):
    try:
        import pandas as pd
    except ImportError:
        print("Pandas is required to generate mock dataset. Please install pandas.")
        return None
    
    if output_path is None:
        out_dir = RAW_DATA_DIR / "cpcb"
        os.makedirs(out_dir, exist_ok=True)
        output_path = out_dir / "cpcb_india_2021_2024.csv"
        
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    records = []
    
    np.random.seed(42)
    
    print(f"Generating realistic mock CPCB data for {len(TARGET_CITIES)} cities over {n_days} days...")
    
    for city, meta in TARGET_CITIES.items():
        lat = meta['lat']
        is_igp = 'IGP' in meta['archetype'] or 'Biomass' in meta['archetype']
        is_coastal = 'Coastal' in meta['archetype']
        
        # Base pollutant concentration levels
        base_pm25 = 90.0 if is_igp else (45.0 if is_coastal else 60.0)
        base_pm10 = base_pm25 * 1.8
        base_no2 = 40.0
        base_so2 = 25.0 if 'Industrial' in meta['archetype'] else 12.0
        base_co = 1.2
        base_o3 = 35.0
        
        for i, dt in enumerate(dates):
            doy = dt.dayofyear
            # Winter peak (Nov-Jan: doy ~ 300 to 365, and 1 to 30)
            winter_factor = 1.0 + 1.2 * np.cos(2 * np.pi * (doy - 15) / 365) if is_igp else 1.0 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365)
            # Monsoon dip (Jul-Aug: doy ~ 180 to 240)
            monsoon_dip = 0.4 if (180 <= doy <= 240) else 1.0
            
            # Stubble burning episode (Late Oct - Mid Nov: doy 295 to 325)
            stubble_spike = 1.0
            if is_igp and (295 <= doy <= 325):
                stubble_spike = 1.8 + 0.8 * np.sin(np.pi * (doy - 295) / 30)
            
            # Diwali spike (around late Oct / early Nov, random trigger)
            diwali_spike = 1.0
            if (doy == 310 or doy == 311):
                diwali_spike = 2.5
            
            noise = np.random.normal(1.0, 0.15)
            
            pm25 = max(5.0, base_pm25 * winter_factor * monsoon_dip * stubble_spike * diwali_spike * noise)
            pm10 = max(10.0, base_pm10 * winter_factor * monsoon_dip * stubble_spike * diwali_spike * np.random.normal(1.0, 0.12))
            no2 = max(5.0, base_no2 * winter_factor * np.random.normal(1.0, 0.15))
            so2 = max(2.0, base_so2 * np.random.normal(1.0, 0.15))
            co = max(0.2, base_co * winter_factor * stubble_spike * np.random.normal(1.0, 0.1))
            o3 = max(5.0, base_o3 * (2.0 - winter_factor * 0.5) * np.random.normal(1.0, 0.2))
            
            records.append({
                'Date': dt.strftime('%Y-%m-%d'),
                'City': city,
                'PM2.5': round(pm25, 2),
                'PM10': round(pm10, 2),
                'NO2': round(no2, 2),
                'SO2': round(so2, 2),
                'CO': round(co, 2),
                'O3': round(o3, 2)
            })
            
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} records across {len(TARGET_CITIES)} cities.")
    print(f"Saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    generate_mock_cpcb_data()
