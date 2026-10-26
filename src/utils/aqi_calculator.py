"""
Central Pollution Control Board (CPCB) National Air Quality Index (NAQI) Calculator
Vectorized and single-value sub-index calculation according to Indian Ministry of Environment,
Forest and Climate Change (MoEFCC) guidelines.
"""

from typing import Dict, Optional, Union
import numpy as np

# Breakpoints according to CPCB NAQI guidelines
# Format: (C_low, C_high, I_low, I_high)
BREAKPOINTS = {
    'PM2.5': [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    'PM10': [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    'NO2': [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    'SO2': [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500),
    ],
    'CO': [  # mg/m3
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    'O3': [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
    'NH3': [
        (0.0, 200.0, 0, 50),
        (200.1, 400.0, 51, 100),
        (400.1, 800.0, 101, 200),
        (800.1, 1200.0, 201, 300),
        (1200.1, 1800.0, 301, 400),
        (1800.1, 2400.0, 401, 500),
    ]
}

CATEGORY_NAMES = {
    0: 'Good',
    1: 'Satisfactory',
    2: 'Moderate',
    3: 'Poor',
    4: 'Very Poor',
    5: 'Severe'
}

def calculate_sub_index(conc: float, pollutant: str) -> Optional[float]:
    """Calculates sub-index for a single pollutant concentration value.

    The CPCB table leaves small gaps between brackets (e.g. PM2.5 30.0 -> 30.1),
    which averaged concentrations such as 30.05 fall into. Each bracket therefore
    applies up to and including its upper bound, so every value maps to an index.
    """
    if conc is None or np.isnan(conc) or conc < 0:
        return np.nan

    if pollutant not in BREAKPOINTS:
        return np.nan

    brackets = BREAKPOINTS[pollutant]
    for c_low, c_high, i_low, i_high in brackets:
        if conc <= c_high:
            sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
            return round(max(sub, 0.0), 2)

    # Concentrations above the highest bracket are capped at 500 (CPCB scale maximum)
    c_low, c_high, i_low, i_high = brackets[-1]
    sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
    return min(round(sub, 2), 500.0)

def calculate_aqi_from_pollutants(pollutant_dict: Dict[str, float]) -> Optional[float]:
    """
    Computes overall CPCB AQI from a dict of pollutant concentrations.
    Rule: Requires at least 3 pollutants, with at least one particulate matter (PM2.5 or PM10).
    AQI = max(sub_indices)
    """
    sub_indices = {}
    for poll, val in pollutant_dict.items():
        if poll in BREAKPOINTS:
            sub = calculate_sub_index(val, poll)
            if not np.isnan(sub):
                sub_indices[poll] = sub
    
    # CPCB validity rule: no AQI from fewer than 3 pollutants, and one must be PM.
    # A sparse-reading fallback would label days CPCB itself leaves unrated.
    has_pm = ('PM2.5' in sub_indices) or ('PM10' in sub_indices)
    if len(sub_indices) >= 3 and has_pm:
        return max(sub_indices.values())

    return np.nan

def get_aqi_category(aqi: float) -> int:
    """
    Maps AQI value to category index (0 to 5):
    0: Good (0-50)
    1: Satisfactory (51-100)
    2: Moderate (101-200)
    3: Poor (201-300)
    4: Very Poor (301-400)
    5: Severe (401-500)
    """
    if np.isnan(aqi):
        return -1
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 200:
        return 2
    elif aqi <= 300:
        return 3
    elif aqi <= 400:
        return 4
    else:
        return 5
