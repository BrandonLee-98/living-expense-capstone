import pytest
import pandas as pd
from app import calc_geo_distance_miles, calculate_twia_surcharge, find_nearest_arbitrage_zip

# ==========================================
# 1. TEST HAVERSINE SPATIAL MATH
# ==========================================
def test_calc_geo_distance_miles():
    """Verify the Great-Circle distance formula calculates real-world miles accurately."""
    # Coordinates for Houston (29.7604, -95.3698) to Dallas (32.7767, -96.7970)
    # Actual driving distance is ~240 miles, true "as the crow flies" radius is ~225 miles.
    dist = calc_geo_distance_miles(29.7604, -95.3698, 32.7767, -96.7970)
    assert 220 < dist < 230, f"Distance calculated as {dist:.1f}, expected ~225 miles"
    
    # Test identical coordinates (Should be exactly 0.0 miles)
    dist_zero = calc_geo_distance_miles(29.7604, -95.3698, 29.7604, -95.3698)
    assert dist_zero == 0.0

# ==========================================
# 2. TEST DETERMINISTIC TWIA SURCHARGE
# ==========================================
def test_calculate_twia_surcharge():
    """Verify the actuarial rules for mandatory coastal insurance are strictly enforced."""
    # Scenario A: ZIP code is in a TWIA mandatory zone (Monthly View)
    assert calculate_twia_surcharge(twia_flag=1, multiplier=1) == 125.0
    
    # Scenario B: ZIP code is in a TWIA mandatory zone (Annualized View)
    assert calculate_twia_surcharge(twia_flag=1, multiplier=12) == 1500.0
    
    # Scenario C: ZIP code is NOT in a coastal zone
    assert calculate_twia_surcharge(twia_flag=0, multiplier=1) == 0.0

# ==========================================
# 3. TEST SPATIAL ARBITRAGE ROUTING
# ==========================================
def test_find_nearest_arbitrage_zip():
    """Verify the engine correctly abandons high-tax zones for cheaper alternatives."""
    # Mock a mini-database of two ZIP codes
    mock_db = pd.DataFrame({
        'zip_code': ['77591', '77598'],
        'latitude': [29.3838, 29.5312],
        'longitude': [-94.9027, -95.1444],
        'twia_mandatory': [1, 0],       # 77591 has expensive coastal insurance, 77598 does not
        'water_mud_flag': [0, 0],
        'pct_area_100yr_flood': [42.0, 1.0]
    })
    
    target_row = mock_db.iloc[0] # User searches for 77591 (High-Hazard)
    
    # Run the arbitrage engine
    best_alt, arbitrage_type = find_nearest_arbitrage_zip(mock_db, target_row)
    
    # Assert the engine successfully routed the user to 77598 (Webster) to avoid TWIA
    assert best_alt['zip_code'] == '77598'
    assert arbitrage_type == "cost_governance"