import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import math
import sqlite3
import io
import openai
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Living Expense Estimation Engine",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    .stSidebar { background-color: #1A202C; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED RESOURCE LOADING & DATABASE
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'.")
    loaded_model = joblib.load(model_path)
    # Warm up XGBoost C++ thread pool once at startup to eliminate cold-start latency
    warmup_df = pd.DataFrame([{
        'sqft': 1500,
        'occupants': 2,
        'pct_area_100yr_flood': 1.0,
        'twia_mandatory': 0,
        'water_mud_flag': 0,
        'houston_cpi_food_scalar': 1.0,
        'effective_tax_rate_pct': 2.2,
        'five_year_pop_growth_pct': 0.0
    }])
    loaded_model.predict(warmup_df)
    return loaded_model

@st.cache_data(show_spinner=False)
def load_all_db_data(db_path: str):
    """Loads all data from SQLite into Pandas for the Spatial Arbitrage Engine."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at '{db_path}'.")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM zip_data", conn)
    conn.close()
    
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    return df

@st.cache_data(show_spinner=False)
def query_single_zip(db_path: str, zip_query: str):
    """Executes a cached, parameterized SQL query to fetch a specific ZIP code."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at '{db_path}'.")
        
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM zip_data WHERE zip_code = ?"
    df = pd.read_sql(query, conn, params=(zip_query,))
    conn.close()
    
    if not df.empty and 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    return df

# --- HELPER: ROBUST IMAGE LOADER FOR ML TAB ---
def render_image(filename: str, caption: str, base_dir: str):
    """Searches common directories for the image and renders it, or shows an explicit error."""
    search_paths = [
        os.path.join(base_dir, filename),                                    
        os.path.join(base_dir, "figures", filename),                 
        os.path.join(base_dir, "extracted_figures", "figures", filename) 
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            st.image(path, caption=caption)
            return
            
    st.error(f"⚠️ **Image Missing:** Could not find `{filename}`. Please place this image in the same folder as `app.py` or inside a `figures` folder.")

# ==========================================
# 3. SPATIAL ARBITRAGE MATH ENGINE
# ==========================================
def calc_geo_distance_miles(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lat2): 
        return 999.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
    return 2 * math.asin(math.sqrt(a)) * 3958.8

def find_nearest_arbitrage_zip(df, target_row):
    target_zip = target_row['zip_code']
    target_twia = int(target_row.get('twia_mandatory', 0))
    target_mud = int(target_row.get('water_mud_flag', 0))
    target_flood = float(target_row.get('pct_area_100yr_flood', 0.0))
    target_lat = target_row.get('latitude', 0.0)
    target_lon = target_row.get('longitude', 0.0)

    candidates = df[df['zip_code'] != target_zip].copy()
    if candidates.empty: 
        return target_row, "none"

    candidates['geo_dist_miles'] = candidates.apply(
        lambda r: calc_geo_distance_miles(target_lat, target_lon, r.get('latitude', 0.0), r.get('longitude', 0.0)), axis=1
    )

    if target_twia == 1:
        ideal = candidates[(candidates.get('twia_mandatory', 0) == 0) & (candidates.get('water_mud_flag', 0) == 0)]
        if not ideal.empty: 
            return ideal.sort_values(by=['geo_dist_miles']).iloc[0], "cost_governance"
    if target_mud == 1:
        no_mud = candidates[(candidates.get('water_mud_flag', 0) == 0) & (candidates.get('twia_mandatory', 0) == 0)]
        if not no_mud.empty: 
            return no_mud.sort_values(by=['geo_dist_miles']).iloc[0], "mud_only"
    if target_flood > 5.0:
        lower_flood = candidates[(candidates.get('twia_mandatory', 0) == 0) & (candidates.get('pct_area_100yr_flood', target_flood) < target_flood)]
        if not lower_flood.empty: 
            return lower_flood.sort_values(by=['geo_dist_miles']).iloc[0], "flood_mitigation"

    return target_row, "optimal"

# ==========================================
# 4. DETERMINISTIC TWIA SURCHARGE ENGINE
# ==========================================
def calculate_twia_surcharge(twia_flag, multiplier):
    return 125.0 * multiplier if twia_flag == 1 else 0.0

# ==========================================
# 4.5. EXECUTIVE PDF REPORT GENERATOR & AI
# ==========================================
def generate_ai_narrative(city, total, food, utils, twia, is_optimal, alt_zip=None, alt_city=None, alt_total=0.0, horizon_label="Monthly"):
    """Calls OpenAI to generate a short, insightful summary of the costs, including arbitrage comparisons."""
    try:
        if "OPENAI_API_KEY" not in st.secrets:
            return "AI Summary unavailable: OPENAI_API_KEY secret is not configured in Streamlit Cloud."
            
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        base_prompt = f"""
        Write a concise, 3-sentence executive summary for a living expense report in {city}.
        The total {horizon_label.lower()} non-mortgage cost is ${total:,.2f}. 
        Groceries are ${food:,.2f}, Utilities are ${utils:,.2f}, and Coastal Insurance is ${twia:,.2f}.
        Keep it professional, analytical, and mention if the coastal insurance is a significant burden. 
        """
        if not is_optimal:
             base_prompt += f"Also explicitly mention that migrating to nearby ZIP {alt_zip} ({alt_city}) would lower total non-mortgage costs to ${alt_total:,.2f}. Do not use bullet points or welcome greetings."
        else:
             base_prompt += "Also state that this is an optimal governance zone with no recommended arbitrage migrations. Do not use bullet points or welcome greetings."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": base_prompt}],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Summary unavailable at this time. (Error: {e})"

def generate_pdf_report(zip_code, city, archetype, pop_growth, total_cost, food, utils, twia, hoa, 
                        is_optimal, alt_zip, alt_city, alt_total, alt_food, alt_utils, alt_twia, alt_hoa, ai_summary="", horizon_label="Monthly"):
    """Generates a premium, enterprise-styled PDF executive summary with dynamic layout sizing."""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # --- 1. Dark Header Block ---
    p.setFillColorRGB(0.1, 0.12, 0.17) 
    p.rect(0, height - 100, width, 100, fill=1, stroke=0)
    
    p.setFillColorRGB(1, 1, 1) 
    p.setFont("Helvetica-Bold", 22)
    p.drawString(40, height - 45, "Non-Mortgage Living Expense Estimate")
    
    p.setFont("Helvetica", 12)
    p.setFillColorRGB(0.7, 0.8, 0.9) 
    p.drawString(40, height - 70, f"Target Profile: ZIP {zip_code} | {city} ({horizon_label} View)")
    
    # --- 2. Classification & Growth ---
    p.setFillColorRGB(0, 0, 0)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(40, height - 130, "Regional Classification:")
    p.setFont("Helvetica", 12)
    p.drawString(200, height - 130, archetype)
    
    p.setFont("Helvetica-Bold", 12)
    p.drawString(40, height - 150, "5-Year Pop Growth:")
    p.setFont("Helvetica", 12)
    
    if pop_growth > 0:
        p.setFillColorRGB(0.1, 0.6, 0.1) 
    else:
        p.setFillColorRGB(0.8, 0.1, 0.1) 
    p.drawString(200, height - 150, f"{pop_growth:+.1f}%")
    p.setFillColorRGB(0, 0, 0) 
    
    # --- 3. Cost Breakdown Table ---
    y_pos = height - 190
    p.setLineWidth(1.5)
    p.setStrokeColorRGB(0.2, 0.2, 0.2)
    p.line(40, y_pos, width - 40, y_pos)
    
    y_pos -= 20
    p.setFont("Helvetica-Bold", 14)
    p.drawString(40, y_pos, f"Itemized {horizon_label} Cost Breakdown")
    
    # Table Headers
    p.setFont("Helvetica-Bold", 11)
    p.drawRightString(width - 150, y_pos, f"Target ({zip_code})")
    if not is_optimal:
        p.drawRightString(width - 40, y_pos, "Arbitrage Alt.")
        p.setFont("Helvetica-Oblique", 9)
        p.setFillColorRGB(0.4, 0.4, 0.4)
        p.drawRightString(width - 40, y_pos - 12, f"ZIP {alt_zip}")
        p.setFillColorRGB(0, 0, 0)
        
    y_pos -= 25
    p.setLineWidth(0.5)
    p.setStrokeColorRGB(0.7, 0.7, 0.7)
    p.line(40, y_pos, width - 40, y_pos)
    y_pos -= 20
    
    # Rows with Zebra Striping
    items = [
        ("Estimated Groceries & Food", food, alt_food),
        ("Estimated Utilities (Power/Water/Gas)", utils, alt_utils),
        ("Mandatory TWIA Coastal Surcharge", twia, alt_twia),
        ("Optional HOA / Maintenance", hoa, alt_hoa)
    ]
    
    p.setFont("Helvetica", 11)
    for i, (label, tgt_val, alt_val) in enumerate(items):
        if i % 2 == 0:
            p.setFillColorRGB(0.97, 0.97, 0.97) 
            p.rect(40, y_pos - 7, width - 80, 24, fill=1, stroke=0)
        
        p.setFillColorRGB(0, 0, 0)
        p.drawString(45, y_pos, label)
        p.drawRightString(width - 150, y_pos, f"${tgt_val:,.2f}")
        if not is_optimal:
            p.drawRightString(width - 40, y_pos, f"${alt_val:,.2f}")
        y_pos -= 20
        
    # Total Row (Dark Bar)
    y_pos -= 5
    p.setFillColorRGB(0.1, 0.12, 0.17) 
    p.rect(40, y_pos - 20, width - 80, 30, fill=1, stroke=0)
    
    p.setFillColorRGB(1, 1, 1)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(45, y_pos - 10, f"TOTAL ESTIMATED {horizon_label.upper()} COST")
    p.drawRightString(width - 150, y_pos - 10, f"${total_cost:,.2f}")
    if not is_optimal:
        p.drawRightString(width - 40, y_pos - 10, f"${alt_total:,.2f}")
        
    y_pos -= 45 
    
    # Savings Highlight Callout
    p.setFillColorRGB(0, 0, 0)
    if not is_optimal:
        savings = total_cost - alt_total
        box_width = 240
        
        p.setFillColorRGB(0.9, 0.98, 0.9) 
        p.setStrokeColorRGB(0.2, 0.8, 0.2) 
        p.rect(width - 40 - box_width, y_pos - 22, box_width, 34, fill=1, stroke=1)
        
        p.setFillColorRGB(0.1, 0.5, 0.1) 
        p.setFont("Helvetica-Bold", 12)
        p.drawRightString(width - 55, y_pos - 10, f"Est. {horizon_label} Savings: ${savings:,.2f}")
    else:
        p.setFont("Helvetica-Oblique", 10)
        p.setFillColorRGB(0.4, 0.4, 0.4)
        p.drawRightString(width - 40, y_pos - 10, "Optimal Governance Zone: No arbitrage required.")
        
    p.setFillColorRGB(0, 0, 0)
    
    # --- 4. AI Narrative Summary (Dynamic Blockquote) ---
    if ai_summary:
        y_pos -= 60
        p.setFont("Helvetica-Bold", 14)
        p.drawString(40, y_pos, "AI Executive Summary")
        
        y_pos -= 20
        
        words = ai_summary.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + word) > 95:
                lines.append(current_line.strip())
                current_line = word + " "
            else:
                current_line += word + " "
        if current_line:
            lines.append(current_line.strip())
            
        box_height = (len(lines) * 16) + 16 
        
        p.setFillColorRGB(0.95, 0.96, 0.98) 
        p.rect(40, y_pos - box_height, width - 80, box_height, fill=1, stroke=0)
        
        p.setFillColorRGB(0.2, 0.4, 0.7) 
        p.rect(40, y_pos - box_height, 4, box_height, fill=1, stroke=0)
        
        p.setFillColorRGB(0.1, 0.1, 0.1)
        textobject = p.beginText(55, y_pos - 18) 
        textobject.setFont("Helvetica", 10)
        textobject.setLeading(16) 
        
        for line in lines:
            textobject.textLine(line)
        p.drawText(textobject)

    # --- 5. Disclaimer Footer ---
    p.setFont("Helvetica-Oblique", 8)
    p.setFillColorRGB(0.6, 0.6, 0.6)
    p.drawCentredString(width / 2, 40, "Note: This estimate relies on predictive machine learning models and spatial proximity math.")
    p.drawCentredString(width / 2, 30, "Results should be used for informational planning purposes only. Generated by Living Expense Engine.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ==========================================
# 5. MAIN APPLICATION
# ==========================================
def main():
    st.title("🏙️ Non-Mortgage Living Expense Estimation Engine")
    st.markdown("### Harris, Fort Bend, and Galveston Counties")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILENAME = os.path.join(BASE_DIR, "xgboost_cost_model.joblib")
    DB_FILENAME = os.path.join(BASE_DIR, "living_costs.db")

    try:
        model = load_model(MODEL_FILENAME)
        df = load_all_db_data(DB_FILENAME) 
    except Exception as e:
        st.error(f"Critical Initialization Error: {e}")
        st.stop()

    # --- UI SIDEBAR ---
    st.sidebar.header("1. Property Parameters")
    zip_query = st.sidebar.text_input("Enter 5-Digit ZIP Code", value="77598", max_chars=5).strip()
    sqft_input = st.sidebar.slider("Property Square Footage", min_value=800, max_value=4000, value=1500, step=50)
    occupants_input = st.sidebar.slider("Household Occupants", min_value=1, max_value=6, value=2)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Housing Governance & HOA")
    include_hoa = st.sidebar.toggle("Enable Subdivision / HOA Fees", value=False, help="Toggle on if targeting housing within a managed HOA community.")

    st.sidebar.markdown("---")
    st.sidebar.header("3. Household Budget Profiles")

    tier_display_names = [
        "Thrifty (SNAP Baseline / Maximum Savings)",
        "Low-Cost (Budget-Conscious / Sales & Coupons)",
        "Moderate-Cost (Standard Middle-Class Average)",
        "Liberal (Premium / Specialty & Organic Focus)"
    ]
    selected_tier_name = st.sidebar.selectbox("USDA Grocery Plan Tier", tier_display_names, index=2)
    view_mode = st.sidebar.radio("Display Horizon", ["Monthly ($/mo)", "Annualized ($/yr)"], horizontal=True)
    time_multiplier = 12 if view_mode == "Annualized ($/yr)" else 1
    is_monthly = (time_multiplier == 1)
    horizon_label = "Monthly" if is_monthly else "Annual"

    # --- SQL DATABASE LOOKUP PIPELINE ---
    selected_row = query_single_zip(DB_FILENAME, zip_query)
    
    if selected_row.empty:
        st.error(f"❌ ZIP Code '{zip_query}' not found in the regional tri-county dataset.")
        st.stop()
    data = selected_row.iloc[0]

    city_name = data.get('city_name', 'Houston Area')
    county_name = data.get('county_name', 'Unknown')
    archetype = data.get('cluster_archetype_name', 'Unclassified Region')
    pop_growth_pct = float(data.get('five_year_pop_growth_pct', 0.0))

    # Dynamic profile summary tailored to the user's selected ZIP code
    gov_desc = "suburban MUD utility tax governance" if int(data.get('water_mud_flag', 0)) == 1 else "standard municipal utility infrastructure"
    coastal_desc = "within a mandatory TWIA coastal windstorm boundary" if int(data.get('twia_mandatory', 0)) == 1 else "outside coastal TWIA windstorm surcharges"
    flood_val = float(data.get('pct_area_100yr_flood', 0.0))
    profile_note = f"ZIP {zip_query} ({county_name} County) operates under {gov_desc}, sits {coastal_desc}, and has {flood_val:.1f}% 100-year floodplain exposure."

    st.sidebar.markdown("---")
    st.sidebar.header("4. ZIP Classification Profile")
    st.sidebar.markdown(f"""
    <div style="font-size: 14px; line-height: 1.5;">
    <b>ZIP Code:</b> {zip_query}<br>
    <b>City / County:</b> {city_name} ({county_name})<br>
    <b>Archetype:</b> {archetype}<br><br>
    <b>5-Year Population Growth Rate:</b> {pop_growth_pct:+.1f}%<br>
    <i>{profile_note}</i>
    </div>
    """, unsafe_allow_html=True)

    # --- ML DATA PREPARATION ---
    features = {
        'sqft': sqft_input,
        'occupants': occupants_input,
        'pct_area_100yr_flood': float(data.get('pct_area_100yr_flood', 0.0)),
        'twia_mandatory': int(data.get('twia_mandatory', 0)),
        'water_mud_flag': int(data.get('water_mud_flag', 0)),
        'houston_cpi_food_scalar': float(data.get('houston_cpi_food_scalar', 1.0)),
        'effective_tax_rate_pct': float(data.get('effective_tax_rate_pct', 2.2)),
        'five_year_pop_growth_pct': pop_growth_pct
    }

    expected_features = ['sqft', 'occupants', 'pct_area_100yr_flood', 'twia_mandatory',
                         'water_mud_flag', 'houston_cpi_food_scalar', 'effective_tax_rate_pct',
                         'five_year_pop_growth_pct']
    input_df = pd.DataFrame([features])[expected_features]

    # --- PREDICTION EXECUTION ---
    with st.spinner("Running XGBoost Inference..."):
        predictions = model.predict(input_df)[0]

    ml_food = predictions[0]
    ml_elec = predictions[1]
    ml_water = predictions[2]
    ml_gas = predictions[3]
    ml_hoa = predictions[4]

    tier_scalars = {
        "Thrifty (SNAP Baseline / Maximum Savings)": 0.75,
        "Low-Cost (Budget-Conscious / Sales & Coupons)": 0.88,
        "Moderate-Cost (Standard Middle-Class Average)": 1.00,
        "Liberal (Premium / Specialty & Organic Focus)": 1.20
    }
    grocery_multiplier = tier_scalars[selected_tier_name]

    pred_food = (ml_food * grocery_multiplier) * time_multiplier
    pred_utils = (ml_elec + ml_water + ml_gas) * time_multiplier
    pred_hoa = (ml_hoa * time_multiplier) if include_hoa else 0.0
    pred_twia = calculate_twia_surcharge(features['twia_mandatory'], time_multiplier)

    total_cost = pred_food + pred_utils + pred_hoa + pred_twia

    # --- ARBITRAGE EXECUTION ---
    best_alt_data, arbitrage_type = find_nearest_arbitrage_zip(df, data)
    alt_zip = best_alt_data['zip_code']
    alt_city = best_alt_data.get('city_name', 'Nearby')
    dist_miles = calc_geo_distance_miles(data.get('latitude'), data.get('longitude'), best_alt_data.get('latitude'), best_alt_data.get('longitude'))

    is_optimal_target = (alt_zip == zip_query) or (arbitrage_type == "optimal")

    alt_features = features.copy()
    alt_features.update({
        'pct_area_100yr_flood': float(best_alt_data.get('pct_area_100yr_flood', 0.0)),
        'twia_mandatory': int(best_alt_data.get('twia_mandatory', 0)),
        'water_mud_flag': int(best_alt_data.get('water_mud_flag', 0)),
        'houston_cpi_food_scalar': float(best_alt_data.get('houston_cpi_food_scalar', 1.0)),
        'effective_tax_rate_pct': float(best_alt_data.get('effective_tax_rate_pct', features['effective_tax_rate_pct'])),
        'five_year_pop_growth_pct': float(best_alt_data.get('five_year_pop_growth_pct', 0.0))
    })
    alt_df = pd.DataFrame([alt_features])[expected_features]
    alt_preds = model.predict(alt_df)[0]

    alt_pred_food = (alt_preds[0] * grocery_multiplier) * time_multiplier
    alt_pred_utils = (alt_preds[1] + alt_preds[2] + alt_preds[3]) * time_multiplier
    alt_pred_hoa = (alt_preds[4] * time_multiplier) if include_hoa else 0.0
    alt_pred_twia = calculate_twia_surcharge(int(alt_features['twia_mandatory']), time_multiplier)
    
    alt_total_cost = alt_pred_food + alt_pred_utils + alt_pred_hoa + alt_pred_twia

    # --- PDF EXPORT COMPONENT (AI ENHANCED) - MOVED TO SIDEBAR ---
    st.sidebar.markdown("---")
    st.sidebar.header("5. Export Component")
    if st.sidebar.button("📄 Generate Executive Report PDF"):
        with st.spinner("Generating AI Narrative & Building PDF..."):
            narrative = generate_ai_narrative(
                city_name, total_cost, pred_food, pred_utils, pred_twia, 
                is_optimal_target, alt_zip, alt_city, alt_total_cost, horizon_label=horizon_label
            )
            
            pdf_buffer = generate_pdf_report(
                zip_query, city_name, archetype, pop_growth_pct, 
                total_cost, pred_food, pred_utils, pred_twia, pred_hoa,
                is_optimal_target, alt_zip, alt_city, alt_total_cost, alt_pred_food, alt_pred_utils, alt_pred_twia, alt_pred_hoa,
                ai_summary=narrative, horizon_label=horizon_label
            )
            
            st.sidebar.success("Report Generated Successfully!")
            st.sidebar.download_button(
                label="📥 Download PDF",
                data=pdf_buffer,
                file_name=f"AI_Expense_Report_{zip_query}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # --- RENDER DASHBOARD HEADER ---
    if features['twia_mandatory'] == 1:
        st.warning(f"🌊 **Coastal Zone Designation ({city_name}):** Located within TWIA coastal policy eligibility boundary.")
    if features['pct_area_100yr_flood'] > 20.0:
        st.error(f"⚠️ **High Flood Hazard Alert ({city_name}):** {features['pct_area_100yr_flood']:.1f}% of this ZIP area lies within the 100-year floodplain.")

    st.subheader(f"Cost Breakdown for ZIP Code: {zip_query} | {city_name} ({archetype})")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Occupancy Cost", f"${total_cost:,.2f}")
    with col2:
        st.metric(f"Food ({selected_tier_name.split(' ')[0]})", f"${pred_food:,.2f}")
    with col3:
        st.metric("Utilities", f"${pred_utils:,.2f}")
    with col4:
        st.metric("TWIA Surcharge", f"${pred_twia:,.2f}" if features['twia_mandatory'] == 1 else "$0.00")
    with col5:
        st.metric("5-Yr Pop Growth", f"{pop_growth_pct:+.1f}%")

    st.markdown("---")

    # --- DEEP DIVE TABS ---
    tab1, tab2, tab3, tab6, tab4, tab5, tab7, tab8 = st.tabs([
        "💰 Itemized Breakdown & Comparison",
        "🌊 Flood & MUD Risk",
        "⚡ Utility Grid Detail",
        "🎯 Budget Recommender",
        "📊 ML Explainability & Diagnostics",
        "🗺️ Project Execution Roadmap",
        "🧠 AI Architecture & Framework",
        "🧪 Live System Diagnostics"
    ])

    with tab1:
        st.subheader(f"Target vs. Alternative Component Comparison — ZIP {zip_query} ({city_name})")
        if is_optimal_target:
            st.success(f"🌟 **Optimal Non-Mortgage Governance:** ZIP {zip_query} already operates on optimal municipal utilities without MUD taxes or coastal surcharges. No alternative migration required.")
            comp_rows = ["Groceries & Food", "Utilities (Power/Water/Gas)"]
            target_vals = [f"${pred_food:,.2f}", f"${pred_utils:,.2f}"]
            if features['twia_mandatory'] == 1:
                comp_rows.append("Mandatory TWIA Coastal Surcharge")
                target_vals.append(f"${pred_twia:,.2f}")
            if include_hoa:
                comp_rows.append("Optional Subdivision Dues / HOA")
                target_vals.append(f"${pred_hoa:,.2f}")
            comp_rows.extend(["Total Estimated Cost", "5-Year Pop Growth Rate", "100-Yr Flood Exposure", "MUD Governance"])
            target_vals.extend([f"${total_cost:,.2f}", f"{pop_growth_pct:+.1f}%", f"{features['pct_area_100yr_flood']:.1f}%", "Suburban MUD" if features['water_mud_flag']==1 else "Standard Municipal"])

            comp_df = pd.DataFrame({
                "Cost / Metric": comp_rows,
                f"Target & Optimal: {zip_query} ({city_name})": target_vals
            })
            st.table(comp_df)
        else:
            st.info(f"💡 **Arbitrage Found:** Evaluating nearby **ZIP {alt_zip} ({alt_city})** — ~{dist_miles:.1f} miles away. Total Cost: ${alt_total_cost:,.2f}")
            comp_rows = ["Groceries & Food", "Utilities (Power/Water/Gas)"]
            target_vals = [f"${pred_food:,.2f}", f"${pred_utils:,.2f}"]
            alt_vals = [f"${alt_pred_food:,.2f}", f"${alt_pred_utils:,.2f}"]

            if features['twia_mandatory'] == 1 or alt_features['twia_mandatory'] == 1:
                comp_rows.append("Mandatory TWIA Coastal Surcharge")
                target_vals.append(f"${pred_twia:,.2f}")
                alt_vals.append(f"${alt_pred_twia:,.2f}")

            if include_hoa:
                comp_rows.append("Optional Subdivision Dues / HOA")
                target_vals.append(f"${pred_hoa:,.2f}")
                alt_vals.append(f"${alt_pred_hoa:,.2f}")

            comp_rows.extend(["Total Estimated Cost", "5-Year Pop Growth Rate", "100-Yr Flood Exposure", "MUD Governance"])
            target_vals.extend([f"${total_cost:,.2f}", f"{pop_growth_pct:+.1f}%", f"{features['pct_area_100yr_flood']:.1f}%", "Suburban MUD" if features['water_mud_flag']==1 else "Standard Municipal"])
            alt_vals.extend([f"${alt_total_cost:,.2f}", f"{float(best_alt_data.get('five_year_pop_growth_pct', 0.0)):+.1f}%", f"{alt_features['pct_area_100yr_flood']:.1f}%", "Suburban MUD" if alt_features['water_mud_flag']==1 else "Standard Municipal"])

            comp_df = pd.DataFrame({
                "Cost / Metric": comp_rows,
                f"Target: {zip_query} ({city_name})": target_vals,
                f"Alternative: {alt_zip} ({alt_city})": alt_vals
            })
            st.table(comp_df)
    
        # --- 5-YEAR COST & ARBITRAGE PROJECTION ENGINE ---
        import altair as alt

        st.divider()
        st.subheader(f"📈 5-Year Non-Mortgage Expense Projection — ZIP {zip_query} ({city_name})")
        st.caption(
            f"Projects how your estimated non-mortgage living costs for ZIP {zip_query} ({city_name}) could grow over the next 5 years "
            "based on regional food inflation, local utility demand, and coastal windstorm trends."
        )

        scenario = st.selectbox(
            "Select Cost Growth Scenario",
            [
                "Baseline Tri-County Forecast (3.2% Food | 3.8% Utils | 7.5% Coastal Ins.)",
                "Conservative Low-Inflation (2.5% Food | 3.0% Utils | 4.5% Coastal Ins.)",
                "High-Inflation & Coastal Stress Test (4.5% Food | 5.5% Utils | 11.0% Coastal Ins.)"
            ],
            index=0
        )

        if "Conservative" in scenario:
            r_food, r_utils, r_twia, r_hoa = 0.025, 0.030, 0.045, 0.025
        elif "Stress Test" in scenario:
            r_food, r_utils, r_twia, r_hoa = 0.045, 0.055, 0.110, 0.050
        else:
            r_food, r_utils, r_twia, r_hoa = 0.032, 0.038, 0.075, 0.035

        # Modulate utility growth slightly by local 5-year population growth (capped at +1.5%)
        pop_demand_adj = min(max(pop_growth_pct, 0.0) * 0.0002, 0.015)
        r_utils_target = r_utils + pop_demand_adj

        unit_suffix = "/mo" if is_monthly else "/yr"
        y_step = 100 if is_monthly else 1000

        base_food = float(pred_food)
        base_utils = float(pred_utils)
        base_twia = float(pred_twia)
        base_hoa = float(pred_hoa)

        years = [f"Year {y}" for y in range(1, 6)]
        target_series = []
        for y in range(5):
            yr_cost = (
                base_food * ((1 + r_food) ** y)
                + base_utils * ((1 + r_utils_target) ** y)
                + base_twia * ((1 + r_twia) ** y)
                + base_hoa * ((1 + r_hoa) ** y)
            )
            target_series.append(round(float(yr_cost), 2))

        target_col = f"Target {zip_query} ({city_name})".replace(":", "")
        pct_change = ((target_series[-1] / target_series[0]) - 1) * 100 if target_series[0] > 0 else 0.0

        # Row 1: Target ZIP Year 1 vs. Year 5 Cost
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.metric(
                f"Target ({zip_query}) Year 1 {horizon_label} Cost",
                f"${target_series[0]:,.2f}{unit_suffix}"
            )
        with p_col2:
            st.metric(
                f"Target ({zip_query}) Year 5 Projected {horizon_label} Cost",
                f"${target_series[-1]:,.2f}{unit_suffix}",
                delta=f"+${target_series[-1] - target_series[0]:,.2f}{unit_suffix} (+{pct_change:.1f}%)",
                delta_color="inverse"
            )

        if is_optimal_target:
            proj_df = pd.DataFrame({
                "Year": years,
                target_col: target_series
            })
            all_vals = target_series
        else:
            alt_pop_growth = float(best_alt_data.get('five_year_pop_growth_pct', 0.0))
            r_utils_alt = r_utils + min(max(alt_pop_growth, 0.0) * 0.0002, 0.015)

            alt_base_food = float(alt_pred_food)
            alt_base_utils = float(alt_pred_utils)
            alt_base_twia = float(alt_pred_twia)
            alt_base_hoa = float(alt_pred_hoa)

            alt_series = []
            for y in range(5):
                yr_alt = (
                    alt_base_food * ((1 + r_food) ** y)
                    + alt_base_utils * ((1 + r_utils_alt) ** y)
                    + alt_base_twia * ((1 + r_twia) ** y)
                    + alt_base_hoa * ((1 + r_hoa) ** y)
                )
                alt_series.append(round(float(yr_alt), 2))

            alt_col = f"Alternative {alt_zip} ({alt_city})".replace(":", "")
            alt_pct_change = ((alt_series[-1] / alt_series[0]) - 1) * 100 if alt_series[0] > 0 else 0.0
            yr1_gap = target_series[0] - alt_series[0]
            yr5_gap = target_series[-1] - alt_series[-1]

            # Row 2: Identical Year 1 vs. Year 5 Cost layout for the Alternative ZIP
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                st.metric(
                    f"Alternative ({alt_zip}) Year 1 {horizon_label} Cost",
                    f"${alt_series[0]:,.2f}{unit_suffix}"
                )
            with a_col2:
                st.metric(
                    f"Alternative ({alt_zip}) Year 5 Projected {horizon_label} Cost",
                    f"${alt_series[-1]:,.2f}{unit_suffix}",
                    delta=f"+${alt_series[-1] - alt_series[0]:,.2f}{unit_suffix} (+{alt_pct_change:.1f}%)",
                    delta_color="inverse"
                )

            # Monthly (or Annual) Savings displayed cleanly underneath both cost rows
            st.success(
                f"💰 **Estimated {horizon_label} Savings ({zip_query} vs. {alt_zip}):** "
                f"Year 1: **${yr1_gap:,.2f}{unit_suffix}**  |  "
                f"Year 5 Projected: **${yr5_gap:,.2f}{unit_suffix}**"
            )

            proj_df = pd.DataFrame({
                "Year": years,
                target_col: target_series,
                alt_col: alt_series
            })
            all_vals = target_series + alt_series

        # Build zoomed Altair line chart with exact $100 (monthly) or $1,000 (annual) y-axis increments
        y_min = math.floor(min(all_vals) / y_step) * y_step
        y_max = math.ceil(max(all_vals) / y_step) * y_step
        if y_max <= y_min:
            y_max = y_min + y_step
        tick_values = list(range(int(y_min), int(y_max) + y_step, y_step))

        proj_long = proj_df.melt(id_vars=["Year"], var_name="Location", value_name="Cost")
        proj_chart = (
            alt.Chart(proj_long)
            .mark_line(point=alt.OverlayMarkDef(size=75), strokeWidth=3)
            .encode(
                x=alt.X("Year:N", title="Projection Timeline", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "Cost:Q",
                    title=f"Estimated {horizon_label} Cost ($)",
                    scale=alt.Scale(domain=[y_min, y_max], zero=False),
                    axis=alt.Axis(values=tick_values, format="$,.0f")
                ),
                color=alt.Color("Location:N", legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[
                    alt.Tooltip("Year:N", title="Timeline"),
                    alt.Tooltip("Location:N", title="Area"),
                    alt.Tooltip("Cost:Q", title=f"{horizon_label} Cost", format="$,.2f")
                ]
            )
            .properties(height=340)
        )
        st.altair_chart(proj_chart, use_container_width=True)

        if is_optimal_target:
            table_df = pd.DataFrame({
                "Projection Year": years,
                f"Target ({zip_query} - {city_name}) {horizon_label}": [f"${v:,.2f}{unit_suffix}" for v in target_series]
            })
        else:
            table_df = pd.DataFrame({
                "Projection Year": years,
                f"Target ({zip_query} - {city_name}) {horizon_label}": [f"${v:,.2f}{unit_suffix}" for v in target_series],
                f"Alternative ({alt_zip} - {alt_city}) {horizon_label}": [f"${v:,.2f}{unit_suffix}" for v in alt_series],
                f"Projected {horizon_label} Savings": [f"${t - a:,.2f}{unit_suffix}" for t, a in zip(target_series, alt_series)]
            })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # --- TRANSPARENCY & METHODOLOGY DISCLOSURE (PLAIN ENGLISH) ---
        with st.expander("ℹ️ How We Calculate These Numbers (Transparency Guide)"):
            st.markdown(f"""
            #### Two-Step Forecast Approach
            Our machine learning model was trained on current housing and utility data across {len(df)} ZIP codes in Harris, Fort Bend, and Galveston counties—not historical multi-year inflation data. To give you a realistic 5-year outlook without guessing beyond what the model learned, we break the calculation into two parts:
            * **Year 1 (Today's Baseline):** Calculated directly by our AI model using your home size (`{sqft_input:,} sqft`), household size (`{occupants_input}`), and local ZIP `{zip_query}` rules.
            * **Years 2–5 (Annual Cost Growth):** Applies realistic yearly inflation rates to each non-mortgage expense category based on Greater Houston and coastal Texas benchmarks.

            #### Local Population Growth Adjustment (`{zip_query}`)
            Fast-growing neighborhoods have to build new water lines, substations, and power grid capacity, which pushes utility bills up slightly faster over time.
            * **How it works:** We add a small **+0.02%** utility adjustment for every **1%** of 5-year population growth in your ZIP code (capped at **+1.50%** max).
            * **Your ZIP (`{zip_query}`):** Population growth is `{pop_growth_pct:+.1f}%`, which adds **`+{pop_demand_adj*100:.2f}%`** to the base utility rate (`{r_utils*100:.1f}%`), giving an adjusted utility growth rate of **`{r_utils_target*100:.2f}% per year`**.
            """)

            assumptions_df = pd.DataFrame({
                "Expense Category": [
                    "Groceries & Food",
                    "Utilities (Power, Water, Gas)",
                    "Coastal Windstorm Insurance (TWIA)",
                    "HOA / Subdivision Dues (Non-Mortgage)"
                ],
                "Conservative": ["2.5% / yr", "3.0% / yr", "4.5% / yr", "2.5% / yr"],
                "Baseline (Default)": ["3.2% / yr", "3.8% / yr", "7.5% / yr", "3.5% / yr"],
                "Stress Test": ["4.5% / yr", "5.5% / yr", "11.0% / yr", "5.0% / yr"],
                "Where These Rates Come From": [
                    "U.S. Bureau of Labor Statistics (BLS) Houston grocery price trends & USDA food plans (~2.5% historical average; 3.2% accounts for recent supply-chain price swings).",
                    "BLS Houston energy inflation data, Texas residential electricity rates (EIA), and local grid upgrade fees (CenterPoint / TNMP).",
                    "Texas Dept. of Insurance (TDI) filings for coastal catastrophe zones (Galveston & eastern Harris County). Texas law caps standard TWIA rate hikes at 10% per year (15% after major storms), so 7.5% reflects a middle-ground increase.",
                    "BLS Houston household operations trends covering common-area landscaping, community pools/parks, street lighting, and neighborhood management fees."
                ]
            })
            st.table(assumptions_df)
            st.caption(
                "**Note:** Year 1 is based on current local data, while Years 2–5 are planning estimates designed to help you compare how non-mortgage costs grow across different ZIP codes rather than guaranteed future prices."
            )
    
    with tab2:
        st.subheader(f"Actuarial Flood Hazard & Municipal Water Governance — ZIP {zip_query} ({city_name})")
        st.markdown("""
        This tab evaluates structural flood exposure and water service governance for the selected ZIP code. 
        It details historical National Flood Insurance Program (NFIP) claim payouts, 100-year floodplain area percentages, 
        and Municipal Utility District (MUD) unincorporated tax governance rules to assess long-term infrastructure reliability.
        """)
        st.markdown("---")
        
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.metric("100-Yr Flood Zone Area", f"{float(data.get('pct_area_100yr_flood', 0.0)):.1f}%")
        with f_col2:
            st.metric("Water Base Monthly Fee", f"${float(data.get('water_base_monthly_fee', 0.0)):,.2f}")
        with f_col3:
            st.metric("Water Rate per 1k Gal", f"${float(data.get('water_rate_per_1k_gal', 0.0)):,.2f}")

        st.markdown("---")
        st.subheader("Historical FEMA Flood Claims & Payouts")
        st.markdown("""
        The following metrics reflect historical data from the National Flood Insurance Program (NFIP). This data helps quantify the actual severity and frequency of past flood events in this ZIP code:
        * **Historical NFIP Flood Claims:** The total volume of approved flood damage claims. A higher count indicates recurring neighborhood flooding.
        * **Average Building Payout:** The average compensation paid for structural repairs (e.g., walls, foundations, electrical). Higher dollar amounts point to deeper, more destructive floodwaters.
        * **Average Contents Payout:** The average compensation paid for lost personal property (e.g., furniture, appliances, clothing) inside the home.
        """)
        
        st.markdown(f"**Historical NFIP Flood Claims:** {data.get('nfip_total_claims', 'N/A')}")
        st.markdown(f"**Average Building Payout:** ${float(data.get('nfip_avg_building_claim', 0.0)):,.2f}")
        st.markdown(f"**Average Contents Payout:** ${float(data.get('nfip_avg_contents_claim', 0.0)):,.2f}")
        st.markdown("---")
        if features['water_mud_flag'] == 1:
            st.warning("⚠️ **Suburban MUD Territory Notice:** This ZIP code is predominantly located in unincorporated territory served by Suburban Municipal Utility Districts (MUDs), resulting in additional property tax levies.")
        else:
            st.success("✅ **Standard Municipal Jurisdiction:** Serviced directly by municipal infrastructure. No additional MUD property tax levies apply.")

    with tab3:
        st.subheader(f"Utility Delivery Grid Operators — ZIP {zip_query} ({city_name})")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown(f"### ⚡ Electricity")
            st.markdown(f"- **Transmission & Delivery Utility:** `{data.get('elec_provider', 'Unknown TDU')}`")
            st.markdown(f"- **Base Monthly Fee:** `${float(data.get('elec_base_monthly_fee', 0.0)):.2f}`")
            st.markdown(f"- **Delivery Surcharge per kWh:** `${float(data.get('elec_delivery_rate_per_kwh', 0.0)):.4f}`")
        with g_col2:
            st.markdown(f"### 🔥 Natural Gas")
            st.markdown(f"- **Gas Utility Distribution Provider:** `{data.get('gas_provider', 'Unknown TDU')}`")
            st.markdown(f"- **Base Monthly Fee:** `${float(data.get('gas_base_monthly_fee', 0.0)):.2f}`")
            st.markdown(f"- **Distribution Charge per CCF:** `${float(data.get('gas_distribution_rate_per_ccf', 0.0)):.4f}`")

    with tab6:
        st.subheader("Regional Budget Recommender")
        st.markdown(f"Find alternative ZIP codes within a specific radius of **{zip_query} ({city_name})** that fit your maximum occupancy budget.")
        
        default_budget = 1500.0 if is_monthly else 18000.0
        max_cap = 10000.0 if is_monthly else 120000.0
        step_val = 100.0 if is_monthly else 1000.0

        with st.form("budget_recommender_form"):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                max_budget = st.number_input(f"Maximum {horizon_label} Budget ($)", min_value=500.0, max_value=max_cap, value=default_budget, step=step_val)
            with r_col2:
                search_radius = st.slider("Search Radius (Miles)", min_value=1, max_value=50, value=10)
                
            submit_search = st.form_submit_button("Search Affordable Areas")

        if submit_search:
            with st.spinner("Scanning regional ZIP codes..."):
                target_lat = data.get('latitude', 0.0)
                target_lon = data.get('longitude', 0.0)
                df_search = df.copy()
                df_search['dist_to_target'] = df_search.apply(
                    lambda r: calc_geo_distance_miles(target_lat, target_lon, r.get('latitude', 0.0), r.get('longitude', 0.0)), axis=1
                )
                nearby_df = df_search[(df_search['dist_to_target'] <= search_radius) & (df_search['zip_code'] != zip_query)].copy()

                if nearby_df.empty:
                    st.warning(f"No alternative ZIP codes found within {search_radius} miles of {zip_query}.")
                else:
                    batch_input = pd.DataFrame({
                        'sqft': sqft_input,
                        'occupants': occupants_input,
                        'pct_area_100yr_flood': nearby_df['pct_area_100yr_flood'].fillna(0.0),
                        'twia_mandatory': nearby_df['twia_mandatory'].fillna(0),
                        'water_mud_flag': nearby_df['water_mud_flag'].fillna(0),
                        'houston_cpi_food_scalar': nearby_df['houston_cpi_food_scalar'].fillna(1.0),
                        'effective_tax_rate_pct': nearby_df['effective_tax_rate_pct'].fillna(2.2),
                        'five_year_pop_growth_pct': nearby_df['five_year_pop_growth_pct'].fillna(0.0)
                    })

                    batch_preds = model.predict(batch_input[expected_features])
                    
                    food_costs = (batch_preds[:, 0] * grocery_multiplier) * time_multiplier
                    utils_costs = (batch_preds[:, 1] + batch_preds[:, 2] + batch_preds[:, 3]) * time_multiplier
                    hoa_costs = (batch_preds[:, 4] * time_multiplier) if include_hoa else 0.0
                    twia_costs = batch_input['twia_mandatory'].apply(lambda x: calculate_twia_surcharge(int(x), time_multiplier))

                    nearby_df['Estimated Total'] = food_costs + utils_costs + hoa_costs + twia_costs
                    
                    affordable_df = nearby_df[nearby_df['Estimated Total'] <= max_budget]
                    
                    if affordable_df.empty:
                        st.error(f"No ZIP codes within {search_radius} miles met the {horizon_label.lower()} budget of ${max_budget:,.2f}.")
                    else:
                        st.success(f"Found {len(affordable_df)} affordable ZIP codes within {search_radius} miles of {zip_query}!")
                        display_cols = ['zip_code', 'city_name', 'cluster_archetype_name', 'dist_to_target', 'Estimated Total']
                        cost_col_name = f"Total {horizon_label} Cost"
                        res_df = affordable_df[display_cols].sort_values('Estimated Total').rename(columns={
                            'zip_code': 'ZIP',
                            'city_name': 'City',
                            'cluster_archetype_name': 'Archetype',
                            'dist_to_target': 'Distance (Miles)',
                            'Estimated Total': cost_col_name
                        })
                        
                        res_df[cost_col_name] = res_df[cost_col_name].apply(lambda x: f"${x:,.2f}")
                        res_df['Distance (Miles)'] = res_df['Distance (Miles)'].apply(lambda x: f"{x:.1f}")
                        st.dataframe(res_df, use_container_width=True, hide_index=True)

    with tab4:
        st.header("📊 ML Explainability & Diagnostics")
        st.markdown("""
        Welcome to the **"Under the Hood"** section of the engine. Here, we decode the XGBoost model's decision-making 
        process using SHapley Additive exPlanations (SHAP) and evaluate its statistical accuracy against our simulated testing dataset.
        """)
        st.divider()

        st.subheader("1. Household Evaluation Matrix & Baseline Analysis")
        st.markdown("""
        **Household Evaluation Matrix (Scenario Breakdown)**
        * **What it tracks:** A testing dataset of 2,790 simulated household scenarios combining 279 unique ZIP codes with random variations of home sizes (800 to 4,000 sqft) and household sizes (1 to 6 occupants). 
        * **What it means:** This provides a statistical breakdown that confirms the simulated parameters worked exactly as intended. It demonstrates how the engine behaves across a wide spectrum of realistic living situations to ensure accurate testing.
        """)

        matrix_data = {
            "Feature": ["sqft", "occupants", "pct_area_100yr_flood", "twia_mandatory", 
                        "water_mud_flag", "houston_cpi_food_scalar", "effective_tax_rate_pct", "five_year_pop_growth_pct"],
            "mean": ["2399.350", "3.496", "4.265", "0.097", "0.294", "1.045", "2.378", "10.862"],
            "std": ["927.335", "1.701", "9.309", "0.296", "0.456", "0.000", "0.394", "58.923"],
            "min": ["801.000", "1.000", "1.000", "0.000", "0.000", "1.045", "2.130", "-33.330"],
            "max": ["3999.000", "6.000", "42.000", "1.000", "1.000", "1.045", "3.170", "871.700"]
        }
        
        st.table(pd.DataFrame(matrix_data))

        st.markdown("""
        *Baseline Intercepts established by the model:*
        * **Food Cost:** 1,314.46 USD
        * **HOA / Maintenance:** 207.06 USD
        * **Electricity:** 85.66 USD
        * **Water:** 75.25 USD
        * **Natural Gas:** 24.25 USD
        """)
        st.divider()

        st.subheader("2. Feature Impact Chart (Beeswarm Plot)")
        st.markdown("""
        * **What it tracks:** An explainability matrix that calculates exactly how much each individual feature influences the predicted expenses.
        * **What it means:** These charts provide a clear visual guide to how the model makes decisions. It shows exactly how much a specific feature (like a high number of occupants) adds or subtracts from the baseline monthly costs to generate your final custom estimate.
        """)
        render_image("shap_importance_all_targets.png", "Global Feature Importance Across All 5 Cost Targets", BASE_DIR)
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            render_image("shap_beeswarm_food.png", "Food Cost SHAP Beeswarm", BASE_DIR)
            render_image("shap_beeswarm_electricity.png", "Electricity SHAP Beeswarm", BASE_DIR)
            render_image("shap_beeswarm_water.png", "Water SHAP Beeswarm", BASE_DIR)
        with col_b2:
            render_image("shap_beeswarm_gas.png", "Gas SHAP Beeswarm", BASE_DIR)
            render_image("shap_beeswarm_hoa.png", "HOA / Maintenance SHAP Beeswarm", BASE_DIR)
        st.divider()

        st.subheader("3. Regression Metrics by Cost Target")
        st.markdown("""
        * **What it plots:** These three side-by-side bar charts act as our model's report card. They map out three key performance scores across all five of our cost categories (Food, Electricity, Water, Gas, and HOA): R² (overall accuracy), MAE (our average dollar mistake), and RMSE (our average mistake, but penalized more for larger errors).
        * **What it means:** This shows the class at a glance what our model is good at. It visually proves that we are predicting Food, Water, Gas, and Electricity with incredible precision (tall R² bars near 1.0, and very short MAE/RMSE dollar bars). It also highlights that our HOA estimates are our weakest link, showing the highest average dollar error.
        """)
        render_image("metric_summary.png", "Regression Metrics by Cost Target (R², MAE, RMSE)", BASE_DIR)
        st.divider()

        st.subheader("4. Error Distribution Charts: Residual Distributions")
        st.markdown("""
        * **What it plots:** These five charts (histograms) look like bell curves and show the distribution of our errors. The horizontal axis represents the dollar amount we were off by, and the vertical axis (how tall the bars are) shows how many times we made that specific mistake.
        * **What it means:** This shows the spread of our mistakes. A tall, skinny spike right at the zero mark means we are consistently accurate with very few outliers (which is exactly what we see for Food and Water). The HOA chart has a wider, flatter spread, showing the class that when we miss on an HOA prediction, we tend to miss by a wider dollar range.
        """)
        render_image("residual_histograms.png", "Residual Distributions Across All 5 Cost Targets", BASE_DIR)
        st.divider()

        st.subheader("5. Error & Residual Diagnostics")
        st.markdown("""
        **Actual vs. Predicted (The Blue Scatter Plots)**
        * **What it plots:** For each cost category, we plotted the actual historical cost on the horizontal axis and our model's predicted cost on the vertical axis. We also drew a red dashed diagonal line straight through the middle, which represents a "perfect fit" where the prediction exactly matches the actual cost.
        * **What it means:** We want our blue dots to hug that red line as tightly as possible. When presenting this, point out how the dots for Food and Water are essentially perfectly lined up. For Electricity and Gas, you can point out a slight "staircase" effect, this is just a quirk of how our tree-based model groups square footage. For HOA, the dots are grouped in distinct streaks because actual HOA costs in our data are locked into three specific price tiers: 45 USD, 85 USD, and 275 USD.
        
        **Homoscedasticity Check (The Red Scatter Plots)**
        * **What it plots:** This plots our predicted cost on the horizontal axis against our "residuals" (the exact dollar amount we were off by) on the vertical axis. A black dashed line runs across zero, representing a prediction with zero error.
        * **What it means:** This chart checks if our model gets confused by more expensive homes. Ideally, we want the red dots scattered evenly both above and below the zero line, no matter how high the predicted cost gets. For Gas and Electricity, the errors are steady and flat across the board. However, for HOA, the class will notice the errors spread out and get larger as the predictions increase, showing exactly where our model struggles with those higher-tier ZIP codes.
        """)
        
        diag_tabs = st.tabs(["Groceries & Food", "Electricity", "Water", "Natural Gas", "HOA / Maintenance"])
        with diag_tabs[0]:
            render_image("diagnostics_food.png", "Food Cost Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[1]:
            render_image("diagnostics_electricity.png", "Electricity Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[2]:
            render_image("diagnostics_water.png", "Water Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[3]:
            render_image("diagnostics_gas.png", "Gas Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[4]:
            render_image("diagnostics_hoa.png", "HOA / Maintenance Target Validation & Homoscedasticity Check", BASE_DIR)
        st.divider()

    with tab5:
        st.subheader("Phase 04 Execution Roadmap: Working Prototype, Integration & Deployment")

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### Week 11")
                st.caption("Application Development & Component Engineering")
                st.info("**Key Hurdle:** Migrating from flat CSV lookups to a parameterized SQLite backend while integrating external OpenAI API calls into dynamic PDF generation.")
            with col2:
                st.markdown("""
                *   **Database Implementation:** Migrated regional tri-county housing and governance records into a structured SQLite database (`living_costs.db`) queried via parameterized SQL (`query_single_zip()`), preventing injection risks and accelerating single-ZIP lookups.
                *   **API Development:** Integrated the OpenAI API (`gpt-4o-mini` via `generate_ai_narrative()`) to synthesize tailored 3-sentence executive cost summaries and arbitrage migration insights.
                *   **Component Development:** Upgraded the ReportLab `generate_pdf_report()` engine with dynamic text-wrapping containers, zebra-striped breakdown tables, savings callout boxes, and in-memory `io.BytesIO()` streaming.
                *   **UI Implementation:** Expanded the dashboard with sidebar export controls, the radius-based **Budget Recommender** tab, and comprehensive AI/XAI framework documentation.

                **Contributors:** Brandon Matias & Taki Eddine Boubekri
                """)
            st.divider()

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### Week 12")
                st.caption("System Integration, Optimization & Testing")
                st.info("**Key Hurdle:** Fortifying the app against malformed user inputs and directory shifts while locking ML inference latency securely under 150ms.")
            with col2:
                st.markdown("""
                *   **Performance Optimization:** Applied Streamlit's `@st.cache_resource` and `@st.cache_data` decorators strictly to the `.joblib` XGBoost model load and SQLite database ingestion, ensuring sub-150ms inference latency.
                *   **Bug Fixing & Edge-Case Hardening:** Added graceful handling for non-numeric or unmapped ZIP inputs (`st.stop()` validation) and built a multi-path fallback loader (`render_image()`) so SHAP diagnostic plots render regardless of launch directory.
                *   **Automated Unit & Integration Testing (`pytest`):** Authored `test_app.py` and an interactive **Live System Diagnostics** tab (`Tab 8`) to verify Haversine Great-Circle distance math, deterministic TWIA $125/mo coastal surcharges, and spatial arbitrage routing.
                *   **End-to-End Integration:** Validated seamless data flow across SQLite queries, multi-output XGBoost inference, spatial filtering, OpenAI summarization, and ReportLab PDF exports.

                **Contributors:** Jonah Joseph, Khadijah Patience & Brandon Matias
                """)
            st.divider()

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### Week 13")
                st.caption("Deployment Preparation & Final Deliverables")
                st.info("**Key Hurdle:** Configuring cloud environment secrets and packaging a self-contained, reproducible MVP deliverable for live demonstration.")
            with col2:
                st.markdown("""
                *   **Deployment Environment Setup:** Deployed the production prototype to Streamlit Community Cloud with automated GitHub CI/CD pipelines and encrypted `OPENAI_API_KEY` secret management.
                *   **System Documentation Finalized:** Completed comprehensive repository documentation (`README.md`, `requirements.txt`, and in-app XAI explainability guides) for full audit transparency.
                *   **Demonstration & Deliverable Packaging:** Prepared the live "Glass Box" walkthrough and packaged all production source code, SQLite database, serialized model artifacts, and `pytest` suites into the final Phase 04 submission archive.

                **Contributors:** Brandon Matias, Khadijah Patience, Jonah Joseph & Taki Eddine Boubekri
                """)
            st.divider()

    with tab7:
        st.header("🧠 AI Architecture & Framework")
        st.markdown("""
        The **Living Expense Estimation Engine** combines non-linear Machine Learning, Explainable AI (XAI), Spatial Proximity Analytics, 
        and Generative AI summarization to deliver precise, audit-defensible non-mortgage living cost predictions across Harris, Fort Bend, and Galveston Counties.
        """)
        st.divider()

        # Section 1: Multi-Target ML Predictive Engine
        st.markdown("### 1. Multi-Target Machine Learning (Predictive Engine)")
        col_ai1, col_ai2 = st.columns([1, 2])
        with col_ai1:
            st.info("""
            **Technology Used:**  
            XGBoost (`MultiOutputRegressor`)  
            
            **Where It Operates:**  
            Backend Model Pipeline (`xgboost_cost_model.joblib`)
            """)
        with col_ai2:
            st.markdown("""
            * **How It Works:** Rather than relying on simple linear averages or static lookup tables, our core engine runs an ensemble of gradient-boosted decision trees. When a user adjusts sidebar parameters (ZIP code, square footage, household occupants), the model simultaneously predicts all **five core non-mortgage expense targets** (Food, Electricity, Water, Natural Gas, and HOA) in real time.
            * **Why XGBoost:** Regional living costs exhibit sudden "step-function" price jumps across administrative borders (such as Municipal Utility District water tax levies or coastal windstorm zones). Gradient-boosted trees excel at capturing these complex non-linear spatial boundaries with high accuracy.
            """)
        st.divider()

        # Section 2: Explainable AI (XAI)
        st.markdown("### 2. Explainable AI & Model Interpretability (XAI)")
        col_xai1, col_xai2 = st.columns([1, 2])
        with col_xai1:
            st.success("""
            **Technology Used:**  
            SHAP (SHapley Additive exPlanations)  
            
            **Where It Operates:**  
            `📊 ML Explainability & Diagnostics` Tab
            """)
        with col_xai2:
            st.markdown("""
            * **How It Works:** Machine learning models can act as opaque "black boxes." We integrate SHAP, grounded in cooperative game theory, to calculate the exact contribution of each input variable.
            * **Why It Matters:** SHAP quantifies precisely how much each feature (e.g., adding an occupant vs. increasing home square footage) pushes each predicted cost above or below the baseline regional intercept, providing full transparency and auditability.
            """)
        st.divider()

        # Section 3: Spatial Distance Analytics (Arbitrage Engine)
        st.markdown("### 3. Spatial Distance Analytics & Arbitrage Engine")
        col_geo1, col_geo2 = st.columns([1, 2])
        with col_geo1:
            st.warning("""
            **Technology Used:**  
            Spatial Proximity Engine (Haversine Distance Math)  
            
            **Where It Operates:**  
            `💰 Comparison` & `🎯 Budget Recommender` Tabs
            """)
        with col_geo2:
            st.markdown("""
            * **How It Works:** The engine calculates Great-Circle geographic distances between ZIP code centroids across the tri-county region.
            * **What It Accomplishes:** When a target ZIP code contains heavy financial surcharges (like coastal insurance or MUD property tax levies), the spatial engine scans neighboring locations within a set radius, re-evaluating the XGBoost model to find lower-cost alternative ZIP codes just miles away.
            """)
        st.divider()

        # Section 4: Generative AI Executive Summarization (OpenAI API)
        st.markdown("### 4. Generative AI Executive Summarization (OpenAI API)")
        col_llm1, col_llm2 = st.columns([1, 2])
        with col_llm1:
            st.info("""
            **Technology Used:**  
            OpenAI API (`gpt-4o-mini`) + ReportLab  
            
            **Where It Operates:**  
            Sidebar `5. Export Component` (`generate_ai_narrative()`)
            """)
        with col_llm2:
            st.markdown("""
            * **How It Works:** When a user generates a downloadable executive PDF report, the application passes the verified XGBoost cost breakdown, TWIA coastal status, and spatial arbitrage savings to OpenAI's `gpt-4o-mini` endpoint (`temperature=0.5`, `max_tokens=150`) to synthesize a tailored 3-sentence financial briefing.
            * **Hallucination Guardrails & Separation of Concerns:** The LLM performs **zero financial calculations**. All dollar figures are computed deterministically by XGBoost, SQLite, and TWIA rules first, and the LLM is strictly constrained to translating those pre-verified outputs into plain-English executive commentary—with automatic fallback handling if API credentials are unavailable.
            """)
        st.divider()

        # Section 5: Deterministic Hybrid Rules
        st.markdown("### 5. Hybrid Deterministic & Regulatory Rules")
        st.markdown("""
        To maintain absolute financial defensibility, predictive Machine Learning is paired with deterministic regulatory rules:
        * **Coastal Windstorm Surcharges (TWIA):** Calculated deterministically ($125/mo policy surcharge) directly from Texas Department of Insurance (TDI) eligibility boundaries.
        * **USDA Grocery Tiers:** Scaled using official USDA economy-of-scale formulas based on the selected household budget profile (Thrifty, Low-Cost, Moderate, Liberal).
        """)

    with tab8:
        st.header("🧪 Live System Diagnostics & Automated Verification Suite")
        st.markdown("""
        Unlike static mockups, this diagnostic suite executes live Python `assert` statements and wall-clock latency benchmarks 
        directly against our production functions, SQLite database (`living_costs.db`), and serialized XGBoost model (`xgboost_cost_model.joblib`).
        """)
        st.divider()
        
        if st.button("▶️ Execute Live Unit, Integration & Edge-Case Suite", use_container_width=True):
            audit_records = []
            all_passed = True

            # --- TEST 1: HAVERSINE SPATIAL MATH (UNIT TEST) ---
            with st.status("1/5 Unit Test: Haversine Great-Circle Spatial Math...", expanded=True) as s1:
                t0 = time.perf_counter()
                try:
                    dist_hou_dal = calc_geo_distance_miles(29.7604, -95.3698, 32.7767, -96.7970)
                    dist_identity = calc_geo_distance_miles(29.7604, -95.3698, 29.7604, -95.3698)
                    assert 220.0 < dist_hou_dal < 230.0, f"Expected ~224.8 mi, got {dist_hou_dal}"
                    assert dist_identity == 0.0, f"Expected 0.0 mi, got {dist_identity}"
                    ms = (time.perf_counter() - t0) * 1000
                    st.write(f"**Inputs:** Houston `(29.7604, -95.3698)` → Dallas `(32.7767, -96.7970)` & Identity check")
                    st.write(f"**Actual Output:** `{dist_hou_dal:.4f} miles` | Identity: `{dist_identity:.1f} miles` (`{ms:.2f} ms`)")
                    st.caption("**What this proves:** Uses a known Houston-to-Dallas coordinate benchmark (224.80 straight-line miles) to verify our spherical distance math returns accurate miles before scanning local tri-county ZIP codes.")
                    s1.update(label=f"1/5 Spatial Math Unit Test: PASSED ({ms:.2f} ms)", state="complete")
                    audit_records.append(("Unit: Spatial Math", "calc_geo_distance_miles()", "Houston → Dallas coords", "220 < dist < 230 & id == 0", f"{dist_hou_dal:.2f} mi / {dist_identity:.1f} mi", f"{ms:.2f} ms", "PASSED"))
                except AssertionError as e:
                    all_passed = False
                    s1.update(label=f"1/5 Spatial Math Unit Test: FAILED ({e})", state="error")
                    audit_records.append(("Unit: Spatial Math", "calc_geo_distance_miles()", "Houston → Dallas coords", "220 < dist < 230", str(e), "N/A", "FAILED"))

            # --- TEST 2: DETERMINISTIC TWIA REGULATORY GOVERNANCE (UNIT TEST) ---
            with st.status("2/5 Unit Test: Deterministic TWIA Surcharge Rules...", expanded=True) as s2:
                t0 = time.perf_counter()
                try:
                    t_month = calculate_twia_surcharge(twia_flag=1, multiplier=1)
                    t_annual = calculate_twia_surcharge(twia_flag=1, multiplier=12)
                    t_inland = calculate_twia_surcharge(twia_flag=0, multiplier=1)
                    assert t_month == 125.0, f"Expected 125.0, got {t_month}"
                    assert t_annual == 1500.0, f"Expected 1500.0, got {t_annual}"
                    assert t_inland == 0.0, f"Expected 0.0, got {t_inland}"
                    ms = (time.perf_counter() - t0) * 1000
                    st.write("**Inputs:** `twia_flag=1 (mult=1)`, `twia_flag=1 (mult=12)`, `twia_flag=0 (mult=1)`")
                    st.write(f"**Actual Output:** `${t_month:.2f}/mo` | `${t_annual:,.2f}/yr` | `${t_inland:.2f}` inland (`{ms:.2f} ms`)")
                    st.caption("**What this proves:** Guarantees coastal windstorm insurance is strictly enforced at $125/mo ($1,500/yr) in coastal zones and $0.00 inland.")
                    s2.update(label=f"2/5 TWIA Regulatory Unit Test: PASSED ({ms:.2f} ms)", state="complete")
                    audit_records.append(("Unit: TWIA Rules", "calculate_twia_surcharge()", "flags=(1, 1, 0), mult=(1, 12, 1)", "== (125.0, 1500.0, 0.0)", f"${t_month:.0f} / ${t_annual:.0f} / ${t_inland:.0f}", f"{ms:.2f} ms", "PASSED"))
                except AssertionError as e:
                    all_passed = False
                    s2.update(label=f"2/5 TWIA Regulatory Unit Test: FAILED ({e})", state="error")
                    audit_records.append(("Unit: TWIA Rules", "calculate_twia_surcharge()", "flags=(1, 1, 0)", "== (125.0, 1500.0, 0.0)", str(e), "N/A", "FAILED"))

            # --- TEST 3: LIVE SQLITE ARBITRAGE ROUTING (INTEGRATION TEST) ---
            with st.status("3/5 Integration Test: Live SQLite Arbitrage Routing...", expanded=True) as s3:
                t0 = time.perf_counter()
                try:
                    coastal_sample = df[df['twia_mandatory'] == 1].iloc[0]
                    src_zip = coastal_sample['zip_code']
                    best_alt, arb_type = find_nearest_arbitrage_zip(df, coastal_sample)
                    assert int(best_alt['twia_mandatory']) == 0, "Alternative ZIP still requires TWIA"
                    assert int(best_alt['water_mud_flag']) == 0, "Alternative ZIP still carries MUD tax"
                    assert arb_type == "cost_governance", f"Unexpected routing reason: {arb_type}"
                    ms = (time.perf_counter() - t0) * 1000
                    st.write(f"**Inputs:** Live coastal DB record `ZIP {src_zip}` (`twia_mandatory=1`) scanned against all `{len(df)}` SQLite ZIP records")
                    st.write(f"**Actual Output:** Routed to `ZIP {best_alt['zip_code']} ({best_alt.get('city_name', 'N/A')})` | Reason: `{arb_type}` (`{ms:.2f} ms`)")
                    st.caption("**What this proves:** Verifies the spatial engine scans all regional SQLite ZIP codes and routes a high-cost coastal ZIP to the nearest surcharge-free ZIP.")
                    s3.update(label=f"3/5 Live Arbitrage Integration Test: PASSED ({ms:.2f} ms)", state="complete")
                    audit_records.append(("Integration: Arbitrage", "find_nearest_arbitrage_zip()", f"Live DB ZIP {src_zip} (TWIA=1)", "alt.TWIA==0 & alt.MUD==0", f"ZIP {best_alt['zip_code']} ({arb_type})", f"{ms:.2f} ms", "PASSED"))
                except AssertionError as e:
                    all_passed = False
                    s3.update(label=f"3/5 Live Arbitrage Integration Test: FAILED ({e})", state="error")
                    audit_records.append(("Integration: Arbitrage", "find_nearest_arbitrage_zip()", "Live coastal ZIP", "alt.TWIA==0 & alt.MUD==0", str(e), "N/A", "FAILED"))

            # --- TEST 4: END-TO-END SQLITE + XGBOOST LATENCY BENCHMARK (SYSTEM TEST) ---
            with st.status(f"4/5 System Performance Test: Live SQLite Query (ZIP {zip_query}) + XGBoost Inference (<150ms)...", expanded=True) as s4:
                t0 = time.perf_counter()
                try:
                    bench_df = query_single_zip(DB_FILENAME, zip_query)
                    assert not bench_df.empty, f"SQLite query returned empty for active ZIP {zip_query}"
                    b_row = bench_df.iloc[0]
                    bench_input = pd.DataFrame([{
                        'sqft': sqft_input,
                        'occupants': occupants_input,
                        'pct_area_100yr_flood': float(b_row['pct_area_100yr_flood']),
                        'twia_mandatory': int(b_row['twia_mandatory']),
                        'water_mud_flag': int(b_row['water_mud_flag']),
                        'houston_cpi_food_scalar': float(b_row['houston_cpi_food_scalar']),
                        'effective_tax_rate_pct': float(b_row['effective_tax_rate_pct']),
                        'five_year_pop_growth_pct': float(b_row['five_year_pop_growth_pct'])
                    }])[expected_features]
                    raw_preds = model.predict(bench_input)[0]
                    ms = (time.perf_counter() - t0) * 1000
                    assert len(raw_preds) == 5, f"Expected 5 target outputs, got {len(raw_preds)}"
                    assert all(val >= 0 for val in raw_preds), f"Negative prediction detected: {raw_preds}"
                    assert ms < 150.0, f"Inference latency {ms:.2f}ms exceeded 150ms threshold"
                    st.write(f"**Inputs:** Parameterized SQLite lookup `ZIP {zip_query}` + 8-feature tensor (`{sqft_input} sqft`, `{occupants_input} occ`) → `model.predict()`")
                    st.write(f"**Actual Output:** 5 targets `[Food=${raw_preds[0]:.2f}, Elec=${raw_preds[1]:.2f}, Water=${raw_preds[2]:.2f}, Gas=${raw_preds[3]:.2f}, HOA=${raw_preds[4]:.2f}]` in **`{ms:.2f} ms`** (Threshold: `< 150.00 ms`)")
                    st.caption("**What this proves:** Confirms `@st.cache_data` and `@st.cache_resource` keep live SQLite lookup + 5-target XGBoost prediction for your selected ZIP under our 150ms target.")
                    s4.update(label=f"4/5 End-to-End Inference & Latency (<150ms): PASSED ({ms:.2f} ms)", state="complete")
                    audit_records.append(("System: E2E Latency", "query_single_zip + model.predict", f"ZIP {zip_query}, {sqft_input}sqft, {occupants_input} occ", "len==5, vals>=0, ms<150", f"5 targets in {ms:.2f} ms", f"{ms:.2f} ms", "PASSED"))
                except AssertionError as e:
                    all_passed = False
                    s4.update(label=f"4/5 End-to-End Inference & Latency: FAILED ({e})", state="error")
                    audit_records.append(("System: E2E Latency", "query_single_zip + model.predict", f"ZIP {zip_query}", "ms < 150.0", str(e), "N/A", "FAILED"))

            # --- TEST 5: EDGE-CASE SANITIZATION & ASSET INTEGRITY (BUG FIX VERIFICATION) ---
            with st.status("5/5 Edge-Case & Asset Verification: Malformed Input & SHAP Files...", expanded=True) as s5:
                t0 = time.perf_counter()
                try:
                    bad_zip_df = query_single_zip(DB_FILENAME, "ABCDE")
                    assert bad_zip_df.empty, "Non-numeric ZIP 'ABCDE' should return empty DataFrame cleanly"
                    shap_path_found = any(
                        os.path.exists(p) for p in [
                            os.path.join(BASE_DIR, "shap_importance_all_targets.png"),
                            os.path.join(BASE_DIR, "figures", "shap_importance_all_targets.png"),
                            os.path.join(BASE_DIR, "extracted_figures", "figures", "shap_importance_all_targets.png")
                        ]
                    )
                    assert shap_path_found, "Primary SHAP diagnostic asset missing from all fallback paths"
                    ms = (time.perf_counter() - t0) * 1000
                    st.write("**Inputs:** Malformed ZIP `'ABCDE'` → `query_single_zip()` & Multi-directory scan for `shap_importance_all_targets.png`")
                    st.write(f"**Actual Output:** Empty DataFrame returned safely (`rows=0`, zero SQL exceptions) | SHAP asset verified (`{ms:.2f} ms`)")
                    st.caption("**What this proves:** Validates our Week 12 bug fixes—letters in the ZIP input fail gracefully without crashing, and SHAP diagnostic images resolve properly.")
                    s5.update(label=f"5/5 Edge-Case & Asset Integrity: PASSED ({ms:.2f} ms)", state="complete")
                    audit_records.append(("Edge-Case: Sanitization", "query_single_zip('ABCDE')", "Non-numeric string 'ABCDE'", "df.empty & SHAP exists", "0 rows (safe), SHAP found", f"{ms:.2f} ms", "PASSED"))
                except AssertionError as e:
                    all_passed = False
                    s5.update(label=f"5/5 Edge-Case & Asset Integrity: FAILED ({e})", state="error")
                    audit_records.append(("Edge-Case: Sanitization", "query_single_zip('ABCDE')", "Non-numeric string 'ABCDE'", "df.empty & SHAP exists", str(e), "N/A", "FAILED"))

            st.divider()
            st.subheader("📋 Live Diagnostic Audit Log")
            audit_df = pd.DataFrame(
                audit_records,
                columns=["Test Layer", "Target Function", "Input Payload", "Assertion Rule", "Actual Output", "Latency", "Status"]
            )
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
            st.caption(
                "**How to Read This Log:** Each row records a live Python `assert` test executed when you clicked the button above. "
                "**Test 1** checks straight-line spherical math using a Houston-to-Dallas reference check (`224.80 mi`). "
                "**Test 2** enforces the `$125/mo` coastal TWIA rule. **Test 3** scans all 313 SQLite ZIP records to verify coastal-to-inland arbitrage routing. "
                f"**Test 4** benchmarks live SQLite + XGBoost prediction speed for your active ZIP `{zip_query}` against our `<150 ms` requirement. "
                "**Test 5** verifies graceful handling of malformed text inputs (`'ABCDE'`) and SHAP chart availability."
            )

            with st.expander("🔍 Inspect Executed Python Assertions (Code Transparency)"):
                st.code(f"""
# 1. Haversine Spatial Math (Houston -> Dallas Control Check)
assert 220.0 < calc_geo_distance_miles(29.7604, -95.3698, 32.7767, -96.7970) < 230.0
assert calc_geo_distance_miles(29.7604, -95.3698, 29.7604, -95.3698) == 0.0

# 2. Deterministic TWIA Governance
assert calculate_twia_surcharge(twia_flag=1, multiplier=1) == 125.0
assert calculate_twia_surcharge(twia_flag=1, multiplier=12) == 1500.0
assert calculate_twia_surcharge(twia_flag=0, multiplier=1) == 0.0

# 3. Live SQLite Spatial Arbitrage Routing
best_alt, arb_type = find_nearest_arbitrage_zip(df, coastal_sample)
assert int(best_alt['twia_mandatory']) == 0 and int(best_alt['water_mud_flag']) == 0
assert arb_type == "cost_governance"

# 4. End-to-End SQLite + XGBoost Latency (< 150ms) for Active ZIP {zip_query}
raw_preds = model.predict(bench_input)[0]
assert len(raw_preds) == 5 and all(val >= 0 for val in raw_preds) and latency_ms < 150.0

# 5. Edge-Case Input Sanitization & SHAP Asset Fallback
assert query_single_zip(DB_FILENAME, "ABCDE").empty
                """, language="python")

            if all_passed:
                st.success("🌟 ALL 5 LIVE UNIT, INTEGRATION & PERFORMANCE ASSERTIONS PASSED.")
            else:
                st.error("⚠️ ONE OR MORE SYSTEM ASSERTIONS FAILED. Review the audit log above.")
            
if __name__ == "__main__":
    main()
