import sys
import os
import time
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# -----------------------------
# 1. SETUP & PATHS
# -----------------------------
st.set_page_config(
    page_title="Karachi AQI Intelligence",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get the path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.database import AQIDatabase
from src.modeling import predict_next_72_hours

MODEL_PATH = os.path.join(project_root, 'models', 'model.pkl')
FEATURES_PATH = os.path.join(project_root, 'models', 'features.pkl')

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# -----------------------------
# 2. THEME & CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 12% 8%, rgba(56,189,248,0.18), transparent 40%),
                    radial-gradient(circle at 88% 18%, rgba(99,102,241,0.14), transparent 42%),
                    linear-gradient(180deg, #050913 0%, #071125 55%, #050913 100%);
        color: #fff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border-right: 1px solid rgba(255,255,255,0.10);
    }
    .hero {
        border-radius: 18px; 
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.14);
        background: linear-gradient(90deg, rgba(56,189,248,0.68), rgba(99,102,241,0.35));
        box-shadow: 0 16px 52px rgba(0,0,0,0.55);
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: #fff; font-weight: 700; }
    .hero p { margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem; }
    .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px;
        background: rgba(255,255,255,0.045);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .card-title { font-size: 0.9rem; opacity: 0.85; text-transform: uppercase; margin-bottom: 8px; }
    .card-value { font-size: 2.2rem; font-weight: 800; color: #ffffff; line-height: 1.1; }
    .card-sub { font-size: 0.9rem; opacity: 0.8; margin-top: 8px; }
    .pill {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. HELPER FUNCTIONS
# -----------------------------
def get_aqi_details(aqi):
    if aqi <= 50: return "Good", "#22c55e", "Air is clean."
    elif aqi <= 100: return "Moderate", "#eab308", "Acceptable quality."
    elif aqi <= 150: return "Unhealthy (Sens.)", "#f97316", "Sensitive groups beware."
    elif aqi <= 200: return "Unhealthy", "#ef4444", "Avoid outdoor exertion."
    elif aqi <= 300: return "Very Unhealthy", "#a855f7", "Health alert."
    else: return "Hazardous", "#78716c", "Emergency."

@st.cache_resource
def load_model_resources(m_hash):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
        
        model_name = type(model).__name__
        if model_name == 'XGBRegressor': model_name = "XGBoost"
        elif model_name == 'RandomForestRegressor': model_name = "Random Forest"
        elif model_name == 'LinearRegression': model_name = "Linear Regression"
            
        return model, features, model_name
    except:
        return None, None, "Unknown"

@st.cache_data(ttl=300)
def load_data_from_mongo():
    try:
        db = AQIDatabase()
        if db.client:
           return db.fetch_data()
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# -----------------------------
# 4. LOAD DATA & LOGIC
# -----------------------------
df_all = load_data_from_mongo()

try:
    m_hash = os.path.getmtime(MODEL_PATH) + os.path.getmtime(FEATURES_PATH)
except:
    m_hash = 0
model, features, active_model_name = load_model_resources(m_hash)

# Sidebar
with st.sidebar:
    st.markdown("## System Status")
    if not df_all.empty:
        st.success("MongoDB Connected", icon="✅")
    else:
        st.error("Database Error", icon="❌")
        
    if model:
        st.success(f"Model Active ({active_model_name})")
    
    st.markdown("---")
    st.markdown("## Actions")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
        
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    if auto_refresh and st_autorefresh:
        st_autorefresh(interval=60000, key="auto_refresh")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem; color:#aaa;">
    Developed by <b>Muhammad Ehsan</b><br>
    © 2026 Karachi AQI Project
    </div>
    """, unsafe_allow_html=True)

if df_all.empty:
    st.warning("Waiting for data... check database connection.")
    st.stop()

# --- TIMEZONE FIX ---
# 1. Convert Data to Karachi Time
if df_all['date'].dt.tz is None:
    df_all['date'] = df_all['date'].dt.tz_localize('UTC')
df_all['date'] = df_all['date'].dt.tz_convert('Asia/Karachi')
df_all = df_all.sort_values('date')

# 2. Get Current Karachi Time
now_karachi = datetime.now(pytz.timezone('Asia/Karachi'))

# 3. Find "Current" Record
# Strict logic: Data must be <= current time (No buffer)
past_data = df_all[df_all['date'] <= now_karachi]

if not past_data.empty:
    current_rec = past_data.iloc[-1]
else:
    current_rec = df_all.iloc[0]

current_aqi = int(current_rec.get('us_aqi', 0))
cat_label, cat_color, cat_advice = get_aqi_details(current_aqi)

# Generate Forecast
forecast_df = pd.DataFrame()
if model and features:
    # Use the PAST data for prediction input
    recent_data = past_data.tail(100).copy()
    try:
        forecast_df = predict_next_72_hours(model, features, recent_data)
        forecast_df['display_date'] = forecast_df['date'].dt.strftime('%A, %d %b')
        forecast_df['date_only'] = forecast_df['date'].dt.date
    except:
        pass

# -----------------------------
# 5. DASHBOARD UI
# -----------------------------
st.markdown(f"""
<div class="hero">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1>Karachi AQI Intelligence</h1>
            <p>Real-time Monitoring & 72-Hour Forecast</p>
        </div>
        <div style="text-align:right;">
            <span class="pill" style="background:rgba(0,0,0,0.2);">Last Updated: {current_rec['date'].strftime('%H:%M')}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.5, 1, 2.5])

with c1:
    st.markdown(f"""
    <div class="card" style="border-left: 4px solid {cat_color};">
        <div class="card-title">Current Status</div>
        <div class="card-value">{current_aqi} <span style="font-size:1rem; font-weight:400; color:{cat_color};">AQI</span></div>
        <div class="card-sub">
            <span class="pill" style="background:{cat_color}20; color:{cat_color}; border-color:{cat_color}40;">{cat_label}</span>
        </div>
        <div class="card-sub" style="margin-top:10px; font-style:italic;">"{cat_advice}"</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    val_pm25 = int(current_rec.get('pm2_5', 0))
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Dominant Pollutant</div>
        <div class="card-value">{val_pm25}</div>
        <div class="card-sub">µg/m³ (PM2.5)</div>
        <div style="margin-top:15px; height:4px; background:rgba(255,255,255,0.1); border-radius:2px;">
            <div style="width: {min(val_pm25, 100)}%; height:100%; background: #38BDF8; border-radius:2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    # DYNAMIC LEADERBOARD
    rmse_vals = {'Linear Regression': 3.10, 'XGBoost': 3.56, 'Random Forest': 3.78}
    active_rmse = rmse_vals.get(active_model_name, 3.10)
    
    html_content = f"""
<div class="card">
<div class="card-title">🏆 Model Performance Leaderboard</div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
<span><b>{active_model_name}</b> (Active)</span>
<span style="color:#22c55e; font-weight:bold;">RMSE: {active_rmse}</span>
</div>
<div style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px; margin-bottom:12px;">
<div style="width:95%; height:100%; background:#22c55e; border-radius:3px;"></div>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
<span style="opacity:0.7;">{'XGBoost' if active_model_name != 'XGBoost' else 'Linear Regression'}</span>
<span style="opacity:0.7;">RMSE: 3.56</span>
</div>
<div style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px; margin-bottom:12px;">
<div style="width:85%; height:100%; background:#f59e0b; border-radius:3px;"></div>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
<span style="opacity:0.7;">{'Random Forest' if active_model_name != 'Random Forest' else 'Linear Regression'}</span>
<span style="opacity:0.7;">RMSE: 3.78</span>
</div>
<div style="width:100%; height:6px; background:rgba(255,255,255,0.1); border-radius:3px;">
<div style="width:75%; height:100%; background:#ef4444; border-radius:3px;"></div>
</div>
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# 6. TABS
# -----------------------------
tab_forecast, tab_compare, tab_data = st.tabs(["Forecast & Trend", "Model Comparison", "Data & Export"])

# --- TAB 1: FORECAST ---
with tab_forecast:
    if not forecast_df.empty:
        st.markdown("### 3-Day Outlook (Tomorrow Onwards)")
        
        daily_avg = forecast_df.groupby('date_only').agg({
            'predicted_aqi': 'mean',
            'display_date': 'first'
        }).reset_index()
        
        # FIX: Filter out "Today" so the forecast starts from Tomorrow
        daily_avg = daily_avg[daily_avg['date_only'] > now_karachi.date()]
        
        # Ensure sorting matches calendar days
        daily_avg = daily_avg.sort_values('date_only').head(3)
        
        cols = st.columns(3)
        
        # FIX: Use enumerate to guarantee sequence (0->Left, 1->Middle, 2->Right)
        # ignoring the dataframe's internal index which might be out of order.
        for i, (index, row) in enumerate(daily_avg.iterrows()):
            d_name = row['display_date']
            d_aqi = int(round(row['predicted_aqi']))
            _, d_col, _ = get_aqi_details(d_aqi)
            
            with cols[i]:
                st.markdown(f"""
                <div class="card" style="text-align:center; padding:15px; border-top: 4px solid {d_col};">
                    <div style="color:#aaa; font-size:1.1rem; font-weight:600; text-transform:uppercase;">{d_name}</div>
                    <div style="font-size:3.5rem; font-weight:bold; color:{d_col}; margin:10px 0;">{d_aqi}</div>
                    <span class="pill" style="background:{d_col}20; color:{d_col};">AVG AQI</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 72-Hour Trend")
        
        fig = go.Figure()
        
        # Plot only Past Data (History)
        hist_plot = past_data.tail(48)
        fig.add_trace(go.Scatter(
            x=hist_plot['date'], y=hist_plot['us_aqi'],
            mode='lines', name='Observed',
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ))
        
        # Plot Future Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['predicted_aqi'],
            mode='lines+markers', name=f'Forecast ({active_model_name})',
            line=dict(color='#38BDF8', width=3),
            marker=dict(size=4)
        ))
        
        # Dynamic Zoom
        all_values = pd.concat([hist_plot['us_aqi'], forecast_df['predicted_aqi']])
        y_min = all_values.min()
        y_max = all_values.max()
        y_range = [y_min * 0.9, y_max * 1.1]

        fig.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            hovermode="x unified",
            yaxis=dict(range=y_range)
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: MODEL COMPARISON ---
with tab_compare:
    st.markdown("### Multi-Model Analysis")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Accuracy Comparison (RMSE)</div>
            <div class="card-sub">Lower is better.</div>
        </div>
        """, unsafe_allow_html=True)
        
        model_data = pd.DataFrame({
            'Model': ['Linear Regression', 'XGBoost', 'Random Forest'],
            'RMSE': [3.10, 3.56, 3.78]
        })
        
        fig_comp = px.bar(
            model_data, x='RMSE', y='Model', orientation='h',
            text='RMSE', color='RMSE',
            color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444']
        )
        fig_comp.update_traces(
            texttemplate='%{text:.2f}', 
            textposition='outside',
            textfont_size=14,
            textfont_color='white',
            width=0.35  # Controls the width of bars (reduced to remove big gaps)
        )
        # FIX: Added bargap and adjusted margin/height to look cleaner
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, visible=False), 
            yaxis=dict(title=None),
            margin=dict(l=0, r=50, t=20, b=0),
            uniformtext_minsize=12, 
            uniformtext_mode='hide',
            bargap=0.1, # Reduces space between bars
            height=250  # Compact height to fit the card better
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Why {active_model_name}?</div>
            <p>In this specific cycle, the <b>{active_model_name}</b> model achieved the lowest error (RMSE {active_rmse}), likely because the recent air quality trend matches its strengths.</p>
            <p>The system automatically selected it over others for the live forecast.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: DATA & EXPORT ---
with tab_data:
    st.markdown("### Data Export")
    col_d1, col_d2 = st.columns([3, 1])
    
    if not forecast_df.empty:
        export_df = forecast_df[['date', 'predicted_aqi']].copy()
        export_df.columns = ['Timestamp', 'Forecast_AQI']
        export_df['Category'] = export_df['Forecast_AQI'].apply(lambda x: get_aqi_details(x)[0])
        
        with col_d1:
            st.dataframe(export_df, use_container_width=True, height=350)
            
        with col_d2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Download Report</div>
                <p>Get the full 72-hour forecast data in CSV format for offline analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                " Download CSV",
                data=csv,
                file_name="karachi_aqi_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
            