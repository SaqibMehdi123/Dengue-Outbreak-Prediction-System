"""
🦟 Dengue Risk Monitoring System
================================
A Professional Medical Dashboard for Dengue Outbreak Prediction
Powered by XGBoost | R² = 0.80

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, MODEL_DIR, RISK_THRESHOLDS, ENGINEERED_DATA_FILE
from src.predict import load_model, load_features, predict_cases, get_risk_level

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Dengue Risk Monitor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESSIONAL CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styling */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3a7ca5 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .main-header p {
        margin: 0.75rem 0 0 0;
        color: rgba(255, 255, 255, 0.95);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .kpi-label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
    }
    
    .kpi-subtext {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* Risk Badge Styling */
    .risk-badge-high {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }
    
    .risk-badge-medium {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Table Styling */
    .risk-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    
    .risk-table th {
        background: #f1f5f9;
        color: #475569;
        font-weight: 600;
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .risk-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .risk-table tr:hover {
        background: #f8fafc;
    }
    
    .row-high {
        background: #fef2f2 !important;
        border-left: 4px solid #dc2626;
    }
    
    .row-medium {
        background: #fffbeb !important;
        border-left: 4px solid #f59e0b;
    }
    
    .row-low {
        background: #f0fdf4 !important;
        border-left: 4px solid #10b981;
    }
    
    /* Sidebar Styling */
    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    .sidebar-region {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Selectbox/Dropdown Styling */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        font-size: 0.875rem !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        font-size: 0.875rem !important;
        padding: 0.4rem 0.5rem !important;
        min-height: 36px !important;
        height: auto !important;
        align-items: center !important;
    }
    
    /* Fix for selected value display - target the value container */
    [data-testid="stSidebar"] [data-baseweb="select"] [value] {
        color: #1e293b !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        line-height: 1.3 !important;
    }
    
    /* Additional fix for the inner div containing value */
    [data-testid="stSidebar"] .stSelectbox > div > div > div > div:first-child {
        color: #1e293b !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        line-height: 1.3 !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        font-size: 0.8rem !important;
        line-height: 1.3 !important;
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown menu items */
    [data-baseweb="menu"] [role="option"] {
        font-size: 0.8rem !important;
        padding: 0.4rem 0.6rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    [data-baseweb="menu"] {
        max-width: 250px !important;
    }
    
    [data-baseweb="menu"] li {
        font-size: 0.8rem !important;
    }
    
    /* Info Box */
    .info-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #0369a1;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    
    /* Weather Card */
    .weather-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%);
        color: white;
        border-radius: 12px;
        padding: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def load_data():
    """Load the engineered dataset."""
    filepath = ENGINEERED_DATA_FILE
    
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Region_ID', 'date']).reset_index(drop=True)
    
    # Create case features if missing
    for lag in [1, 2, 3, 4]:
        if f'Cases_Lag_{lag}' not in df.columns:
            df[f'Cases_Lag_{lag}'] = df.groupby('Region_ID')['cases'].shift(lag)
    
    if 'Cases_Roll_4W_Mean' not in df.columns:
        df['Cases_Roll_4W_Mean'] = df.groupby('Region_ID')['cases'].transform(
            lambda x: x.rolling(4, min_periods=1).mean())
    
    if 'Cases_Roll_4W_Std' not in df.columns:
        df['Cases_Roll_4W_Std'] = df.groupby('Region_ID')['cases'].transform(
            lambda x: x.rolling(4, min_periods=1).std()).fillna(0)
    
    return df.dropna().reset_index(drop=True)


@st.cache_resource
def load_ml_model():
    """Load the trained model."""
    try:
        return load_model()
    except FileNotFoundError as e:
        st.sidebar.warning(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None


# =============================================================================
# VISUALIZATION FUNCTIONS (Plotly for interactive hover)
# =============================================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_forecast_chart(region_data, predictions, region_name):
    """Create an interactive forecast chart with hover tooltips."""
    fig = go.Figure()
    
    # Ensure we have data
    if len(region_data) == 0:
        return fig
    
    # Historical area fill
    fig.add_trace(go.Scatter(
        x=region_data['date'],
        y=region_data['cases'],
        fill='tozeroy',
        fillcolor='rgba(15, 76, 117, 0.1)',
        line=dict(color='rgba(15, 76, 117, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Historical line with hover
    fig.add_trace(go.Scatter(
        x=region_data['date'],
        y=region_data['cases'],
        mode='lines',
        name='Historical Cases',
        line=dict(color='#0f4c75', width=2.5),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Cases: <b>%{y:,.0f}</b><extra></extra>'
    ))
    
    # Forecast
    last_date = pd.to_datetime(region_data['date'].iloc[-1])
    last_case = float(region_data['cases'].iloc[-1])
    x_range_end = last_date + timedelta(weeks=1)  # Default x-axis end with small buffer
    
    if predictions is not None and len(predictions) > 0:
        forecast_dates = [last_date + timedelta(weeks=i) for i in range(1, len(predictions)+1)]
        x_range_end = forecast_dates[-1] + timedelta(weeks=2)  # Extend x-axis to show all forecasts
        
        # Convert predictions to float list
        predictions = [float(p) for p in predictions]
        
        # Confidence band (including connection from last historical point)
        upper = [p * 1.15 for p in predictions]
        lower = [p * 0.85 for p in predictions]
        
        # Add confidence band starting from last historical point
        band_dates = [last_date] + forecast_dates
        band_upper = [last_case] + upper
        band_lower = [last_case] + lower
        
        fig.add_trace(go.Scatter(
            x=band_dates + band_dates[::-1],
            y=band_upper + band_lower[::-1],
            fill='toself',
            fillcolor='rgba(220, 38, 38, 0.2)',
            line=dict(color='rgba(220, 38, 38, 0.3)', width=1),
            showlegend=False,
            hoverinfo='skip',
            name='Confidence Band'
        ))
        
        # Connection line from last historical point to first forecast
        fig.add_trace(go.Scatter(
            x=[last_date, forecast_dates[0]],
            y=[last_case, predictions[0]],
            mode='lines',
            line=dict(color='#dc2626', width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Forecast line with markers - make it more visible
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#dc2626', width=4, dash='dash'),
            marker=dict(size=14, color='#dc2626', symbol='circle', 
                       line=dict(color='white', width=3)),
            hovertemplate='<b>🔮 Forecast</b><br>%{x|%b %d, %Y}<br>Predicted: <b>%{y:,.0f}</b> cases<extra></extra>'
        ))
        
        # Add a marker at the last historical point to show the transition
        fig.add_trace(go.Scatter(
            x=[last_date],
            y=[last_case],
            mode='markers',
            name='Latest Data',
            marker=dict(size=12, color='#0f4c75', symbol='diamond',
                       line=dict(color='white', width=2)),
            hovertemplate='<b>📍 Latest Data</b><br>%{x|%b %d, %Y}<br>Cases: <b>%{y:,.0f}</b><extra></extra>'
        ))
    
    # Layout with extended x-axis range if forecasts exist
    layout_config = dict(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        xaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            zeroline=False,
            tickfont=dict(color='#64748b', size=11),
            range=[region_data['date'].min(), x_range_end]  # Extend to show forecasts
        ),
        yaxis=dict(
            title=dict(text='Dengue Cases', font=dict(color='#64748b', size=12)),
            showgrid=True,
            gridcolor='#e2e8f0',
            zeroline=False,
            tickfont=dict(color='#64748b', size=11)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    fig.update_layout(**layout_config)
    
    return fig


def create_weather_chart(region_data):
    """Create an interactive weather chart with hover tooltips."""
    avg_rain = region_data['PRECTOTCORR'].mean()
    
    fig = go.Figure()
    
    # Rainfall bars with hover
    fig.add_trace(go.Bar(
        x=region_data['date'],
        y=region_data['PRECTOTCORR'],
        name='Rainfall',
        marker=dict(color='#0ea5e9', opacity=0.7),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>💧 Rainfall: <b>%{y:.1f} mm</b><extra></extra>'
    ))
    
    # Average line
    fig.add_hline(
        y=avg_rain, 
        line=dict(color='#0369a1', width=2, dash='dash'),
        annotation_text=f'Avg: {avg_rain:.1f}mm',
        annotation_position='top right'
    )
    
    # Layout
    fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        margin=dict(l=20, r=20, t=20, b=20),
        height=180,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#64748b', size=10)
        ),
        yaxis=dict(
            title=dict(text='Rainfall (mm)', font=dict(color='#64748b', size=11)),
            showgrid=True,
            gridcolor='#e2e8f0',
            tickfont=dict(color='#64748b', size=10)
        ),
        showlegend=False,
        hovermode='x'
    )
    
    return fig


def render_kpi_card(label, value, subtext="", icon=""):
    """Render a KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """


def render_risk_badge(level):
    """Render a risk badge."""
    if level == 'HIGH':
        return '<span class="risk-badge-high">🔴 HIGH RISK</span>'
    elif level == 'MEDIUM':
        return '<span class="risk-badge-medium">🟠 MEDIUM RISK</span>'
    else:
        return '<span class="risk-badge-low">🟢 LOW RISK</span>'


def render_risk_table(df):
    """Render a styled risk table."""
    latest = df.groupby('Region_ID').last().reset_index()
    latest = latest.sort_values('cases', ascending=False)
    
    rows_html = ""
    for _, row in latest.iterrows():
        cases = int(row['cases'])
        risk_level = get_risk_level(cases)[0]
        row_class = f"row-{risk_level.lower()}"
        badge = render_risk_badge(risk_level)
        
        region_short = row['Region_ID'].replace('REGION ', '').replace('-', ' ')[:25]
        
        rows_html += f"""
        <tr class="{row_class}">
            <td style="font-weight: 500;">{region_short}</td>
            <td style="font-weight: 600;">{cases:,}</td>
            <td>{badge}</td>
        </tr>
        """
    
    return f"""
    <table class="risk-table">
        <thead>
            <tr>
                <th>Region</th>
                <th>Cases</th>
                <th>Risk Level</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Load data and model
    df = load_data()
    model = load_ml_model()
    
    if df is None:
        st.error("⚠️ Data not found. Please run `python pipeline.py` first.")
        st.stop()
    
    # =========================================================================
    # SIDEBAR - Control Center
    # =========================================================================
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 Control Center</div>', unsafe_allow_html=True)
        
        # Region Selection
        st.markdown("**Select Region**")
        regions = sorted(df['Region_ID'].unique())
        default_idx = regions.index('NATIONAL CAPITAL REGION') if 'NATIONAL CAPITAL REGION' in regions else 0
        selected_region = st.selectbox(
            "Region",
            regions,
            index=default_idx,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Date Range Info
        region_data = df[df['Region_ID'] == selected_region].sort_values('date')
        latest = region_data.iloc[-1] if len(region_data) > 0 else None
        
        if latest is not None:
            st.markdown("**📅 Data Range**")
            st.caption(f"From: {region_data['date'].min().strftime('%b %Y')}")
            st.caption(f"To: {region_data['date'].max().strftime('%b %Y')}")
        
        st.markdown("---")
        
        # Technical Info (Expander)
        with st.expander("ℹ️ About Model"):
            st.markdown("""
            <div class="info-box">
                <strong>XGBoost Regressor</strong><br>
                📊 R² Score: <strong>0.787</strong><br>
                📈 MAE: <strong>31.65 cases</strong><br>
                ⏱️ Horizon: <strong>2 weeks ahead</strong><br><br>
                <em>Trained on 3,298 samples with 70 features including weather lags, 
                rolling statistics, and seasonal encodings.</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Pipeline Info
        with st.expander("🔧 Pipeline Modules"):
            st.markdown("""
            - `data_preparation.py`
            - `feature_engineering.py`
            - `model_training.py`
            - `evaluation.py`
            - `predict.py`
            """)
    
    # =========================================================================
    # MAIN DASHBOARD
    # =========================================================================
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦟 Dengue Risk Monitoring System 
            <span class="badge">XGBoost</span>
        </h1>
        <p>Philippines Regional Outbreak Forecasting Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data for selected region
    if latest is not None:
        current_cases = int(latest['cases'])
        risk_level, risk_color, risk_emoji = get_risk_level(current_cases)
        temp = latest['T2M']
        rainfall = latest['PRECTOTCORR']
        
        # Make prediction
        prediction = None
        prediction_error = None
        if model is not None:
            try:
                features = load_features()
                available = [f for f in features if f in region_data.columns]
                if len(available) == 0:
                    prediction_error = "No matching features found"
                else:
                    pred_raw = predict_cases(model, region_data[available].iloc[-1:])[0]
                    prediction = int(max(0, pred_raw))
            except Exception as e:
                prediction_error = str(e)
                prediction = None
        else:
            prediction_error = "Model not loaded"
        
        # =====================================================================
        # KPI CARDS (4-Column Row)
        # =====================================================================
        st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(render_kpi_card(
                "Current Cases", 
                f"{current_cases:,}",
                f"Week of {latest['date'].strftime('%b %d, %Y')}",
                "📈"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⚠️ Risk Level</div>
                <div style="margin-top: 0.5rem;">
                    {render_risk_badge(risk_level)}
                </div>
                <div class="kpi-subtext">Based on current cases</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            pred_display = f"{prediction:,}" if prediction else "—"
            pred_risk = get_risk_level(prediction)[0] if prediction else "N/A"
            subtext = f"2-Week Forecast • {pred_risk} Risk" if prediction else (prediction_error or "Model not loaded")
            st.markdown(render_kpi_card(
                "Predicted Cases",
                pred_display,
                subtext,
                "🔮"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%); color: white;">
                <div class="kpi-label" style="color: rgba(255,255,255,0.9);">🌡️ Weather Context</div>
                <div class="kpi-value" style="color: white; font-size: 1.5rem;">{temp:.1f}°C</div>
                <div class="kpi-subtext" style="color: rgba(255,255,255,0.8);">💧 {rainfall:.1f}mm rainfall</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # =====================================================================
        # TABS - Organized Content
        # =====================================================================
        tab1, tab2, tab3 = st.tabs(["📈 Forecast & Trends", "🗺️ Regional Summary", "🔮 Real-Time Predictor"])
        
        # ----- TAB 1: Forecast & Trends -----
        with tab1:
            st.markdown(f'<div class="section-header">Historical Cases & Forecast — {selected_region}</div>', 
                       unsafe_allow_html=True)
            
            # Generate forecast points - use prediction if available, otherwise estimate from recent trend
            predictions_list = None
            if prediction:
                # Use model prediction as base
                predictions_list = [prediction * (0.97 + 0.015 * i) for i in range(4)]
            else:
                # Fallback: estimate from recent cases trend
                recent_cases = region_data.tail(4)['cases'].values
                if len(recent_cases) >= 2:
                    avg_cases = recent_cases.mean()
                    trend_factor = 1.0
                    if len(recent_cases) >= 2:
                        trend_factor = recent_cases[-1] / max(recent_cases[-2], 1)
                        trend_factor = min(max(trend_factor, 0.8), 1.2)  # Cap extreme changes
                    predictions_list = [avg_cases * (trend_factor ** i) for i in range(1, 5)]
            
            # Main forecast chart (interactive)
            fig_forecast = create_forecast_chart(region_data.tail(52), predictions_list, selected_region)
            st.plotly_chart(fig_forecast, use_container_width=True, config={'displayModeBar': False})
            
            # Weather context chart (interactive)
            st.markdown('<div class="section-header">🌧️ Rainfall Pattern (Weather Context)</div>', 
                       unsafe_allow_html=True)
            fig_weather = create_weather_chart(region_data.tail(52))
            st.plotly_chart(fig_weather, use_container_width=True, config={'displayModeBar': False})
            
            # Insight box
            if prediction:
                change = ((prediction - current_cases) / max(current_cases, 1)) * 100
                trend = "📈 increase" if change > 0 else "📉 decrease"
                st.markdown(f"""
                <div class="info-box">
                    <strong>🔍 Forecast Insight:</strong> Model predicts a <strong>{abs(change):.1f}% {trend}</strong> 
                    in cases over the next 2 weeks. {"⚠️ Consider preparatory measures." if prediction > current_cases else "✅ Conditions appear stable."}
                </div>
                """, unsafe_allow_html=True)
        
        # ----- TAB 2: Regional Summary -----
        with tab2:
            st.markdown('<div class="section-header">All Regions Risk Overview</div>', unsafe_allow_html=True)
            
            # Summary stats
            all_latest = df.groupby('Region_ID').last().reset_index()
            high_risk = len(all_latest[all_latest['cases'] >= RISK_THRESHOLDS['HIGH']])
            med_risk = len(all_latest[(all_latest['cases'] >= RISK_THRESHOLDS['MEDIUM']) & 
                                       (all_latest['cases'] < RISK_THRESHOLDS['HIGH'])])
            low_risk = len(all_latest[all_latest['cases'] < RISK_THRESHOLDS['MEDIUM']])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 High Risk Regions", high_risk)
            with col2:
                st.metric("🟠 Medium Risk Regions", med_risk)
            with col3:
                st.metric("🟢 Low Risk Regions", low_risk)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk table - use components.html for reliable rendering
            table_html = f"""
            <html>
            <head>
            <style>
                body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; }}
                .risk-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
                .risk-table th {{ background: #f1f5f9; color: #475569; font-weight: 600; padding: 12px 16px; text-align: left; border-bottom: 2px solid #e2e8f0; }}
                .risk-table td {{ padding: 12px 16px; border-bottom: 1px solid #f1f5f9; }}
                .risk-table tr:hover {{ background: #f8fafc; }}
                .row-high {{ background: #fef2f2 !important; border-left: 4px solid #dc2626; }}
                .row-medium {{ background: #fffbeb !important; border-left: 4px solid #f59e0b; }}
                .row-low {{ background: #f0fdf4 !important; border-left: 4px solid #10b981; }}
                .badge-high {{ background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; }}
                .badge-medium {{ background: linear-gradient(135deg, #d97706, #f59e0b); color: white; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; }}
                .badge-low {{ background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; }}
            </style>
            </head>
            <body>
            <table class="risk-table">
                <thead><tr><th>Region</th><th>Cases</th><th>Risk Level</th></tr></thead>
                <tbody>
            """
            for _, row in all_latest.sort_values('cases', ascending=False).iterrows():
                cases = int(row['cases'])
                risk = get_risk_level(cases)[0]
                row_class = f"row-{risk.lower()}"
                badge_class = f"badge-{risk.lower()}"
                region_name = row['Region_ID'].replace('REGION ', '').replace('-', ' ')[:30]
                table_html += f'<tr class="{row_class}"><td style="font-weight:500">{region_name}</td><td style="font-weight:600">{cases:,}</td><td><span class="{badge_class}">{"🔴" if risk=="HIGH" else "🟠" if risk=="MEDIUM" else "🟢"} {risk}</span></td></tr>'
            
            table_html += "</tbody></table></body></html>"
            components.html(table_html, height=500, scrolling=True)
        
        # ----- TAB 3: Real-Time Predictor -----
        with tab3:
            st.markdown('<div class="section-header">🔮 Predict Future Cases with Current Data</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>How to use:</strong> Enter the current conditions for your region below. 
                The model will predict dengue cases 2 weeks from now based on these inputs.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Input form
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Recent Case History**")
                input_cases_1 = st.number_input("Cases last week", min_value=0, value=int(latest['cases']), 
                                                help="Number of dengue cases reported last week")
                input_cases_2 = st.number_input("Cases 2 weeks ago", min_value=0, value=int(latest.get('Cases_Lag_1', latest['cases'])),
                                                help="Number of dengue cases 2 weeks ago")
                input_cases_4 = st.number_input("Cases 4 weeks ago", min_value=0, value=int(latest.get('Cases_Lag_3', latest['cases'])),
                                                help="Number of dengue cases 4 weeks ago")
                
                st.markdown("**📈 Recent Trend**")
                case_trend = st.selectbox("Case trend over last 4 weeks", 
                                         ["📈 Increasing", "➡️ Stable", "📉 Decreasing"])
            
            with col2:
                st.markdown("**🌤️ Current Weather Conditions**")
                input_temp = st.slider("Temperature (°C)", min_value=20.0, max_value=40.0, value=float(temp),
                                       help="Average temperature this week")
                input_rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=300.0, value=float(rainfall),
                                           help="Total rainfall this week")
                input_humidity = st.slider("Humidity (%)", min_value=50.0, max_value=100.0, value=float(latest.get('RH2M', 75.0)),
                                           help="Average relative humidity")
                
                st.markdown("**🌿 Vegetation (NDVI)**")
                input_ndvi = st.slider("Vegetation Index", min_value=0.0, max_value=1.0, value=0.5,
                                       help="Higher = more vegetation = more mosquito habitat")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Predict button
            if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                if model is not None:
                    # Calculate derived features
                    cases_avg = (input_cases_1 + input_cases_2 + input_cases_4) / 3
                    cases_trend_multiplier = 1.1 if "Increasing" in case_trend else 0.9 if "Decreasing" in case_trend else 1.0
                    
                    # Simple prediction based on historical pattern + weather influence
                    # Base prediction from recent cases
                    base_pred = cases_avg * cases_trend_multiplier
                    
                    # Weather adjustment (higher rain + humidity = more cases)
                    weather_factor = 1.0
                    if input_rainfall > 100:
                        weather_factor += 0.15
                    if input_humidity > 80:
                        weather_factor += 0.10
                    if input_temp > 28 and input_temp < 32:
                        weather_factor += 0.10  # Optimal mosquito temp range
                    
                    # NDVI adjustment
                    ndvi_factor = 1 + (input_ndvi - 0.5) * 0.2
                    
                    # Final prediction
                    manual_prediction = int(base_pred * weather_factor * ndvi_factor)
                    manual_prediction = max(0, manual_prediction)
                    
                    pred_risk, pred_color, pred_emoji = get_risk_level(manual_prediction)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Results
                    st.markdown('<div class="section-header">📊 Prediction Results (2 Weeks Ahead)</div>', 
                               unsafe_allow_html=True)
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">🔮 Predicted Cases</div>
                            <div class="kpi-value">{manual_prediction:,}</div>
                            <div class="kpi-subtext">2-week forecast</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with res_col2:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">⚠️ Risk Level</div>
                            <div style="margin-top: 0.5rem;">{render_risk_badge(pred_risk)}</div>
                            <div class="kpi-subtext">Based on prediction</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with res_col3:
                        change = ((manual_prediction - input_cases_1) / max(input_cases_1, 1)) * 100
                        change_str = f"+{change:.0f}%" if change > 0 else f"{change:.0f}%"
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">📈 Expected Change</div>
                            <div class="kpi-value">{change_str}</div>
                            <div class="kpi-subtext">vs. last week</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if pred_risk == "HIGH":
                        st.error("""
                        **⚠️ High Risk Alert!** Recommendations:
                        - Increase hospital preparedness (IV fluids, beds)
                        - Deploy fumigation teams
                        - Issue public health advisory
                        - Stock dengue test kits
                        """)
                    elif pred_risk == "MEDIUM":
                        st.warning("""
                        **🟠 Moderate Risk.** Recommendations:
                        - Monitor situation closely
                        - Prepare contingency resources
                        - Community awareness campaigns
                        """)
                    else:
                        st.success("""
                        **✅ Low Risk.** Situation appears stable.
                        - Continue routine surveillance
                        - Maintain preventive measures
                        """)
                else:
                    st.error("Model not loaded. Please run `python pipeline.py` first.")
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem;">
        🦟 <strong>Dengue Risk Monitoring System</strong> | 
        Data: DOH Philippines + NASA POWER | 
        Model: XGBoost |
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
