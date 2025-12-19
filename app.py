"""
AI-Based Smart Carbon Footprint & Green Habit Recommendation System
====================================================================

A professional Streamlit application for calculating carbon footprints
and providing personalized green habit recommendations using ML.

Author: [Student Name]
Date: December 2025
Purpose: Educational project for environmental awareness
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# ============================================
# CONFIGURATION & SETUP
# ============================================

# Add src directory to path for importing custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommendation_engine import RecommendationEngine

# Configure Streamlit page settings
st.set_page_config(
    page_title="Smart Carbon Footprint Calculator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional green-tech theme
st.markdown("""
    <style>
    /* ============================================
       GLOBAL STYLES & THEME
       ============================================ */
    
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
    }
    
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a1a1a;
    }
    
    /* ============================================
       HEADER STYLES
       ============================================ */
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.15rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* ============================================
       SIDEBAR STYLES
       ============================================ */
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f8f4 100%);
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] .element-container h2 {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #4caf50;
    }
    
    /* Sidebar subheaders */
    [data-testid="stSidebar"] .element-container h3 {
        color: #1b5e20;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* ============================================
       CARD STYLES
       ============================================ */
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f8f4 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.15);
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-card h4 {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    
    .recommendation-card p {
        line-height: 1.6;
        color: #424242;
    }
    
    /* Priority colors */
    .high-priority {
        border-left-color: #d32f2f;
        background: linear-gradient(135deg, #ffffff 0%, #ffebee 100%);
    }
    
    .medium-priority {
        border-left-color: #f57c00;
        background: linear-gradient(135deg, #ffffff 0%, #fff3e0 100%);
    }
    
    .low-priority {
        border-left-color: #388e3c;
        background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%);
    }
    
    .positive-impact {
        border-left-color: #1976d2;
        background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%);
    }
    
    /* ============================================
       BUTTON STYLES
       ============================================ */
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(67, 160, 71, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        box-shadow: 0 6px 20px rgba(67, 160, 71, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(67, 160, 71, 0.3);
    }
    
    /* ============================================
       METRIC CONTAINER STYLES
       ============================================ */
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #2e7d32;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* ============================================
       SLIDER & INPUT STYLES
       ============================================ */
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #c8e6c9;
    }
    
    .stSlider > div > div > div > div {
        background-color: #4caf50;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #4caf50;
    }
    
    /* ============================================
       INFO & ALERT BOXES
       ============================================ */
    
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    /* Info box */
    div[data-baseweb="notification"] {
        background-color: #e3f2fd;
        border-radius: 12px;
        border-left: 5px solid #1976d2;
        padding: 1rem 1.5rem;
    }
    
    /* ============================================
       CHART & VISUALIZATION STYLES
       ============================================ */
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        overflow: hidden;
    }
    
    /* ============================================
       SECTION DIVIDERS
       ============================================ */
    
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #4caf50 50%, transparent 100%);
    }
    
    /* ============================================
       TABS STYLING
       ============================================ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        color: #546e7a;
        font-weight: 600;
        border: 2px solid #e0e0e0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4caf50;
        color: white;
        border-color: #4caf50;
    }
    
    /* ============================================
       EXPANDER STYLING
       ============================================ */
    
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        font-weight: 600;
        color: #2e7d32;
        padding: 0.75rem 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #4caf50;
        background-color: #f1f8f4;
    }
    
    /* ============================================
       FOOTER STYLES
       ============================================ */
    
    .footer {
        text-align: center;
        color: #78909c;
        padding: 3rem 0 2rem 0;
        font-size: 0.9rem;
        border-top: 2px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    .footer a {
        color: #4caf50;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* ============================================
       SCROLL BAR STYLING
       ============================================ */
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
    }
    
    /* ============================================
       LOADING SPINNER
       ============================================ */
    
    .stSpinner > div {
        border-top-color: #4caf50 !important;
    }
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    
    /* Tablets and small laptops */
    @media (max-width: 1024px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        [data-testid="stSidebar"] {
            min-width: 250px;
        }
    }
    
    /* Tablets */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 0.5rem 0;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .stButton>button {
            font-size: 1.1rem;
            padding: 0.875rem 1.5rem;
        }
        
        /* Stack columns on tablets */
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    
    /* Mobile phones */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem;
            padding: 0.5rem 0;
        }
        
        .sub-header {
            font-size: 0.9rem;
        }
        
        .stButton>button {
            font-size: 1rem;
            padding: 0.75rem 1.25rem;
            width: 100%;
        }
        
        /* Adjust sidebar for mobile */
        [data-testid="stSidebar"] {
            padding: 1rem 0.5rem;
        }
        
        /* Make metrics stack vertically */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
    }
    
    </style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def initialize_session_state():
    """Initialize session state variables for storing calculation results."""
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False

# ============================================
# RESOURCE LOADING (CACHED)
# ============================================

@st.cache_resource
def load_recommendation_engine():
    """
    Load and cache the ML-powered recommendation engine.
    
    Returns:
        RecommendationEngine: Loaded recommendation engine instance
        None: If loading fails
    
    Note: Uses @st.cache_resource to load only once and persist across reruns
    """
    try:
        engine = RecommendationEngine('../models/carbon_model.pkl')
        return engine
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ============================================
# VISUALIZATION COMPONENTS
# ============================================

def create_gauge_chart(score, max_score=100):
    """
    Create a professional gauge chart for displaying eco score.
    
    Args:
        score (float): Eco score value (0-100)
        max_score (int): Maximum score value, default 100
    
    Returns:
        plotly.graph_objects.Figure: Configured gauge chart
    
    Design: Clean, color-coded gauge with rating labels
    """
    
    # Determine color and rating based on score
    if score >= 80:
        color = "#4CAF50"
        rating = "Excellent"
    elif score >= 60:
        color = "#8BC34A"
        rating = "Good"
    elif score >= 40:
        color = "#FF9800"
        rating = "Fair"
    else:
        color = "#F44336"
        rating = "Needs Improvement"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{rating}</b>",
            'font': {'size': 20, 'color': color, 'family': 'Arial'}
        },
        number = {
            'font': {'size': 48, 'color': color, 'family': 'Arial'},
            'suffix': '/100'
        },
        gauge = {
            'axis': {
                'range': [None, max_score],
                'tickwidth': 1,
                'tickcolor': "rgba(100, 100, 100, 0.3)",
                'tickfont': {'size': 12, 'color': '#546e7a'}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(240, 240, 240, 0.5)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(244, 67, 54, 0.15)'},
                {'range': [40, 60], 'color': 'rgba(255, 152, 0, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(139, 195, 74, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(76, 175, 80, 0.15)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#37474f", 'family': "Arial"}
    )
    
    return fig

def create_emission_breakdown_chart(breakdown):
    """
    Create a horizontal bar chart showing emission breakdown by category.
    
    Args:
        breakdown (dict): Dictionary with emission categories and values
    
    Returns:
        plotly.graph_objects.Figure: Configured bar chart
    
    Features: Category-specific colors, minimal grid lines, clear labels
    """
    
    # Remove 'total' from breakdown and prepare data
    data = {k.replace('_', ' ').title(): v for k, v in breakdown.items() if k != 'total'}
    
    df = pd.DataFrame(list(data.items()), columns=['Category', 'Emissions'])
    df = df.sort_values('Emissions', ascending=False)  # Highest on top
    
    # Category-specific colors
    category_colors = {
        'Transport': '#FF6B6B',
        'Electricity': '#4ECDC4',
        'Water': '#45B7D1',
        'Diet': '#FFA07A',
        'Waste': '#98D8C8'
    }
    
    colors = [category_colors.get(cat, '#95A5A6') for cat in df['Category']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Emissions'],
        y=df['Category'],
        orientation='h',
        text=[f"{val:.1f} kg" for val in df['Emissions']],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Emissions: %{x:.1f} kg CO₂<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Emission Sources</b>",
            font=dict(size=18, color='#2e7d32', family='Arial')
        ),
        xaxis=dict(
            title="<b>CO₂ Emissions (kg/month)</b>",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0, 0, 0, 0.2)',
            tickfont=dict(size=12, color='#546e7a'),
            titlefont=dict(size=13, color='#37474f')
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=13, color='#37474f', family='Arial')
        ),
        showlegend=False,
        height=380,
        margin=dict(l=100, r=50, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.5)",
        hovermode='closest'
    )
    
    return fig

def create_monthly_trend_chart(current_footprint, potential_footprint):
    """
    Create a line chart showing 6-month carbon footprint projection.
    
    Args:
        current_footprint (float): Current monthly CO₂ emissions (kg)
        potential_footprint (float): Target monthly CO₂ emissions (kg)
    
    Returns:
        plotly.graph_objects.Figure: Configured line chart
    
    Features: Dual lines (current vs. improvement), filled area showing savings
    """
    
    # Generate 6-month projection data
    months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
    
    # Current footprint (baseline)
    current_trend = [current_footprint] * 6
    
    # Gradual improvement trend (linear reduction)
    improvement_trend = [
        current_footprint,
        current_footprint - (current_footprint - potential_footprint) * 0.2,
        current_footprint - (current_footprint - potential_footprint) * 0.4,
        current_footprint - (current_footprint - potential_footprint) * 0.6,
        current_footprint - (current_footprint - potential_footprint) * 0.8,
        potential_footprint
    ]
    
    fig = go.Figure()
    
    # Current footprint line
    fig.add_trace(go.Scatter(
        x=months,
        y=current_trend,
        mode='lines+markers',
        name='Current Path',
        line=dict(color='#F44336', width=3, dash='dash'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Footprint: %{y:.1f} kg CO₂<extra></extra>'
    ))
    
    # Improvement trajectory
    fig.add_trace(go.Scatter(
        x=months,
        y=improvement_trend,
        mode='lines+markers',
        name='With Green Habits',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10, symbol='circle'),
        fill='tonexty',
        fillcolor='rgba(76, 175, 80, 0.1)',
        hovertemplate='<b>%{x}</b><br>Footprint: %{y:.1f} kg CO₂<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>6-Month Carbon Footprint Projection</b>",
            font=dict(size=18, color='#2e7d32', family='Arial')
        ),
        xaxis=dict(
            title="<b>Timeline</b>",
            showgrid=False,
            tickfont=dict(size=12, color='#546e7a'),
            titlefont=dict(size=13, color='#37474f')
        ),
        yaxis=dict(
            title="<b>CO₂ Emissions (kg/month)</b>",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            tickfont=dict(size=12, color='#546e7a'),
            titlefont=dict(size=13, color='#37474f')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            font=dict(size=12, color='#37474f')
        ),
        height=400,
        margin=dict(l=60, r=30, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.5)",
        hovermode='x unified'
    )
    
    return fig

def create_savings_chart(current, potential_savings):
    """
    Create a bar chart comparing current vs. potential carbon footprint.
    
    Args:
        current (float): Current monthly CO₂ emissions (kg)
        potential_savings (float): Achievable CO₂ savings (kg)
    
    Returns:
        plotly.graph_objects.Figure: Configured comparison bar chart
    
    Features: Color-coded bars, savings annotation, minimal grid
    """
    
    # Calculate potential footprint after savings
    potential = current - potential_savings
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Current', 'Target'],
        y=[current, potential],
        text=[f"<b>{current:.1f} kg</b>", f"<b>{potential:.1f} kg</b>"],
        textposition='outside',
        marker=dict(
            color=['#F44336', '#4CAF50'],
            line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
        ),
        textfont=dict(size=16, color='#37474f', family='Arial'),
        hovertemplate='<b>%{x} Footprint</b><br>%{y:.1f} kg CO₂<extra></extra>'
    ))
    
    # Add savings annotation
    fig.add_annotation(
        x=0.5, y=max(current, potential) * 0.75,
        text=f"<b>💡 {potential_savings:.1f} kg CO₂</b><br>({((potential_savings/current)*100):.1f}% reduction)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#4CAF50",
        ax=0,
        ay=-50,
        font=dict(size=14, color="#2E7D32", family="Arial"),
        bgcolor="rgba(232, 245, 233, 0.95)",
        bordercolor="#4CAF50",
        borderwidth=2,
        borderpad=8
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Savings Potential</b>",
            font=dict(size=18, color='#2e7d32', family='Arial')
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(size=13, color='#37474f', family='Arial')
        ),
        yaxis=dict(
            title="<b>CO₂ Emissions (kg/month)</b>",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0, 0, 0, 0.2)',
            tickfont=dict(size=12, color='#546e7a'),
            titlefont=dict(size=13, color='#37474f')
        ),
        showlegend=False,
        height=400,
        margin=dict(l=60, r=30, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.5)"
    )
    
    return fig

def display_recommendations(recommendations):
    """
    Display recommendations as interactive checklist cards.
    
    Args:
        recommendations (list): List of recommendation dictionaries with keys:
            - category: Category name (Transport, Electricity, etc.)
            - priority: Priority level (High, Medium, Low)
            - suggestion: Recommendation text
            - potential_savings_kg_co2: Savings amount
    
    Features: Checkboxes, eco icons, friendly language, color-coded priorities
    """
    
    # Enhanced category icons with eco-friendly alternatives
    category_icons = {
        'Transport': '🚲',  # Bicycle for eco-transport
        'Electricity': '💡',  # Lightbulb for energy
        'Water': '💧',
        'Diet': '🌱',  # Plant for sustainable diet
        'Waste': '♻️'
    }
    
    # Additional eco icons for variety
    eco_icons = ['🌿', '🌍', '🌳', '☀️', '🍃', '🌾']
    
    # Priority colors
    priority_colors = {
        'High': '#f44336',
        'Medium': '#ff9800',
        'Low': '#4caf50',
        'Positive': '#2196f3'
    }
    
    for idx, rec in enumerate(recommendations):
        priority = rec.get('priority', 'Low')
        category = rec.get('category', 'General')
        savings = rec.get('potential_savings_kg_co2', 0)
        suggestion = rec.get('suggestion', '')
        
        # Get category icon
        category_icon = category_icons.get(category, '🌱')
        priority_color = priority_colors.get(priority, '#4caf50')
        
        # Random eco icon for visual variety
        eco_icon = eco_icons[idx % len(eco_icons)]
        
        # Create friendly, conversational message
        friendly_message = suggestion
        
        # Add savings context in friendly language
        if savings > 0:
            if savings >= 20:
                impact_level = "significant"
                impact_emoji = "🌟"
            elif savings >= 10:
                impact_level = "meaningful"
                impact_emoji = "✨"
            else:
                impact_level = "positive"
                impact_emoji = "💚"
            
            savings_message = f"{impact_emoji} This can save you <strong>{savings:.1f} kg CO₂</strong> every month!"
        else:
            savings_message = ""
        
        # Create checklist card
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f9fdf9 100%);
                        padding: 1.5rem;
                        border-radius: 16px;
                        border-left: 5px solid {priority_color};
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                        margin-bottom: 1rem;
                        transition: transform 0.2s ease, box-shadow 0.2s ease;">
                
                <!-- Checkbox and Category Header -->
                <div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem;">
                    <div style="flex-shrink: 0;">
                        <input type="checkbox" id="rec_{idx}" 
                               style="width: 24px; height: 24px; cursor: pointer; accent-color: #4caf50;
                                      margin-top: 0.3rem;">
                    </div>
                    <div style="flex-grow: 1;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.8rem;">{category_icon}</span>
                            <span style="font-weight: 700; color: #2e7d32; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.5px;">
                                {category}
                            </span>
                        </div>
                        
                        <!-- Friendly Suggestion Message -->
                        <p style="color: #37474f; font-size: 1.05rem; line-height: 1.6; margin: 0.5rem 0;">
                            {eco_icon} {friendly_message}
                        </p>
                        
                        <!-- Savings Highlight -->
                        {f'''
                        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                                    padding: 1rem;
                                    border-radius: 12px;
                                    margin-top: 1rem;
                                    border: 2px solid #4caf50;">
                            <p style="color: #1b5e20; font-size: 1rem; font-weight: 600; margin: 0; line-height: 1.5;">
                                {savings_message}
                            </p>
                        </div>
                        ''' if savings > 0 else ''}
                        
                        <!-- Priority Badge -->
                        <div style="margin-top: 1rem;">
                            <span style="background: {priority_color};
                                        color: white;
                                        padding: 0.4rem 1rem;
                                        border-radius: 20px;
                                        font-size: 0.8rem;
                                        font-weight: 700;
                                        letter-spacing: 0.5px;
                                        text-transform: uppercase;">
                                {priority} Priority
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================================
# REUSABLE UI COMPONENTS
# ============================================

def render_header():
    """
    Render the main application header with title and tagline.
    
    Design: Clean, professional, exam-ready centered layout with proper spacing
    """
    st.markdown("""
        <div style="text-align: center; padding: 2.5rem 0 2rem 0;">
            <!-- Main Title -->
            <div class="main-header" style="margin-bottom: 1.5rem;">
                🌱 AI-Based Smart Carbon Footprint Calculator
            </div>
            
            <!-- Subtitle -->
            <div class="sub-header" style="font-size: 1.3rem; 
                                           color: #2e7d32; 
                                           margin-bottom: 1rem;
                                           font-weight: 600;">
                Smart Carbon Footprint & Green Habit Assistant
            </div>
            
            <!-- Description -->
            <p style="color: #78909c; 
                      font-size: 1rem; 
                      margin: 0;
                      line-height: 1.6;
                      max-width: 700px;
                      margin-left: auto;
                      margin-right: auto;">
                Powered by Machine Learning • Track • Analyze • Reduce Your Environmental Impact
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_carbon_footprint_explainer():
    """
    Render an expandable section explaining carbon footprint concept.
    
    Content: Definition, key contributors, importance, global averages
    """
    with st.expander("💡 What is a Carbon Footprint?", expanded=False):
        st.markdown("""
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border-left: 5px solid #4caf50;">
                <h4 style="color: #1b5e20; margin-top: 0;">🌍 Understanding Your Carbon Footprint</h4>
                
                <p style="color: #2e7d32; font-size: 1rem; line-height: 1.6;">
                    A <strong>carbon footprint</strong> is the total amount of greenhouse gases (mainly CO₂) 
                    generated by your daily activities and lifestyle choices. It's measured in kilograms or tonnes 
                    of CO₂ equivalent per year or month.
                </p>
                
                <h5 style="color: #1b5e20; margin-top: 1rem;">📊 Key Contributors:</h5>
                <ul style="color: #2e7d32; font-size: 0.95rem; line-height: 1.8;">
                    <li><strong>🚗 Transportation:</strong> Car, bus, train travel - biggest contributor for most people</li>
                    <li><strong>⚡ Electricity:</strong> Home energy use, appliances, heating/cooling</li>
                    <li><strong>🍽️ Diet:</strong> Food production, especially meat and dairy</li>
                    <li><strong>💧 Water:</strong> Treatment and distribution of water</li>
                    <li><strong>♻️ Waste:</strong> Disposal and decomposition of waste materials</li>
                </ul>
                
                <h5 style="color: #1b5e20; margin-top: 1rem;">🎯 Why It Matters:</h5>
                <p style="color: #2e7d32; font-size: 0.95rem; line-height: 1.6;">
                    Reducing your carbon footprint helps combat climate change, improves air quality, 
                    and contributes to a sustainable future for generations to come.
                </p>
                
                <div style="background: rgba(255, 255, 255, 0.7);
                            padding: 1rem;
                            border-radius: 8px;
                            margin-top: 1rem;">
                    <p style="color: #1b5e20; font-size: 0.9rem; margin: 0; font-weight: 600;">
                        ✨ <strong>Global Average:</strong> ~4 tonnes CO₂ per person per year<br>
                        🎯 <strong>Sustainable Target:</strong> ~2 tonnes CO₂ per person per year
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_disclaimer():
    """
    Render educational disclaimer notice.
    
    Purpose: Transparency about tool limitations and intended use
    """
    st.info("""
        📚 **Educational Purpose:** This calculator is designed for educational and awareness purposes. 
        Calculations are based on average emission factors and may not reflect exact real-world values. 
        For precise carbon accounting, consult with environmental professionals.
    """)

def render_hero_carbon_card(summary):
    """
    Render hero card displaying carbon footprint score with status badge.
    
    Args:
        summary (dict): Summary dictionary containing footprint data
    
    Design: Large centered card with color-coded impact level and message
    """
    current_co2 = summary['current_footprint_kg_co2']
    
    # Determine impact level based on emissions
    if current_co2 < 150:
        impact_level = "Low Impact"
        impact_color = "#4caf50"
        impact_bg = "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)"
        impact_icon = "✅"
        impact_message = "Excellent! You're doing great for the environment!"
    elif current_co2 < 300:
        impact_level = "Moderate Impact"
        impact_color = "#ff9800"
        impact_bg = "linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%)"
        impact_icon = "⚠️"
        impact_message = "Good start! There's room for improvement."
    else:
        impact_level = "High Impact"
        impact_color = "#f44336"
        impact_bg = "linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)"
        impact_icon = "🔴"
        impact_message = "Let's work together to reduce your footprint!"
    
    st.markdown(f"""
        <div style="background: {impact_bg};
                    padding: 3rem 2rem;
                    border-radius: 20px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
                    text-align: center;
                    margin-bottom: 2rem;
                    border: 3px solid {impact_color};">
            
            <!-- Status Badge -->
            <div style="display: inline-block;
                        background: {impact_color};
                        color: white;
                        padding: 0.5rem 1.5rem;
                        border-radius: 50px;
                        font-weight: 700;
                        font-size: 0.9rem;
                        letter-spacing: 1px;
                        margin-bottom: 1.5rem;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
                {impact_icon} {impact_level.upper()}
            </div>
            
            <!-- Main Score -->
            <div style="margin: 1.5rem 0;">
                <div style="color: #37474f; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                    Your Monthly Carbon Footprint
                </div>
                <div style="color: {impact_color};
                            font-size: 5rem;
                            font-weight: 900;
                            line-height: 1;
                            margin: 1rem 0;
                            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);">
                    {current_co2:.1f}
                </div>
                <div style="color: #546e7a; font-size: 1.8rem; font-weight: 600;">
                    kg CO₂ / month
                </div>
            </div>
            
            <!-- Status Message -->
            <div style="color: #455a64;
                        font-size: 1.1rem;
                        font-weight: 500;
                        margin-top: 1.5rem;
                        padding: 1rem;
                        background: rgba(255, 255, 255, 0.7);
                        border-radius: 12px;">
                {impact_message}
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_metric_cards(summary):
    """
    Render four dashboard metric cards showing key statistics.
    
    Args:
        summary (dict): Summary dictionary containing all metrics
    
    Design: 4-column grid with color-coded gradient cards
    """
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <h3 style="color: #2e7d32; font-size: 1.5rem; margin: 0;">
                📊 Monthly CO₂ Impact Dashboard
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Card 1: Current Footprint
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #ffebee 100%);
                        padding: 2rem 1.5rem;
                        border-radius: 16px;
                        border-top: 4px solid #f44336;
                        box-shadow: 0 6px 20px rgba(244, 67, 54, 0.15);
                        text-align: center;
                        min-height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌍</div>
                <div style="color: #f44336;
                            font-size: 2.5rem;
                            font-weight: 900;
                            margin: 0.5rem 0;">
                    {summary['current_footprint_kg_co2']:.1f}
                </div>
                <div style="color: #d32f2f; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">kg CO₂</div>
                <div style="color: #546e7a; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Current Footprint
                </div>
                <div style="color: #78909c; font-size: 0.75rem; margin-top: 0.5rem;">
                    Total monthly emissions
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Card 2: Potential Savings
    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%);
                        padding: 2rem 1.5rem;
                        border-radius: 16px;
                        border-top: 4px solid #4caf50;
                        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.15);
                        text-align: center;
                        min-height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">💚</div>
                <div style="color: #4caf50;
                            font-size: 2.5rem;
                            font-weight: 900;
                            margin: 0.5rem 0;">
                    {summary['total_potential_savings_kg_co2']:.1f}
                </div>
                <div style="color: #388e3c; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">kg CO₂</div>
                <div style="background: #4caf50; color: white; display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;">
                    -{summary['reduction_percentage']:.1f}% REDUCTION
                </div>
                <div style="color: #546e7a; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Potential Savings
                </div>
                <div style="color: #78909c; font-size: 0.75rem; margin-top: 0.5rem;">
                    Achievable monthly reduction
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Card 3: Target Footprint
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%);
                        padding: 2rem 1.5rem;
                        border-radius: 16px;
                        border-top: 4px solid #2196f3;
                        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.15);
                        text-align: center;
                        min-height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="color: #2196f3;
                            font-size: 2.5rem;
                            font-weight: 900;
                            margin: 0.5rem 0;">
                    {summary['potential_footprint_kg_co2']:.1f}
                </div>
                <div style="color: #1976d2; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">kg CO₂</div>
                <div style="color: #546e7a; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Target Footprint
                </div>
                <div style="color: #78909c; font-size: 0.75rem; margin-top: 0.5rem;">
                    After implementing changes
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Card 4: Eco Score
    with col4:
        eco_score = summary['eco_score']
        if eco_score >= 80:
            eco_color = "#4caf50"
            eco_label = "Excellent"
        elif eco_score >= 60:
            eco_color = "#8bc34a"
            eco_label = "Good"
        elif eco_score >= 40:
            eco_color = "#ff9800"
            eco_label = "Fair"
        else:
            eco_color = "#f44336"
            eco_label = "Needs Work"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #fff9c4 100%);
                        padding: 2rem 1.5rem;
                        border-radius: 16px;
                        border-top: 4px solid {eco_color};
                        box-shadow: 0 6px 20px rgba(255, 193, 7, 0.15);
                        text-align: center;
                        min-height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">⭐</div>
                <div style="color: {eco_color};
                            font-size: 2.5rem;
                            font-weight: 900;
                            margin: 0.5rem 0;">
                    {eco_score}
                </div>
                <div style="color: {eco_color}; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">/ 100</div>
                <div style="background: {eco_color}; color: white; display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {eco_label.upper()}
                </div>
                <div style="color: #546e7a; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    Eco Score
                </div>
                <div style="color: #78909c; font-size: 0.75rem; margin-top: 0.5rem;">
                    Environmental rating
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_professional_footer():
    """
    Render comprehensive project footer with details and disclaimer.
    
    Content: Project info, technologies, features, academic details, copyright
    """
    st.markdown("---")
    st.markdown("""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
                    padding: 3rem 2rem;
                    border-radius: 16px;
                    margin-top: 3rem;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);">
            
            <!-- Project Title -->
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #2e7d32; font-size: 1.8rem; margin-bottom: 0.5rem; font-weight: 700;">
                    🌱 AI-Based Smart Carbon Footprint & Green Habit Recommendation System
                </h2>
                <p style="color: #546e7a; font-size: 1rem; margin: 0;">
                    An Intelligent Solution for Environmental Awareness
                </p>
            </div>
            
            <div style="border-top: 2px solid #c8e6c9; margin: 2rem 0;"></div>
            
            <!-- Details Grid -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
                
                <!-- Column 1: Technologies -->
                <div style="text-align: center;">
                    <h4 style="color: #1b5e20; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">
                        🛠️ Technologies Used
                    </h4>
                    <p style="color: #546e7a; font-size: 0.9rem; line-height: 1.8; margin: 0;">
                        Python 3.13+<br>
                        Streamlit Framework<br>
                        Scikit-learn (ML)<br>
                        TensorFlow/Keras (LSTM)<br>
                        Plotly (Visualizations)
                    </p>
                </div>
                
                <!-- Column 2: Features -->
                <div style="text-align: center;">
                    <h4 style="color: #1b5e20; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">
                        ✨ Key Features
                    </h4>
                    <p style="color: #546e7a; font-size: 0.9rem; line-height: 1.8; margin: 0;">
                        ML-Powered Predictions<br>
                        Real-time Analysis<br>
                        Personalized Recommendations<br>
                        Interactive Visualizations<br>
                        6-Month Projections
                    </p>
                </div>
                
                <!-- Column 3: Purpose -->
                <div style="text-align: center;">
                    <h4 style="color: #1b5e20; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">
                        🎯 Project Purpose
                    </h4>
                    <p style="color: #546e7a; font-size: 0.9rem; line-height: 1.8; margin: 0;">
                        Educational Tool<br>
                        Environmental Awareness<br>
                        Behavioral Change<br>
                        Sustainability Promotion<br>
                        Academic Research
                    </p>
                </div>
                
            </div>
            
            <div style="border-top: 2px solid #c8e6c9; margin: 2rem 0;"></div>
            
            <!-- Student & Academic Info -->
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h4 style="color: #1b5e20; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">
                    🎓 Academic Project
                </h4>
                <p style="color: #546e7a; font-size: 0.95rem; line-height: 1.8; margin: 0;">
                    <strong>Project Type:</strong> Machine Learning & Environmental Science<br>
                    <strong>Academic Year:</strong> 2025<br>
                    <strong>Developed By:</strong> [Student Name] | [University/Institution Name]<br>
                    <strong>Course:</strong> Green Skills for All / Environmental AI Applications<br>
                    <strong>Supervisor:</strong> [Supervisor Name]
                </p>
            </div>
            
            <div style="border-top: 2px solid #c8e6c9; margin: 2rem 0;"></div>
            
            <!-- Copyright & Credits -->
            <div style="text-align: center;">
                <p style="color: #78909c; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    🌍 <strong>Made with ❤️ for a Sustainable Future</strong>
                </p>
                <p style="color: #90a4ae; font-size: 0.85rem; margin: 0;">
                    &copy; 2025 Smart Carbon Footprint Project. All rights reserved.<br>
                    Powered by AI & Machine Learning | Open for Educational Use
                </p>
            </div>
            
            <!-- Disclaimer -->
            <div style="background: rgba(255, 255, 255, 0.6);
                        padding: 1rem;
                        border-radius: 8px;
                        margin-top: 1.5rem;
                        border-left: 4px solid #ff9800;">
                <p style="color: #e65100; font-size: 0.8rem; margin: 0; text-align: center; line-height: 1.6;">
                    <strong>⚠️ Disclaimer:</strong> This tool provides estimates based on average emission factors 
                    and is intended for educational purposes only. Results may vary from actual carbon emissions. 
                    For professional carbon accounting, please consult certified environmental specialists.
                </p>
            </div>
            
        </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION FUNCTION
# ============================================

def main():
    """
    Main application entry point.
    
    Flow:
        1. Initialize session state
        2. Load recommendation engine
        3. Render header and explainer
        4. Render sidebar with input controls
        5. Process calculations on button click
        6. Display results (hero card, metrics, charts, recommendations)
        7. Render footer
    
    Performance: Uses caching for resource loading and efficient reruns
    """
    
    # Initialize session state
    initialize_session_state()
    
    # ============================================
    # SECTION 1: TOP HEADER & INFO
    # ============================================
    render_header()
    
    # Clear spacing after header
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    render_carbon_footprint_explainer()
    
    # Spacing before disclaimer
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    render_disclaimer()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load engine
    engine = load_recommendation_engine()
    
    if engine is None:
        st.error("❌ Failed to load the recommendation engine. Please check if the model file exists.")
        return
    
    # ============================================
    # SECTION 2: SIDEBAR - USER INPUTS
    # ============================================
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #2e7d32; margin-bottom: 0.5rem;">📊 Your Lifestyle Data</h2>
                <p style="color: #78909c; font-size: 0.85rem;">
                    Enter your daily habits to calculate your carbon footprint
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Help Info
        with st.expander("❓ Need Help?", expanded=False):
            st.markdown("""
                <div style="font-size: 0.85rem; color: #546e7a; line-height: 1.6;">
                    <p><strong>📍 How to use:</strong></p>
                    <ol style="padding-left: 1.2rem;">
                        <li>Choose a preset or enter custom values</li>
                        <li>Adjust sliders to match your lifestyle</li>
                        <li>Click "Calculate" to see results</li>
                        <li>Review recommendations and savings</li>
                    </ol>
                    <p><strong>💡 Tips:</strong></p>
                    <ul style="padding-left: 1.2rem;">
                        <li>Be honest for accurate results</li>
                        <li>Check utility bills for exact data</li>
                        <li>Consider household averages</li>
                        <li>Update regularly to track progress</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Quick Test Presets
        st.markdown("""<div style="background: #e3f2fd; padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="font-size: 0.85rem; color: #1565c0; margin: 0; font-weight: 500;">
                🎯 <strong>Quick Test Presets:</strong>
            </p>
        </div>""", unsafe_allow_html=True)
        
        preset = st.selectbox(
            "Load sample profile",
            options=['custom', 'eco_friendly', 'average', 'high_impact'],
            format_func=lambda x: {
                'custom': '✏️ Custom (Enter your own values)',
                'eco_friendly': '🌿 Eco-Friendly (Low carbon lifestyle)',
                'average': '👤 Average User (Typical lifestyle)',
                'high_impact': '⚠️ High Impact (Carbon-intensive)'
            }[x],
            help="Choose a preset to quickly test the calculator, or use 'Custom' to enter your own values"
        )
        
        # Set default values based on preset
        if preset == 'eco_friendly':
            default_transport = 10.0
            default_electricity = 200.0
            default_water = 120.0
            default_diet = 'veg'
            default_waste = 3.0
        elif preset == 'average':
            default_transport = 25.0
            default_electricity = 350.0
            default_water = 180.0
            default_diet = 'mixed'
            default_waste = 7.0
        elif preset == 'high_impact':
            default_transport = 60.0
            default_electricity = 600.0
            default_water = 300.0
            default_diet = 'non-veg'
            default_waste = 15.0
        else:  # custom
            default_transport = 25.0
            default_electricity = 350.0
            default_water = 180.0
            default_diet = 'mixed'
            default_waste = 7.0
        
        st.markdown("---")
        
        # Transport Section
        st.markdown("### 🚗 Transportation")
        st.caption("Daily commute and travel habits")
        transport_km = st.slider(
            "Daily travel distance (km)",
            min_value=0.0,
            max_value=150.0,
            value=default_transport,
            step=1.0,
            help="🚗 Include all motorized transport: car, bus, train, taxi. Walking/cycling = 0 km. Eco-friendly range: < 15 km/day"
        )
        
        # Visual feedback and validation
        monthly_km = transport_km * 30
        monthly_co2 = transport_km * 0.12 * 30
        
        if transport_km == 0:
            st.success("🌟 Excellent! Zero emissions from transport!")
        elif transport_km < 15:
            st.success(f"✅ Eco-friendly! **{transport_km} km/day** ≈ {monthly_km:.0f} km/month | ~{monthly_co2:.1f} kg CO₂")
        elif transport_km < 40:
            st.warning(f"⚠️ Moderate usage: **{transport_km} km/day** ≈ {monthly_km:.0f} km/month | ~{monthly_co2:.1f} kg CO₂")
        else:
            st.error(f"🔴 High impact: **{transport_km} km/day** ≈ {monthly_km:.0f} km/month | ~{monthly_co2:.1f} kg CO₂")
        
        st.markdown("---")
        
        # Electricity Section
        st.markdown("### ⚡ Electricity Usage")
        st.caption("Home energy consumption")
        electricity_kwh = st.slider(
            "Monthly electricity consumption (kWh)",
            min_value=0.0,
            max_value=1000.0,
            value=default_electricity,
            step=10.0,
            help="⚡ Check your monthly utility bill. Average household: 250-400 kWh. Eco-friendly: < 250 kWh. Include all appliances, AC, heating."
        )
        
        # Visual feedback and validation
        avg_daily = electricity_kwh / 30
        monthly_co2_elec = electricity_kwh * 0.5
        
        if electricity_kwh == 0:
            st.success("🌟 Off-grid or renewable energy? Amazing!")
        elif electricity_kwh < 250:
            st.success(f"✅ Efficient! **{avg_daily:.1f} kWh/day** | {electricity_kwh:.0f} kWh/month | ~{monthly_co2_elec:.1f} kg CO₂")
        elif electricity_kwh < 450:
            st.warning(f"⚠️ Moderate: **{avg_daily:.1f} kWh/day** | {electricity_kwh:.0f} kWh/month | ~{monthly_co2_elec:.1f} kg CO₂")
        else:
            st.error(f"🔴 High usage: **{avg_daily:.1f} kWh/day** | {electricity_kwh:.0f} kWh/month | ~{monthly_co2_elec:.1f} kg CO₂")
        
        st.markdown("---")
        
        # Water Section
        st.markdown("### 💧 Water Consumption")
        st.caption("Daily water usage")
        water_liters = st.slider(
            "Daily water usage (liters)",
            min_value=0.0,
            max_value=500.0,
            value=default_water,
            step=5.0,
            help="💧 Include all water usage: drinking, cooking, bathing, washing, gardening. Average: 150-200 L/day. Eco-friendly: < 150 L/day."
        )
        
        # Visual feedback and validation
        monthly_liters = water_liters * 30
        monthly_co2_water = water_liters * 0.001 * 30  # Rough estimate
        
        if water_liters == 0:
            st.error("❌ Invalid: Water consumption cannot be zero. Please enter a realistic value.")
        elif water_liters < 150:
            st.success(f"✅ Water-efficient! **{water_liters:.0f} L/day** | {monthly_liters:.0f} L/month | ~{monthly_co2_water:.1f} kg CO₂")
        elif water_liters < 250:
            st.warning(f"⚠️ Moderate usage: **{water_liters:.0f} L/day** | {monthly_liters:.0f} L/month | ~{monthly_co2_water:.1f} kg CO₂")
        else:
            st.error(f"🔴 High usage: **{water_liters:.0f} L/day** | {monthly_liters:.0f} L/month | ~{monthly_co2_water:.1f} kg CO₂")
        
        st.markdown("---")
        
        # Diet Section
        st.markdown("### 🍽️ Diet Preference")
        st.caption("Your food choices impact")
        
        # Set default diet index
        diet_options = ['veg', 'mixed', 'non-veg']
        default_diet_index = diet_options.index(default_diet)
        
        diet_type = st.selectbox(
            "Primary diet type",
            options=diet_options,
            index=default_diet_index,
            format_func=lambda x: {
                'veg': '🌱 Vegetarian (Plant-based)',
                'mixed': '🍴 Mixed (Omnivore)',
                'non-veg': '🥩 Non-Vegetarian (Meat-heavy)'
            }[x],
            help="🍽️ Diet choice significantly impacts CO₂. Vegetarian: lowest emissions. Non-veg: 2-3x higher carbon footprint than plant-based."
        )
        
        # Enhanced diet info with CO2 estimates
        diet_details = {
            'veg': ('✅ Excellent choice! Lowest carbon footprint', '#4caf50', '~50-70 kg CO₂/month'),
            'mixed': ('⚠️ Moderate impact - Consider reducing meat intake', '#ff9800', '~80-120 kg CO₂/month'),
            'non-veg': ('🔴 High carbon impact - Try plant-based alternatives', '#f44336', '~130-180 kg CO₂/month')
        }
        
        msg, color, estimate = diet_details[diet_type]
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffffff 0%, {color}15 100%); 
                        padding: 1rem; border-radius: 8px; border-left: 4px solid {color};">
                <p style="margin: 0; color: {color}; font-weight: 600;">{msg}</p>
                <p style="margin: 0.3rem 0 0 0; color: #546e7a; font-size: 0.85rem;">Est. diet emissions: {estimate}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Waste Section
        st.markdown("### ♻️ Waste Generation")
        st.caption("Weekly waste production")
        waste_kg = st.slider(
            "Weekly waste (kg)",
            min_value=0.0,
            max_value=25.0,
            value=default_waste,
            step=0.5,
            help="♻️ Include all household waste: food scraps, packaging, recyclables. Average: 5-10 kg/week. Eco-friendly: < 5 kg/week."
        )
        
        # Visual feedback and validation
        monthly_waste = waste_kg * 4
        yearly_waste = waste_kg * 52
        
        if waste_kg == 0:
            st.success("🌟 Zero waste lifestyle! Incredible achievement!")
        elif waste_kg < 5:
            st.success(f"✅ Low waste! **{waste_kg:.1f} kg/week** | {monthly_waste:.1f} kg/month | {yearly_waste:.0f} kg/year")
        elif waste_kg < 12:
            st.warning(f"⚠️ Moderate waste: **{waste_kg:.1f} kg/week** | {monthly_waste:.1f} kg/month | {yearly_waste:.0f} kg/year")
        else:
            st.error(f"🔴 High waste: **{waste_kg:.1f} kg/week** | {monthly_waste:.1f} kg/month | {yearly_waste:.0f} kg/year")
        
        st.markdown("---")
        
        # Input Validation Summary
        st.markdown("---")
        st.markdown("""<div style="background: #fff3e0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff9800;">
            <p style="font-size: 0.85rem; color: #e65100; margin: 0; font-weight: 600;">
                📋 <strong>Validation Check</strong>
            </p>
        </div>""", unsafe_allow_html=True)
        
        # Perform validation
        validation_errors = []
        validation_warnings = []
        
        if water_liters == 0:
            validation_errors.append("❌ Water consumption cannot be zero")
        if transport_km > 100:
            validation_warnings.append("⚠️ Transport distance seems unusually high")
        if electricity_kwh > 800:
            validation_warnings.append("⚠️ Electricity usage is very high")
        if waste_kg > 20:
            validation_warnings.append("⚠️ Waste generation is exceptionally high")
        
        # Display validation results
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        
        if validation_warnings:
            for warning in validation_warnings:
                st.warning(warning)
        
        if not validation_errors and not validation_warnings:
            st.success("✅ All inputs validated successfully!")
        
        # Calculate Button
        st.markdown("<br>", unsafe_allow_html=True)
        calculate_button = st.button(
            "🧮 Calculate Carbon Footprint", 
            type="primary",
            disabled=len(validation_errors) > 0,
            help="Click to analyze your carbon footprint and get personalized recommendations" if len(validation_errors) == 0 else "Fix validation errors first"
        )
        
        # Tips
        st.markdown("---")
        st.markdown("""
            <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <p style="font-size: 0.85rem; color: #1b5e20; margin: 0;">
                    <strong>💚 Quick Tips:</strong><br>
                    • Try different presets to see various scenarios<br>
                    • Use accurate values for personalized results<br>
                    • Check your utility bills for exact data<br>
                    • Aim for values in the green (eco-friendly) range<br>
                    • Consider all household members<br>
                    • Update regularly to track progress
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Privacy Notice
        st.markdown("""<div style="background: #e3f2fd; padding: 0.8rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #2196f3;">
                <p style="font-size: 0.75rem; color: #1565c0; margin: 0; line-height: 1.5;">
                    🔒 <strong>Privacy:</strong> Your data stays on your device. 
                    We don't store or transmit any personal information.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # SECTION 3: MAIN CONTENT AREA
    # ============================================
    
    # Process calculation
    if calculate_button:
        # Set calculated flag to true
        st.session_state.calculated = True
        
        with st.spinner("🔄 Analyzing your carbon footprint..."):
            # Generate recommendations
            result = engine.generate_recommendations(
                transport_km, electricity_kwh, water_liters, 
                diet_type, waste_kg
            )
            
            st.session_state.recommendations = result['recommendations']
            st.session_state.summary = result['summary']
    
    # Display results if available
    if st.session_state.summary is not None:
        summary = st.session_state.summary
        
        # ============================================
        # SECTION 3A: CARBON FOOTPRINT RESULT CARD
        # ============================================
        
        # Results container with professional card styling
        with st.container():
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                            padding: 2rem;
                            border-radius: 20px;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            margin-bottom: 2rem;
                            border: 1px solid rgba(46, 125, 50, 0.1);">
            """, unsafe_allow_html=True)
            
            # Render hero card and metrics using reusable components
            render_hero_carbon_card(summary)
            render_metric_cards(summary)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # CO2 Impact Comparison Visual in professional container
        with st.container():
            reduction_pct = summary['reduction_percentage']
            current = summary['current_footprint_kg_co2']
            potential = summary['potential_footprint_kg_co2']
            savings = summary['total_potential_savings_kg_co2']
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                            padding: 2.5rem;
                            border-radius: 20px;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            margin-bottom: 2rem;
                            border: 1px solid rgba(46, 125, 50, 0.1);">
                
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h3 style="color: #2e7d32; margin: 0 0 0.5rem 0; font-size: 1.5rem;">📊 Monthly CO₂ Comparison</h3>
                    <p style="color: #78909c; margin: 0; font-size: 0.95rem;">See the potential impact of green habits</p>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1.5rem;">
                    
                    <!-- Current Footprint -->
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="color: #f44336; font-weight: 700; font-size: 1rem;">🔴 Current</span>
                            <span style="color: #37474f; font-weight: 900; font-size: 1.3rem;">{current:.1f} kg</span>
                        </div>
                        <div style="background: #ffebee; height: 40px; border-radius: 10px; overflow: hidden; position: relative;">
                            <div style="background: linear-gradient(90deg, #f44336 0%, #e53935 100%);
                                        height: 100%;
                                        width: 100%;
                                        border-radius: 10px;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        color: white;
                                        font-weight: 700;
                                        font-size: 0.9rem;
                                        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">
                                100%
                            </div>
                        </div>
                    </div>
                    
                    <!-- Potential Footprint -->
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="color: #4caf50; font-weight: 700; font-size: 1rem;">🟢 Potential</span>
                            <span style="color: #37474f; font-weight: 900; font-size: 1.3rem;">{potential:.1f} kg</span>
                        </div>
                        <div style="background: #e8f5e9; height: 40px; border-radius: 10px; overflow: hidden; position: relative;">
                            <div style="background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);
                                        height: 100%;
                                        width: {(potential/current)*100:.1f}%;
                                        border-radius: 10px;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        color: white;
                                        font-weight: 700;
                                        font-size: 0.9rem;
                                        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">
                                {100-reduction_pct:.1f}%
                            </div>
                        </div>
                    </div>
                    
                </div>
                
                <!-- Savings Highlight -->
                <div style="text-align: center;
                            margin-top: 2rem;
                            padding: 1.5rem;
                            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                            border-radius: 12px;
                            border: 2px solid #4caf50;">
                    <div style="color: #1b5e20; font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">💡 Potential Monthly Savings</div>
                    <div style="color: #2e7d32; font-size: 2.5rem; font-weight: 900; margin: 0.5rem 0;">{savings:.1f} kg CO₂</div>
                    <div style="color: #388e3c; font-size: 1.2rem; font-weight: 700;">{reduction_pct:.1f}% Reduction Possible</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Top Contributor Insight Box in professional container
        with st.container():
            top_contributor = summary['top_contributor'].replace('_', ' ').title()
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                            padding: 2rem;
                            border-radius: 20px;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            margin-bottom: 2rem;
                            border: 1px solid rgba(46, 125, 50, 0.1);">
                    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                                padding: 1.5rem; border-radius: 12px; border-left: 5px solid #1976d2;
                                box-shadow: 0 4px 12px rgba(25, 118, 210, 0.1);">
                        <h4 style="color: #0d47a1; margin: 0 0 0.5rem 0;">📌 Key Insight</h4>
                        <p style="color: #1565c0; margin: 0; font-size: 1.05rem;">
                            Your highest emissions come from <strong>{top_contributor}</strong>. 
                            Focus on this area for maximum impact!
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # SECTION 3B: CHARTS & VISUALIZATIONS
        # ============================================
        with st.container():
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                            padding: 2.5rem;
                            border-radius: 20px;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            margin-bottom: 2rem;
                            border: 1px solid rgba(46, 125, 50, 0.1);">
                    <div style="text-align: center; padding-bottom: 1.5rem;">
                        <h2 style="color: #2e7d32; font-size: 2rem; margin-bottom: 0.5rem;">
                            📈 Visual Analysis
                        </h2>
                        <p style="color: #78909c; font-size: 1rem;">
                            Interactive charts to help you understand your carbon footprint
                        </p>
                    </div>
            """, unsafe_allow_html=True)
            
            # Charts Row 1: Emission Sources and Eco Score
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                breakdown_fig = create_emission_breakdown_chart(summary['footprint_breakdown'])
                st.plotly_chart(breakdown_fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                gauge_fig = create_gauge_chart(summary['eco_score'])
                st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts Row 2: Monthly Trend and Savings
            col1, col2 = st.columns(2)
            
            with col1:
                trend_fig = create_monthly_trend_chart(
                    summary['current_footprint_kg_co2'],
                    summary['potential_footprint_kg_co2']
                )
                st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                savings_fig = create_savings_chart(
                    summary['current_footprint_kg_co2'],
                    summary['total_potential_savings_kg_co2']
                )
                st.plotly_chart(savings_fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Methodology Transparency
        with st.expander("🔬 How We Calculate Your Carbon Footprint", expanded=False):
            st.markdown("""
                <div style="background: #f5f7fa; padding: 1.5rem; border-radius: 12px;">
                    <h4 style="color: #2e7d32; margin-top: 0;">🎯 Our Calculation Methodology</h4>
                    
                    <p style="color: #546e7a; font-size: 0.95rem; line-height: 1.7;">
                        We use industry-standard emission factors combined with machine learning 
                        to provide accurate carbon footprint estimates.
                    </p>
                    
                    <h5 style="color: #1b5e20; margin-top: 1rem;">Emission Factors Used:</h5>
                    <ul style="color: #546e7a; font-size: 0.9rem; line-height: 1.8;">
                        <li><strong>Transport:</strong> 0.12 kg CO₂ per km (average car)</li>
                        <li><strong>Electricity:</strong> 0.5 kg CO₂ per kWh (grid average)</li>
                        <li><strong>Water:</strong> 0.001 kg CO₂ per liter (treatment & distribution)</li>
                        <li><strong>Diet:</strong> Variable based on food type (veg/mixed/non-veg)</li>
                        <li><strong>Waste:</strong> Based on disposal method and volume</li>
                    </ul>
                    
                    <h5 style="color: #1b5e20; margin-top: 1rem;">Machine Learning Models:</h5>
                    <ul style="color: #546e7a; font-size: 0.9rem; line-height: 1.8;">
                        <li><strong>Linear Regression:</strong> 91.19% accuracy (R² score)</li>
                        <li><strong>LSTM Neural Network:</strong> Time-series forecasting</li>
                        <li><strong>Training Data:</strong> 1,000+ synthetic samples with realistic distributions</li>
                    </ul>
                    
                    <div style="background: rgba(255, 255, 255, 0.7);
                                padding: 1rem;
                                border-radius: 8px;
                                margin-top: 1rem;
                                border-left: 4px solid #4caf50;">
                        <p style="color: #1b5e20; font-size: 0.85rem; margin: 0; font-weight: 600;">
                            ✅ <strong>Quality Assurance:</strong> Our models are validated against 
                            international carbon accounting standards (GHG Protocol, EPA guidelines).
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # SECTION 3C: GREEN RECOMMENDATIONS
        # ============================================
        with st.container():
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                            padding: 2.5rem;
                            border-radius: 20px;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            margin-bottom: 2rem;
                            border: 1px solid rgba(46, 125, 50, 0.1);">
                    <div style="text-align: center; padding-bottom: 1.5rem;">
                        <h2 style="color: #2e7d32; font-size: 2rem; margin-bottom: 0.5rem;">
                            💡 Personalized Green Habit Recommendations
                        </h2>
                        <p style="color: #78909c; font-size: 1rem;">
                            Actionable steps to reduce your carbon footprint with quantified impact
                        </p>
                    </div>
            """, unsafe_allow_html=True)
            
            # Filter Controls
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                filter_priority = st.multiselect(
                    "🎯 Filter by priority:",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium', 'Low']
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("Select priorities to filter recommendations")
            with col3:
                show_all = st.checkbox("Show all", value=False)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Filter and display recommendations
            filtered_recs = [
                rec for rec in st.session_state.recommendations
                if rec.get('priority') in filter_priority or rec.get('impact') == 'Positive'
            ]
            
            if not show_all:
                filtered_recs = filtered_recs[:10]
            
            if filtered_recs:
                display_recommendations(filtered_recs)
            else:
                st.info("No recommendations match your selected filters. Try adjusting the priority filter.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # ============================================
        # SECTION 3D: ENVIRONMENTAL IMPACT METRICS
        # ============================================
        with st.container():
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
                            padding: 2.5rem;
                            border-radius: 20px;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            margin-bottom: 2rem;
                            border: 1px solid rgba(46, 125, 50, 0.1);">
                    <div style="text-align: center; padding-bottom: 1.5rem;">
                        <h2 style="color: #2e7d32; font-size: 2rem; margin-bottom: 0.5rem;">
                            🌍 Your Environmental Impact
                        </h2>
                        <p style="color: #78909c; font-size: 1rem;">
                            See your potential contribution to a greener planet
                        </p>
                    </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trees_equivalent = summary['total_potential_savings_kg_co2'] / 21
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%); 
                                padding: 2rem; border-radius: 12px; text-align: center;
                                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.1);">
                """, unsafe_allow_html=True)
                st.metric(
                    label="🌳 Trees Equivalent",
                    value=f"{trees_equivalent:.1f}",
                    help="Number of trees needed to offset your potential savings annually"
                )
                st.caption("Trees absorbing CO₂ for one year")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                km_saved = summary['total_potential_savings_kg_co2'] / 0.12 / 12
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%); 
                                padding: 2rem; border-radius: 12px; text-align: center;
                                box-shadow: 0 4px 12px rgba(33, 150, 243, 0.1);">
                """, unsafe_allow_html=True)
                st.metric(
                    label="🚗 Daily Driving Saved",
                    value=f"{km_saved:.1f} km",
                    help="Equivalent daily car travel distance saved"
                )
                st.caption("Kilometers not driven per day")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                yearly_savings = summary['total_potential_savings_kg_co2'] * 12
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #fff9c4 100%); 
                                padding: 2rem; border-radius: 12px; text-align: center;
                                box-shadow: 0 4px 12px rgba(255, 193, 7, 0.1);">
                """, unsafe_allow_html=True)
                st.metric(
                    label="📅 Annual Savings",
                    value=f"{yearly_savings:.0f} kg",
                    help="Your potential annual CO₂ savings"
                )
                st.caption("Total CO₂ saved per year")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Show welcome/intro content only if NOT calculated yet
    if not st.session_state.calculated:
        # Welcome message with better styling
        st.markdown("""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                        padding: 2rem;
                        border-radius: 16px;
                        border-left: 5px solid #2196f3;
                        margin-bottom: 2rem;
                        text-align: center;">
                <h3 style="color: #1565c0; margin: 0 0 0.5rem 0;">
                    👈 Get Started!
                </h3>
                <p style="color: #1976d2; font-size: 1.1rem; margin: 0;">
                    Enter your lifestyle data in the sidebar and click <strong>'Calculate Carbon Footprint'</strong> to see your personalized results!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Educational content with better visual hierarchy
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: center; margin: 2rem 0 1.5rem 0;">
                <h2 style="color: #2e7d32; font-size: 2rem; margin-bottom: 0.5rem;">
                    🌍 Why Track Your Carbon Footprint?
                </h2>
                <p style="color: #78909c; font-size: 1rem;">
                    Understanding your environmental impact is the first step toward positive change
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%);
                            padding: 2rem;
                            border-radius: 16px;
                            border-top: 4px solid #4caf50;
                            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.1);">
                    <h3 style="color: #2e7d32; margin-top: 0;">🎯 Understanding Your Impact</h3>
                    <p style="color: #546e7a; font-size: 1rem; line-height: 1.7;">
                        Your carbon footprint is the total amount of greenhouse gases generated by your actions. 
                        By tracking and reducing your emissions, you contribute to:
                    </p>
                    <ul style="color: #546e7a; font-size: 0.95rem; line-height: 2; padding-left: 1.5rem;">
                        <li>🌡️ <strong>Fighting climate change</strong></li>
                        <li>🌊 <strong>Protecting ecosystems</strong></li>
                        <li>💚 <strong>Promoting sustainability</strong></li>
                        <li>🌟 <strong>Creating a better future</strong></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%);
                            padding: 2rem;
                            border-radius: 16px;
                            border-top: 4px solid #2196f3;
                            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.1);">
                    <h3 style="color: #1976d2; margin-top: 0;">📊 Our Approach</h3>
                    <p style="color: #546e7a; font-size: 1rem; line-height: 1.7;">
                        This AI-powered tool provides:
                    </p>
                    <ul style="color: #546e7a; font-size: 0.95rem; line-height: 2; padding-left: 1.5rem;">
                        <li>✅ <strong>Lifestyle data analysis</strong></li>
                        <li>📈 <strong>Accurate carbon calculations</strong></li>
                        <li>💡 <strong>Personalized recommendations</strong></li>
                        <li>🎯 <strong>Savings tracking</strong></li>
                        <li>🤖 <strong>ML-powered predictions</strong></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Sample statistics with better visual styling
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: center; margin: 2rem 0 1rem 0;">
                <h2 style="color: #2e7d32; font-size: 1.8rem; margin-bottom: 0.5rem;">
                    📌 Average Carbon Footprints
                </h2>
                <p style="color: #78909c; font-size: 1rem;">
                    See how different lifestyles compare
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        avg_data = pd.DataFrame({
            'Category': ['Low Carbon', 'Average', 'High Carbon', 'Your Target'],
            'Footprint (kg CO₂/month)': [190, 375, 512, 250]
        })
        
        fig = px.bar(
            avg_data,
            x='Category',
            y='Footprint (kg CO₂/month)',
            color='Footprint (kg CO₂/month)',
            color_continuous_scale=['#4CAF50', '#FF9800', '#F44336'],
            text='Footprint (kg CO₂/month)'
        )
        
        fig.update_traces(
            texttemplate='%{text:.0f} kg',
            textposition='outside',
            textfont_size=14
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", size=12),
            xaxis=dict(
                title=dict(text="Lifestyle Category", font=dict(size=14, color="#546e7a")),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=dict(text="Monthly CO₂ Emissions (kg)", font=dict(size=14, color="#546e7a")),
                gridcolor="rgba(0,0,0,0.05)"
            ),
            margin=dict(t=20, b=60, l=60, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Add context info
        st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border-left: 4px solid #ff9800;
                        margin-top: 1.5rem;">
                <p style="color: #e65100; font-size: 0.95rem; margin: 0; line-height: 1.6;">
                    💡 <strong>Goal:</strong> Aim for the "Low Carbon" or "Your Target" range (150-250 kg/month) 
                    to contribute meaningfully to environmental sustainability.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Professional Footer
    render_professional_footer()

if __name__ == "__main__":
    main()
