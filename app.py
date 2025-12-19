"""
═══════════════════════════════════════════════════════════════════════════════
Smart Carbon Footprint Calculator - Professional Production Version
═══════════════════════════════════════════════════════════════════════════════
A modern, user-friendly web application for calculating personal carbon footprints
and providing actionable recommendations for environmental impact reduction.

Author: [Your Name]
Institution: [Your Institution]
Date: December 2025
Version: 2.0.0 - Production Ready
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Carbon Footprint Calculator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS STYLING - Modern Green-Tech Theme
# ═══════════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    """
    Inject custom CSS for professional styling.
    Includes: background gradients, card styles, button effects, typography
    """
    st.markdown("""
    <style>
    /* ========== GLOBAL STYLES ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main app background - Light green to white gradient */
    .stApp {
        background: linear-gradient(180deg, #e8f5e9 0%, #f1f8f4 30%, #ffffff 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* ========== CARD STYLES ========== */
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(46, 125, 50, 0.1);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(46, 125, 50, 0.12);
    }
    
    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%);
        padding: 3.5rem;
        border-radius: 24px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
        text-align: center;
        border: 4px solid;
    }
    
    /* ========== SIDEBAR STYLES ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fdf9 50%, #f1f8f4 100%);
        border-right: 2px solid #c8e6c9;
        box-shadow: 4px 0 16px rgba(46, 125, 50, 0.08);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1b5e20;
        font-size: 1.6rem;
        font-weight: 800;
        text-align: center;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #4caf50;
    }
    
    /* ========== BUTTON STYLES ========== */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 800;
        padding: 1.25rem 2.5rem;
        border-radius: 16px;
        border: none;
        box-shadow: 0 6px 16px rgba(67, 160, 71, 0.35);
        transition: all 0.4s ease;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        box-shadow: 0 10px 28px rgba(67, 160, 71, 0.45);
        transform: translateY(-4px) scale(1.02);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* ========== TYPOGRAPHY ========== */
    h1 {
        color: #1b5e20;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    h2 {
        color: #2e7d32;
        font-weight: 700;
    }
    
    h3 {
        color: #388e3c;
        font-weight: 600;
    }
    
    /* ========== METRIC STYLES ========== */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2e7d32;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* ========== INPUT STYLES ========== */
    .stNumberInput input, .stSlider {
        border-radius: 8px;
    }
    
    /* ========== TEXT COLOR RULES ========== */
    /* Default dark text for light backgrounds */
    .stMarkdown p, .stMarkdown li {
        color: #333333;
    }
    
    /* Labels - dark green */
    label {
        color: #1b5e20 !important;
        font-weight: 600 !important;
    }
    
    /* Ensure white text on colored backgrounds */
    .stButton>button {
        color: white !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state variables for conditional rendering."""
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'result' not in st.session_state:
        st.session_state.result = None

# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_carbon_footprint(transport_km, electricity_kwh, water_liters, diet_type, waste_kg):
    """
    Calculate total carbon footprint and breakdown by category.
    Returns dict with total, breakdown, status, color, icon, message.
    """
    # Emission factors (kg CO2)
    TRANSPORT_FACTOR = 0.21  # per km
    ELECTRICITY_FACTOR = 0.527  # per kWh
    WATER_FACTOR = 0.001  # per liter
    DIET_FACTORS = {'veg': 50, 'mixed': 100, 'non-veg': 150}
    WASTE_FACTOR = 0.5  # per kg per week
    
    # Calculate components
    transport_co2 = transport_km * TRANSPORT_FACTOR
    electricity_co2 = electricity_kwh * ELECTRICITY_FACTOR
    water_co2 = water_liters * 30 * WATER_FACTOR
    diet_co2 = DIET_FACTORS.get(diet_type, 100)
    waste_co2 = waste_kg * 4 * WASTE_FACTOR
    
    total = transport_co2 + electricity_co2 + water_co2 + diet_co2 + waste_co2
    
    # Determine status
    if total < 200:
        status, color, icon, message = "Low Impact", "#4caf50", "✅", "Excellent! You're doing great!"
    elif total < 400:
        status, color, icon, message = "Moderate Impact", "#ff9800", "⚠️", "Good start! Room for improvement."
    else:
        status, color, icon, message = "High Impact", "#f44336", "🔴", "Let's reduce your footprint!"
    
    return {
        'total': round(total, 1),
        'breakdown': {
            'Transport': round(transport_co2, 1),
            'Electricity': round(electricity_co2, 1),
            'Water': round(water_co2, 1),
            'Diet': round(diet_co2, 1),
            'Waste': round(waste_co2, 1)
        },
        'status': status,
        'color': color,
        'icon': icon,
        'message': message
    }

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render professional header with centered title and subtitle."""
    st.markdown("""
        <div style="text-align: center; padding: 2.5rem 0 2rem 0;">
            <h1 style="font-size: 3.5rem; 
                       margin-bottom: 1.5rem;
                       color: #1b5e20;
                       font-weight: 800;">
                🌱 Smart Carbon Footprint Calculator
            </h1>
            <h3 style="font-size: 1.5rem; color: #2e7d32; margin-bottom: 1rem; font-weight: 600;">
                Track • Analyze • Reduce Your Environmental Impact
            </h3>
            <p style="color: #546e7a; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
                Calculate your carbon footprint and get personalized recommendations
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar_inputs():
    """Render sidebar with all input controls. Returns all input values."""
    st.sidebar.markdown("<h2>📊 Your Lifestyle Data</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align:center; color:#546e7a; font-size:0.9rem;'>Enter your monthly habits</p>", unsafe_allow_html=True)
    
    # Transport
    st.sidebar.markdown("<h3 style='color:#1b5e20; font-weight:700;'>🚗 Transportation</h3>", unsafe_allow_html=True)
    transport_km = st.sidebar.number_input("Distance by car (km/month)", 0, 5000, 300, 50)
    st.sidebar.markdown("---")
    
    # Electricity
    st.sidebar.markdown("<h3 style='color:#1b5e20; font-weight:700;'>⚡ Electricity</h3>", unsafe_allow_html=True)
    electricity_kwh = st.sidebar.number_input("Electricity (kWh/month)", 0, 2000, 200, 10)
    st.sidebar.markdown("---")
    
    # Water
    st.sidebar.markdown("<h3 style='color:#1b5e20; font-weight:700;'>💧 Water</h3>", unsafe_allow_html=True)
    water_liters = st.sidebar.slider("Daily water (liters)", 0, 500, 150, 10)
    st.sidebar.markdown("---")
    
    # Diet
    st.sidebar.markdown("<h3 style='color:#1b5e20; font-weight:700;'>🍽️ Diet</h3>", unsafe_allow_html=True)
    diet_type = st.sidebar.selectbox("Diet type", ['veg', 'mixed', 'non-veg'],
                                      format_func=lambda x: {'veg':'🌱 Vegetarian','mixed':'🍴 Mixed','non-veg':'🥩 Non-Veg'}[x])
    st.sidebar.markdown("---")
    
    # Waste
    st.sidebar.markdown("<h3 style='color:#1b5e20; font-weight:700;'>♻️ Waste</h3>", unsafe_allow_html=True)
    waste_kg = st.sidebar.slider("Weekly waste (kg)", 0.0, 30.0, 8.0, 0.5)
    
    return transport_km, electricity_kwh, water_liters, diet_type, waste_kg

def render_welcome_screen():
    """Render welcome screen before calculation."""
    with st.container():
        st.markdown("""
            <div class="card" style="text-align:center; background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%); border-left:5px solid #2196f3;">
                <h2 style="color:#0d47a1; margin-bottom:1rem; font-weight:700;">👈 Get Started!</h2>
                <p style="color:#1565c0; font-size:1.2rem; font-weight:500;">
                    Enter your data in the <strong>sidebar</strong> and click <strong>'Calculate'</strong>!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
                <div class="card" style="background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%); border-top:4px solid #4caf50;">
                    <h3 style="color:#1b5e20; font-weight:700;">🎯 What is Carbon Footprint?</h3>
                    <p style="color:#2e7d32; line-height:1.7; font-weight:500;">
                        Total greenhouse gas emissions from your daily activities.
                    </p>
                    <ul style="color:#2c6e49; line-height:1.8; font-weight:500;">
                        <li><strong>Transportation:</strong> Vehicle emissions</li>
                        <li><strong>Energy:</strong> Electricity use</li>
                        <li><strong>Diet:</strong> Food production</li>
                        <li><strong>Waste:</strong> Disposal impact</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card" style="background:linear-gradient(135deg,#fff3e0 0%,#ffe0b2 100%); border-top:4px solid #ff9800;">
                    <h3 style="color:#e65100; font-weight:700;">📊 Why Track It?</h3>
                    <ul style="color:#d84315; line-height:1.8; font-weight:500;">
                        <li>Identify high-impact areas</li>
                        <li>Set reduction goals</li>
                        <li>Make informed decisions</li>
                        <li>Contribute to climate action</li>
                        <li>Save money through efficiency</li>
                    </ul>
                    <p style="color:#bf360c; font-weight:700; margin-top:1rem;">
                        🌍 Goal: &lt;200 kg CO₂/month
                    </p>
                </div>
            """, unsafe_allow_html=True)

def render_results(result):
    """Render results section with hero card and breakdown."""
    # Hero Card
    with st.container():
        st.markdown(f"""
            <div class="hero-card" style="border-color: {result['color']};">
                <div style="display:inline-block; background:{result['color']}; color:white; 
                            padding:0.75rem 2rem; border-radius:50px; font-weight:800; 
                            font-size:1rem; margin-bottom:2rem; box-shadow:0 6px 16px rgba(0,0,0,0.2);">
                    {result['icon']} {result['status'].upper()}
                </div>
                <div>
                    <div style="color:#1f4e35; font-size:1.5rem; font-weight:800; margin-bottom:1rem; text-transform:uppercase;">
                        Your Monthly Carbon Footprint
                    </div>
                    <div style="color:{result['color']}; font-size:6.5rem; font-weight:900; 
                                margin:1.5rem 0; text-shadow:3px 3px 6px rgba(0,0,0,0.1);">
                        {result['total']}
                    </div>
                    <div style="color:#2c6e49; font-size:2rem; font-weight:800;">
                        kg CO₂ / month
                    </div>
                </div>
                <div style="margin-top:2.5rem; background:linear-gradient(135deg,#f5f5f5 0%,#e0e0e0 100%); 
                            padding:1.5rem 2rem; border-radius:16px; border-left:5px solid {result['color']};">
                    <p style="color:#1f4e35; font-size:1.2rem; font-weight:700; margin:0;">
                        {result['icon']} {result['message']}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Breakdown metrics
    st.markdown("""
        <div style="text-align:center; padding:1.5rem 0;">
            <h2 style="color:#1b5e20; font-size:2rem; font-weight:800;">📊 Detailed Breakdown</h2>
            <p style="color:#546e7a; font-weight:500;">Carbon footprint by category</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    categories = ['Transport', 'Electricity', 'Water', 'Diet', 'Waste']
    icons = ['🚗', '⚡', '💧', '🍽️', '♻️']
    colors = ['#f44336', '#2196f3', '#00bcd4', '#ff9800', '#4caf50']
    
    for idx, (col, category) in enumerate(zip(cols, categories)):
        with col:
            value = result['breakdown'][category]
            st.markdown(f"""
                <div class="card" style="text-align:center; border-top:4px solid {colors[idx]}; min-height:180px;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">{icons[idx]}</div>
                    <div style="color:{colors[idx]}; font-size:2.5rem; font-weight:900; margin:0.5rem 0;">
                        {value}
                    </div>
                    <div style="color:#1f4e35; font-size:0.9rem; font-weight:800; text-transform:uppercase;">
                        {category}
                    </div>
                    <div style="color:#546e7a; font-size:0.8rem; margin-top:0.3rem; font-weight:600;">kg CO₂</div>
                </div>
            """, unsafe_allow_html=True)

def render_charts(result):
    """Render interactive Plotly charts."""
    with st.container():
        st.markdown("""
            <div style="text-align:center; padding:2rem 0 1rem 0;">
                <h2 style="color:#1b5e20; font-weight:800;">📈 Visual Analysis</h2>
                <p style="color:#546e7a; font-weight:500;">Interactive charts</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            # Pie Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(result['breakdown'].keys()),
                values=list(result['breakdown'].values()),
                hole=0.4,
                marker=dict(colors=['#f44336', '#2196f3', '#00bcd4', '#ff9800', '#4caf50']),
                textinfo='label+percent'
            )])
            fig_pie.update_layout(
                title=dict(text="Emission Breakdown", font=dict(color="#1b5e20", size=20, family="Inter")),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#333333", size=14, family="Inter")
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Bar Chart
            fig_bar = go.Figure(data=[go.Bar(
                x=list(result['breakdown'].keys()),
                y=list(result['breakdown'].values()),
                marker=dict(color=list(result['breakdown'].values()), colorscale='Greens'),
                text=list(result['breakdown'].values()),
                texttemplate='%{text:.1f} kg',
                textposition='outside'
            )])
            fig_bar.update_layout(
                title=dict(text="CO₂ by Category", font=dict(color="#1b5e20", size=20, family="Inter")),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(color="#333333", size=14, family="Inter"),
                xaxis=dict(tickfont=dict(color="#333333", size=12)),
                yaxis=dict(tickfont=dict(color="#333333", size=12))
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

def render_recommendations(result):
    """Render personalized recommendations."""
    with st.container():
        st.markdown("""
            <div style="text-align:center; padding:2rem 0 1rem 0;">
                <h2 style="color:#1b5e20; font-weight:800;">💡 Recommendations</h2>
                <p style="color:#546e7a; font-weight:500;">Actionable steps to reduce your footprint</p>
            </div>
        """, unsafe_allow_html=True)
        
        recs = {
            'Transport': ["🚌 Use public transport", "🚲 Bike for short distances", "⚡ Consider electric vehicles"],
            'Electricity': ["💡 Switch to LED bulbs", "🔌 Unplug unused devices", "☀️ Install solar panels"],
            'Water': ["🚿 Take shorter showers", "🌱 Install efficient fixtures", "♻️ Reuse greywater"],
            'Diet': ["🌱 Increase plant-based meals", "🏪 Buy local produce", "♻️ Reduce food waste"],
            'Waste': ["♻️ Separate recyclables", "🗑️ Compost organic waste", "🛍️ Use reusable bags"]
        }
        
        sorted_categories = sorted(result['breakdown'].items(), key=lambda x: x[1], reverse=True)
        colors = {'Transport': '#f44336', 'Electricity': '#2196f3', 'Water': '#00bcd4', 'Diet': '#ff9800', 'Waste': '#4caf50'}
        
        for idx, (category, value) in enumerate(sorted_categories[:3], 1):
            recommendations_html = "".join([f"<li style='margin:0.5rem 0; color:#2c3e50; font-weight:500;'>{rec}</li>" for rec in recs[category]])
            st.markdown(f"""
                <div class="card" style="border-left:5px solid {colors[category]};">
                    <h3 style="color:{colors[category]}; font-weight:800;">#{idx} Priority: {category} ({value} kg CO₂)</h3>
                    <ul style="margin-top:1rem; padding-left:1.5rem; color:#2c3e50; font-weight:500; line-height:1.8;">
                        {recommendations_html}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application orchestrating the entire UI flow."""
    init_session_state()
    inject_custom_css()
    
    render_header()
    st.markdown("<div style='margin:2rem 0;'></div>", unsafe_allow_html=True)
    
    # Sidebar inputs - always accessible
    transport_km, electricity_kwh, water_liters, diet_type, waste_kg = render_sidebar_inputs()
    
    # Calculate button
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("🧮 Calculate Carbon Footprint"):
        result = calculate_carbon_footprint(transport_km, electricity_kwh, water_liters, diet_type, waste_kg)
        st.session_state.calculated = True
        st.session_state.result = result
    
    # Conditional rendering
    if not st.session_state.calculated:
        render_welcome_screen()
    else:
        result = st.session_state.result
        render_results(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_charts(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_recommendations(result)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align:center; padding:2rem; border-top:2px solid #e0e0e0; color:#546e7a;">
            <p style="margin:0; font-size:0.9rem; font-weight:600;">🌍 <strong>Made with ❤️ for a Sustainable Future</strong></p>
            <p style="margin:0.5rem 0 0 0; font-size:0.85rem; font-weight:500;">
                © 2025 Smart Carbon Footprint Calculator | Educational Purpose
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
