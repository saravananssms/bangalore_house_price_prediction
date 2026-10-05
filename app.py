import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="Bangalore House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main-title {
        font-size: 2.5rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 0.5rem 0; margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem; color: #888; text-align: center; margin-bottom: 2rem;
    }
    .price-card {
        padding: 2rem; border-radius: 16px; text-align: center;
        margin: 1rem 0; box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .price-value {
        font-size: 3rem; font-weight: 700; margin: 0.5rem 0;
    }
    .price-label {
        font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 2px;
    }
    .badge {
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; margin-top: 8px;
    }
    .badge-low { background: #d4edda; color: #155724; }
    .badge-medium { background: #fff3cd; color: #856404; }
    .badge-high { background: #f8d7da; color: #721c24; }
    .badge-premium { background: #e2d4f0; color: #4a235a; }
    .stat-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border: 1px solid #e9ecef; border-radius: 12px;
        padding: 1.2rem; text-align: center;
    }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #333; }
    .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .gradient-card-1 { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .gradient-card-2 { background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }
    .gradient-card-3 { background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; }
    .gradient-card-4 { background: linear-gradient(135deg, #43e97b, #38f9d7); color: #1a1a1a; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #333;
        border-left: 4px solid #667eea; padding-left: 12px; margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    try:
        import keras
        model = keras.models.load_model(
            os.path.join(BASE_DIR, "model.h5"), compile=False
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    except Exception:
        return None
    return model


@st.cache_data
def load_artifacts():
    with open(os.path.join(BASE_DIR, "columns.json")) as f:
        columns_data = json.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "location_stats.json")) as f:
        stats = json.load(f)
    config = {}
    cp = os.path.join(BASE_DIR, "model_config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            config = json.load(f)
    return columns_data, scaler, stats, config


def predict_price(model, scaler, columns_data, location, sqft, bath, bhk):
    feature_cols = columns_data['feature_columns']
    x = np.zeros(len(feature_cols))
    x[feature_cols.index('total_sqft')] = sqft
    x[feature_cols.index('bath')] = bath
    x[feature_cols.index('bhk')] = bhk
    x[feature_cols.index('balcony')] = min(bhk, 2)
    loc_col = location
    if loc_col in feature_cols:
        x[feature_cols.index(loc_col)] = 1
    x_scaled = scaler.transform([x])
    return model.predict(x_scaled, verbose=0)[0][0]


def get_price_category(price, ranges):
    if price <= ranges['low_threshold']:
        return 'Budget', 'badge-low', '#28a745'
    elif price <= ranges['medium_threshold']:
        return 'Mid-Range', 'badge-medium', '#ffc107'
    elif price <= ranges['high_threshold']:
        return 'Premium', 'badge-high', '#dc3545'
    else:
        return 'Luxury', 'badge-premium', '#6f42c1'


# ==========================
# SIDEBAR
# ==========================
st.sidebar.markdown("## 🏠 Bangalore House Price")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["Price Prediction", "Market Analytics", "How It Works"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("Python | Keras | Streamlit | Plotly")
st.sidebar.markdown("**Dataset:** ~13,000 Bangalore properties")
st.sidebar.markdown("**Model:** Dense Neural Network (.h5)")

# ==========================
# LOAD EVERYTHING
# ==========================
model = load_model()
if model is None:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

columns_data, scaler, stats, config = load_artifacts()
locations = sorted(columns_data['locations'])
price_ranges = stats['price_ranges']
loc_stats = pd.DataFrame(stats['location_stats'])

# ==========================
# PAGE: PRICE PREDICTION
# ==========================
if page == "Price Prediction":
    st.markdown('<div class="main-title">Bangalore House Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered real estate price estimation for 100+ Bangalore locations</div>', unsafe_allow_html=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card gradient-card-1"><div class="stat-value" style="color:white">{stats["total_records"]:,}</div><div class="stat-label" style="color:#ddd">Properties Analyzed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card gradient-card-2"><div class="stat-value" style="color:white">{len(locations)}</div><div class="stat-label" style="color:#ddd">Locations Covered</div></div>', unsafe_allow_html=True)
    with col3:
        r2 = config.get('r2_score', 0.9)
        st.markdown(f'<div class="stat-card gradient-card-3"><div class="stat-value" style="color:white">{r2:.1%}</div><div class="stat-label" style="color:#ddd">Model Accuracy (R²)</div></div>', unsafe_allow_html=True)
    with col4:
        mae = config.get('mae', 10)
        st.markdown(f'<div class="stat-card gradient-card-4"><div class="stat-value">₹{mae:.1f}L</div><div class="stat-label">Avg Error (MAE)</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Input form
    st.markdown('<div class="section-header">Enter Property Details</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        location = st.selectbox("📍 Location", locations, index=locations.index('Whitefield') if 'Whitefield' in locations else 0)
    with col_b:
        bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5], index=2)
    with col_c:
        sqft = st.slider("📐 Total Sqft", 300, 6000, 1200, step=50)
    with col_d:
        bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5], index=1)

    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        predicted = predict_price(model, scaler, columns_data, location, sqft, bath, bhk)
        predicted = max(predicted, 5)
        category, badge_class, badge_color = get_price_category(predicted, price_ranges)

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            st.markdown(f"""
            <div class="price-card" style="border: 3px solid {badge_color};">
                <div class="price-label">Estimated Price</div>
                <div class="price-value" style="color: {badge_color};">₹ {predicted:.2f} Lakhs</div>
                <div class="price-label">({predicted/100:.2f} Crores)</div>
                <div><span class="badge {badge_class}">{category}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        det1, det2, det3 = st.columns(3)
        with det1:
            st.metric("Price per Sqft", f"₹{(predicted * 100000 / sqft):,.0f}")
        with det2:
            loc_avg = loc_stats[loc_stats['location'] == location]['avg_price'].values
            if len(loc_avg) > 0:
                diff = predicted - loc_avg[0]
                st.metric("vs Location Average", f"₹{predicted:.1f}L", f"{diff:+.1f}L")
            else:
                st.metric("Location", location)
        with det3:
            st.metric("Configuration", f"{bhk} BHK | {sqft} sqft | {bath} bath")

        # Price range visual
        st.markdown('<div class="section-header">Price Range Classification</div>', unsafe_allow_html=True)

        if HAS_PLOTLY:
            ranges_labels = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
            ranges_values = [
                price_ranges['low_threshold'],
                price_ranges['medium_threshold'] - price_ranges['low_threshold'],
                price_ranges['high_threshold'] - price_ranges['medium_threshold'],
                price_ranges['max_price'] - price_ranges['high_threshold'],
            ]
            colors = ['#28a745', '#ffc107', '#dc3545', '#6f42c1']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted,
                title={'text': "Property Value (Lakhs)", 'font': {'size': 18}},
                number={'prefix': "₹", 'suffix': "L", 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, min(price_ranges['max_price'], 300)], 'tickprefix': '₹', 'ticksuffix': 'L'},
                    'bar': {'color': badge_color, 'thickness': 0.3},
                    'steps': [
                        {'range': [0, price_ranges['low_threshold']], 'color': '#d4edda'},
                        {'range': [price_ranges['low_threshold'], price_ranges['medium_threshold']], 'color': '#fff3cd'},
                        {'range': [price_ranges['medium_threshold'], price_ranges['high_threshold']], 'color': '#f8d7da'},
                        {'range': [price_ranges['high_threshold'], min(price_ranges['max_price'], 300)], 'color': '#e2d4f0'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.8, 'value': predicted},
                }
            ))
            fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=40, r=40))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Comparable locations
            st.markdown('<div class="section-header">How This Location Compares</div>', unsafe_allow_html=True)
            top_locs = loc_stats.head(15).copy()
            top_locs['is_selected'] = top_locs['location'] == location
            if location not in top_locs['location'].values:
                sel_row = loc_stats[loc_stats['location'] == location]
                if len(sel_row):
                    sel_row = sel_row.copy()
                    sel_row['is_selected'] = True
                    top_locs = pd.concat([top_locs, sel_row])
            top_locs = top_locs.sort_values('avg_price', ascending=True)
            colors_bar = ['#667eea' if not s else '#f5576c' for s in top_locs['is_selected']]
            fig_comp = go.Figure(go.Bar(
                x=top_locs['avg_price'], y=top_locs['location'],
                orientation='h', marker_color=colors_bar,
                text=[f"₹{p:.0f}L" for p in top_locs['avg_price']], textposition='outside',
            ))
            fig_comp.update_layout(
                title="Average Price by Location (Top 15 + Your Selection)",
                xaxis_title="Average Price (Lakhs)", yaxis_title="",
                height=500, margin=dict(l=150, r=40, t=50, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ==========================
# PAGE: MARKET ANALYTICS
# ==========================
elif page == "Market Analytics":
    st.markdown('<div class="main-title">Bangalore Real Estate Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comprehensive market insights powered by data from 13,000+ properties</div>', unsafe_allow_html=True)

    if not HAS_PLOTLY:
        st.warning("Install plotly for interactive charts: `pip install plotly`")
        st.stop()

    bhk_data = pd.DataFrame(stats['bhk_price_data'])

    # Row 1: Price Distribution + Price Range Donut
    st.markdown('<div class="section-header">Price Distribution Overview</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig_hist = px.histogram(
            bhk_data, x='price', nbins=60, color_discrete_sequence=['#667eea'],
            title="Property Price Distribution (Lakhs)",
            labels={'price': 'Price (Lakhs)', 'count': 'Number of Properties'}
        )
        fig_hist.update_layout(bargap=0.05, height=400, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)

    with r1c2:
        low_t = price_ranges['low_threshold']
        med_t = price_ranges['medium_threshold']
        high_t = price_ranges['high_threshold']
        budget = len(bhk_data[bhk_data['price'] <= low_t])
        mid = len(bhk_data[(bhk_data['price'] > low_t) & (bhk_data['price'] <= med_t)])
        premium = len(bhk_data[(bhk_data['price'] > med_t) & (bhk_data['price'] <= high_t)])
        luxury = len(bhk_data[bhk_data['price'] > high_t])

        fig_donut = go.Figure(go.Pie(
            labels=['Budget<br>(< ₹{:.0f}L)'.format(low_t),
                    'Mid-Range<br>(₹{:.0f}-{:.0f}L)'.format(low_t, med_t),
                    'Premium<br>(₹{:.0f}-{:.0f}L)'.format(med_t, high_t),
                    'Luxury<br>(> ₹{:.0f}L)'.format(high_t)],
            values=[budget, mid, premium, luxury],
            hole=0.55,
            marker_colors=['#28a745', '#ffc107', '#dc3545', '#6f42c1'],
            textinfo='label+percent', textfont_size=11,
        ))
        fig_donut.update_layout(title="Price Range Classification", height=400,
                                annotations=[dict(text='Price<br>Segments', x=0.5, y=0.5,
                                                   font_size=14, showarrow=False)])
        st.plotly_chart(fig_donut, use_container_width=True)

    # Row 2: BHK vs Price + Sqft vs Price
    st.markdown('<div class="section-header">Price Drivers Analysis</div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        bhk_avg = bhk_data.groupby('bhk')['price'].agg(['mean', 'median', 'count']).reset_index()
        fig_bhk = go.Figure()
        fig_bhk.add_trace(go.Bar(
            x=bhk_avg['bhk'], y=bhk_avg['mean'], name='Average Price',
            marker_color='#667eea', text=[f"₹{v:.0f}L" for v in bhk_avg['mean']], textposition='outside'
        ))
        fig_bhk.add_trace(go.Bar(
            x=bhk_avg['bhk'], y=bhk_avg['median'], name='Median Price',
            marker_color='#f093fb', text=[f"₹{v:.0f}L" for v in bhk_avg['median']], textposition='outside'
        ))
        fig_bhk.update_layout(
            title="Average & Median Price by BHK", barmode='group',
            xaxis_title="BHK", yaxis_title="Price (Lakhs)",
            height=420, plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_bhk, use_container_width=True)

    with r2c2:
        sample = bhk_data.sample(min(2000, len(bhk_data)), random_state=42)
        fig_scatter = px.scatter(
            sample, x='total_sqft', y='price', color='bhk',
            title="Sqft vs Price (colored by BHK)",
            labels={'total_sqft': 'Total Sqft', 'price': 'Price (Lakhs)', 'bhk': 'BHK'},
            color_continuous_scale='Viridis', opacity=0.6,
        )
        fig_scatter.update_layout(height=420, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 3: Top locations bar chart
    st.markdown('<div class="section-header">Location-wise Price Analysis</div>', unsafe_allow_html=True)

    top_n = st.slider("Number of locations to show", 10, 40, 20)
    top = loc_stats.head(top_n).sort_values('avg_price', ascending=True)

    fig_loc = go.Figure()
    fig_loc.add_trace(go.Bar(
        x=top['avg_price'], y=top['location'], orientation='h',
        marker=dict(color=top['avg_price'], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Avg Price (L)")),
        text=[f"₹{p:.0f}L ({c:.0f} props)" for p, c in zip(top['avg_price'], top['count'])],
        textposition='outside',
    ))
    fig_loc.update_layout(
        title=f"Top {top_n} Most Expensive Locations in Bangalore",
        xaxis_title="Average Price (Lakhs)", yaxis_title="",
        height=max(500, top_n * 28), margin=dict(l=160, r=80, t=50, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_loc, use_container_width=True)

    # Row 4: Price per sqft by location + BHK distribution
    st.markdown('<div class="section-header">Detailed Insights</div>', unsafe_allow_html=True)
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        top_ppsqft = loc_stats.head(15).sort_values('avg_price_per_sqft', ascending=True)
        fig_ppsqft = go.Figure(go.Bar(
            x=top_ppsqft['avg_price_per_sqft'], y=top_ppsqft['location'],
            orientation='h', marker_color='#f5576c',
            text=[f"₹{p:,.0f}/sqft" for p in top_ppsqft['avg_price_per_sqft']],
            textposition='outside',
        ))
        fig_ppsqft.update_layout(
            title="Price per Sqft by Location (Top 15)",
            xaxis_title="Avg Price per Sqft (₹)", height=450,
            margin=dict(l=150), plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_ppsqft, use_container_width=True)

    with r4c2:
        bhk_counts = bhk_data['bhk'].value_counts().sort_index()
        fig_pie = go.Figure(go.Pie(
            labels=[f"{b} BHK" for b in bhk_counts.index],
            values=bhk_counts.values,
            hole=0.4, marker_colors=px.colors.qualitative.Set2,
            textinfo='label+percent+value',
        ))
        fig_pie.update_layout(title="Property Distribution by BHK", height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Row 5: Heatmap — BHK x Top Locations
    st.markdown('<div class="section-header">BHK × Location Price Heatmap</div>', unsafe_allow_html=True)

    heat_locs = loc_stats.head(12)['location'].tolist()
    heat_data = bhk_data[bhk_data['location'].isin(heat_locs)]
    pivot = heat_data.pivot_table(values='price', index='location', columns='bhk', aggfunc='median')
    pivot = pivot.reindex(heat_locs)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{c} BHK" for c in pivot.columns], y=pivot.index,
        colorscale='YlOrRd', texttemplate='₹%{z:.0f}L', textfont={'size': 11},
        colorbar=dict(title="Median Price (L)"),
    ))
    fig_heat.update_layout(
        title="Median Price: Top 12 Locations × BHK",
        height=450, margin=dict(l=150, t=50),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ==========================
# PAGE: HOW IT WORKS
# ==========================
elif page == "How It Works":
    st.markdown('<div class="main-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Understanding the ML pipeline behind the predictions</div>', unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-header">Machine Learning Pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("1. Data Collection", "13,000+ Bangalore property records", "gradient-card-1"),
        ("2. Data Cleaning", "Handle nulls, parse sqft, fix BHK", "gradient-card-2"),
        ("3. Feature Engineering", "Location encoding, outlier removal", "gradient-card-3"),
        ("4. Model Training", "Keras Dense NN with BatchNorm", "gradient-card-4"),
        ("5. Prediction", "Real-time price estimation", "gradient-card-1"),
    ]
    for col, (title, desc, css) in zip([p1, p2, p3, p4, p5], steps):
        with col:
            st.markdown(f'<div class="stat-card {css}" style="color:white;min-height:120px"><div style="font-weight:600;font-size:0.9rem">{title}</div><div style="font-size:0.75rem;margin-top:8px;opacity:0.9">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Data Cleaning
    st.markdown('<div class="section-header">Data Cleaning Steps</div>', unsafe_allow_html=True)
    st.code("""# 1. Drop unnecessary columns
df = df.drop(['area_type', 'society', 'availability'], axis=1)

# 2. Extract BHK from size column
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# 3. Parse sqft ranges (e.g., "1200 - 1500" -> 1350)
def parse_sqft(x):
    if '-' in str(x):
        parts = str(x).split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    return float(x)

# 4. Remove outliers
df = df[df['total_sqft'] / df['bhk'] >= 300]     # Min 300 sqft per BHK
df = df[df['bath'] <= df['bhk'] + 2]             # Reasonable bath count
df = df[df['price_per_sqft'] < 50000]            # Remove extreme prices""", language="python")

    # Model Architecture
    st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
    arch_df = pd.DataFrame({
        'Layer': ['Dense(128, relu)', 'BatchNormalization', 'Dropout(0.3)',
                  'Dense(64, relu)', 'BatchNormalization', 'Dropout(0.2)',
                  'Dense(32, relu)', 'Dense(1, linear)'],
        'Purpose': [
            'First hidden layer - learns price patterns from features',
            'Normalizes activations for stable training',
            'Drops 30% neurons to prevent overfitting',
            'Second hidden layer - refines learned patterns',
            'Normalizes second layer activations',
            'Light regularization (20% dropout)',
            'Third hidden layer - final feature compression',
            'Output: predicted price in Lakhs (regression)'
        ]
    })
    st.table(arch_df)

    st.markdown('<div class="section-header">Feature Engineering</div>', unsafe_allow_html=True)
    st.code("""# Features used for prediction:
# 1. total_sqft  — Total area in square feet
# 2. bath        — Number of bathrooms
# 3. bhk         — Number of bedrooms (1-5)
# 4. balcony     — Number of balconies
# 5. location_*  — One-hot encoded location (100+ dummies)

# TF-IDF is NOT used — this is a tabular regression problem
# StandardScaler normalizes all numeric features""", language="python")

    # Model performance
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    if config:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("R² Score", f"{config.get('r2_score', 0):.4f}")
        with mc2:
            st.metric("MAE", f"₹{config.get('mae', 0):.2f} Lakhs")
        with mc3:
            st.metric("Training Samples", f"{config.get('training_samples', 0):,}")
        with mc4:
            st.metric("Features", f"{config.get('features', 0)}")

    st.markdown('<div class="section-header">Price Range Classification Logic</div>', unsafe_allow_html=True)
    st.markdown(f"""
    | Category | Range | Color |
    |----------|-------|-------|
    | **Budget** | Below ₹{price_ranges['low_threshold']:.0f} Lakhs | 🟢 Green |
    | **Mid-Range** | ₹{price_ranges['low_threshold']:.0f}L - ₹{price_ranges['medium_threshold']:.0f}L | 🟡 Yellow |
    | **Premium** | ₹{price_ranges['medium_threshold']:.0f}L - ₹{price_ranges['high_threshold']:.0f}L | 🔴 Red |
    | **Luxury** | Above ₹{price_ranges['high_threshold']:.0f} Lakhs | 🟣 Purple |

    These thresholds are computed from the dataset's 25th, 50th, and 75th percentiles.
    """)
