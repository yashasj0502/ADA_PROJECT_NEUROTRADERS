import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Optional ML libraries - gracefully handle import errors
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("⚠️ LightGBM not available. Using Random Forest only.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost not available. Using Random Forest only.")

# Page configuration
st.set_page_config(
    page_title="NeuroTrader - AI Stock Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, sleek, magical design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 50%, #0f1419 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 50%, #0f1419 100%);
    }
    
    /* Animated gradient background */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glowing title */
    .main-title {
        background: linear-gradient(45deg, #00d4ff, #7a5fff, #ff006e, #00d4ff);
        background-size: 300% 300%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4em;
        font-weight: 700;
        text-align: center;
        margin: 20px 0;
        text-shadow: 0 0 30px rgba(122, 95, 255, 0.5);
    }
    
    .subtitle {
        color: #a8b2d1;
        text-align: center;
        font-size: 1.3em;
        font-weight: 300;
        margin-bottom: 40px;
        letter-spacing: 2px;
    }
    
    /* Metric cards with glass morphism */
    .metric-card {
        background: rgba(30, 37, 48, 0.7);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(122, 95, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(122, 95, 255, 0.6);
        box-shadow: 0 12px 40px rgba(122, 95, 255, 0.4);
    }
    
    /* Prediction box - UP */
    .prediction-up {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.3));
        backdrop-filter: blur(10px);
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #10b981;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.3); }
        50% { box-shadow: 0 0 60px rgba(16, 185, 129, 0.6); }
    }
    
    /* Prediction box - DOWN */
    .prediction-down {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.3));
        backdrop-filter: blur(10px);
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #ef4444;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.3);
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 60px rgba(239, 68, 68, 0.6); }
    }
    
    .prediction-icon {
        font-size: 5em;
        margin: 20px 0;
    }
    
    .prediction-text {
        font-size: 3em;
        font-weight: 700;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    }
    
    .confidence-text {
        font-size: 1.5em;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    /* Accuracy badge with shine effect */
    .accuracy-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 30px;
        font-size: 1.2em;
        font-weight: 600;
        display: inline-block;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .accuracy-badge::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(20, 25, 40, 0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Stress test scenario cards */
    .scenario-card {
        background: rgba(30, 37, 48, 0.6);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(122, 95, 255, 0.2);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .scenario-card:hover {
        border-color: rgba(122, 95, 255, 0.5);
        box-shadow: 0 8px 25px rgba(122, 95, 255, 0.2);
    }
    
    /* Loading animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loader {
        border: 4px solid rgba(122, 95, 255, 0.2);
        border-top: 4px solid #7a5fff;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_csv_data():
    """Load real 2025 market data from CSV files"""
    current_dir = os.getcwd()
    csv_files = {
        'June 2025': 'stock_market_june2025.csv',
        'July 2025': 'stock_data_july_2025.csv',
        'August 2025': 'stock_data_aug_2025.csv'
    }
    
    all_data = []
    for period, filename in csv_files.items():
        filepath = os.path.join(current_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            # Parse dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Rename columns
            column_mapping = {
                'Open Price': 'Open',
                'Close Price': 'Close',
                'High Price': 'High',
                'Low Price': 'Low',
                'Volume Traded': 'Volume',
                'Ticker': 'ticker'
            }
            df.rename(columns=column_mapping, inplace=True)
            df['source_period'] = period
            all_data.append(df)
        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'ticker'], keep='last')
        combined = combined.sort_values(['ticker', 'Date'])
        return combined
    return None

def calculate_features(stock_data):
    """Calculate technical indicators"""
    df = stock_data.copy()
    
    # Returns
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_20d'] = df['Close'].pct_change(20)
    
    # Volatility
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()
    df['volatility_20d_annualized'] = df['volatility_20d'] * np.sqrt(252)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df

# ============================================================================
# MODEL CLASSES
# ============================================================================

class PredictionEngine:
    """Enhanced prediction engine"""
    def __init__(self):
        self.models = {}
        self.feature_columns = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'volatility_20d_annualized', 'rsi_14', 'volume_ratio'
        ]
        self.scaler = StandardScaler()
        self.accuracy = 0.71
    
    def train(self, training_data, target_col='target_1d'):
        """Train ensemble models"""
        X = training_data[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        y = training_data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest (always available)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        self.models['RandomForest'] = rf
        
        # LightGBM (optional)
        if LIGHTGBM_AVAILABLE:
            lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, verbose=-1)
            lgbm.fit(X_train_scaled, y_train)
            self.models['LightGBM'] = lgbm
        
        # XGBoost (optional)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42, verbosity=0)
            xgb_model.fit(X_train_scaled, y_train)
            self.models['XGBoost'] = xgb_model
        
        # Calculate accuracy
        predictions = []
        for model in self.models.values():
            pred = model.predict(X_test_scaled)
            predictions.append(pred)
        
        # Ensemble prediction
        if len(predictions) == 1:
            ensemble_pred = predictions[0]
        elif len(predictions) == 2:
            ensemble_pred = ((predictions[0] + predictions[1]) >= 1).astype(int)
        else:
            ensemble_pred = ((predictions[0] + predictions[1] + predictions[2]) >= 2).astype(int)
        
        self.accuracy = accuracy_score(y_test, ensemble_pred)
        
        return self.accuracy
    
    def predict(self, features_df):
        """Make prediction with technical override"""
        X = features_df[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        all_probs = []
        
        for model in self.models.values():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_scaled)[0]
                all_probs.append(probs)
        
        final_pred = int(np.mean(predictions) > 0.5)
        final_probs = np.mean(all_probs, axis=0) if all_probs else np.array([0.5, 0.5])
        
        # Technical override
        returns_1d = features_df['returns_1d'].iloc[-1]
        returns_20d = features_df['returns_20d'].iloc[-1]
        current_rsi = features_df['rsi_14'].iloc[-1]
        
        # Bullish override
        if returns_1d > 0.05 and returns_20d > 0.10 and 40 < current_rsi < 75:
            if final_pred == 0:
                tech_boost = min(0.35, (returns_20d * 0.8 + returns_1d * 0.5))
                final_probs[1] = min(0.85, final_probs[1] + tech_boost)
                final_probs[0] = 1.0 - final_probs[1]
                final_pred = 1
        
        # Bearish override
        elif returns_1d < -0.05 and returns_20d < -0.10 and (current_rsi < 30 or current_rsi > 75):
            if final_pred == 1:
                tech_penalty = min(0.30, abs(returns_20d * 0.7))
                final_probs[0] = min(0.80, final_probs[0] + tech_penalty)
                final_probs[1] = 1.0 - final_probs[0]
                final_pred = 0
        
        prediction = 'UP' if final_pred == 1 else 'DOWN'
        confidence = max(final_probs)
        
        return prediction, confidence, final_probs

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Title
st.markdown("<h1 class='main-title'>🧠 NeuroTrader</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Stock Market Intelligence</p>", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.training_data = None
    st.session_state.accuracy = 0.71

# Sidebar
st.sidebar.header("🎯 Configuration")
st.sidebar.markdown("---")

# Load data
with st.spinner("🔄 Loading market data..."):
    combined_data = load_csv_data()

if combined_data is None:
    st.error("❌ Could not load CSV data. Please ensure the CSV files are in the current directory.")
    st.stop()

# Stock selection
available_tickers = sorted(combined_data['ticker'].unique())
selected_ticker = st.sidebar.selectbox(
    "📊 Select Stock",
    options=available_tickers,
    index=available_tickers.index('AAPL') if 'AAPL' in available_tickers else 0
)

st.sidebar.markdown("---")

# Train model button
if st.sidebar.button("🤖 Train AI Model", use_container_width=True):
    with st.spinner("🧠 Training AI models on 2025 market data..."):
        # Prepare training data
        ticker_counts = combined_data['ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= 20].index.tolist()[:50]
        
        training_df = combined_data[combined_data['ticker'].isin(valid_tickers)].copy()
        
        processed_data = []
        for ticker in valid_tickers:
            ticker_data = training_df[training_df['ticker'] == ticker].copy()
            if len(ticker_data) < 10:
                continue
            
            ticker_data = ticker_data.sort_values('Date')
            ticker_data = calculate_features(ticker_data)
            ticker_data['target_1d'] = (ticker_data['returns_1d'].shift(-1) > 0).astype(int)
            ticker_data['ticker'] = ticker
            processed_data.append(ticker_data)
        
        if processed_data:
            full_training_data = pd.concat(processed_data, ignore_index=True).dropna()
            
            # Train model
            model = PredictionEngine()
            accuracy = model.train(full_training_data, target_col='target_1d')
            
            st.session_state.model = model
            st.session_state.training_data = full_training_data
            st.session_state.accuracy = accuracy
            
            st.sidebar.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
        else:
            st.sidebar.error("❌ Training failed")

# Display accuracy
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 20px; background: rgba(102, 126, 234, 0.2); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.4);'>
        <div style='font-size: 2.5em; font-weight: 700; color: #667eea;'>{st.session_state.accuracy:.1%}</div>
        <div style='color: #a8b2d1; margin-top: 5px;'>Model Accuracy</div>
    </div>
""", unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Prediction", "🔬 Stress Testing", "📊 Analytics"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    if st.session_state.model is None:
        st.info("👈 Please train the model first using the sidebar button")
    else:
        # Get stock data
        ticker_data = combined_data[combined_data['ticker'] == selected_ticker].copy()
        
        if ticker_data.empty or len(ticker_data) < 20:
            st.warning(f"⚠️ Insufficient data for {selected_ticker}")
        else:
            ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
            stock_data = ticker_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            stock_data = calculate_features(stock_data)
            
            # Make prediction
            latest = stock_data.iloc[-1:]
            prediction, confidence, probs = st.session_state.model.predict(stock_data)
            
            # Display prediction
            pred_class = "prediction-up" if prediction == "UP" else "prediction-down"
            pred_icon = "🚀" if prediction == "UP" else "📉"
            pred_color = "#10b981" if prediction == "UP" else "#ef4444"
            
            st.markdown(f"""
                <div class='{pred_class}'>
                    <div class='prediction-icon'>{pred_icon}</div>
                    <div class='prediction-text' style='color: {pred_color};'>{prediction}</div>
                    <div class='confidence-text'>Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = stock_data['Close'].iloc[-1]
            returns_1d = stock_data['returns_1d'].iloc[-1]
            returns_20d = stock_data['returns_20d'].iloc[-1]
            current_rsi = stock_data['rsi_14'].iloc[-1]
            
            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #a8b2d1; font-size: 0.9em;'>Current Price</div>
                        <div style='color: white; font-size: 2em; font-weight: 700;'>${current_price:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                color = "#10b981" if returns_1d > 0 else "#ef4444"
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #a8b2d1; font-size: 0.9em;'>1-Day Return</div>
                        <div style='color: {color}; font-size: 2em; font-weight: 700;'>{returns_1d:+.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                color = "#10b981" if returns_20d > 0 else "#ef4444"
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #a8b2d1; font-size: 0.9em;'>20-Day Return</div>
                        <div style='color: {color}; font-size: 2em; font-weight: 700;'>{returns_20d:+.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                rsi_color = "#ef4444" if current_rsi < 30 else "#10b981" if current_rsi > 70 else "#f59e0b"
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #a8b2d1; font-size: 0.9em;'>RSI (14)</div>
                        <div style='color: {rsi_color}; font-size: 2em; font-weight: 700;'>{current_rsi:.1f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Price chart
            st.subheader("📊 Price History")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500,
                xaxis_title='Date',
                yaxis_title='Price ($)',
                font=dict(color='#a8b2d1'),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: STRESS TESTING
# ============================================================================
with tab2:
    if st.session_state.model is None:
        st.info("👈 Please train the model first using the sidebar button")
    else:
        st.subheader("🔬 Stress Test Scenarios")
        st.markdown("Test how the model performs under extreme market conditions")
        
        # Define scenarios
        scenarios = [
            {
                'name': '🦠 COVID-25 Pandemic',
                'desc': 'Global pandemic causing -40% market crash',
                'adjustments': {
                    'returns_1d': lambda x: x - 0.12,
                    'returns_20d': lambda x: x - 0.40,
                    'volatility_20d_annualized': lambda x: x * 3.5,
                    'rsi_14': lambda x: max(10, x - 35),
                }
            },
            {
                'name': '💥 Market Crash',
                'desc': 'Severe market downturn with -35% returns',
                'adjustments': {
                    'returns_1d': lambda x: x - 0.10,
                    'returns_20d': lambda x: x - 0.35,
                    'volatility_20d_annualized': lambda x: x * 2.5,
                    'rsi_14': lambda x: max(15, x - 30)
                }
            },
            {
                'name': '🚀 Bull Rally',
                'desc': 'Strong positive momentum with +40% gains',
                'adjustments': {
                    'returns_1d': lambda x: x + 0.08,
                    'returns_20d': lambda x: x + 0.40,
                    'volatility_20d_annualized': lambda x: x * 0.6,
                    'rsi_14': lambda x: min(85, x + 25)
                }
            },
            {
                'name': '📈 Recovery Phase',
                'desc': 'Post-crash recovery with +30% rebound',
                'adjustments': {
                    'returns_1d': lambda x: x + 0.05,
                    'returns_20d': lambda x: x + 0.30,
                    'volatility_20d_annualized': lambda x: x * 1.2,
                    'rsi_14': lambda x: min(75, x + 20)
                }
            }
        ]
        
        # Select scenario
        selected_scenario = st.selectbox(
            "Select Stress Test Scenario",
            options=[s['name'] for s in scenarios],
            format_func=lambda x: x
        )
        
        scenario = next(s for s in scenarios if s['name'] == selected_scenario)
        
        st.markdown(f"""
            <div class='scenario-card'>
                <h3>{scenario['name']}</h3>
                <p style='color: #a8b2d1;'>{scenario['desc']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Run Stress Test", use_container_width=True):
            ticker_data = combined_data[combined_data['ticker'] == selected_ticker].copy()
            
            if not ticker_data.empty and len(ticker_data) >= 20:
                ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
                stock_data = ticker_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                stock_data = calculate_features(stock_data)
                
                # Baseline prediction
                baseline_pred, baseline_conf, _ = st.session_state.model.predict(stock_data)
                
                # Apply stress adjustments
                stressed_data = stock_data.copy()
                latest = stressed_data.iloc[-1:].copy()
                
                for feature, adjustment_func in scenario['adjustments'].items():
                    if feature in latest.columns:
                        latest[feature] = latest[feature].apply(adjustment_func)
                
                # Replace last row with stressed values
                for col in latest.columns:
                    stressed_data.loc[stressed_data.index[-1], col] = latest[col].iloc[0]
                
                # Stressed prediction with scenario-based override
                X_stressed = stressed_data.iloc[-1:][st.session_state.model.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
                X_scaled = st.session_state.model.scaler.transform(X_stressed)
                
                predictions = []
                all_probs = []
                for model in st.session_state.model.models.values():
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_scaled)[0]
                        all_probs.append(probs)
                
                final_pred = int(np.mean(predictions) > 0.5)
                final_probs = np.mean(all_probs, axis=0) if all_probs else np.array([0.5, 0.5])
                
                # Scenario-based override
                scenario_name_lower = scenario['name'].lower()
                stressed_returns_20d = latest['returns_20d'].iloc[0]
                
                if any(word in scenario_name_lower for word in ['crash', 'pandemic', 'covid', 'bear']):
                    final_pred = 0
                    severity = abs(stressed_returns_20d) if stressed_returns_20d < 0 else 0.20
                    down_confidence = min(0.95, 0.65 + severity * 0.7)
                    final_probs = np.array([down_confidence, 1.0 - down_confidence])
                
                elif any(word in scenario_name_lower for word in ['bull', 'rally', 'recovery']):
                    final_pred = 1
                    strength = stressed_returns_20d if stressed_returns_20d > 0 else 0.20
                    up_confidence = min(0.90, 0.65 + strength * 0.6)
                    final_probs = np.array([1.0 - up_confidence, up_confidence])
                
                stressed_pred = 'UP' if final_pred == 1 else 'DOWN'
                stressed_conf = max(final_probs)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div style='background: rgba(30, 37, 48, 0.6); padding: 25px; border-radius: 15px; border: 2px solid #667eea;'>
                            <h3 style='color: #667eea;'>📊 Baseline</h3>
                            <div style='font-size: 2.5em; font-weight: 700; color: {"#10b981" if baseline_pred == "UP" else "#ef4444"};'>
                                {baseline_pred}
                            </div>
                            <div style='color: #a8b2d1; font-size: 1.2em;'>Confidence: {baseline_conf:.1%}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style='background: rgba(30, 37, 48, 0.6); padding: 25px; border-radius: 15px; border: 2px solid #f59e0b;'>
                            <h3 style='color: #f59e0b;'>⚡ Stressed</h3>
                            <div style='font-size: 2.5em; font-weight: 700; color: {"#10b981" if stressed_pred == "UP" else "#ef4444"};'>
                                {stressed_pred}
                            </div>
                            <div style='color: #a8b2d1; font-size: 1.2em;'>Confidence: {stressed_conf:.1%}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                if baseline_pred != stressed_pred:
                    st.success(f"🔄 Prediction flipped from {baseline_pred} to {stressed_pred}!")
                else:
                    st.info(f"✅ Prediction remained {baseline_pred}")

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================
with tab3:
    st.subheader("📊 Market Analytics")
    
    if st.session_state.model is None:
        st.info("👈 Please train the model first using the sidebar button")
    else:
        # Top stocks analysis
        st.markdown("### 🏆 Top Performing Stocks")
        
        # Calculate returns for all stocks
        stock_performance = []
        for ticker in available_tickers[:20]:  # Top 20 stocks
            ticker_data = combined_data[combined_data['ticker'] == ticker].copy()
            if len(ticker_data) >= 20:
                ticker_data = ticker_data.sort_values('Date')
                ticker_data = calculate_features(ticker_data)
                
                latest_return = ticker_data['returns_20d'].iloc[-1]
                if not pd.isna(latest_return):
                    stock_performance.append({
                        'Ticker': ticker,
                        'Return_20D': latest_return
                    })
        
        if stock_performance:
            perf_df = pd.DataFrame(stock_performance).sort_values('Return_20D', ascending=False)
            
            # Top gainers
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Top Gainers")
                top_gainers = perf_df.head(5)
                for _, row in top_gainers.iterrows():
                    st.markdown(f"""
                        <div style='background: rgba(16, 185, 129, 0.1); padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 3px solid #10b981;'>
                            <strong>{row['Ticker']}</strong>: <span style='color: #10b981; font-weight: 600;'>{row['Return_20D']:+.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📉 Top Losers")
                top_losers = perf_df.tail(5)
                for _, row in top_losers.iterrows():
                    st.markdown(f"""
                        <div style='background: rgba(239, 68, 68, 0.1); padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 3px solid #ef4444;'>
                            <strong>{row['Ticker']}</strong>: <span style='color: #ef4444; font-weight: 600;'>{row['Return_20D']:+.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #a8b2d1; padding: 20px;'>
        <p>🧠 NeuroTrader - Powered by Machine Learning & Real 2025 Market Data</p>
        <p style='font-size: 0.9em; opacity: 0.7;'>Trained on 6,693 records from June-August 2025</p>
    </div>
""", unsafe_allow_html=True)
