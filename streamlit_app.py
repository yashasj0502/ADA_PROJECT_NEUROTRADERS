
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the Python path to import our classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our NeuroTrader classes (assuming they are in the same directory)
try:
    from neurotrader_system import EnhancedNeuroTraderSystem, enhanced_analysis_with_shap
except ImportError:
    st.error("⚠️ Could not import NeuroTrader system. Please ensure neurotrader_system.py is available.")

st.set_page_config(
    page_title="NeuroTrader - AI Stock Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    border: 1px solid #e6e9ef;
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
}
.prediction-up {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
}
.prediction-down {
    background-color: #f8d7da;
    border-color: #f5c6cb;
    color: #721c24;
}
.confidence-high {
    border-left: 5px solid #28a745;
}
.confidence-medium {
    border-left: 5px solid #ffc107;
}
.confidence-low {
    border-left: 5px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'neurotrader' not in st.session_state:
    st.session_state.neurotrader = None
    st.session_state.initialized = False

# Header
st.title("🚀 NeuroTrader: AI-Powered Stock Prediction System")
st.markdown("**Multimodal • Explainable • Stress-Tested**")

# Sidebar
st.sidebar.header("🎛️ Control Panel")

# System initialization
if not st.session_state.initialized:
    st.sidebar.warning("⚠️ System needs to be initialized")

    if st.sidebar.button("🚀 Initialize NeuroTrader System"):
        with st.spinner("Initializing NeuroTrader with real sentiment data..."):
            try:
                st.session_state.neurotrader = EnhancedNeuroTraderSystem()

                # Initialize with comprehensive data
                training_data = st.session_state.neurotrader.initialize_with_comprehensive_data(
                    tickers=['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL'],
                    start_date='2020-01-01'
                )

                if training_data is not None:
                    st.session_state.initialized = True
                    st.sidebar.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to initialize system")

            except Exception as e:
                st.sidebar.error(f"❌ Initialization error: {str(e)}")
else:
    st.sidebar.success("✅ System Ready")

# Main interface (only show if initialized)
if st.session_state.initialized and st.session_state.neurotrader:

    # Stock selection
    available_tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    selected_ticker = st.sidebar.selectbox(
        "📊 Select Stock",
        available_tickers,
        index=0
    )

    # Stress test scenario
    stress_scenarios = [
        'None',
        'market_dip',
        'volatility_spike', 
        'sentiment_crash',
        'liquidity_shock',
        'systemic_crisis',
        'black_swan'
    ]

    selected_scenario = st.sidebar.selectbox(
        "🧪 Stress Test Scenario",
        stress_scenarios,
        index=0
    )

    # Analysis button
    if st.sidebar.button("🎯 Run Analysis"):

        # Create three columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with st.spinner(f"Analyzing {selected_ticker}..."):
            try:
                # Get enhanced analysis with SHAP
                analysis = enhanced_analysis_with_shap(st.session_state.neurotrader, selected_ticker)

                if analysis:
                    # Column 1: Prediction Results
                    with col1:
                        st.subheader("🎯 Prediction Results")

                        prediction = analysis['prediction']
                        confidence = analysis['confidence']
                        prob_up = analysis['probability_up']
                        prob_down = analysis['probability_down']

                        # Prediction display
                        pred_class = "prediction-up" if prediction == "UP" else "prediction-down"
                        conf_class = ("confidence-high" if confidence > 0.75 else 
                                     "confidence-medium" if confidence > 0.65 else "confidence-low")

                        st.markdown(f"""
                        <div class="metric-container {pred_class} {conf_class}">
                            <h3>Direction: {prediction} {'📈' if prediction == 'UP' else '📉'}</h3>
                            <h4>Confidence: {confidence:.2%}</h4>
                            <p>Probability UP: {prob_up:.2%}</p>
                            <p>Probability DOWN: {prob_down:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Model accuracy
                        model_acc = analysis.get('model_accuracy', 0)
                        st.metric("Model Accuracy", f"{model_acc:.2%}")

                    # Column 2: Explanation
                    with col2:
                        st.subheader("💡 AI Explanation")

                        # Enhanced rationale
                        enhanced_rationale = analysis.get('enhanced_rationale', analysis.get('rationale', 'No rationale available'))
                        st.write(enhanced_rationale)

                        # Feature importance
                        feature_analysis = analysis.get('feature_analysis', pd.DataFrame())
                        if not feature_analysis.empty:
                            st.subheader("🔍 Top Features")
                            top_features = feature_analysis.head(5)

                            fig = px.bar(
                                top_features,
                                x='importance_normalized',
                                y='feature',
                                orientation='h',
                                title="Feature Importance"
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                        # SHAP explanation
                        shap_explanation = analysis.get('shap_explanation')
                        if shap_explanation is not None and not shap_explanation.empty:
                            st.subheader("🧠 SHAP Analysis")
                            top_shap = shap_explanation.head(3)
                            for _, row in top_shap.iterrows():
                                impact = "Positive" if row['shap_value'] > 0 else "Negative"
                                st.write(f"• **{row['feature'].replace('_', ' ').title()}**: {impact} impact ({row['shap_value']:.3f})")

                    # Column 3: Stress Testing
                    with col3:
                        st.subheader("🧪 Stress Test Results")

                        stress_results = analysis.get('stress_test_results', {})
                        if stress_results:

                            # Overall resilience
                            resilience = stress_results.get('resilience_score', 'N/A')
                            if isinstance(resilience, (int, float)):
                                st.metric("Resilience Score", f"{resilience:.1f}%")

                            # Predictions flipped
                            flipped = stress_results.get('predictions_flipped', 'N/A')
                            total_tests = stress_results.get('total_tests', 'N/A')
                            st.metric("Predictions Flipped", f"{flipped}/{total_tests}")

                            # Individual scenario results
                            if selected_scenario != 'None':
                                st.subheader(f"🎭 {selected_scenario.replace('_', ' ').title()} Scenario")

                                scenario_result = stress_results.get(selected_scenario, {})
                                if scenario_result:
                                    orig_conf = stress_results.get('original', {}).get('confidence', 0)
                                    stress_conf = scenario_result.get('confidence', 0)
                                    conf_change = stress_conf - orig_conf

                                    st.metric(
                                        "Confidence Change", 
                                        f"{conf_change:+.2%}",
                                        delta=f"{conf_change:.2%}"
                                    )

                                    if scenario_result.get('prediction_flipped', False):
                                        st.warning("⚠️ Prediction flipped under stress!")
                                    else:
                                        st.success("✅ Prediction held under stress")

                        # Risk metrics
                        st.subheader("⚡ Risk Metrics")
                        confidence_level = "High" if confidence > 0.75 else "Medium" if confidence > 0.65 else "Low"
                        risk_color = "green" if confidence > 0.75 else "orange" if confidence > 0.65 else "red"

                        st.markdown(f"**Confidence Level:** :{risk_color}[{confidence_level}]")

                        if isinstance(resilience, (int, float)):
                            resilience_level = "Excellent" if resilience > 80 else "Good" if resilience > 60 else "Weak"
                            res_color = "green" if resilience > 80 else "orange" if resilience > 60 else "red"
                            st.markdown(f"**Resilience:** :{res_color}[{resilience_level}]")

                else:
                    st.error("Failed to analyze the selected stock")

            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    # Additional tabs
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Market Data", "ℹ️ About"])

    with tab1:
        st.subheader("🤖 Model Performance Metrics")

        if hasattr(st.session_state.neurotrader.predictor, 'performance_metrics'):
            metrics = st.session_state.neurotrader.predictor.performance_metrics

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
            with col2:
                st.metric("Best Model", metrics.get('best_model', 'N/A'))
            with col3:
                st.metric("Feature Count", metrics.get('feature_count', 0))
            with col4:
                st.metric("Training Samples", metrics.get('train_size', 0))

    with tab2:
        st.subheader("📈 Market Overview")
        st.info("Real-time market data integration would go here")

        # Placeholder for market data visualization
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'Price': np.cumsum(np.random.randn(30)) + 100
        })

        fig = px.line(sample_data, x='Date', y='Price', title='Sample Stock Price')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("ℹ️ About NeuroTrader")

        st.markdown("""
        ### 🚀 NeuroTrader Features

        **Multimodal Analysis:**
        - 📊 Technical indicators and price patterns
        - 📰 Real news sentiment analysis using FinBERT
        - 🌐 Market-wide indicators (VIX, SPY)

        **Explainable AI:**
        - 🧠 SHAP-based feature importance
        - 💡 Natural language explanations
        - 🔍 Transparent decision making

        **Stress Testing:**
        - 📉 Market dip scenarios
        - 📈 Volatility spike testing
        - 😰 Sentiment crash simulation
        - 💧 Liquidity shock analysis
        - 🌪️ Black swan event testing

        **Machine Learning:**
        - 🎯 Ensemble models (LightGBM, XGBoost, Random Forest)
        - 📈 70%+ target accuracy
        - 🔄 Time-series aware validation
        """)

else:
    # Show introduction if not initialized
    st.markdown("""
    ## Welcome to NeuroTrader! 🚀

    A cutting-edge AI system that combines:

    ### 🔬 **Multimodal Analysis**
    - Technical indicators from price data
    - Sentiment analysis from financial news
    - Market-wide risk indicators

    ### 🧠 **Explainable AI** 
    - SHAP-based feature importance
    - Natural language explanations
    - Transparent prediction rationale

    ### 🧪 **Stress Testing**
    - Market crash scenarios
    - Volatility spikes
    - Sentiment crashes
    - Liquidity shocks

    **Click "Initialize NeuroTrader System" in the sidebar to begin!**
    """)

    # Show sample visualization
    st.subheader("📊 Sample Analysis Preview")

    sample_data = pd.DataFrame({
        'Feature': ['Price Momentum', 'Sentiment Score', 'Volume Ratio', 'RSI', 'Market Correlation'],
        'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
    })

    fig = px.bar(sample_data, x='Importance', y='Feature', orientation='h', 
                 title='Sample Feature Importance Analysis')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This is an educational AI system. Always conduct your own research before making investment decisions.")
