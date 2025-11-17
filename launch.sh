#!/bin/bash

# NeuroTrader Launch Script
# Sets up environment and launches Streamlit app

echo "🧠 NeuroTrader - Launching AI Stock Prediction GUI..."
echo ""

# Set library path for LightGBM
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Launch Streamlit
streamlit run app.py
