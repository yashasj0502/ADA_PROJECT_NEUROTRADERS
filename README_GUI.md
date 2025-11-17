# NeuroTrader - AI-Powered Stock Prediction GUI

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation Steps

1. **Navigate to the project directory:**
```bash
cd /Users/srijanjha/Desktop/NT
```

2. **Install required packages:**
```bash
pip install streamlit pandas numpy plotly scikit-learn lightgbm xgboost
```

OR install from requirements file (if available):
```bash
pip install -r requirements.txt
```

### Running the Application

**Method 1: Using Streamlit CLI (Recommended)**
```bash
streamlit run app.py
```

**Method 2: Using Python**
```bash
python -m streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

---

## 📊 Application Features

### 🎯 Main Features:
1. **Stock Prediction** - AI-powered UP/DOWN predictions with confidence scores
2. **Stress Testing** - Test predictions under extreme market scenarios:
   - 🦠 COVID-25 Pandemic (-40% crash)
   - 💥 Market Crash (-35% decline)
   - 🚀 Bull Rally (+40% gains)
   - 📈 Recovery Phase (+30% rebound)
3. **Market Analytics** - Top gainers and losers analysis

### 🎨 Modern UI Features:
- ✨ Animated gradient backgrounds
- 🌟 Glowing effects and smooth animations
- 📱 Responsive glass-morphism cards
- 🎭 Pulsing prediction boxes
- 🔮 Magical hover effects

---

## 🔧 How to Use

### Step 1: Train the Model
1. Click the **"🤖 Train AI Model"** button in the sidebar
2. Wait for training to complete (~30 seconds)
3. Model accuracy will be displayed

### Step 2: Make Predictions
1. Select a stock from the dropdown menu (sidebar)
2. View the prediction (UP/DOWN) with confidence score
3. Analyze key metrics:
   - Current Price
   - 1-Day Return
   - 20-Day Return  
   - RSI (14)
4. Review the price history chart

### Step 3: Stress Testing
1. Go to the **"🔬 Stress Testing"** tab
2. Select a market scenario
3. Click **"🔬 Run Stress Test"**
4. Compare baseline vs stressed predictions
5. See if predictions flip under extreme conditions

### Step 4: Market Analytics
1. Go to the **"📊 Analytics"** tab
2. View top gainers (📈) and losers (📉)
3. Analyze 20-day returns across stocks

---

## 📁 Required Files

Make sure these CSV files are in the same directory as `app.py`:
- `stock_market_june2025.csv`
- `stock_data_july_2025.csv`
- `stock_data_aug_2025.csv`

---

## 🎯 Model Details

- **Architecture**: Ensemble of 3 models
  - Random Forest
  - LightGBM
  - XGBoost
- **Accuracy**: ~71% on 2025 market data
- **Features**: 6 technical indicators
  - Returns (1-day, 5-day, 20-day)
  - Volatility (20-day annualized)
  - RSI (14-period)
  - Volume Ratio
- **Training Data**: 6,693 records from June-August 2025

---

## 🐛 Troubleshooting

### Issue: Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Issue: ModuleNotFoundError
```bash
pip install --upgrade streamlit pandas numpy plotly scikit-learn lightgbm xgboost
```

### Issue: CSV files not found
- Ensure all 3 CSV files are in the same directory as `app.py`
- Check file names match exactly (case-sensitive)

### Issue: Browser doesn't open
- Manually navigate to: `http://localhost:8501`
- Or check terminal output for the correct URL

---

## 💡 Tips for Best Experience

1. **Train the model first** before making predictions
2. **Use Chrome or Firefox** for best compatibility
3. **Full screen mode** recommended for optimal UI experience
4. **Retrain periodically** for updated predictions
5. **Try different scenarios** in stress testing to understand model robustness

---

## 🎨 UI Theme
- **Dark mode** with gradient backgrounds
- **Animated elements** for magical experience
- **Glass morphism** cards with blur effects
- **Color coding**:
  - 🟢 Green = UP/Positive
  - 🔴 Red = DOWN/Negative
  - 🟡 Yellow = Warning/Neutral

---

## 🔄 Stopping the Application

Press `Ctrl+C` in the terminal where the app is running

---

## 📝 Notes

- First run may take longer due to model training
- Predictions update in real-time
- All calculations happen locally (no internet required after package installation)
- Model uses real June-August 2025 market data

---

## 🚀 Quick Command Summary

```bash
# Navigate to directory
cd /Users/srijanjha/Desktop/NT

# Install packages
pip install streamlit pandas numpy plotly scikit-learn lightgbm xgboost

# Run app
streamlit run app.py

# Alternative port
streamlit run app.py --server.port 8502

# Stop app
Ctrl+C
```

---

**Enjoy predicting stocks with NeuroTrader! 🧠📈**
