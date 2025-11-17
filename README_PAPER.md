# NeuroTrader Research Paper - Graph Export Guide

## 📊 Generating Graphs for the Paper

Your LaTeX paper (`NeuroTrader_Paper.tex`) now includes three figures that need to be generated from your Jupyter notebook.

### Method 1: Automatic (Recommended) ✨

The notebook has been updated to **automatically save** the graphs when you run the analysis cells.

1. **Open the notebook:**
   ```bash
   code NeuroTrader.ipynb
   ```

2. **Run Cell 5** (Comprehensive Stock Analysis):
   - This will generate and automatically save:
     - `market_dashboard.png` (Figure 1)
     - `stock_cards.png` (Figure 2)

3. **Run Cell 6** (Stress Testing Framework):
   - Add this line at the very end of the cell (after `plt.show()`):
   ```python
   plt.savefig('stress_testing.png', dpi=300, bbox_inches='tight', facecolor='white')
   print("💾 Saved stress_testing.png")
   ```
   - Re-run the cell to save the stress testing figure

4. **Verify the files:**
   ```bash
   ls -lh *.png
   ```
   You should see:
   - `market_dashboard.png` (~2-3 MB)
   - `stock_cards.png` (~2-3 MB)
   - `stress_testing.png` (~1-2 MB)

### Method 2: Manual Export

If automatic saving doesn't work:

1. Run Cell 5 and Cell 6 in the notebook
2. Right-click on each figure
3. Select "Save Image As..."
4. Save with the exact filenames:
   - `market_dashboard.png`
   - `stock_cards.png`
   - `stress_testing.png`

## 📝 Compiling the LaTeX Paper

### Prerequisites

Install LaTeX if you haven't already:
- **macOS:** `brew install --cask mactex`
- **Or use Overleaf:** Upload all files to [overleaf.com](https://www.overleaf.com)

### Compilation

1. **Make sure graphs are in the same folder:**
   ```bash
   cd /Users/srijanjha/Desktop/NT
   ls *.png *.tex
   ```

2. **Compile the paper:**
   ```bash
   pdflatex NeuroTrader_Paper.tex
   pdflatex NeuroTrader_Paper.tex  # Run twice for references
   ```

3. **Open the PDF:**
   ```bash
   open NeuroTrader_Paper.pdf
   ```

### Using Overleaf (Easier)

1. Go to [overleaf.com](https://www.overleaf.com) and create account
2. Create new project → "Upload Project"
3. Zip these files:
   ```bash
   zip neurotrader_paper.zip NeuroTrader_Paper.tex market_dashboard.png stock_cards.png stress_testing.png
   ```
4. Upload the zip file
5. Click "Recompile" in Overleaf
6. Download the PDF

## 🖼️ Figure Descriptions

### Figure 1: Market Dashboard (`market_dashboard.png`)
- **Position in paper:** Section 5.3, Results subsection
- **Contents:** 8-panel comprehensive dashboard
  - Market sentiment pie chart
  - Confidence rankings bar chart
  - High-value stocks
  - Returns momentum map (scatter plot)
  - RSI technical indicators
  - Volatility risk profile
  - Risk-return matrix
  - Detailed metrics table

### Figure 2: Stock Cards (`stock_cards.png`)
- **Position in paper:** Section 5.3, Results subsection
- **Contents:** 16 individual stock performance cards
  - Color-coded by prediction (green=UP, red=DOWN)
  - Shows prediction, confidence, price, returns, RSI, volatility
  - Premium card-based layout

### Figure 3: Stress Testing (`stress_testing.png`)
- **Position in paper:** Section 5.2, Stress Testing Results
- **Contents:** 4-panel stress testing analysis
  - Flip rate by scenario (horizontal bar)
  - Confidence impact (horizontal bar)
  - Model resilience pie chart
  - Impact heatmap (scenario × stock matrix)

## 📋 Quick Checklist

- [ ] Run Cell 5 in notebook → generates `market_dashboard.png` and `stock_cards.png`
- [ ] Add save line to Cell 6 → generates `stress_testing.png`
- [ ] Verify all 3 PNG files exist in `/Users/srijanjha/Desktop/NT`
- [ ] Compile LaTeX: `pdflatex NeuroTrader_Paper.tex` (twice)
- [ ] Open PDF: `open NeuroTrader_Paper.pdf`
- [ ] Check that all figures appear correctly in the paper

## 🎨 Figure Quality Settings

All figures are saved with:
- **DPI:** 300 (publication quality)
- **Format:** PNG with white background
- **Size:** 24×16 inches (dashboard), 24×18 inches (cards), ~18×10 inches (stress test)
- **Bounding:** Tight (no extra whitespace)

## 🚨 Troubleshooting

**Problem:** "File not found" error when compiling LaTeX
- **Solution:** Make sure PNG files are in the same directory as the .tex file

**Problem:** Figures look pixelated in PDF
- **Solution:** Increase DPI to 400 or 600 in the `savefig()` calls

**Problem:** Cell 5/6 doesn't generate figures
- **Solution:** Make sure you ran Cell 4 first to train the model

**Problem:** "Memory error" when saving figures
- **Solution:** Reduce DPI to 200 or reduce figure size (figsize parameter)

## 📚 Paper Structure

Your paper now includes:
- ✅ Title and authors
- ✅ Abstract (accurate description of your system)
- ✅ Introduction with motivation
- ✅ Related work (13 references)
- ✅ System architecture
- ✅ Methodology (ensemble, technical override, stress testing)
- ✅ Experimental results with real numbers
- ✅ 3 high-quality figures with detailed captions
- ✅ Discussion and limitations
- ✅ Conclusion and future work
- ✅ Bibliography

**Total pages:** ~8-10 pages in IEEE conference format

Perfect for your academic presentation! 🎓
