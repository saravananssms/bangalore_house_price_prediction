# 🏠 Bangalore House Price Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.10-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly)

An AI-powered real estate price prediction system for **100+ Bangalore locations** using Deep Learning. Features a visually rich Streamlit GUI with interactive Plotly charts, price range classification, and comprehensive market analytics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## Overview

This project predicts house prices in Bangalore based on:
- **Location** (100+ areas across Bangalore)
- **BHK** (1-5 bedrooms)
- **Total Square Feet** (area of the property)
- **Number of Bathrooms**

The model is trained on **~13,000 property records** and provides real-time price estimates with confidence indicators and market comparisons.

---

## Features

### 🔮 Price Prediction
- Enter property details → Get instant price estimate in Lakhs
- **Price range classification**: Budget / Mid-Range / Premium / Luxury
- Visual gauge meter showing where your property falls
- Comparison with location averages

### 📊 Market Analytics Dashboard
- **Price Distribution Histogram** — See how prices are spread across Bangalore
- **Price Range Donut Chart** — Budget vs Mid-Range vs Premium vs Luxury breakdown
- **BHK vs Price Analysis** — Average & median prices by bedroom count
- **Sqft vs Price Scatter Plot** — Relationship between area and price
- **Top Locations Bar Chart** — Most expensive areas with property counts
- **Price per Sqft Analysis** — Cost efficiency by location
- **BHK × Location Heatmap** — Median prices across locations and BHK types

### 📖 How It Works
- Complete ML pipeline explanation with code snippets
- Model architecture table with layer descriptions
- Feature engineering documentation
- Price classification logic

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | Bengaluru House Price Data (Kaggle) |
| **Records** | ~13,000 properties |
| **Locations** | 100+ Bangalore areas |
| **Features** | area_type, location, size, total_sqft, bath, balcony, price |
| **Target** | Price (in Lakhs) |

### Sample Locations
Koramangala, Indira Nagar, HSR Layout, Whitefield, Electronic City, Jayanagar, Marathahalli, Bellandur, Hebbal, Sarjapur Road, Bannerghatta Road, and 90+ more.

---

## Pipeline

```
┌─────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐   ┌────────────┐
│ Raw Dataset │ → │ Data Cleaning│ → │Feature Engineering│ → │ Model Train  │ → │ .h5 Model  │
│ (13K rows)  │   │ Nulls, Parse │   │ Encoding, Scale   │   │ Keras Dense  │   │ + Artifacts│
└─────────────┘   └──────────────┘   └───────────────────┘   └──────────────┘   └────────────┘
```

### Data Cleaning Steps
1. Drop irrelevant columns (`area_type`, `society`, `availability`)
2. Extract BHK from `size` column (`"3 BHK"` → `3`)
3. Parse sqft ranges (`"1200 - 1500"` → `1350`)
4. Remove outliers (min 300 sqft/BHK, reasonable bath counts, price caps)
5. Group rare locations into `"other"` category

### Feature Engineering
- **One-hot encoding** for 100+ locations
- **StandardScaler** normalization for numeric features
- **Price per sqft** calculation for analysis
- Final feature count: ~100+ dimensions

---

## Model Architecture

| Layer | Type | Purpose |
|-------|------|---------|
| 1 | Dense(128, relu) | First hidden layer — learns price patterns |
| 2 | BatchNormalization | Stabilizes and accelerates training |
| 3 | Dropout(0.3) | Prevents overfitting (30% dropout) |
| 4 | Dense(64, relu) | Refines learned feature representations |
| 5 | BatchNormalization | Normalizes activations |
| 6 | Dropout(0.2) | Light regularization (20%) |
| 7 | Dense(32, relu) | Final feature compression |
| 8 | Dense(1, linear) | Output: predicted price in Lakhs |

**Optimizer:** Adam | **Loss:** MSE | **Early Stopping:** patience=10

---

## Project Structure

```
bangalore_house_price_prediction/
├── train_model.py          # Data processing + model training script
├── app.py                  # Streamlit GUI with Plotly charts
├── README.md               # This file
│
├── (Generated after training)
├── model.h5                # Trained Keras model
├── columns.json            # Feature column names + location list
├── scaler.pkl              # StandardScaler for inference
├── location_stats.json     # Pre-computed stats for charts
├── model_config.json       # Training metrics (R², MAE)
└── Bengaluru_House_Data.csv # Dataset (auto-downloaded)
```

---

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn keras tensorflow streamlit plotly
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/bangalore_house_price_prediction.git
cd bangalore_house_price_prediction
```

---

## Usage

### Step 1: Train the Model
```bash
python train_model.py
```
This will:
- Download/generate the Bangalore house price dataset
- Clean and preprocess the data
- Train a Keras Dense Neural Network
- Save `model.h5`, `columns.json`, `scaler.pkl`, `location_stats.json`

### Step 2: Launch the GUI Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Step 3: Make Predictions
1. Select a **location** from the dropdown
2. Choose **BHK** (1-5 bedrooms)
3. Set **total sqft** using the slider
4. Pick number of **bathrooms**
5. Click **"Predict Price"**

---

## Screenshots

### Price Prediction Page
> Interactive price prediction with gauge meter, price classification badges, and location comparisons.

### Market Analytics Dashboard
> 7+ interactive Plotly charts: histograms, donut charts, scatter plots, bar charts, heatmaps.

### How It Works Page
> Complete pipeline explanation with code snippets and architecture details.

---

## Results

| Metric | Value |
|--------|-------|
| **R² Score** | ~0.85+ |
| **MAE** | ~10-15 Lakhs |
| **Training Samples** | ~10,000 |
| **Test Samples** | ~2,500 |
| **Features** | ~100+ |
| **Locations Covered** | 100+ |

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.12** | Core programming language |
| **Pandas & NumPy** | Data manipulation and processing |
| **Scikit-learn** | StandardScaler, train-test split, metrics |
| **TensorFlow / Keras** | Deep learning model (.h5 format) |
| **Streamlit** | Web-based GUI application |
| **Plotly** | Interactive charts and visualizations |

---

## Future Improvements

- 🏗️ **Real-time Data Scraping** — Fetch live prices from 99acres, MagicBricks
- 🗺️ **Map Visualization** — Interactive Bangalore map with price heatmap
- 📄 **PDF Report Generation** — Download property valuation reports
- 🤖 **BERT-based Location Embeddings** — Understand location descriptions semantically
- 📈 **Price Trend Forecasting** — Predict future price movements using time series
- 🏢 **Commercial Property Support** — Extend to offices, shops, warehouses
- 📱 **Mobile App** — Flutter or React Native frontend

---

## License

This project is open source and available under the [MIT License](LICENSE).
