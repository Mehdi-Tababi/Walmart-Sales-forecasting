# Retail Weekly Sales Forecasting using Temporal Fusion Transformer (TFT)

## 📌 Project Overview
This project leverages deep learning to forecast weekly retail sales across multiple store locations. By utilizing the **Temporal Fusion Transformer (TFT)** architecture, the model effectively combines static store metadata, known future inputs (like scheduled holidays), and historical macroeconomic indicators to generate highly accurate, multi-horizon quantile forecasts.

### 🏆 Key Results
The model achieved exceptional accuracy on the validation holdout set:
* **MAPE (Mean Absolute Percentage Error):** 5.81%
* **MAE (Mean Absolute Error):** $59,138.59
* **RMSE (Root Mean Squared Error):** $82,524.46

---

## 📊 Dataset & Features
The dataset consists of weekly sales records grouped by individual stores. 

**Target Variable:** * `Weekly_Sales` (Continuous, evaluated with Log Transformation)

**Features Used:**
* **Static Categoricals:** `Store` (Unique store identifiers)
* **Time-Varying Known Reals:** `time_idx`, `Date`, `Holiday_Flag`, `CPI`, `Unemployment`, `Fuel_Price`, `Temperature`

### 💡 Key EDA Insights
Extensive Exploratory Data Analysis (EDA) revealed:
* **Macroeconomic Resilience:** Factors like `Fuel_Price` and `Unemployment` have near-zero linear correlation with day-to-day weekly sales, proving the core retail business is highly resilient to standard economic fluctuations.
* **Regional CPI Clusters:** The Consumer Price Index (CPI) data is split into two distinct, massive clusters (Low Cost: ~125-145 vs. High Cost: ~180-225), indicating stores operate in two completely different economic regions.

---

## 🧠 Model Architecture & Training

### Tech Stack
* **Frameworks:** PyTorch, PyTorch Lightning, PyTorch Forecasting
* **Data Processing:** Pandas, NumPy

### Model Configuration
* **Architecture:** Temporal Fusion Transformer (`TemporalFusionTransformer`)
* **Lookback Window (Encoder Length):** 52 weeks (1 full year)
* **Forecast Horizon (Prediction Length):** 5 weeks
* **Loss Function:** `QuantileLoss` (Predicting 10th, 50th, and 90th percentiles for confidence intervals)
* **Hyperparameters:** `learning_rate=0.01`, `hidden_size=32`, `attention_head_size=2`, `dropout=0.3`

### Training Pipeline
The model is trained using a PyTorch Lightning `Trainer` with the following safeguards:
* **Max Epochs:** 36
* **Gradient Clipping:** 0.5 (to prevent exploding gradients)
* **Callbacks:** * `EarlyStopping` (monitoring `val_loss` with a patience of 8 epochs)
  * `ModelCheckpoint` (saving the top 1 model based on minimal validation loss)
* **Logging:** Tracked via `CSVLogger`

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch pytorch-lightning pytorch-forecasting pandas numpy matplotlib
