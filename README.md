# Walmart Weekly Sales Forecasting (XGBoost)

This project forecasts Walmart weekly sales using an XGBoost time-series pipeline with leakage-safe feature engineering and log-transformed target modeling.

## Project Structure

- `walmart_forecasting_weekly_sales.ipynb`: End-to-end workflow (EDA, modeling, tuning, inference)
- `Walmart_Sales.csv`: Main dataset
- `walmart_sales_data_profiling_report.html`: Automated data profiling report

## Notebook Workflow

1. Data loading and quality checks
2. Exploratory data analysis (trend, distribution, correlations, holiday effect)
3. XGBoost feature engineering:
	 - Store encoding
	 - Lag features (1, 2, 4, 8)
	 - Rolling statistics (4, 8, 12 windows)
4. Leakage-safe setup:
	 - Rolling features are based on past values only (`shift(1)`)
	 - Time-based train/validation split
5. XGBoost default model (log target)
6. Hyperparameter tuning with `RandomizedSearchCV` + `TimeSeriesSplit`
7. Recursive multi-step inference for future weeks per store

## Why Log Transform Is Used for XGBoost

`Weekly_Sales` is right-skewed. Training XGBoost on `log1p(Weekly_Sales)` helps stabilize variance and often improves error metrics.

Prediction flow:

1. Train on `log1p(y)`
2. Predict in log space
3. Convert back to original scale with `expm1`
4. Clip to non-negative values for business-valid forecasts

## XGBoost Validation Results (Current Pipeline)

Using log-target training and inverse-transform evaluation:

- Default XGBoost (log target):
	- MAE: 30,874.64
	- RMSE: 42,309.77
- Tuned XGBoost (log target):
	- MAE: 30,673.78
	- RMSE: 41,406.77

Tuned XGBoost (log target) is currently the best XGBoost variant in this notebook.

## Inference Output

The inference stage performs recursive forecasting for all stores.

- Forecast horizon: 8 weeks
- Stores: 45
- Total predictions: 360

Output columns:

- `Date`
- `Store`
- `Predicted_Weekly_Sales`
- `Model_Used`

## Environment and Dependencies

Core Python packages used:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- statsmodels
- ydata-profiling

## How To Run

1. Open `walmart_forecasting_weekly_sales.ipynb`.
2. Install required packages in the notebook environment.
3. Run cells from top to bottom.
4. Check model metrics in the XGBoost section.
5. Run the final inference cell to generate `forecast_df`.

## Notes

- Keep cell execution order intact because later cells depend on variables created earlier.
- If you change feature engineering or model settings, rerun training, tuning, and inference cells.
- Future exogenous variables (CPI, unemployment, fuel price, temperature, holiday flag) are currently carried forward from latest known values for inference scenarios.
