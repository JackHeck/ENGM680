# ENGM680
Final LSTM model for ENGM680 G-6
Overview
This repository contains a complete data-processing and LSTM-based regression pipeline designed to predict % Silica Concentrate from the Mining Process Flotation Plant dataset.
The project includes:
Automated CSV loading with multi-encoding fallback
Timestamp cleaning and unique ordering for time-series consistency
Automatic creation of datasets with and without %Iron Concentrate
*Final model trained with %Iron Concentrate
Full LSTM pipeline (dataset creation, scaling, training, validation, early stopping)
Visualization of prediction results and training curves
Metrics computation (MAE, MSE, RMSE, R²)
This repo is structured for clarity and reproducibility, allowing you to run preprocessing, training, and evaluation in a single script.
