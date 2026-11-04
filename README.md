# House Price Prediction: Regularization, Cross-Validation and Model Comparison

A regression project that predicts house prices (in INR) and compares linear, tree-based and kernel models.

## Dataset

`Advanced_Regression_HousePrice_Dataset_3800.csv` has 3800 rows and 12 columns.

| Column | Description |
|---|---|
| property_id | Unique ID (dropped before modelling) |
| sale_date | Date of sale (dropped, used only for Time Series Split) |
| area_sqft | Built-up area |
| bedrooms, bathrooms | Room counts |
| location_score | Locality quality score (1 to 10) |
| property_age | Age of the property in years |
| distance_city_km | Distance from the city centre |
| near_school, near_metro | 1 if nearby, else 0 |
| crime_rate_index | Crime rate of the area |
| house_price_inr | **Target**: sale price in INR |

## Project Video
[Watch the project explanation here](https://drive.google.com/file/d/1Cov2HsxS3j9K6LgdwlQT3KTjM2mfRxHL/view?usp=sharing)

## Project Structure (Notebook: `Project_2.ipynb`)

- **Part A:** Theory (regularization, Ridge vs Lasso, cross-validation, why trees ignore scaling)
- **Part B:** Data loading, preprocessing (standardization), 80/20 train-test split (`random_state=42`)
- **Part C:** Ridge and Lasso, alpha tuning with 5-fold CV (`RidgeCV`, `LassoCV`), coefficient comparison
- **Part D:** K-Fold, Stratified K-Fold (binned target), Leave-One-Out and Time Series Split
- **Part E:** Decision Tree vs Random Forest
- **Part F:** Support Vector Regression (linear and RBF kernels)
- **Part G:** Evaluation of all models, over/underfitting analysis, final report

## How to Run

1. Open `Project_2.ipynb` in Google Colab or Jupyter.
2. Upload the CSV and update the file path in the first read cell.
3. Run all cells from top to bottom (Runtime, then Run all).

Requirements: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Important Notes

- Ridge, Lasso and SVR were trained on **standardized** prices, while the tree models used raw rupee prices. Predictions of the scaled models are converted back with `pred * std + mean` before computing MSE, MAE and RMSE, so all models are compared in the same unit.
- R² is unit-free and is the same either way.
- Time Series Split requires the data to be sorted by `sale_date` first.
- Leave-One-Out fits the model 3800 times and is much slower than the other methods.
