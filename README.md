# NYC Airbnb Price Prediction

![Airbnb cover](./README-images/airbnb-cover.png)

This project is a quick walkthrough of how I built a price prediction model for Airbnb listings in New York City using the publicly available NYC Airbnb dataset.

I kept the notebook structured as steps, so it’s easier to see what happened at each stage: data loading, cleaning, feature engineering, encoding, and then model training + evaluation.

## Project files

- `Airbnb.ipynb` — step-by-step notebook (EDA, preprocessing, model, evaluation)
- `data/AB_NYC_2019.csv` — dataset used by the notebook

## How to run

1. Open `Airbnb.ipynb` in Cursor (or any Jupyter environment).
2. Run the cells from top to bottom.

The notebook expects the dataset at `data/AB_NYC_2019.csv` (relative to the project folder).

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Step by step (what I did)

### 1) Load the dataset

The notebook loads `AB_NYC_2019.csv` from `data/`.
After loading, it prints the dataset shape and checks missing values and duplicates.

### 2) Clean the data

I removed duplicate rows and inspected missing values.
Then I dropped columns with too many missing values (based on the notebook’s threshold).

After that, I filtered out rows that look like outliers for the price task using these simplified rules:

I removed listings where `price` is 0 (placeholder / bad data) or where `price > 1000`.
I also removed listings where `minimum_nights > 30` since those behave more like long-term rentals than typical short stays.

### 3) Feature engineering

I created two additional features:

`min_booking_cost` = `price × minimum_nights`
`host_type` = `"commercial"` if `calculated_host_listings_count > 5`, otherwise `"individual"`

### 4) Encode categorical variables

For modeling, I treated `room_type` as an ordinal feature using a fixed order:
Shared room < Private room < Entire home/apt

For the remaining categorical variables, I used one-hot encoding later inside the pipeline (so the model can learn without forcing an order).

### 5) Build the preprocessing + model pipeline

The pipeline handles:

Missing numeric values (median imputation) and numeric scaling
Missing categorical values (most-frequent imputation) and one-hot encoding for nominal categories

For the regression model, I used **Ridge Regression**.

### 6) Evaluate with cross-validation

To make the evaluation more reliable, I used K-fold cross-validation (k = 10) on the training setup.

## Results (from the notebook)

Cross-validation summary (k = 10):

MAE (mean): **48.58**
RMSE (mean): **79.71**
R2 (mean): **0.5356**

I also printed per-fold values for MAE, RMSE, and R2 so you can see how stable the model is across splits.


## Future Work

To push the accuracy further, here are some improvements I can try next:

1. Tune the Ridge model more carefully
   Try different `alpha` values (and compare with plain Linear Regression or Elastic Net) using the same cross-validation setup.

2. Improve preprocessing choices
   Re-check how missing values are handled and whether dropping columns with many missing values is removing useful signal.

3. Add better features
   Add features that capture location patterns more directly (for example: distances to landmarks, latitude/longitude binning, or interaction features like `neighbourhood_group × room_type`).
   You can also create more realistic “pricing context” features than just `min_booking_cost`.

4. Try a stronger model
   Consider tree-based models (e.g., Random Forest, Gradient Boosting, XGBoost/LightGBM) that often handle non-linear effects and categorical signals better than linear models.

5. Use a more robust evaluation
   If your goal is generalization, try repeated K-fold cross-validation or different random seeds to see if the results stay stable.
