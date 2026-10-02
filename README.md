# Simple Linear Regression: Canada Per Capita Income

## What we did

**Setup and troubleshooting**
- Fixed `NameError: name 'pd' is not defined`: the import cell was never run.
- Fixed `ModuleNotFoundError: No module named 'pandas'`: installed packages with `pip install pandas numpy matplotlib scikit-learn`.
- Fixed the typo `matplotlib.pylot` -> `matplotlib.pyplot`.
- Fixed `FileNotFoundError` by using the full path to the CSV with a raw string (`r"..."`).
- Learned `%matplotlib inline` only works in Jupyter. In a `.py` file, use `plt.show()` instead.

**Loading and plotting**
- Loaded `canada_per_capita_income.csv` into a DataFrame (`df`).
- Made a scatter plot of `year` vs `per capita income (US$)` with axis labels.
- Fixed `KeyError` by replacing the `area` / `price` columns (from a different tutorial) with `year` and `per capita income (US$)`.

**Model**
- Created the model: `reg = linear_model.LinearRegression()`.
- Trained it: `reg.fit(df[['year']], df['per capita income (US$)'])`.
  - X needs double brackets (2D), y uses single brackets (1D).
- Predicted income for 2016 with `reg.predict(pd.DataFrame({'year': [2016]}))`.
  - Passing a bare `2016` fails because sklearn expects 2D input.
- Inspected the learned parameters:
  - `reg.coef_` is the slope `m` (about 828.47).
  - `reg.intercept_` is `b` (about -1,632,210.76).
- Started verifying the prediction by hand with `y = m*x + b`.

## Issue to fix in the last cell

```python
828.46507522 * 37974.83379353 + -1632210.7578554575
```

`37974.83` is the **predicted income** for 2016, not the year. In `y = m*x + b`, `x` should be the year:

```python
828.46507522 * 2016 + -1632210.7578554575   # ≈ 37974.83, matches reg.predict
```

Better still, use the variables instead of pasted numbers:

```python
reg.coef_[0] * 2016 + reg.intercept_
```

## What we haven't done yet

- [ ] Explore the data first: `df.head()`, `df.info()`, `df.describe()`, and check for missing values.
- [ ] Plot the regression line over the scatter plot: `plt.plot(df['year'], reg.predict(df[['year']]))`.
- [ ] Evaluate the model: R² (`reg.score(...)`), MSE / RMSE / MAE.
- [ ] Split data into train and test sets (`train_test_split`) to check how well it generalizes.
- [ ] Predict multiple or future years in one call, e.g. 2020, 2025, 2030.
- [ ] Add predictions to a new CSV with a `per capita income (US$)` column and export it.
- [ ] Check residuals to see whether a straight line is a good fit for this data.
- [ ] Save the trained model (`joblib` or `pickle`) so it can be reused without retraining.
- [ ] Clean up the notebook: remove the stray `%matplotlib inline` if moving to a `.py` script, and fix the last cell.
- [ ] Write a short conclusion on what the slope means (average income increase per year).
