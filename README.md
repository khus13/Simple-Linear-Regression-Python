# Simple Linear Regression: Canada Per Capita Income

Predicts Canada's per capita income (US$) from the year using simple linear regression.

## Project Files

| File | Description |
|---|---|
| `Simple Linear Regression.ipynb` | Notebook with the full workflow |
| `main.py` | Script version (load data, print head) |
| `canada_per_capita_income.csv` | Dataset (`year`, `per capita income (US$)`) |

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Workflow

1. Import libraries
2. Load the CSV into a DataFrame
3. Plot the data (scatter)
4. Create and train the model (`fit`)
5. Predict (`predict`)
6. Inspect the slope and intercept
7. Plot the regression line over the data

## Core Formula

**Line equation**

```
y = m * x + b
```

| Symbol | Meaning | In this project |
|---|---|---|
| `y` | Predicted value | per capita income (US$) |
| `x` | Input feature | year |
| `m` | Slope (coefficient) | `reg.coef_` |
| `b` | Intercept | `reg.intercept_` |

**How `m` and `b` are found (Ordinary Least Squares)**

```
m = Σ (xᵢ - x̄)(yᵢ - ȳ) / Σ (xᵢ - x̄)²
b = ȳ - m * x̄
```

OLS picks the line that minimizes the sum of squared residuals:

```
Residual  = yᵢ - ŷᵢ
SSE       = Σ (yᵢ - ŷᵢ)²
```

**Example from this project**

```
m ≈ 828.465
b ≈ -1,632,210.758
income(2016) = 828.465 * 2016 + (-1,632,210.758) ≈ 37,974.83
```

---

## Library Reference

### 1. pandas

Used for loading and handling tabular data.

```python
import pandas as pd
```

| Syntax | Definition |
|---|---|
| `import pandas as pd` | Imports pandas and gives it the alias `pd`. |
| `pd.read_csv("file.csv")` | Reads a CSV file into a DataFrame. Use a raw string `r"C:\path\file.csv"` for Windows full paths so backslashes are not treated as escapes. |
| `df` | A DataFrame: a 2D table with labeled rows and columns. |
| `df.head()` | Returns the first 5 rows. Useful to confirm data loaded. |
| `df.columns` | Lists the column names. Check this to get exact names. |
| `df['col']` | Selects one column as a Series (1D). Used for the target `y`. |
| `df[['col']]` | Selects one column as a DataFrame (2D). Used for the feature `X`. |
| `pd.DataFrame({'year': [2016]})` | Builds a DataFrame from a dictionary. Used to give `predict` a 2D input with the same column name used in `fit`. |

### 2. numpy

Imported for numerical work. It is not directly called yet in this project, but scikit-learn and pandas use it internally.

```python
import numpy as np
```

| Syntax | Definition |
|---|---|
| `import numpy as np` | Imports NumPy with the alias `np`. |
| `np.array([[2016]])` | Creates an array. A nested list like `[[2016]]` is the 2D shape `predict` expects. |

### 3. matplotlib

Used for plotting.

```python
import matplotlib.pyplot as plt
```

| Syntax | Definition |
|---|---|
| `import matplotlib.pyplot as plt` | Imports the plotting module as `plt`. (Note: it is `pyplot`, not `pylot`.) |
| `%matplotlib inline` | Jupyter-only magic command that shows plots inside the notebook. Causes a SyntaxError in a `.py` file. |
| `plt.scatter(x, y, color='red', marker='+')` | Draws individual data points. `color` sets the color, `marker` sets the point shape. |
| `plt.plot(x, y, color='blue')` | Draws a connected line. Used for the regression line: `plt.plot(df['year'], reg.predict(df[['year']]))`. |
| `plt.xlabel('text', fontsize=20)` | Sets the x-axis label and font size. |
| `plt.ylabel('text', fontsize=20)` | Sets the y-axis label and font size. |
| `plt.show()` | Displays the figure. Required in scripts. |

### 4. scikit-learn

Used for building the machine learning model.

```python
from sklearn import linear_model
```

| Syntax | Definition |
|---|---|
| `from sklearn import linear_model` | Imports the module containing linear models. |
| `reg = linear_model.LinearRegression()` | Creates an untrained linear regression model object. |
| `reg.fit(X, y)` | Trains the model on the data. `X` must be 2D (`df[['year']]`), `y` must be 1D (`df['per capita income (US$)']`). Fitting finds the best `m` and `b`. |
| `reg.predict(X_new)` | Returns predicted `y` values for new inputs. `X_new` must be 2D, e.g. `pd.DataFrame({'year': [2016]})`. Returns an array. |
| `reg.coef_` | The learned slope `m`. Returned as an array (one value per feature). |
| `reg.intercept_` | The learned intercept `b`. |

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `NameError: name 'pd' is not defined` | Import cell was not run | Run the import cell first |
| `ModuleNotFoundError: No module named 'pandas'` | Package not installed in that Python | `pip install pandas` (use `%pip install` inside Jupyter) |
| `No module named 'matplotlib.pylot'` | Typo | Use `matplotlib.pyplot` |
| `FileNotFoundError` | CSV not in working directory | Use the full path or move the file next to the notebook |
| `KeyError: 'area'` | Wrong column name | Check `df.columns` and match exactly, including case, spaces, and symbols |
| `ValueError: Expected 2D array, got scalar` | Passing a bare number to `predict` | Pass `pd.DataFrame({'year': [2016]})` or `[[2016]]` |
| `SyntaxError` on `%matplotlib inline` | Jupyter magic used in a `.py` file | Remove it and use `plt.show()` |

## Next Steps

- Evaluate the model (`reg.score`, MSE, RMSE)
- Train/test split
- Predict multiple future years and export to CSV
- Save the trained model with `joblib`
