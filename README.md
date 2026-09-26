# Ames House Price Prediction

A machine learning project that predicts the sale price of a house in **Ames, Iowa** from its
physical and quality characteristics. It contains a training notebook, the saved model
artifacts, and a **Streamlit web app** that turns the trained model into an interactive
price estimator.

The dataset is the well-known [Ames Housing](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
dataset: **2,930 sales** recorded between **2006 and 2010**, with **82 raw columns** per
property and a target `SalePrice` ranging from **$12,789 to $755,000** (mean $180,796,
median $160,000).

---

## Project structure

```
House Prices Predection/
│
├── app.py                          # Streamlit web app (the prediction UI)
├── model.ipynb                     # Training notebook: cleaning → encoding → training
│
├── dataset/
│   └── AmesHousing.csv             # Source data (2,930 rows × 82 columns)
│
├── model finished files/           # Trained artifacts consumed by app.py
│   ├── house_price_model.pkl       # Fitted LinearRegression model
│   ├── scaler.pkl                  # Fitted MinMaxScaler
│   └── feature_columns.pkl         # The 76 pre-encoding column names
│
├── README.md                       # Project documentation
└── .venv/                          # Local virtual environment
```

The three `.pkl` files are produced by the last cell of `model.ipynb` and are the only
bridge between training and serving. `app.py` never refits anything — it loads these
artifacts and only performs inference.

---

## Model performance

Measured on a held-out test split of 586 houses (20%):

| Metric | Value |
| --- | --- |
| **R² Score** | **0.8948** |
| **RMSE** | **$29,046** |
| Training rows | 2,344 |
| Test rows | 586 |

An R² of ~0.89 means the model explains roughly **89% of the variation** in sale price. The
typical prediction error is about **$29k** on a dataset where the average house costs
$181k — a reasonable error band for tabular real-estate data, where square footage, quality
ratings and neighborhood explain most but not all of the price.

---

## Methods used, and why

### 1. Column selection

```python
data = data.drop(columns=['Alley', 'Mas Vnr Type', 'Bsmt Full Bath', 'Bsmt Half Bath', 'SalePrice', 'Order'])
```

`SalePrice` is the prediction target and `Order` is just a row counter — neither belongs in
the feature set. `Alley` and `Mas Vnr Type` are dropped because they are **over 60% empty**
(93% and 61% respectively), so imputing them would invent far more than it recovers. The
remaining two, `Bsmt Full Bath` and `Bsmt Half Bath`, are essentially complete (0.1%
missing) — keeping them would have been reasonable, and dropping them was a choice rather
than a necessity. The net effect is **76 raw columns**.

**Why it matters:** the dropped columns are also why `app.py` must apply the *exact same*
drop list. Any drift between the two would silently shift every feature by one position.

### 2. One-hot encoding — `pd.get_dummies(drop_first=True)`

Ames is a **categorical-heavy** dataset: 41 of the 76 remaining columns are text labels
(`Neighborhood`, `Exter Qual`, `Bldg Type`, …). Linear regression cannot read strings, so
each one is expanded into indicator columns.

`drop_first=True` removes one dummy per category — the **reference level**. This prevents
the classic *dummy variable trap*: with all levels present plus an intercept, the columns
would be perfectly collinear and the model coefficients would be unstable and
uninterpretable. It also keeps the design matrix compact.

**Why it matters:** this expands **76 → 255 features**: 35 true numeric columns plus **220
one-hot columns**. The reference level is the *first alphabetically* (e.g. `Blmngtn` is
dropped for `Neighborhood`, `1Fam` for `Bldg Type`, so `NAmes` and `Twnhs` do have
columns). In the app, choosing a reference level therefore means "leave all of that
feature's dummies at 0" — which is exactly what the code does.

### 3. Missing-value imputation — `KNNImputer(n_neighbors=5)`

27 columns still contain gaps after the drops, including `Lot Frontage` (490 missing) and
`Fireplace Qu` (1,422). Some are **structural**: `Pool QC` is empty for 2,917 of the 2,930 houses
because only 13 have a pool, so the missing value is itself the information.

`KNNImputer` fills each gap from the **5 most similar rows** rather than a flat column
average, so a missing garage quality borrows from houses that also have garages.

**Why it matters:** imputation happens **before** the train/test split, which is a mild
form of **data leakage** — the imputer has technically seen the test rows. It is a
conscious simplification for a learning project rather than a production-grade choice;
in a real pipeline the imputer belongs inside a `Pipeline` fit only on the training fold.

### 4. Feature scaling — `MinMaxScaler`

Scales every feature into `[0, 1]` by subtracting the column minimum and dividing by its
range. Tree-based models would ignore this step, but `LinearRegression` uses gradient-based
optimization, which converges far faster when all features share one scale. Otherwise
`PID` and `Lot Area` (measured in the hundreds of thousands) would dominate the
neighborhood labels (0 or 1) purely because of their magnitude.

**Why it matters:** the scaler is fit on the **training split only** and then applied to
both splits — the correct order. Any prediction must pass through this same saved scaler,
since the model's coefficients were learned against the scaled values.

### 5. Train/test split — `test_size=0.2, random_state=42`

An 80/20 split with a **fixed seed** so the reported metrics are reproducible. Without a
fixed `random_state`, every run would produce a slightly different R² and the evaluation
would not be trustworthy.

### 6. Linear regression — `LinearRegression`

```python
model = LinearRegression()
```

The final estimator. Each of the 255 coefficients is the average change in sale price,
in dollars, associated with moving one unit along that scaled feature — which is why
`Gr Liv Area` comes out strongly positive and `Overall Qual` next.

**Why linear regression:** it is a strong baseline for this kind of tabular data because
the Ames price function is close to additive — each feature contributes roughly a fixed
amount, and the dataset has no meaningful nonlinear interactions. It also trains in
seconds and its coefficients stay directly interpretable, which matters when a
non-technical person has to trust an estimated price.

**Honest limitations of this model:**

- **Effect sizes from small categories are unreliable.** `Roof Matl` dummies carry
  coefficients around $600k, but that is collinearity noise from a handful of houses, not
  a real signal. Rare categories (`Exter Qual_Ex`, 107 homes) behave the same way. The
  app avoids exposing these prominently for this reason.
- **`PID` is still in the feature set.** It is a record ID with no causal meaning, kept
  only because removing it would change the saved model's input width. It is a genuine
  modeling flaw worth fixing in a retrain.
- **Linear regression cannot express interactions.** A gradient-boosted tree model
  (`HistGradientBoostingRegressor` or `RandomForest`) would likely score better by learning
  thresholds like "quality matters more above 2,000 sq ft."

### 7. Persistence — `joblib.dump`

`joblib` serializes the fitted model and scaler to `.pkl` files. Inference code must use a
*fitted* object, and refitting at app startup would be slow and would risk the app
predicting with a slightly different model than the one that was evaluated. `joblib` is the
right tool here because it preserves the full Python object — including the learned
`coef_` array and the scaler's `min_`/`max_` — which plain JSON or CSV cannot.

### 8. Web serving — `Streamlit` with `@st.cache_resource`

Streamlit turns a Python script directly into a web app: the script reruns top to bottom
on every widget interaction, and each `st.number_input` / `st.selectbox` becomes a browser
widget. No Flask routes, templates, or JavaScript are needed, which is why it is a good fit
for a single-page estimator.

`@st.cache_resource` loads the model, scaler, and CSV **once per server process** and reuses
them on every rerun. Without it, every keystroke and every button press would reload the
artifacts and re-run `get_dummies` on 2,930 rows — slow and wasteful.

The `app.py` logic mirrors the notebook exactly:

1. Rebuild the 255-column layout from the raw CSV using the same drop list and
   `get_dummies(drop_first=True)`.
2. **Assert** the column count matches `model.n_features_in_` and `scaler.n_features_in_`,
   raising a clear error otherwise instead of letting scikit-learn fail obscurely.
3. Fill untouched features with **training-set medians**, not zeros. After `MinMaxScaler`,
   a `0` means *"the smallest value in the training data"* — a 0 sq ft house with no
   basement, which would produce badly under-priced estimates.
4. Reindex to the exact training column order before calling `scaler.transform`, since
   scikit-learn validates only the **number** of columns, never their names or order. A
   single out-of-order column would corrupt every prediction silently.
5. Pass a plain NumPy array to the scaler, matching how it was fitted, which avoids a
   spurious "feature names" warning.

### 9. Evaluation — R², RMSE, and residual plots

`r2_score` measures how much variance the model explains; `mean_squared_error` (square-rooted
into RMSE) expresses the error in the same units as the target, so it can be read directly
as dollars. The notebook also plots **actual vs. predicted** and a **residual plot**. The
residual plot is the important one: a random scatter around zero indicates the linear
assumption is reasonable, while visible curves or fans would mean the model is
systematically wrong and needs more features or a nonlinear learner.

---

## Tech stack

| Tool | Version | Role |
| --- | --- | --- |
| Python | 3.14.7 | Language |
| pandas | 3.0.6 | Data loading and manipulation |
| NumPy | 2.5.3 | Array operations |
| scikit-learn | 1.9.1 | Imputation, scaling, splitting, `LinearRegression` |
| Streamlit | 1.64.0 | Web app framework |
| joblib | 1.6.0 | Model serialization |
| matplotlib / seaborn | — | Evaluation plots in the notebook |

---

## How to run the web app

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate

# 2. Make sure the dependencies are installed
pip install pandas numpy scikit-learn streamlit joblib

# 3. Launch the app
streamlit run app.py
```

Streamlit serves the app at `http://localhost:8501`.

To retrain the model from scratch, open `model.ipynb` in Jupyter and run all cells
top to bottom — the final cell regenerates the three `.pkl` files.

---

## Using the app

1. Fill in the house's characteristics under **Categorical features** (neighborhood,
   building type, quality ratings) and **Numeric features** (living area, lot size, garage,
   basement, dates).
2. Fields you do not touch are filled with training-set medians, so an estimate always
   describes a complete, realistic house rather than an empty one.
3. Click **Predict Price** to see the estimated sale price.
4. Sensitivity behaves as expected: a larger, newer, higher-quality house is valued higher
   than a small, older one, and premium neighborhoods such as `NoRidge` price above
   `IDOTRR`.

## Possible future improvements

- Move the imputer and scaler into a scikit-learn `Pipeline` to eliminate the current data
  leakage and guarantee train/serve consistency.
- Drop `PID` and other meaningless columns, then retrain and re-save the artifacts.
- Swap in a tree-based ensemble (`HistGradientBoostingRegressor`, `GradientBoostingRegressor`)
  or compare against them; the residual plot suggests nonlinearity is being left on the table.
- Wrap rare high-variance categories (`Roof Matl`, `Misc Feature`) or use regularization such
  as `Ridge`/`Lasso` to stabilize their coefficients.
- Add cross-validation for a more reliable estimate than a single 80/20 split provides.
- Deploy the Streamlit app to Streamlit Community Cloud so the estimate is publicly reachable.
