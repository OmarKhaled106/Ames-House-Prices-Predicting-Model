import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model finished files"
DATA_PATH = BASE_DIR / "dataset" / "AmesHousing.csv"

# Columns removed in the notebook before encoding. The model was trained on whatever is
# left, so the app has to reproduce this drop list exactly.
DROPPED_COLUMNS = [
    "Alley",
    "Mas Vnr Type",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "SalePrice",
    "Order",
]

# column -> (label, min, max, step)
NUMERIC_INPUTS = {
    "Gr Liv Area": ("Above ground living area (sq ft)", 300, 6000, 10),
    "TotRms AbvGrd": ("Total rooms above grade", 2, 20, 1),
    "Bedroom AbvGr": ("Bedrooms above grade", 0, 10, 1),
    "Kitchen AbvGr": ("Kitchens above grade", 0, 5, 1),
    "Full Bath": ("Full bathrooms", 0, 6, 1),
    "Half Bath": ("Half bathrooms", 0, 4, 1),
    "1st Flr SF": ("1st floor area (sq ft)", 0, 4000, 10),
    "2nd Flr SF": ("2nd floor area (sq ft)", 0, 4000, 10),
    "Low Qual Fin SF": ("Low quality finished area (sq ft)", 0, 1200, 10),
    "Overall Qual": ("Overall quality", 1, 10, 1),
    "Overall Cond": ("Overall condition", 1, 10, 1),
    "MS SubClass": ("Building class code", 20, 190, 10),
    "Year Built": ("Year built", 1800, 2026, 1),
    "Year Remod/Add": ("Year remodelled / added", 1800, 2026, 1),
    "Fireplaces": ("Fireplaces", 0, 5, 1),
    "Total Bsmt SF": ("Total basement area (sq ft)", 0, 6000, 10),
    "Bsmt Unf SF": ("Unfinished basement area (sq ft)", 0, 2500, 10),
    "BsmtFin SF 1": ("Finished basement area type 1 (sq ft)", 0, 3000, 10),
    "BsmtFin SF 2": ("Finished basement area type 2 (sq ft)", 0, 2000, 10),
    "Garage Cars": ("Garage capacity (cars)", 0, 5, 1),
    "Garage Area": ("Garage area (sq ft)", 0, 2000, 10),
    "Garage Yr Blt": ("Garage year built", 1900, 2026, 1),
    "Lot Area": ("Lot area (sq ft)", 500, 50000, 50),
    "Lot Frontage": ("Lot frontage (ft)", 0, 350, 1),
    "Mas Vnr Area": ("Masonry veneer area (sq ft)", 0, 1500, 1),
    "Wood Deck SF": ("Wood deck area (sq ft)", 0, 1000, 1),
    "Open Porch SF": ("Open porch area (sq ft)", 0, 500, 1),
    "Enclosed Porch": ("Enclosed porch area (sq ft)", 0, 500, 1),
    "Screen Porch": ("Screen porch area (sq ft)", 0, 500, 1),
    "3Ssn Porch": ("Three season porch area (sq ft)", 0, 500, 1),
    "Pool Area": ("Pool area (sq ft)", 0, 800, 1),
}

NUMERIC_GROUPS = {
    "Size & rooms": [
        "Gr Liv Area",
        "TotRms AbvGrd",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "Full Bath",
        "Half Bath",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
    ],
    "Age & quality": [
        "Overall Qual",
        "Overall Cond",
        "MS SubClass",
        "Year Built",
        "Year Remod/Add",
        "Fireplaces",
    ],
    "Basement": ["Total Bsmt SF", "Bsmt Unf SF", "BsmtFin SF 1", "BsmtFin SF 2"],
    "Garage": ["Garage Cars", "Garage Area", "Garage Yr Blt"],
    "Lot & outdoor": [
        "Lot Area",
        "Lot Frontage",
        "Mas Vnr Area",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "Screen Porch",
        "3Ssn Porch",
        "Pool Area",
    ],
}

# column -> label
CATEGORICAL_INPUTS = {
    "Neighborhood": "Neighborhood",
    "Bldg Type": "Building type",
    "House Style": "House style",
    "Lot Config": "Lot configuration",
    "Lot Shape": "Lot shape",
    "Land Contour": "Land contour",
    "Land Slope": "Land slope",
    "Condition 1": "Primary condition",
    "Exter Qual": "Exterior quality",
    "Exter Cond": "Exterior condition",
    "Kitchen Qual": "Kitchen quality",
    "Foundation": "Foundation",
    "Bsmt Qual": "Basement quality",
    "Bsmt Cond": "Basement condition",
    "Bsmt Exposure": "Basement exposure",
    "BsmtFin Type 1": "Basement finish type 1",
    "Heating": "Heating system",
    "Heating QC": "Heating quality",
    "Central Air": "Central air",
    "Electrical": "Electrical system",
    "Functional": "Functionality",
    "Fireplace Qu": "Fireplace quality",
    "Garage Type": "Garage type",
    "Garage Finish": "Garage finish",
    "Garage Qual": "Garage quality",
    "Paved Drive": "Paved driveway",
    "Roof Style": "Roof style",
    "Pool QC": "Pool quality",
    "Fence": "Fence",
    "Misc Feature": "Miscellaneous feature",
    "Sale Condition": "Sale condition",
}


@st.cache_resource
def load_artifacts():
    """Load the model/scaler and rebuild the exact feature layout they were trained on."""
    model = joblib.load(ARTIFACT_DIR / "house_price_model.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")

    raw = pd.read_csv(DATA_PATH)
    features = raw.drop(columns=DROPPED_COLUMNS)

    # Same encoding as the notebook: get_dummies(drop_first=True) -> 255 columns.
    # get_dummies replaces every object column with its dummies, so column kinds are read
    # from `features`, not from the encoded frame.
    numeric_kinds = ("int64", "float64")
    column_kinds = features.dtypes.astype(str).to_dict()
    encoded = pd.get_dummies(features, drop_first=True).astype(float)
    feature_columns = encoded.columns.tolist()

    expected = int(getattr(model, "n_features_in_", len(feature_columns)))
    if len(feature_columns) != expected or int(scaler.n_features_in_) != expected:
        raise RuntimeError(
            f"Feature mismatch: this app builds {len(feature_columns)} columns but the "
            f"model/scaler expect {expected}. Re-run the notebook so the saved model matches "
            "the app's encoding."
        )

    # Filling unset features with 0 would be wrong: after MinMaxScaler 0 means "smallest
    # value in the training set", i.e. an impossible house. Use the training median instead.
    typical = encoded.median()

    defaults = {}
    levels_map = {}
    dummy_map = {}
    for column in features.columns:
        kind = column_kinds[column]
        if kind in numeric_kinds:
            typical_value = typical[column]
            defaults[column] = (
                int(round(typical_value)) if kind == "int64" else float(typical_value)
            )
        else:
            levels = sorted({str(value) for value in features[column].dropna().unique()})
            modes = features[column].dropna().mode()
            defaults[column] = str(modes.iat[0]) if not modes.empty else levels[0]
            levels_map[column] = levels
            # Columns holding this feature's one-hot dummies. The first level of every
            # categorical is the drop_first reference level and has no dummy column.
            dummy_map[column] = [c for c in feature_columns if c.startswith(f"{column}_")]

    return model, scaler, feature_columns, typical, defaults, dummy_map, levels_map


def build_input_frame(feature_columns, typical, dummy_map, numeric_values, category_values):
    """Turn the widget values into a single row in the model's exact feature order."""
    row = typical.copy()

    for column, value in numeric_values.items():
        if column in row.index:
            row[column] = float(value)

    for column, value in category_values.items():
        for dummy in dummy_map.get(column, []):
            row[dummy] = 0.0
        selected = f"{column}_{value}"
        if selected in row.index:
            row[selected] = 1.0

    return row.reindex(feature_columns).to_frame().T.astype(float)


model, scaler, feature_columns, typical, defaults, dummy_map, levels_map = load_artifacts()

st.title("Ames House Price Predictor")
st.caption(
    f"Linear regression trained on {len(feature_columns)} encoded features. "
    "Fields you leave untouched are set to the training-set median, so the estimate "
    "always describes a complete, typical house."
)

st.subheader("Categorical features")
category_values = {}
category_items = list(CATEGORICAL_INPUTS.items())
for start in range(0, len(category_items), 3):
    columns = st.columns(3)
    for column, (name, label) in zip(columns, category_items[start : start + 3]):
        if name not in dummy_map:
            continue
        # Every observed level is offered. The drop_first reference level has no dummy
        # column, so selecting it simply leaves all of this feature's dummies at 0.
        options = [defaults[name]] + [
            level for level in levels_map[name] if level != defaults[name]
        ]
        with column:
            category_values[name] = st.selectbox(
                label, options, key=f"cat_{name}"
            )

st.subheader("Numeric features")
numeric_values = {}
for group, names in NUMERIC_GROUPS.items():
    with st.expander(group, expanded=group == "Size & rooms"):
        for start in range(0, len(names), 3):
            columns = st.columns(3)
            for column, name in zip(columns, names[start : start + 3]):
                label, minimum, maximum, step = NUMERIC_INPUTS[name]
                with column:
                    default = defaults[name]
                    minimum_t = type(default)(minimum) if not isinstance(default, float) else float(minimum)
                    maximum_t = type(default)(maximum) if not isinstance(default, float) else float(maximum)
                    step_t = type(default)(step) if not isinstance(default, float) else float(step)
                    numeric_values[name] = st.number_input(
                        label,
                        min_value=minimum_t,
                        max_value=maximum_t,
                        value=default,
                        step=step_t,
                        key=f"num_{name}",
                    )

if st.button("Predict Price", type="primary"):
    input_frame = build_input_frame(
        feature_columns, typical, dummy_map, numeric_values, category_values
    )
    # The scaler was fitted on a plain numpy array, so feed it one (avoids a
    # "feature names" warning from mismatched inputs).
    prediction = float(model.predict(scaler.transform(input_frame.to_numpy(dtype=float)))[0])
    prediction = max(prediction, 0.0)

    st.success(f"Estimated House Price: ${prediction:,.0f}")
    st.caption(
        "Trained on Ames, Iowa sales from 2006-2010 with a linear model, so treat this "
        "as a rough estimate rather than an appraisal."
    )
