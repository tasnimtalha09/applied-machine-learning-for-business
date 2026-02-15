# importing the necessary libraries
from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple


# loading the saved artifacts from the artifacts directory


# resolving the path to the artifacts folder relative to this script
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# loading the tuned CatBoost model
cat_model = joblib.load(ARTIFACT_DIR / "catboost_tuned.joblib")

# loading the fitted ordinal encoder (employee_estimate, credibility_level, purchase_stage)
ordinal_encoder = joblib.load(ARTIFACT_DIR / "ordinal_encoder.joblib")

# loading the fitted one-hot encoder (nominal categorical columns)
onehot_encoder = joblib.load(ARTIFACT_DIR / "onehot_encoder.joblib")

# loading the fitted standard scaler (numeric columns)
scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")

# loading the metadata dictionary (column lists, category orders, thresholds)
metadata = joblib.load(ARTIFACT_DIR / "metadata.joblib")


# extracting metadata into module-level constants


# column groups used during preprocessing
ORDINAL_COLS = metadata["ordinal_cols"]
ONEHOT_COLS = metadata["onehot_cols"]
NUMERIC_COLS = metadata["numeric_cols"]

# exact feature column order the model expects at prediction time
FEATURE_COLUMNS = metadata["feature_columns"]

# category orderings for ordinal columns (must match training)
EMP_ORDER = metadata["employee_order"]
CRED_ORDER = metadata["credibility_order"]
PURCHASE_ORDER = metadata["purchase_stage_order"]

# mapping for Yes/No and True/False fields to integer (1/0)
YN_MAP = metadata["yn_map"]

# business action thresholds based on QualifiedScore (P(Hot) + P(Warm))
THR_PRIORITIZE = metadata["qualified_thresholds"]["prioritize"]
THR_NURTURE = metadata["qualified_thresholds"]["nurture"]


# creating some helper functions

# converting a Yes/No or True/False series to integer (1/0)
def _yes_no_to_int(series: pd.Series) -> pd.Series:

    # mapping values using the YN_MAP dictionary; unknown values become 0
    return series.map(YN_MAP).fillna(0).astype(int)

# applying the exact same preprocessing pipeline used during training: boolean conversion → ordinal encoding → one-hot encoding → scaling → column alignment
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:

    # creating a copy to avoid mutating the original dataframe
    df = raw_df.copy()

    # converting Yes/No boolean fields to integer (1/0)
    for col in ["has_google_listing", "has_phone"]:
        if col in df.columns:
            df[col] = _yes_no_to_int(df[col])

    # setting ordinal categories to match the exact ordering used during training
    df["employee_estimate"] = pd.Categorical(df["employee_estimate"], categories=EMP_ORDER, ordered=True)
    df["credibility_level"] = pd.Categorical(df["credibility_level"], categories=CRED_ORDER, ordered=True)
    df["purchase_stage"] = pd.Categorical(df["purchase_stage"], categories=PURCHASE_ORDER, ordered=True)

    # applying the fitted ordinal encoder to ordinal columns
    df[ORDINAL_COLS] = ordinal_encoder.transform(X=df[ORDINAL_COLS])

    # applying the fitted one-hot encoder to nominal categorical columns
    oh = pd.DataFrame(
        onehot_encoder.transform(X=df[ONEHOT_COLS]),
        columns=onehot_encoder.get_feature_names_out(ONEHOT_COLS),
        index=df.index,
    )

    # dropping the original nominal columns and joining the one-hot encoded columns
    df = df.drop(columns=ONEHOT_COLS).join(oh)

    # applying the fitted standard scaler to numeric columns
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # reindexing to match the exact training feature columns and order
    # any missing columns are filled with 0 (e.g., unseen one-hot categories)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df

# mapping a QualifiedScore (0–100) to a business action based on thresholds: >= 70: Prioritize (Sales); >= 40: Nurture (Follow-up); < 40: Deprioritize
def score_to_action(score: float) -> str:

    # checking if the score meets the prioritize threshold
    if score >= THR_PRIORITIZE:
        return "Prioritize (Sales)"

    # checking if the score meets the nurture threshold
    if score >= THR_NURTURE:
        return "Nurture (Follow-up)"

    # if neither threshold is met, deprioritize the lead
    return "Deprioritize"


# creating the main prediction function

# scoring the new leads end-to-end: preprocess → predict bucket → compute QualifiedScore → assign action. Returns a copy of the input dataframe with pred_bucket, lead_score, and action columns appended.
def predict_leads(raw_df: pd.DataFrame) -> pd.DataFrame:

    # preprocessing the raw input dataframe using the training pipeline
    X = preprocess(raw_df)

    # generating predicted class probabilities from the tuned CatBoost model
    proba = cat_model.predict_proba(X)

    # retrieving the class labels in the order the model outputs probabilities
    classes = cat_model.classes_

    # building a mapping from class name to column index in the probability matrix
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # finding the column indices for Hot and Warm (the "qualified" classes)
    hot_idx = class_to_idx["Hot"]
    warm_idx = class_to_idx["Warm"]

    # predicting the most likely lead bucket for each row (flattened to 1D)
    pred_bucket = np.array(cat_model.predict(X)).ravel()

    # computing the QualifiedScore as P(Hot) + P(Warm), scaled to 0–100
    lead_score = 100.0 * (proba[:, hot_idx] + proba[:, warm_idx])

    # rounding the score to one decimal place
    lead_score = np.round(lead_score, 1)

    # mapping each score to a business action (Prioritize / Nurture / Deprioritize)
    action = np.array([score_to_action(s) for s in lead_score])

    # appending the predictions to a copy of the original dataframe
    out = raw_df.copy()
    out["pred_bucket"] = pred_bucket
    out["lead_score"] = lead_score
    out["action"] = action

    return out