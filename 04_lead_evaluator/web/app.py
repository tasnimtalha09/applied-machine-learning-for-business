"""
Lead Evaluator — MVP Web Interface
===================================
Flask backend that wraps the existing predict_leads pipeline,
adds SHAP explanations (via CatBoost built-in), and serves a
clean business-friendly UI for single- and multi-company scoring.

Why Flask?
    • Built-in Jinja2 templating — perfect for an HTML-based MVP.
    • Minimal boilerplate compared to FastAPI (no async needed).
    • Easier to ship a self-contained demo in one command.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

# ---------------------------------------------------------------------------
# Make the parent directory (04_lead_evaluator) importable so we can reuse
# pipeline.py and its artifacts without duplicating anything.
# ---------------------------------------------------------------------------
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from catboost import Pool                       # noqa: E402
from pipeline import (                          # noqa: E402
    cat_model,
    preprocess,
    predict_leads,
    FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
SAVED_CSV = Path(__file__).resolve().parent / "saved_leads.csv"

# ---------------------------------------------------------------------------
# Column metadata (used by the front-end form)
# ---------------------------------------------------------------------------
INPUT_COLUMNS = [
    "company_name",
    "industry_category",
    "sub_industry",
    "location_area",
    "district",
    "zone_type",
    "distance_km",
    "company_type",
    "employee_estimate",
    "materials_used",
    "adhesive_type_needed",
    "product_needed",
    "application_type",
    "purchase_stage",
    "expected_monthly_volume_liters",
    "order_frequency",
    "urgency_days",
    "payment_terms_expected",
    "has_google_listing",
    "google_review_count",
    "has_phone",
    "credibility_level",
    "lead_source",
]

NUMERIC_FIELDS = {
    "distance_km",
    "expected_monthly_volume_liters",
    "urgency_days",
    "google_review_count",
}

BOOLEAN_FIELDS = {
    "has_google_listing",
    "has_phone",
}

# Dropdown options (derived from training data)
DROPDOWN_OPTIONS: dict[str, list[str]] = {
    "industry_category": [
        "Automotive/Transport", "Construction", "Footwear", "Furniture",
        "Other", "Packaging", "Retail/Trading", "Upholstery/Foam",
    ],
    "sub_industry": [
        "Auto Seat Upholstery", "Building Materials Shop",
        "Bus/Coach Body Shop", "Car Accessories Unit", "Carpentry Unit",
        "Corrugated Box Factory", "Cushion Producer",
        "Door & Board Manufacturer", "Foam Cutting Unit",
        "Footwear Accessories Supplier", "General Trading Store",
        "General Workshop", "Hardware & Adhesive Shop",
        "Interior Materials Trader", "Interior Workshop",
        "Label Printing Unit", "Leather Goods Workshop",
        "Mattress Manufacturer", "Miscellaneous Supplier",
        "Modular Furniture Manufacturer", "PVC Sheet Workshop",
        "Paper Bag Producer", "Plastic Packaging Manufacturer",
        "Shoe Upper Factory", "Small Enterprise", "Small Repair Unit",
        "Sofa Upholstery Workshop", "Sole Manufacturer",
        "Vehicle Interior Workshop", "Wholesale Store",
        "Wood Furniture Factory", "Wood Panel Unit",
    ],
    "location_area": [
        "Gazipur", "Gulshan", "Keraniganj", "Mirpur", "Motijheel",
        "Narayanganj", "Savar", "Tejgaon", "Tongi", "Uttara",
    ],
    "district": ["Dhaka", "Gazipur", "Narayanganj"],
    "zone_type": [
        "Central Dhaka", "Industrial Zone", "Mixed Zone", "Near Highway",
    ],
    "company_type": ["Distributor", "Manufacturer", "Retailer", "Trader"],
    "employee_estimate": [
        "Micro (1-9)", "Small (10-49)", "Medium (50-249)", "Large (250+)",
    ],
    "materials_used": [
        "Fabric", "Foam", "Leather", "Mixed/Unknown", "PVC/EVA",
        "Paper/Board", "Plastic", "Rubber", "Wood",
    ],
    "adhesive_type_needed": ["PU", "SR", "Unknown"],
    "product_needed": [
        "General Purpose Adhesive", "PU Adhesive", "SR Adhesive",
    ],
    "application_type": [
        "Assembly", "Bonding", "Lamination", "Repair/Maintenance",
    ],
    "purchase_stage": ["Trial Order", "Regular Reorder", "Annual Contract"],
    "order_frequency": ["Monthly", "One-time", "Weekly"],
    "payment_terms_expected": ["Cash", "Credit", "Unknown"],
    "credibility_level": ["Low", "Medium", "High"],
    "lead_source": [
        "Distributor Referral", "Facebook Page", "Phone Call",
        "Referral", "Walk-in", "Website Inquiry", "WhatsApp",
    ],
    "has_google_listing": ["Yes", "No"],
    "has_phone": ["Yes", "No"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bucket → colour mapping
BUCKET_COLORS = {
    "Hot":            "#1E7F3F",
    "Warm":           "#6DBE45",
    "Save For Later": "#F4C430",
    "Cold":           "#F28B82",
    "Reject":         "#B00020",
}


def _compute_shap(X_processed: pd.DataFrame) -> list[dict]:
    """Return per-row SHAP data for the QualifiedScore (P_Hot + P_Warm).

    Each element:  {
        "base_value": float,
        "features": [{"name": str, "value": float}, ...],  # top-10 by |value|
    }
    """
    pool = Pool(X_processed)
    raw = np.array(
        cat_model.get_feature_importance(type="ShapValues", data=pool)
    )
    # raw shape: (n_samples, n_classes, n_features+1)
    classes = list(cat_model.classes_)
    hot_idx = classes.index("Hot")
    warm_idx = classes.index("Warm")
    n_feat = len(FEATURE_COLUMNS)

    results = []
    for i in range(raw.shape[0]):
        combined = raw[i, hot_idx, :n_feat] + raw[i, warm_idx, :n_feat]
        base = float(raw[i, hot_idx, n_feat] + raw[i, warm_idx, n_feat])
        top_idx = np.argsort(np.abs(combined))[::-1][:10]
        features = [
            {"name": FEATURE_COLUMNS[j], "value": round(float(combined[j]), 4)}
            for j in top_idx
        ]
        results.append({"base_value": round(base, 4), "features": features})
    return results


def _build_results(raw_df: pd.DataFrame) -> list[dict]:
    """Run prediction + SHAP and return a list of per-company dicts."""
    results_df = predict_leads(raw_df)
    X_processed = preprocess(raw_df)
    shap_data = _compute_shap(X_processed)

    rows: list[dict] = []
    for idx in range(len(results_df)):
        row = results_df.iloc[idx]
        entry: dict = {
            "company_name": str(row["company_name"]),
            "pred_bucket": str(row["pred_bucket"]),
            "lead_score": round(float(row["lead_score"]), 1),
            "action": str(row["action"]),
            "bucket_color": BUCKET_COLORS.get(str(row["pred_bucket"]), "#888"),
            "shap": shap_data[idx],
        }
        # Attach original input features
        for col in INPUT_COLUMNS:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    entry[col] = ""
                else:
                    entry[col] = val if not isinstance(val, (np.integer, np.floating)) else val.item()
        rows.append(entry)
    return rows


def _try_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort fuzzy column mapping to expected feature names."""
    rename_map: dict[str, str] = {}
    expected = {c.lower().replace("_", " "): c for c in INPUT_COLUMNS}
    for col in df.columns:
        key = col.strip().lower().replace("_", " ")
        if key in expected:
            rename_map[col] = expected[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        columns=INPUT_COLUMNS,
        dropdown_options=DROPDOWN_OPTIONS,
        numeric_fields=list(NUMERIC_FIELDS),
        boolean_fields=list(BOOLEAN_FIELDS),
    )


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Accept JSON body with a list of company dicts → return scored results."""
    payload = request.get_json(force=True)
    companies = payload.get("companies", [])
    if not companies:
        return jsonify({"error": "No data provided."}), 400

    raw_df = pd.DataFrame(companies)
    raw_df = _try_map_columns(raw_df)

    # Cast numeric columns
    for col in NUMERIC_FIELDS:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0)

    try:
        rows = _build_results(raw_df)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"results": rows})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept .csv / .xlsx file upload → return parsed rows as JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    filename = f.filename or ""

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(f.stream)
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(f.stream, engine="openpyxl")
        else:
            return jsonify({"error": "Unsupported file type. Use .csv or .xlsx"}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to parse file: {exc}"}), 400

    df = _try_map_columns(df)

    # Drop lead_bucket/lead_score if present (they're targets, not inputs)
    for col in ("lead_bucket", "lead_score"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Identify missing expected columns
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in INPUT_COLUMNS]

    records = df.to_dict(orient="records")
    # Clean NaN → None
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None

    return jsonify({
        "records": records,
        "missing_columns": missing,
        "extra_columns": extra,
        "row_count": len(records),
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    """Append a single lead (with predictions) to saved_leads.csv."""
    payload = request.get_json(force=True)
    lead = payload.get("lead", {})
    if not lead:
        return jsonify({"error": "No lead data."}), 400

    fieldnames = INPUT_COLUMNS + ["pred_bucket", "lead_score"]
    file_exists = SAVED_CSV.exists()

    with open(SAVED_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(lead)

    return jsonify({"success": True, "message": "Lead saved successfully."})


@app.route("/api/download-results", methods=["POST"])
def api_download_results():
    """Accept evaluated results JSON → return downloadable CSV."""
    payload = request.get_json(force=True)
    results = payload.get("results", [])
    if not results:
        return jsonify({"error": "No results to download."}), 400

    fieldnames = ["company_name", "pred_bucket", "lead_score", "action"] + [
        c for c in INPUT_COLUMNS if c != "company_name"
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        writer.writerow(row)

    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="evaluated_leads.csv",
    )


@app.route("/api/form-options", methods=["GET"])
def api_form_options():
    """Return dropdown options, field types, etc. for front-end form."""
    return jsonify({
        "columns": INPUT_COLUMNS,
        "dropdown_options": DROPDOWN_OPTIONS,
        "numeric_fields": list(NUMERIC_FIELDS),
        "boolean_fields": list(BOOLEAN_FIELDS),
    })


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 80)
