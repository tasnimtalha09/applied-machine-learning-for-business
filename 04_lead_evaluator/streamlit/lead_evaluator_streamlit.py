# importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import Pool
import matplotlib.pyplot as plt

# --- Load Artifacts ---
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
cat_model = joblib.load(ARTIFACT_DIR / "catboost_tuned.joblib")
ordinal_encoder = joblib.load(ARTIFACT_DIR / "ordinal_encoder.joblib")
onehot_encoder = joblib.load(ARTIFACT_DIR / "onehot_encoder.joblib")
scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
metadata = joblib.load(ARTIFACT_DIR / "metadata.joblib")

ORDINAL_COLS = metadata["ordinal_cols"]
ONEHOT_COLS = metadata["onehot_cols"]
NUMERIC_COLS = metadata["numeric_cols"]
FEATURE_COLUMNS = metadata["feature_columns"]
EMP_ORDER = metadata["employee_order"]
CRED_ORDER = metadata["credibility_order"]
PURCHASE_ORDER = metadata["purchase_stage_order"]
YN_MAP = metadata["yn_map"]
THR_PRIORITIZE = metadata["qualified_thresholds"]["prioritize"]
THR_NURTURE = metadata["qualified_thresholds"]["nurture"]

# --- Helper Functions ---
def _yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.map(YN_MAP).fillna(0).astype(int)

def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    drop_cols = ["company_name", "lead_bucket", "lead_score", "actual_lead_score"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for col in ["has_google_listing", "has_phone"]:
        if col in df.columns:
            df[col] = _yes_no_to_int(df[col])
    df["employee_estimate"] = pd.Categorical(df["employee_estimate"], categories=EMP_ORDER, ordered=True)
    df["credibility_level"] = pd.Categorical(df["credibility_level"], categories=CRED_ORDER, ordered=True)
    df["purchase_stage"] = pd.Categorical(df["purchase_stage"], categories=PURCHASE_ORDER, ordered=True)
    df[ORDINAL_COLS] = ordinal_encoder.transform(df[ORDINAL_COLS])
    oh = pd.DataFrame(
        onehot_encoder.transform(df[ONEHOT_COLS]),
        columns=onehot_encoder.get_feature_names_out(ONEHOT_COLS),
        index=df.index,
    )
    df = df.drop(columns=ONEHOT_COLS).join(oh)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df

def score_to_action(score: float) -> str:
    if score >= THR_PRIORITIZE:
        return "Prioritize (Sales)"
    if score >= THR_NURTURE:
        return "Nurture (Follow-up)"
    return "Deprioritize"

def predict_leads(raw_df: pd.DataFrame) -> pd.DataFrame:
    X = preprocess(raw_df)
    proba = cat_model.predict_proba(X)
    classes = cat_model.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}
    hot_idx = class_to_idx["Hot"]
    warm_idx = class_to_idx["Warm"]
    pred_bucket = np.array(cat_model.predict(X)).ravel()
    lead_score = 100.0 * (proba[:, hot_idx] + proba[:, warm_idx])
    lead_score = np.round(lead_score, 1)
    action = np.array([score_to_action(s) for s in lead_score])
    out = raw_df.copy()
    out["pred_bucket"] = pred_bucket
    out["lead_score"] = lead_score
    out["action"] = action
    return out

def compute_shap(X_processed: pd.DataFrame):
    pool = Pool(X_processed)
    raw = np.array(cat_model.get_feature_importance(type="ShapValues", data=pool))
    classes = list(cat_model.classes_)
    hot_idx = classes.index("Hot")
    warm_idx = classes.index("Warm")
    n_feat = len(FEATURE_COLUMNS)
    shap_data = []
    for i in range(raw.shape[0]):
        combined = raw[i, hot_idx, :n_feat] + raw[i, warm_idx, :n_feat]
        base = float(raw[i, hot_idx, n_feat] + raw[i, warm_idx, n_feat])
        top_idx = np.argsort(np.abs(combined))[::-1][:10]
        features = [(FEATURE_COLUMNS[j], combined[j]) for j in top_idx]
        shap_data.append((base, features))
    return shap_data

# --- Streamlit UI ---
st.set_page_config(page_title="Lead Evaluator", layout="wide")
st.title("Lead Evaluator — Swan Chemical")
st.markdown("""
A business-friendly interface for scoring and explaining company leads.\
Upload a CSV or fill the form below.\
**All predictions and explanations run on the server — no data leaves your browser.**
""")

# --- Sidebar: File Upload ---
st.sidebar.header("Upload Leads File")
file = st.sidebar.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx", "xls"])

# --- Main: Manual Input Form ---
with st.expander("Manual Input (Single Company)", expanded=True):
    form = st.form("manual_form")
    company_name = form.text_input("Company Name")
    industry_category = form.selectbox("Industry Category", ["Automotive/Transport", "Construction", "Footwear", "Furniture", "Other", "Packaging", "Retail/Trading", "Upholstery/Foam"])
    sub_industry = form.selectbox("Sub Industry", ["Auto Seat Upholstery", "Building Materials Shop", "Bus/Coach Body Shop", "Car Accessories Unit", "Carpentry Unit", "Corrugated Box Factory", "Cushion Producer", "Door & Board Manufacturer", "Foam Cutting Unit", "Footwear Accessories Supplier", "General Trading Store", "General Workshop", "Hardware & Adhesive Shop", "Interior Materials Trader", "Interior Workshop", "Label Printing Unit", "Leather Goods Workshop", "Mattress Manufacturer", "Miscellaneous Supplier", "Modular Furniture Manufacturer", "PVC Sheet Workshop", "Paper Bag Producer", "Plastic Packaging Manufacturer", "Shoe Upper Factory", "Small Enterprise", "Small Repair Unit", "Sofa Upholstery Workshop", "Sole Manufacturer", "Vehicle Interior Workshop", "Wholesale Store", "Wood Furniture Factory", "Wood Panel Unit"])
    location_area = form.selectbox("Location Area", ["Gazipur", "Gulshan", "Keraniganj", "Mirpur", "Motijheel", "Narayanganj", "Savar", "Tejgaon", "Tongi", "Uttara"])
    district = form.selectbox("District", ["Dhaka", "Gazipur", "Narayanganj"])
    zone_type = form.selectbox("Zone Type", ["Central Dhaka", "Industrial Zone", "Mixed Zone", "Near Highway"])
    distance_km = form.number_input("Distance (km)", min_value=0.0, step=0.1)
    company_type = form.selectbox("Company Type", ["Distributor", "Manufacturer", "Retailer", "Trader"])
    employee_estimate = form.selectbox("Employee Estimate", ["Micro (1-9)", "Small (10-49)", "Medium (50-249)", "Large (250+)"])
    materials_used = form.selectbox("Materials Used", ["Fabric", "Foam", "Leather", "Mixed/Unknown", "PVC/EVA", "Paper/Board", "Plastic", "Rubber", "Wood"])
    adhesive_type_needed = form.selectbox("Adhesive Type Needed", ["PU", "SR", "Unknown"])
    product_needed = form.selectbox("Product Needed", ["General Purpose Adhesive", "PU Adhesive", "SR Adhesive"])
    application_type = form.selectbox("Application Type", ["Assembly", "Bonding", "Lamination", "Repair/Maintenance"])
    purchase_stage = form.selectbox("Purchase Stage", ["Trial Order", "Regular Reorder", "Annual Contract"])
    expected_monthly_volume_liters = form.number_input("Expected Monthly Volume (L)", min_value=0.0, step=1.0)
    order_frequency = form.selectbox("Order Frequency", ["Monthly", "One-time", "Weekly"])
    urgency_days = form.number_input("Urgency (days)", min_value=0.0, step=1.0)
    payment_terms_expected = form.selectbox("Payment Terms Expected", ["Cash", "Credit", "Unknown"])
    has_google_listing = form.selectbox("Has Google Listing", ["Yes", "No"])
    google_review_count = form.number_input("Google Review Count", min_value=0.0, step=1.0)
    has_phone = form.selectbox("Has Phone", ["Yes", "No"])
    credibility_level = form.selectbox("Credibility Level", ["Low", "Medium", "High"])
    lead_source = form.selectbox("Lead Source", ["Distributor Referral", "Facebook Page", "Phone Call", "Referral", "Walk-in", "Website Inquiry", "WhatsApp"])
    submit = form.form_submit_button("Evaluate Lead")

    if submit and company_name:
        input_dict = {k: v for k, v in locals().items() if k in FEATURE_COLUMNS or k in ["company_name"]}
        df = pd.DataFrame([input_dict])
        results = predict_leads(df)
        st.success(f"Predicted Bucket: {results['pred_bucket'][0]}")
        st.write(f"Lead Score: {results['lead_score'][0]}")
        X_proc = preprocess(df)
        shap_data = compute_shap(X_proc)[0]
        base, features = shap_data
        st.write("#### SHAP Waterfall (Top 10 Features)")
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = [v for _, v in features]
        labels = [f for f, _ in features]
        colors = ["#dc4c3e" if v > 0 else "#4285f4" for v in vals]
        ax.barh(labels, vals, color=colors)
        ax.axvline(0, color="#888", lw=1)
        ax.set_xlabel("SHAP Value (impact on score)")
        st.pyplot(fig)
        st.write("#### Input Details")
        st.json(input_dict)

# --- File Upload Mode ---
if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    st.write(f"Loaded {len(df)} leads from file.")
    results = predict_leads(df)
    st.dataframe(results[["company_name", "pred_bucket", "lead_score", "action"]])
    st.write("#### Download Results")
    st.download_button("Download CSV", results.to_csv(index=False), file_name="evaluated_leads.csv")
    # SHAP for first row
    st.write("#### SHAP Waterfall for First Lead")
    X_proc = preprocess(df.head(1))
    shap_data = compute_shap(X_proc)[0]
    base, features = shap_data
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [v for _, v in features]
    labels = [f for f, _ in features]
    colors = ["#dc4c3e" if v > 0 else "#4285f4" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#888", lw=1)
    ax.set_xlabel("SHAP Value (impact on score)")
    st.pyplot(fig)
