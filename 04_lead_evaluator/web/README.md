# Lead Evaluator — Web Interface (MVP)

A clean, business-friendly web UI for the Swan Chemical lead scoring system.  
Built for **demo / presentation purposes**.

---

## Tech Stack

| Layer    | Technology                     |
| -------- | ------------------------------ |
| Backend  | Python · **Flask**             |
| Frontend | HTML + CSS + Vanilla JS        |
| Charts   | Chart.js (CDN)                 |
| ML Model | CatBoost (SHAP via built-in)   |

**Why Flask?** Single `app.py`, built-in Jinja2 templating, zero async complexity — ideal for an MVP demo served from one command.

---

## Folder Structure

```
web/
├── app.py                  # Flask backend (routes + SHAP computation)
├── saved_leads.csv         # Auto-created on first Save Lead
├── templates/
│   └── index.html          # Single-page UI (form, results, modals)
├── static/
│   └── styles.css          # Light theme + optional dark mode
└── README.md               # ← You are here
```

The app imports `pipeline.py` and model artifacts from the **parent directory** (`04_lead_evaluator/`).  
Nothing is duplicated.

---

## How to Run

```bash
# 1  Navigate to the web folder
cd Deliverables/04_lead_evaluator/web

# 2  Make sure dependencies are installed (same Python that has catboost)
pip install flask openpyxl

# 3  Start the server
python app.py
```

Then open **http://127.0.0.1:5001** in your browser.

---

## Features

### Single-Company Mode
- Fill the 23-field form manually **or** upload a 1-row CSV/XLSX
- Click **Evaluate Lead** → see predicted bucket, score, SHAP waterfall, and all inputs
- **Save Lead** button opens an editable modal → appends to `saved_leads.csv`

### Multi-Company Mode (auto-detected on file upload)
- Upload a CSV/XLSX with **multiple rows**
- **Summary Dashboard** tab: donut chart of bucket distribution, percentage stats, hover to see company names
- **Individual Results** tab: sortable table with per-company View / Save buttons
- Click a row → expand detailed view with SHAP waterfall
- **Download CSV** button to export all evaluated results

### Bonus
- Loading spinner during prediction
- Form validation (Company Name required)
- Column-mismatch warning on upload
- Dark mode toggle (top-right)
- CSV download of evaluated results

---

## Design

- Light theme by default (white, soft greys, subtle borders)
- Font: **Inter** (Google Fonts)
- Bucket colour coding:
  - 🟢 Hot `#1E7F3F` · 🟩 Warm `#6DBE45` · 🟡 Save For Later `#F4C430` · 🟠 Cold `#F28B82` · 🔴 Reject `#B00020`
- SHAP waterfall: red bars = push score up, blue bars = push score down
