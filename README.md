<div style="text-align: justify;">

# Introduction

This report consists of a number of projects done for the academic course **Applied Machine Learning for Business** under the **BBA** program (Data Analytics minor) at **Institute of Business Administration (IBA), University of Dhaka**.




# Projects Overview

## Project 01: Understanding & Explaining a Machine Learning Algorithm

In [this project](01_understanding_and_explaining_a_machine_learning_algorithm), we explored the three Naive Bayes Classifier algorithms—Multinomial, Bernoulli, and Gaussian—using the SMS Spam Collection Dataset. 

* **Goal:** To understand and explain the working of Naive Bayes algorithms in the context of spam detection.

* **Dataset:** SMS Spam Collection Dataset from UCI Machine Learning Repository. Link to the dataset [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

* **README**: A detailed README.md file dedicated only for this project can be found [here](01_understanding_and_explaining_a_machine_learning_algorithm/README.md).

* **Notebook:** The Jupyter Notebook can be found [here](01_understanding_and_explaining_a_machine_learning_algorithm/naive_bayes_classifiers.ipynb).

* **Video Presentation:** A video presentation explaining the project can be found [here](https://www.linkedin.com/posts/tasnimtalha09_have-you-ever-wondered-how-your-messaging-activity-7396386394614460416-3SYG?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADzHX0oB0t4dpMRgmmkbZZhC5z3UQS7rZrw).

* **Tools Used:** Python, Jupyter Notebook, scikit-learn, pandas, numpy, matplotlib, seaborn.

* **One-Line Results:** Multinomial Naive Bayes performed the best among the three models whereas Gaussian Naive Bayes performed the worst.

* **Workflow:**
    1. We loaded the dataset either from the local file or fetched it online if the local file was not found.
    2. We encoded the target variable.
    3. We conducted a stratified train-test split at a 70:30 ratio.
    4. We encoded the data based on the requirements of each Naive Bayes algorithms—`CountVectorizer` for Multinomial and Bernoulli, and `TfidfVectorizer` for Gaussian.
    5. We trained the models.
    6. We evaluated the models using accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and ROC-AUC score.
    7. We conducted an overfitting and a cross-validation check.
    8. We finally selected the best dataset.



## Project 02: Predicting Box Office Proceeds

In [this project](02_predicting_box_office_proceeds), we developed a production-ready machine learning system to forecast first-week box office revenue using 26 years of theatrical performance data and advanced gradient boosting algorithms

* **Goal:** To build a predictive model that accurately forecasts opening-week box office revenue using pre-release film attributes, enabling data-driven investment decisions for studios, distributors, and theater chains. The judgement criteria will be the first 7-day box office revenue of **Avatar: Fire & Ash** releasing on December 19, 2025.

* **Dataset:** 26 years of box office history (2000-2025) extracted from TMDB API, IMDb ratings, and theatrical performance data. Over **4,300 films** with **160+ engineered features** after preprocessing.

* **README**: A detailed README.md file dedicated only for this project can be found [here](02_predicting_box_office_proceeds/README.md).

* **Notebook:** The Jupyter Notebook can be found [here](02_predicting_box_office_proceeds/box_office_prediction.ipynb).

* **Tools Used:** Python, Jupyter Notebook, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, pandas, numpy, matplotlib, seaborn, SHAP, TensorFlow.

* **One-Line Results:** CatBoost (Optuna-optimized) achieved **78.35% R²** with **$3.42M MAE**, demonstrating that theater distribution and audience sentiment outweigh production budget and star power in predicting opening-week revenue.

* **Workflow:**
    1. **Data Extraction**: Scraped 26 years of box office data using custom API scripts, extracting TMDB IDs, metadata, ratings, and theatrical performance metrics.
    2. **Data Aggregation**: Aggregated weekly performance data to focus on first 7 days of release, removing irrelevant time-series components.
    3. **Data Cleaning**: Handled missing values, corrected erroneous entries, dropped irrelevant columns, and standardized feature naming conventions.
    4. **Feature Engineering**: Created 150+ derived features including `theater_penetration`, `log_budget`, `is_franchise`, `release_month`, and temporal indicators.
    5. **Encoding**: Applied multiple encoding strategies—One-Hot Encoding for categorical variables, kFold Encoding for high-cardinality features (directors, distributors), Ordinal Encoding for ranked categories, and Count Encoding for frequency-based features.
    6. **Standardization**: Normalized numerical features using `StandardScaler()` to ensure consistent scaling across models.
    7. **Model Training**: Trained 9 regression algorithms (LightGBM, XGBoost, CatBoost, Gradient Boosting, Random Forest, AdaBoost, Decision Tree, ElasticNet, MLP) and compared performance using R², RMSE, and MAE.
    8. **Model Selection**: Advanced top 4 gradient boosting models to rigorous validation using 5-fold cross-validation, fold-wise performance analysis, and generalization testing.
    9. **Hyperparameter Tuning**: Developed CatBoost v2 (RandomizedSearchCV) and v3 (Optuna with gap-penalty optimization) variants through depth adjustment, L2 regularization, learning rate tuning, and early stopping strategies.
    10. **Model Evaluation**: Selected CatBoost v3 (Optuna) as production model based on superior test performance (78.35% R², $3.42M MAE, $7.80M RMSE) with robust generalization.
    11. **Explainability Analysis**: Conducted SHAP analysis to identify revenue drivers, revealing theater distribution and audience sentiment as dominant predictors.
    12. **Validation**: Confirmed deployment readiness through cross-validation stability (77.96% ± 1.44%) and acceptable train-test gap (18.99%).



## Project 03: Comparing Naive Bayes Models with Neural Network Models
In [this project](03_comparing_naive_bayes_models_with_neural_network_models), we compare the models used in [project 01](01_understanding_and_explaining_a_machine_learning_algorithm) (Naive Bayes Classifiers—Multinomial, Bernoulli, and Gaussian) with various neural network algorithms to see whether neural networks can outperform Naive Bayes classifiers in spam detection using the SMS Spam Collection Dataset.

* **Goal:** To compare the performance of Naive Bayes classifiers with various neural network architectures for spam detection and see whether neural networks can outperform the Naive Bayes classifiers.

* **Dataset:** SMS Spam Collection Dataset from UCI Machine Learning Repository. Link to the dataset [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

* **README**: A detailed README.md file dedicated only for this project can be found [here](03_comparing_naive_bayes_models_with_neural_network_models/README.md).

* **Notebook:** The Jupyter Notebook can be found [here](03_comparing_naive_bayes_models_with_neural_network_models/neural_network.ipynb).

* **Tools Used:** Python, Jupyter Notebook, scikit-learn, pandas, numpy, matplotlib, seaborn, Tensorflow.

* **One-Line Results:** The **CNN + LSTM Hybrid** model outperformed all others, achieving **99.04% accuracy**, **99.06% precision**, **93.75% recall**, and a **96.33% F1-score**, surpassing Multinomial Naive Bayes across all metrics while maintaining identical overfitting levels (0.67%).

* **Workflow:**
    1. Loaded the dataset either from the local file or fetched it online if the local file was not found.
    2. Encoded the target variable (`ham` = 0, `spam` = 1).
    3. Conducted a stratified train-test split at a 70:30 ratio.
    4. Encoded the data based on the requirements of each model—`CountVectorizer` for Multinomial Naive Bayes, `TfidfVectorizer` for Multi-Layer Perceptron, and `Tokenizer` with `pad_sequences` for CNN, GRU, LSTM, BiLSTM, and CNN + LSTM Hybrid.
    5. Trained **seven** models: Multinomial Naive Bayes, Multi-Layer Perceptron (MLP), 1D Convolutional Neural Network (1D CNN), Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), Bidirectional LSTM (BiLSTM), and CNN + LSTM Hybrid.
    6. Evaluated the models using accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and ROC-AUC score.
    7. Conducted an overfitting check by comparing train-test accuracy differences.
    8. Compiled all performance metrics into a comparative table and selected the best model (1-Dimensional Convolutional Neural Network).



## Project 04: Lead Evaluator

In [this project](04_lead_evaluator), we built a machine learning–driven lead evaluation system for **Swan Chemical Ltd.**, a leading adhesive manufacturer. The system classifies incoming company inquiries into actionable lead buckets (Hot, Warm, Cold, Save For Later, and Reject) and generates a probability-based lead score to support sales prioritization.

* **Goal:** To develop a practical, business-ready lead evaluation MVP that predicts each incoming company lead's lead bucket and produces a probability-based QualifiedScore (**P(Hot) + P(Warm)**) to guide sales follow-up decisions.

* **Dataset:** A synthetic dataset created to mimic real-world lead evaluation scenarios, containing features related to company inquiries such as company size, industry, inquiry source, location, urgency, expected demand volume, purchase stage, and trust signals.

* **README**: A detailed README.md file dedicated only for this project can be found [here](04_lead_evaluator/README.md).

* **Notebook:** The Jupyter Notebook can be found [here](04_lead_evaluator/Lead_evaluator.ipynb).

* **Scoring Script:** A separate script for production-style lead scoring can be found [here](04_lead_evaluator/scoring.py).

* **Tools Used:** Python, Jupyter Notebook, scikit-learn, CatBoost, LightGBM, pandas, numpy, matplotlib, seaborn, SHAP, joblib.

* **One-Line Results:** Tuned CatBoost achieved **78.15% Macro F1**, **95.63% OVR Macro AUC**, **99.16% Qualified-vs-Not AUC**, and **82.13% Hot+Warm Recall** with only **5.96% overfitting gap**, making it a reliable and deployment-ready lead scoring model.

* **Workflow:**
    1. **Data Loading & Cleaning**: Loaded the synthetic lead dataset, converted boolean columns (`has_google_listing`, `has_phone`) to binary integers, and verified data types and shape.
    2. **Ordinal Structuring**: Established meaningful ordinal orders for `employee_estimate`, `credibility_level`, and `purchase_stage` using pandas Categorical types.
    3. **Exploratory Data Analysis**: Conducted EDA across **six** thematic areas—Target & Scoring, Geography & Feasibility, Business Identity, Capacity & Credibility, Digital Signal, Need & Product Fit, Purchase Behavior & Urgency, and Commercial Terms & Source.
    4. **Train-Test Split**: Performed a stratified **70:30** train-test split to preserve class distribution across all five lead buckets.
    5. **Encoding**: Applied Ordinal Encoding for ranked features (`employee_estimate`, `credibility_level`, `purchase_stage`) and One-Hot Encoding (with `drop = "first"`) for nominal categorical features.
    6. **Standardization**: Normalized numerical features (`distance_km`, `expected_monthly_volume_liters`, `urgency_days`, `google_review_count`) using `StandardScaler`.
    7. **Model Training**: Trained five multi-class classification models—Multinomial Logistic Regression, Random Forest, LightGBM, CatBoost, and HistGradientBoosting.
    8. **Model Evaluation**: Evaluated all models using Macro F1, Weighted F1, Accuracy, OVR Macro/Weighted ROC-AUC, Qualified-vs-Not ROC-AUC, and Hot + Warm Recall.
    9. **Overfitting Check**: Compared train–test Macro F1 gaps across all models to assess generalization risk.
    10. **Cross-Validation**: Conducted 5-fold cross-validation (Macro F1) with boxplot and heatmap visualizations to confirm performance stability across splits.
    11. **Model Selection**: Selected CatBoost as the final model based on highest Macro F1 (78.38%), best CV Mean F1 (0.7804), highest Qualified-vs-Not AUC (99.01%), and strongest Hot+Warm Recall (81.63%).
    12. **Hyperparameter Tuning**: Tuned CatBoost using RandomizedSearchCV (100 iterations, StratifiedKFold), reducing overfitting from 21.50% to 5.96% while improving Macro F1 to 78.15%.
    13. **Explainability Analysis**: Conducted feature importance analysis using both CatBoost's built-in importance and SHAP values (global bar chart and individual waterfall plot), identifying `urgency_days`, `expected_monthly_volume_liters`, and `google_review_count` as the top three drivers.
    14. **Deployment**: Saved all fitted artifacts (model, encoders, scaler, metadata) to an `artifacts/` folder and built a lightweight scoring module (`lead_evaluator.py`) with a separate testing script (`scoring.py`) for production-style lead scoring.


</div>