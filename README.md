<div style="text-align: justify;">

# Introduction

This report consists of a number of projects done for the academic course **Applied Machine Learning for Business** under the **BBA** program (Data Analytics minor) at **Institute of Business Administration (IBA), University of Dhaka**.




# Projects Overview

## Project 01: Understanding & Explaining a Machine Learning Algorithm

In [this project](project_01_understanding_&_explaining_a_machine_learning_algorithm), we explored the three Naive Bayes Classifier algorithms—Multinomial, Bernoulli, and Gaussian—using the SMS Spam Collection Dataset. 

* **Goal:** To understand and explain the working of Naive Bayes algorithms in the context of spam detection.

* **Dataset:** SMS Spam Collection Dataset from UCI Machine Learning Repository. Link to the dataset [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

* **README**: A detailed README.md file dedicated only for this project can be found [here](project_01_understanding_&_explaining_a_machine_learning_algorithm/README.md).

* **Notebook:** The Jupyter Notebook can be found [here](project_01_understanding_&_explaining_a_machine_learning_algorithm/Naive_Bayes_Classifiers.ipynb).

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

In [this project](project_02_predicting_box_office_proceeds), we developed a production-ready machine learning system to forecast first-week box office revenue using 26 years of theatrical performance data and advanced gradient boosting algorithms

* **Goal:** To build a predictive model that accurately forecasts opening-week box office revenue using pre-release film attributes, enabling data-driven investment decisions for studios, distributors, and theater chains. The judgement criteria will be the first 7-day box office revenue of **Avatar: Fire & Ash** releasing on December 19, 2025.

* **Dataset:** 26 years of box office history (2000-2025) extracted from TMDB API, IMDb ratings, and theatrical performance data. Over **4,300 films** with **160+ engineered features** after preprocessing.

* **README**: A detailed README.md file dedicated only for this project can be found [here](project_02_predicting_box_office_proceeds/README.md).
* **Notebook:** The Jupyter Notebook can be found [here](project_02_predicting_box_office_proceeds/Box_Office_Proceeds_Prediction.ipynb).

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
In [this project](project_03_comparing_naive_bayes_models_with_neural_network_models), we compare the models used in [project 01](project_01_understanding_&_explaining_a_machine_learning_algorithm) (Naive Bayes Classifiers—Multinomial, Bernoulli, and Gaussian) with various neural network algorithms to see whether neural networks can outperform Naive Bayes classifiers in spam detection using the SMS Spam Collection Dataset.

* **Goal:** To compare the performance of Naive Bayes classifiers with various neural network architectures for spam detection and see whether neural networks can outperform the Naive Bayes classifiers.

* **Dataset:** SMS Spam Collection Dataset from UCI Machine Learning Repository. Link to the dataset [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

* **README**: A detailed README.md file dedicated only for this project can be found [here](project_03_comparing_naive_bayes_models_with_neural_network_models/README.md).

* **Notebook:** The Jupyter Notebook can be found [here](project_03_comparing_naive_bayes_models_with_neural_network_models/neural_network.ipynb).

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

</div>