<div style="text-align: justify;">

# Introduction

This report consists of a number of projects done for the academic course **Applied Machine Learning for Business** under the **BBA** program (Data Analytics minor) at **Institute of Business Administration (IBA), University of Dhaka**.


# Projects Overview

## Project 01: Understanding & Explaining a Machine Learning Algorithm

In [this project](Project%2001%20—%20Understanding%20&%20Explaining%20a%20Machine%20Learning%20Algorithm), we explored the three Naive Bayes Classifier algorithms—Multinomial, Bernoulli, and Gaussian—using the SMS Spam Collection Dataset. 

* **Goal:** To understand and explain the working of Naive Bayes algorithms in the context of spam detection.
* **Dataset:** SMS Spam Collection Dataset from UCI Machine Learning Repository. Link to the dataset [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
* **README**: A detailed README.md file dedicated only for this project can be found [here](Project%2001%20—%20Understanding%20&%20Explaining%20a%20Machine%20Learning%20Algorithm/README.md).
* **Notebook:** [Naive Bayes Classifiers](Project%2001%20—%20Understanding%20&%20Explaining%20a%20Machine%20Learning%20Algorithm/Naive%20Bayes%20Classifiers.ipynb)
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

</div>