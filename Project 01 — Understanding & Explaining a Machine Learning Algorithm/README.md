<div style="text-align: justify;">

# Executive Summary

This project explores and compares three Naive Bayes classifiers—**Multinomial**, **Bernoulli**, and **Gaussian**—using the **SMS Spam Collection** Dataset from the UCI Machine Learning Repository. The goal was to classify text messages as spam or ham (non-spam) through preprocessing, encoding, and applying probabilistic machine learning models.

The dataset contained roughly **87% ham** and **13% spam**, showing a natural imbalance. The analysis focused on model performance, overfitting checks, and generalization ability through metrics like **accuracy, precision, recall, F1-score**, and **ROC-AUC**.

![Ham vs. Spam Distribution](Assets/ham_vs_spam.png)
***Figure 01:** Ham vs. Spam Distribution.*


# Business Problem
With the rising volume of spam in digital communication, automated detection is critical for both user safety and system efficiency. The business problem addressed is:  
> *Can we accurately classify SMS messages as spam or ham using probabilistic learning methods while balancing false positives (legitimate messages flagged as spam) and false negatives (spam messages passing through)?*

This study evaluates three Naive Bayes variants to determine which provides the best trade-off between accuracy, recall, and precision for real-world spam filtering applications.

# Methodology

## Data Preparation
As can be seen from the pie chart above, the dataset is imbalanced with a higher proportion of ham messages compared to spam messages. So, we will preserve this imbalance in our training and testing datasets with the code `stratify = sms["label_encoding"]`.

The **three** naive bayes models require different types of encoders:
1. **Multinomial Naive Bayes**: `CountVectorizer()`
2. **Bernoulli Naive Bayes**: `CountVectorizer(binary = True)`
3. **Gaussian Naive Bayes**: `TfidfVectorizer()`

## Model Training & Performance Evaluation
Now, we import the necessary libraries from scikit-learn, build the model objects, fit the train data, and predict the test data. Then, we evaluate the models using a bunch of metrics. Some of them are: accuracy, precision, recall, F1-score, and ROC-AUC score.

### Accuracy & Classification Report
A summary of the accuracy scores along with their classification reports (precision, recall, F1-score) of the three models on the test dataset are as follows:

***Table 01:** Model Performance Comparison of Spam (Class 1).*
| *Model* | Accuracy | Precision | Recall | F1-Score |
|:------|:----------|:-----------|:--------|:----------|
| ***MultinomialNB*** | 98.68% | 0.97 | 0.93 | 0.95 |
| ***BernoulliNB*** | 97.19% | 0.99 | 0.79 | 0.88 |
| ***GaussianNB*** | 87.74% | 0.52 | 0.92 | 0.67 |

#### Key Insights
For detecting spam messages (class 1), **Multinomial Naive Bayes** achieved the best balance with precision = 0.97, recall = 0.93, and F1 = 0.95, correctly identifying most spam messages with very few false alarms. **Bernoulli Naive Bayes** was ranked second, and **Gaussian Naive Bayes** performed the worst.

### Confusion Matrices
We generated the three confusion matrices for the models and displayed them side-by-side for an easier comparison.

![Confusion Matrices](Assets/confusion_matrices.png)
***Figure 02:** Confusion Matrices.*

#### Key Insights
Our goal is to minimize either false positives or false negatives, depending on what matters more for the application.

* If we want to avoid flagging legitimate (ham) messages as spam, the best model is **Bernoulli Naive Bayes**, which produced only 1 false positive out of 1448. This is ideal when missing important emails (like business or bank messages) is unacceptable.

* If we want to avoid letting spam slip through, the best model is **Multinomial Naive Bayes**, with only 15 false negatives out of 224. This model suits scenarios where security and spam filtering are top priorities.

* For a balanced trade-off, **Multinomial Naive Bayes** again performs best, catching almost all spam messages (209 true positives) while keeping both error types low.


### ROC Curves & ROC-AUC Scores
We also plotted the ROC curves of all three models together for a visual comparison. The ROC-AUC scores are also mentioned in the legend.

![ROC Curves](Assets/roc_curves.png)
***Figure 03:** ROC Curves & ROC-AUC Scores.*

#### Key Insights
The ROC curves show that **Multinomial Naive Bayes** and **Bernoulli Naive Bayes** both achieved excellent discrimination ability between spam and ham messages, with AUC scores of 99.066% and 99.128% respectively. They can be considered almost equal if we account for the decimal places. In contrast, **Gaussian Naive Bayes** performed notably worse, with an AUC of 89.53%, showing that it is less effective at distinguishing between the two classes.

### Overfitting Check
Before finally selecting on a model, we want to check whether any of the models have any overfitting issues. Usually one of the telltale signs of overfitting is **a large difference between the training and testing accuracies of a model**. If a model has a very high training accuracy but a significantly lower testing accuracy, it may be overfitting the training data.

```
Train & Test Accuracy and their Differences

For Multinomial Naive Bayes
Train Accuracy: 99.36%
Test Accuracy: 98.68%%
Difference in accuracy: 0.67%

For Bernoulli Naive Bayes
Train Accuracy: 98.51%
Test Accuracy: 97.19%
Difference in accuracy: 1.32%

For Gaussian Naive Bayes
Train Accuracy: 93.77%
Test Accuracy: 87.74%
Difference in accuracy: 6.03%
```

#### Key Insights
The train–test comparison shows that **Multinomial Naive Bayes** generalizes best, with nearly identical training (99.36%) and testing (98.68%) accuracies—a minimal 0.67% gap. **Bernoulli Naive Bayes** also performs consistently, with only a slightly higher gap (1.32%). In contrast, **Gaussian Naive Bayes** shows a large 6.03% gap, indicating overfitting and poor generalization to unseen data. Overall, **MultinomialNB** is the most reliable for spam detection, while **GaussianNB** struggles due to its mismatch with text-based features.


### Cross-Validation Check

Now, we want to have another check on how our models perform on new and unseen data. Another method of testing this is through cross-validation. For this dataset, we perform a **k-Fold Cross Validation**, For our use case, we use a `k = 5`, i.e., a 5-Fold Cross Validation.

```
Cross Validation Results of the Three Models

Multinomial Naive Bayes: 97.59% ± 0.53%
Bernoulli Naive Bayes: 97.08% ± 0.38%
Gaussian Naive Bayes: 87.05% ± 1.00%
```

![test](Assets/cross_val.png)
***Figure 04:** Cross-Validation Results.*

#### Key Insights
Among the three, **Multinomial Naive Bayes** performed best, followed closely by **Bernoulli Naive Bayes**, while **Gaussian Naive Bayes** lagged behind. Multinomial and Bernoulli models showed strong generalization and consistency across folds, confirming no overfitting. GaussianNB underperformed because its assumption of continuous, normally distributed features doesn’t fit text-based data.

To check their performance across the folds, we can visualize the results using a heatmap.

![Heatmap](Assets/heatmap.png)
***Figure 05:** Cross-Validation Heatmap.*

### Best Model Selection
Finally, we want to compare the performance of the three models across **three** specific metrics: Accuracy, F1-Score, and ROC-AUC Score. We will create a bar plot for each metric to visually compare the models.

![Final Model Comparison](Assets/model_comparison.png)
***Figure 06:** Final Model Comparison.*

#### Key Insights
As we can see from the barchart above, **Multinomial Naive Bayes** scored the highest in Accuracy and F1 Score (Spam) whereas **Bernoulli Naive Bayes** scored the highest in the ROC-AUC Score with MultinomialNB trailing very closely behind. **Gaussian Naive Bayes** scored the lowest in all three metrics. 

Comparing from this chart and all the above insights, we can conclude that **Multinomial Naive Bayes** is the best model among the three.

# Conclusion

So, from the above comparison(s), graphs, and confusion matrixes, we can see that **Multinomial Naive Bayes** emerged as the most effective spam detector, combining high accuracy (98.68%) with consistent generalization across folds. Its probabilistic handling of word frequency data makes it the optimal choice for text classification tasks of this nature.

It makes the perfect sense since it accounts for how many times one word appears in a text. Gaussian Naive Bayes performed poorly because this dataset was a text-based one and GNB performs the best in numeric features. Although the dataset was encoded using TF-IDF, this model was not the best one for this dataset.

**Performance of MNB**:

| Metric | Value |
|:-------|:------|
| Accuracy |  98.68%|
| Precision |  97%|
| Recall |  93%|
| F1-score |  95%|
| False Positive Rate |  0.48%|
| False Negative Rate |  6.70%|
| ROC-AUC |  99.07%|

So, to rank the performance of the **three** models:
1. Multinomial Naive Bayes
2. Bernoulli Naive Bayes
3. Gaussian Naive Bayes

</div>