# Bias Detection and Fairness Mitigation in Epidemiological Machine Learning Models

## Project Overview

Machine learning models are increasingly used in healthcare and epidemiology to support disease prediction and risk assessment. However, predictive performance alone does not guarantee that these models produce equitable outcomes across different population groups.

This project explores the detection and mitigation of gender-related prediction disparities in a machine learning model trained on diabetes-related demographic and clinical data.

Using Python and Fairlearn, I developed a logistic regression classification model and investigated whether its predictions differed between male and female patients. I subsequently applied fairness-constrained learning techniques to examine the extent to which demographic parity disparities could be reduced.

The project demonstrates the practical application of responsible AI principles in epidemiological modelling and highlights the importance of evaluating algorithmic fairness alongside predictive performance.

## Research Objectives

The project aimed to:

- Develop a machine learning model for diabetes classification using demographic and health-related variables.
- Explore the distribution of diabetes classifications across gender groups.
- Identify potential gender-related disparities in model predictions.
- Quantify prediction disparities using Demographic Parity Difference (DPD).
- Apply fairness mitigation techniques using the Fairlearn library.
- Compare fairness metrics before and after mitigation.
- Explore model performance through cross-validation and hyperparameter tuning.

## Dataset

The analysis uses the **Early Stage Diabetes Risk Prediction Dataset**, obtained from the UCI Machine Learning Repository.

Dataset characteristics:

| Attribute | Description |
|---|---|
| Observations | 520 |
| Variables | 17 |
| Target variable | Diabetes classification |
| Sensitive attribute | Gender |
| Target classes | Positive and Negative |
| Missing values | None identified |
| Duplicate observations | 269 identified |

The dataset contains demographic information and clinical indicators, including age, gender, polyuria, polydipsia, sudden weight loss, weakness, polyphagia, visual blurring and other diabetes-associated symptoms.

**Dataset source:** https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset

## Technologies and Libraries

The project was implemented in Python using:

- **Pandas:** Data manipulation and exploratory analysis.
- **NumPy:** Numerical operations.
- **Scikit-learn:** Logistic regression, preprocessing, cross-validation and hyperparameter tuning.
- **Fairlearn:** Fairness assessment and bias mitigation.
- **Matplotlib:** Data visualisation.
- **Seaborn:** Exploratory and statistical visualisation.

## Methodology

### 1. Exploratory Data Analysis

Initial analysis examined the dataset structure, variable distributions, missing values, duplicate observations and relationships between demographic and clinical characteristics.

Visualisations were developed to explore:

- Gender distribution across diabetes classifications.
- Clinical indicators in relation to diabetes classification.
- Age distribution.
- Relationships between selected features.
- Correlations between encoded variables.

### 2. Feature Engineering

Categorical variables were converted into numerical representations using one-hot encoding.

The target variable was transformed into a binary classification outcome:

- Positive = 1
- Negative = 0

Gender was retained as the sensitive attribute for fairness evaluation.

### 3. Model Development

A Logistic Regression classifier was implemented as the baseline predictive model.

The model was subsequently evaluated using five-fold cross-validation, with feature scaling incorporated into the modelling pipeline.

### 4. Fairness Assessment

Fairness was evaluated using the **Demographic Parity Difference (DPD)** metric from Fairlearn.

DPD measures the difference in positive prediction rates between demographic groups.

A lower absolute DPD indicates a smaller disparity in positive prediction rates.

### 5. Fairness Mitigation

Fairness-constrained learning was explored using Fairlearn's reduction algorithms:

- Exponentiated Gradient with Demographic Parity constraints.
- Fairlearn GridSearch with Demographic Parity constraints.

The objective was to investigate whether fairness constraints could reduce disparities in model predictions.

### 6. Model Evaluation and Hyperparameter Tuning

Five-fold cross-validation was used to assess classification accuracy.

GridSearchCV was also applied to explore different regularisation strengths for Logistic Regression.

## Results and Key Findings

### Predictive Performance

The initial cross-validation produced the following results:

| Evaluation | Result |
|---|---:|
| Mean cross-validated accuracy | 92.50% |
| Best tuned cross-validation accuracy | 93.08% |
| Best regularisation parameter (C) | 0.1 |

### Fairness Assessment

The initial baseline model demonstrated a Demographic Parity Difference of 0.50.

Following fairness-constrained learning with Exponentiated Gradient, the reported difference decreased to 0.02.

| Model | Demographic Parity Difference |
|---|---:|
| Baseline Logistic Regression | 0.50 |
| Exponentiated Gradient mitigation | 0.02 |
| Fairlearn GridSearch mitigation | 0.26 |

These results illustrate that fairness constraints can substantially change the distribution of positive predictions across demographic groups.

However, demographic parity alone does not establish overall model fairness or clinical suitability.

## Key Takeaways

This project highlights several important considerations in healthcare machine learning:

1. High predictive accuracy does not necessarily imply equitable model behaviour.
2. Demographic characteristics can be important dimensions of algorithmic fairness assessment.
3. Fairness metrics provide additional information beyond conventional predictive performance measures.
4. Fairness-constrained learning offers practical approaches to investigating and reducing demographic disparities.
5. Responsible healthcare AI requires consideration of both predictive performance and potential differences in outcomes across population groups.

## Limitations and Future Improvements

This notebook represents an exploratory implementation of fairness assessment and mitigation.

Further work should address:

- Removing duplicate observations and assessing their influence on model performance.
- Implementing stratified train-test splitting and evaluating models on independent test data.
- Preventing data leakage during model development and fairness assessment.
- Comparing additional fairness metrics, including Equalized Odds and demographic-group-specific error rates.
- Evaluating sensitivity, specificity, precision, recall and calibration across gender groups.
- Assessing the predictive performance and fairness trade-offs introduced by mitigation.
- Investigating intersectional fairness across multiple demographic characteristics.

The reported fairness metrics should therefore be interpreted as exploratory results rather than evidence of validated clinical fairness.

## Repository Contents

| File | Description |
|---|---|
| `Bias_in_Epidemiology.ipynb` | Jupyter Notebook containing the analysis, modelling, fairness evaluation and visualisations. |
| `README.md` | Project documentation and methodology. |

## Reproducibility

To run this project:

1. Clone the repository.
2. Install the required Python libraries.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Execute the notebook cells sequentially.

Install the dependencies:

```bash
pip install pandas numpy scikit-learn fairlearn matplotlib seaborn jupyter
```

The dataset is retrieved directly from the UCI Machine Learning Repository within the notebook.

## Conclusion

This project provided an opportunity to explore the intersection of epidemiology, machine learning and responsible artificial intelligence.

By combining predictive modelling with fairness assessment and mitigation, the analysis demonstrates why evaluating algorithmic disparities is an important consideration when developing machine learning approaches for healthcare applications.

**Research interests:** Health Data Science | Epidemiological Modelling | Responsible AI | Machine Learning in Healthcare | Algorithmic Fairness

---

*Developed independently by Anne Amina Freitas.*
