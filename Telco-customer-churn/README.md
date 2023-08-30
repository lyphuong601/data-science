# Telco Customer Churn Analysis

<p align="center"><img src="img/Telco customer.png" height="300" width="400"></p>

## 📌 Introduction
- Telco is a telecomunication company. To grow its business, it priortizes resources on keeping existing customers happy and reducing churn rate. The company is currently looking for a model to predict customer churn so as to put them on its customer retention program.
- Motivation: Use data available, predict whether a customer will churn or not

## Data Overview:
- Dependent variable: churn (0 or 1) to denote whether customers left within last month
- Independent variables: services that each customer has signed up for, customer account information and basic demographics characteristics.

## Technology Used

<ul>
  <li>Merge</li>
  <li>Handle imbalanced dataset</li>
  <li>Classification models</li>
  <li>Grid search CV, hyperparameter tuning</li>
  <li>ROC AUC, AUC</li>
</ul>

## Contents

<h3>1. Merge dataset</h3>
<h3>2. Data descriptions</h3>
<h3>3. Oversampling minority class</h3>
<h3>4. Model training and performance</h3>
  Base model (Logistics Regression):
  <ul>
    <li>Test set F1 score: 0.591</li> 
    <li>Test set accuracy score: 0.539</li>
  </ul>
  Model selection: 
  <p align="center"><img src="img/model_comparison.png" height="300" width="550"></p>

  I choose GradientBoostingClassifier as the final model and use it to tune the hyperparameter because it gives the best performance

  Final tuned model (GradientBoostingClassifier):
  <ul>
    <li>Test set F1 score: 0.645</li> 
    <li>Test set accuracy score: 0.847</li>
  </ul>

## Conclusion

- Sucessfully predict 64.5% of customer churning
- Final tuned model improved overall model performance by 9.1% in F1 Score and by 57.14% in Accuarcy score, compared to base model

## Projects Completed

1. <a href="https://github.com/lyphuong601/job-postings-data-cleaning">Job Posting Data Cleaning</a>
2. <a href="https://github.com/lyphuong601/data-science/tree/main/linear-regression-BGD-deployment">House Price Predictions</a>
3. <a href="https://github.com/lyphuong601/adventuework-inc-da-project"> Adventuework Inc DA Project</a>

More projects coming up soon. Do drop a ⭐ if you like it.