# Loan Default Prediction using Classification Algorithms

## Goal: compare 5 major ML models for Classification (Logistic Regression, KNN, Random Forest (RF), Support Vector Machine (SVM), Multi-layer Perceptron (ANN), based on how well they can predict whether a given loan will default.

- We used PyCaret library to build train and test models on Peer-to-Peer (P2P) loan data from Lending Club
- To evaluate whether a given loan will default, we build classification models based on various features (characteristics of loan and borrower) from a public corpus of loans from 2007-2018 (2.2M loans granted)
- The loans can broadly be in one of 4 statuses: fully paid, charged-off, default, or active (includes 30d, 120d past due). 
- We use the loans for which the final outcome is known (paid-off, default, paid in full or charged-off) to test and train various ML models
- We then use the models to predict how many of the current loans will default and compare the predictions based on accuracy, precision,recall, F-stats and kappa

## Loan Dataset 
- Loan dataset with anonymized historical data on loans with ~150 features spread across loan types, payment data, and borrower characteristics 
- To categorize borrowers based on borrowers attributes to understand the propensity to default E.g. credit history, delinquencies, public records, borrowers are assigned risk assessment grades (A-G based on credit score, delinquencies, public records etc.) by LendingClub
- Loans are typically 3 year or 5 year terms
- Dataset consisted of ~400K records with 145 features (predictor variables) & 1 binary output for whether loan defaults (1=default, 0=Loan is OK/paid) issued between 2013-2015
- We only consider variables that can have a direct or indirect response to a borrower’s potential to default

## Environment and Libraries
- 4vCPU, 16GB RAM machine
- ML models coded in Python and run in Jupyterlab
- NumPy, Pandas, Scikit-learn, PyCaret

![Libraries](libraries.png)

## Data Preparation
- Dataset is “real-life” => noisy, and requiring review, cleanup and preparation before model building 
  - Columns that obviously had no relation to the analysis in question (E.g. Applicant ID, Employee Title etc.)
  - Variables that had missing values in observations need to be “filled” or imputed 
  - If > 90% of values are missing then variable is dropped
  - For numeric variables, impute with the mean, for categorical fill with frequency or mode
  - For columns that had identical relationships to the analysis in question (E.g. funded_amnt and funded_amnt_inv as they are always the same as loan_amt) only 1 variables chosen from each highly correlated set of variables  
- Categorical values need to be converted using “coding” i.e. create numeric value 
  - OneHot Encoding for Levels to create 0/1 dummy variables (e.g. State where borrower resides becomes 50 dummy 0/1 variables)
  - Ordinal Variables such as borrower grade (A > B> C > D > E > F > G)
  - Variables with high cardinality are handled by substituting their frequencies  
  - Converted some continuous variables to range of values to enhance interpretation of results (E.g. loan_amt, int_rate, Annual_income, credit_history_years, revol_util, total_pymnt etc.)
  
## Data Pipeline 
- Dataset loaded from csv as pandas dataframes and filtered to remove missing columns
- Pipeline run with PyCaret's setup
  - missing value imputation for categorical and numeric variables
  - coding of categorical variables
  - normalization (convert each variable to standard normal N($mu$ = 0, $sigma$=1) variables) since numeric variables are in different ranges
  - removes variables from sets that exhibit multicollinearity 
- Final Dataset: 356K loans, 30 features with 80-20 Train-test Split

## ML models, Training & Testing
- 5 major families of ML  models built & tested for best predictive capability: Logistic Regression, K-Nearest Neighbors, Random Forest, Support Vector Machines, and Articial neural netowrk (Multi-layer perceptron)
- Multiple performance criteria used to determine best predictive model:
  - Accuracy: Fraction of predictions that model got right
  - Recall: What proportion of actual loan defaults were identified by model correctly? (A model that produces no false negatives has a recall of 1.0)
  - Precision: What proportion of loan defaults identified by model were actually correct? (A model that produces no false positives has a precision of 1.0)
  - F1 Statistics: harmonic mean of Precision & Recall
  - $\Kappa$: How much better is model performing over a classifier that simply guesses at random according to the frequency of each class? (Important when classes have unequal distribution)
- We compare these values for each model and also use a confusion matrix to show misclassification/ correct classifications
#### Note: A false positive is an outcome where the model incorrectly predicts a default when there is no actual default. A false negative is an outcome where the model incorrectly predicts there is no default when the loan was actually in default


## Results

| Model	| Accuracy	| AUC	| Recall	| Precsion | 	F1	| Kappa |
| ----- | --------  | ---- |------  | -------- | ----- | ------ |
| Logistic Regression	| 0.9530	| 0.9298	| 0.7305	| 1.0000	| 0.8442 |	0.8174|
| K Neighbors Classifier	| 0.8774 | 	0 | 0.3560	| 0.8566 |	0.5029 |	0.4463 |
| SVM - Linear Kernel	| 0.9572 |	0 |	0.7546 |	1.0000 |	0.8602	| 0.8355 |
| MLP Classifier |	0.9492 |	0.9295 |	0.7093 |	0.9989 |	0.8295 |	0.8007 |
| Random Forest Classifier |	0.9671	| 0.9495 |	0.8110	| 1.0000	| 0.8956	| 0.8763 |


- Random Forest performs best: 
  - Highest Accuracy
  - Perfect precision (no False positives!)
  - Lowest False Negative rate
  - Best F1 Score and Kappa
- Other models tried as experiments: Naive Bayes, AbaBoost, QDA, LDA, XGBoost, LightGBM, CatBoost

  
## Key Insights
- Data cleaning and preprocessing is critical for building and tuning machine learning models in real-life 
- All 5 ML models performed well without overfitting except for KNN 
  - RF models select a subset of features in each of its decision trees so have less bias 
  - MLP can distinguish data that is not linearly separable due to multiple layers and non-linear activation functions
  - SVM models are stable and can generalize well since once a hyperplane is found, small changes to data cannot greatly affect the hyperplane 
  - LR model outputs have a nice probabilistic interpretation, can be regularized to avoid overfitting
- 10-fold Cross-validation to choose the best hyper-parameters generally resulted in a rise in accuracy 
- For Loan default prediction, False Negatives Rate is critical to evaluate the model performance as FNs can lead to negative impact on investors and pose challenges to credibility of the lender







