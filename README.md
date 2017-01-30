## Objective

Identify features to use with machine learning in order to predict persons of interest in a scandal.

###Skills applied

    - Python
    - Sklearn:
      - KNeighbors
      - AdaBoost
      - GaussianNB
      - DecisionTree
      - RandomForest

## Summary

I explored an Enron email dataset to discover patterns. I used the following features to help predict persons of interest:
  
    -Salary
    -Total Payments
    -Exercised Stock Options
    -Bonus
    -Total Stock Value
    -Expenses
    -Loan Advances
    -Deferred Income
    -Long Term Incentive

I also created a new feature, To Ratio, which compares the number of emails an indiviudal sent to a person of interest over total emails sent.

## Findings

I used AdaBoost classifier with parameter n_estimators = 100 for optimal results.

  Without added feauture:
  
    -Precision = 0.4888
    
    -Recall = 0.349
  
  With added feature:
  
    -Precision = 0.5325
    
    -Recall = 0.3605

## Resources

    1) http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
   
   
