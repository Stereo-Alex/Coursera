
# Week 2 Python Assessment

This Jupyter Notebook is auxillary to the following assessment in this week.  To complete this assessment, you will complete the 7 questions outlined in this document and use the output from your python cells as answers.

Your goal of this assignment is to construct regression and logistics models and interpret model paramters.

Run the following cell to initialize your environment and begin the assessment.


```python
#### RUN THIS

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import statsmodels.api as sm
import pandas as pd  

from sklearn.datasets import load_boston
boston_dataset = load_boston() 

boston = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
boston["MEDV"] = boston_dataset.target

url = "nhanes_2015_2016.csv"
NHANES = pd.read_csv(url)
vars = ["BPXSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "BMXBMI", "SMQ020"]
NHANES = NHANES[vars].dropna()
NHANES["smq"] = NHANES.SMQ020.replace({2: 0, 7: np.nan, 9: np.nan})
NHANES["RIAGENDRx"] = NHANES.RIAGENDR.replace({1: "Male", 2: "Female"})
NHANES["DMDEDUC2x"] = NHANES.DMDEDUC2.replace({1: "lt9", 2: "x9_11", 3: "HS", 4: "SomeCollege",5: "College", 7: np.nan, 9: np.nan})

np.random.seed(123)
```

Now that your notebook is ready, begin answering the questions below.

### Questions 1-3

The first three questions will be utilizing the Boston housing dataset seen in week 1. 

Here is the description for each column:

* __CRIM:__ Per capita crime rate by town
* __ZN:__ Proportion of residential land zoned for lots over 25,000 sq. ft
* __INDUS:__ Proportion of non-retail business acres per town
* __CHAS:__ Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* __NOX:__ Nitric oxide concentration (parts per 10 million)
* __RM:__ Average number of rooms per dwelling
* __AGE:__ Proportion of owner-occupied units built prior to 1940
* __DIS:__ Weighted distances to five Boston employment centers
* __RAD:__ Index of accessibility to radial highways
* __TAX:__ Full-value property tax rate per $\$10,000$
* __PTRATIO:__ Pupil-teacher ratio by town
* __B:__ $1000(Bk ??? 0.63)^2$, where Bk is the proportion of [people of African American descent] by town
* __LSTAT:__ Percentage of lower status of the population
* __MEDV:__ Median value of owner-occupied homes in $\$1000$s

Uncomment and run the following code to generate a simple linear regression and output the model summary:


```python
#model = sm.OLS.from_formula("MEDV ~ RM + CRIM", data=boston)
#result = model.fit()
#result.summary()
```

Utilizing the above output, answer the following three questions:

#### Question 1 (You'll answer this question within the quiz that follows this notebook)

What is the value of the coefficient for predictor __RM__?

#### Question 2 (You'll answer this question within the quiz that follows this notebook)

Are the predictors for this model statistically significant, yes or no? (Hint: What are their p-values?)

Run the following code for question 3:


```python
## For Question 3
model = sm.OLS.from_formula("MEDV ~ RM + CRIM + LSTAT", data=boston)
result = model.fit()
result.summary()
```

#### Question 3 (You'll answer this question within the quiz that follows this notebook)

What happened to our R-Squared value when we added the third predictor __LSTAT__ to our initial model?
  

#### Question 4 (You'll answer this question within the quiz that follows this notebook)

What type of model should we use when our target outcome, or dependent variable is continuous?

### Questions 5-6

The next two questions will involve the NHANES dataset.

Uncomment and run the following code to generate a logistics regression and output the model summary:


```python
#model = sm.GLM.from_formula("smq ~ RIAGENDRx + RIDAGEYR + DMDEDUC2x", family=sm.families.Binomial(), data=NHANES)
#result = model.fit()
#result.summary()
```

#### Question 5 (You'll answer this question within the quiz that follows this notebook)

Which of our predictors has the largest coefficient?


#### Question 6 (You'll answer this question within the quiz that follows this notebook)

Which values for DMDEDUC2x and RIAGENDRx are represented in our intercept, or what is our reference level?


#### Question 7 (You'll answer this question within the quiz that follows this notebook)

What model should we use when our target outcome, or dependent variable is binary, or only has two outputs, 0 and 1.

