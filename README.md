# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:


```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/bmi.csv")

df.head()
```
<img width="258" height="193" alt="image" src="https://github.com/user-attachments/assets/a6a58b3d-ee8e-4bbe-9b86-d4b1486cc0ad" />

```
df.dropna()
```
<img width="289" height="389" alt="image" src="https://github.com/user-attachments/assets/6f513d6d-7dbe-48f4-8a41-ffa9aad07e15" />

```
max_vals = np.max(np.abs(df[['Height','Weight' ]]))
max_vals
```
<img width="88" height="36" alt="image" src="https://github.com/user-attachments/assets/38914fdc-417b-4fbd-869a-6d50e4c098e6" />

```
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df[['Height','Weight']] = sc.fit_transform(df[['Height','Weight']])

df.head(10)
```

<img width="304" height="334" alt="image" src="https://github.com/user-attachments/assets/b1b75541-c791-49f0-93a2-033917a12d91" />

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])

df.head(10)
```

<img width="303" height="344" alt="image" src="https://github.com/user-attachments/assets/369134dd-b9e2-4dfd-ad2a-3f6be2fceb47" />

```
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df[['Height','Weight' ]]=scaler.fit_transform(df[['Height','Weight']])

df
```
<img width="306" height="399" alt="image" src="https://github.com/user-attachments/assets/82b4121a-958c-408e-9e5c-24a25a43ee30" />



```
df4=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4[['Height', 'Weight' ]]= scaler.fit_transform(df4[['Height','Weight' ]])
df4.head()
```
<img width="298" height="193" alt="image" src="https://github.com/user-attachments/assets/3bf81144-dfe5-4f7f-b62d-c6ffed9326ca" />


```
df3=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3[['Height','Weight' ]]=scaler.fit_transform(df3[['Height','Weight' ]])
df3
```

<img width="308" height="389" alt="image" src="https://github.com/user-attachments/assets/50cca165-676d-4ad8-87a9-82531c53a81c" />

```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset (1).csv")

df.columns
```

<img width="569" height="84" alt="image" src="https://github.com/user-attachments/assets/27910358-fc2d-4a2c-bccf-960864b23b31" />



```
df.shape
```
<img width="188" height="36" alt="image" src="https://github.com/user-attachments/assets/258b79c8-5825-4a01-855a-5217428652f4" />

```
X = df.drop("Survived", axis=1)
y = df['Survived']

df1=df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)

df1.columns
```

<img width="758" height="37" alt="image" src="https://github.com/user-attachments/assets/9708b614-d11a-4761-8a1c-d74e02b35651" />


```
df1['Age'].isnull().sum()
```

<img width="159" height="45" alt="image" src="https://github.com/user-attachments/assets/0fa94045-ce52-4471-8eff-f9652eef126b" />


```
df1['Age'].fillna(method='ffill')
```
<img width="271" height="468" alt="image" src="https://github.com/user-attachments/assets/cb1d922b-e156-42d2-8e53-ab7f21f212d6" />

```
df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()
```
<img width="789" height="59" alt="image" src="https://github.com/user-attachments/assets/a6136170-42bc-425e-83f9-cb0c4836c776" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('/content/titanic_dataset (1).csv')

data=data.dropna()

data=data.dropna()

# Separate the features and target variable
X = data.drop(['Survived','Name','Ticket'], axis=1)
y = data['Survived']
X
```

<img width="647" height="386" alt="image" src="https://github.com/user-attachments/assets/5000dbc9-d190-47d3-b976-11c512f860fa" />

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
```
```
data["sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes
```
```
data
```
<img width="798" height="512" alt="image" src="https://github.com/user-attachments/assets/184ac0cc-f2f0-44ec-81bc-c6dd21e501f4" />
```
k = 5

# Re-create X and y based on the numerically encoded data
# Select only the numerical features after encoding
numerical_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'sex']
X = data[numerical_features]
y = data['Survived']

selector = SelectKBest(score_func=chi2, k=k)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="609" height="42" alt="image" src="https://github.com/user-attachments/assets/7071a28f-69e7-45e9-b5fe-bf6e7df75f6a" />

```
X.info()
```
<img width="340" height="254" alt="image" src="https://github.com/user-attachments/assets/f8e65e3f-7af4-4e36-b50a-93cdd66dc301" />

```
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)


# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="488" height="38" alt="image" src="https://github.com/user-attachments/assets/c80b7399-f476-4bb6-a8b0-caf54cc47b7a" />

```
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="580" height="42" alt="image" src="https://github.com/user-attachments/assets/e5eb2595-d824-48ea-bec5-ea9e2916b2b8" />

```
from sklearn.feature_selection import SelectPercentile, chi2

selector = SelectPercentile(score_func=chi2, percentile=10) # 10% of the features
X_new = selector.fit_transform(X, y)
```

```
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier (you can use any other model)
model = RandomForestClassifier()

# Initialize the SelectFromModel with the model and threshold
sfm = SelectFromModel(model, threshold='mean' )

# Fit the SelectFromModel to your data
sfm.fit(X, y)

# Get the selected features
selected_features = X.columns[sfm.get_support()]

# Print the selected features
print("Selected Features:")
print(selected_features)
```
<img width="483" height="46" alt="image" src="https://github.com/user-attachments/assets/a024184e-9830-4cc4-8288-fbc4b1535296" />

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to your data
model.fit(X, y)

# Get the feature importances
feature_importances = model.feature_importances_

# Set a threshold for feature importance
threshold = 0.15 # Adjust the threshold as needed

# Get the selected features
selected_features = X.columns[feature_importances > threshold]

# Print the selected features
print("selected Features:")
print(selected_features)
```
<img width="488" height="53" alt="image" src="https://github.com/user-attachments/assets/659a31ab-6368-4c59-b14d-da53e9c9857b" />

```
df=pd.read_csv('/content/titanic_dataset (1).csv')

df.columns
```

<img width="547" height="62" alt="image" src="https://github.com/user-attachments/assets/e31c19a5-059a-4ec5-b73d-d1b17ef51b60" />

```
df
```
<img width="788" height="471" alt="image" src="https://github.com/user-attachments/assets/30ad2550-ff51-4cd1-bc17-34ae0c05f13c" />

```
df=df.dropna()

df.isnull().sum()
```
<img width="136" height="379" alt="image" src="https://github.com/user-attachments/assets/1b887e5a-1394-4fdf-9177-e91a0d4b7854" />

```
# Separate features and target
X = df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare' ]]
y = df ['Survived' ]

# SelectKBest with mutual_info_classif for feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected features
selected_features = X.columns[selected_feature_indices]
print("selected Features:")
print (selected_features)

```

<img width="446" height="46" alt="image" src="https://github.com/user-attachments/assets/c2780859-9658-4152-ba2c-55c45b6759c3" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the 'tips' dataset from seaborn
import seaborn as sns
tips = sns.load_dataset('tips')

# Display the first few rows of the dataset
tips.head()
```

<img width="391" height="177" alt="image" src="https://github.com/user-attachments/assets/3b8e9559-a58d-44f5-abed-d5ad0aacc925" />

```
contingency_table = pd.crosstab(tips['sex'], tips['time'])
print(contingency_table)
```
<img width="196" height="80" alt="image" src="https://github.com/user-attachments/assets/a2cb89dc-5275-4e5b-b1c2-a11857eb3cc3" />

```
chi2, p, dof, expected_freq = chi2_contingency(contingency_table)
# Display the results
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
```
<img width="357" height="47" alt="image" src="https://github.com/user-attachments/assets/909b5afb-bf8a-463e-a41a-5b54c284684b" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# Create a sample dataset
data = {
'Feature1': [1, 2, 3, 4, 5],
'Feature2': ['A', 'B', 'C', 'A', 'B'],
'Feature3': [0, 1, 1, 0, 1],
'Target': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[['Feature1', 'Feature3' ]]
y = df ['Target' ]

# SelectKBest with mutual_info_classif for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected features
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="370" height="55" alt="image" src="https://github.com/user-attachments/assets/0881880b-e286-4f7d-8138-58d44409b080" />

# RESULT:
Thus,the Feature Scaling and selection Executed successfully.
