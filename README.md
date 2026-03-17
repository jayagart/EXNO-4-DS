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
<img width="176" height="28" alt="image" src="https://github.com/user-attachments/assets/095f3285-4eab-4f9c-a9bd-57ba74a015fa" />

```
X = df.drop("Survived", axis=1)
y = df['Survived']

df1=df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)

df1.columns
```
<img width="663" height="30" alt="image" src="https://github.com/user-attachments/assets/6567de0e-f689-4d78-90fc-80da0d796f0e" />

```
df1['Age'].isnull().sum()
```

<img width="141" height="35" alt="image" src="https://github.com/user-attachments/assets/febf6ed9-07e3-41c7-b321-8150c619e5fc" />

```
df1['Age'].fillna(method='ffill')

```

<img width="333" height="402" alt="image" src="https://github.com/user-attachments/assets/7c1590cd-30f9-477c-a452-a0118aad5038" />

```
df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()
```

<img width="767" height="50" alt="image" src="https://github.com/user-attachments/assets/330a8d99-a2da-427c-87a9-e370600ecfeb" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('/content/titanic_dataset (1).csv')

data=data.dropna()
```
```
data=data.dropna()

# Separate the features and target variable
X = data.drop(['Survived','Name','Ticket'], axis=1)
y = data['Survived']
X
```

<img width="575" height="350" alt="image" src="https://github.com/user-attachments/assets/71766178-557e-4e0e-9259-2422c14971bd" />

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data
```
<img width="794" height="511" alt="image" src="https://github.com/user-attachments/assets/60fe86d6-555a-4994-868b-ee5b332a0347" />

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
<img width="468" height="46" alt="image" src="https://github.com/user-attachments/assets/c67bda2a-e7dc-4a6c-8637-cff5b68aeb8c" />

```
X.info()
```
<img width="327" height="253" alt="image" src="https://github.com/user-attachments/assets/4eaa6692-761b-4846-90af-a219e297186f" />

```
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)
```
```
# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="510" height="59" alt="image" src="https://github.com/user-attachments/assets/b7a9df77-7de4-4dbc-bdf1-5b56a1025a7e" />

```
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_new = selector.fit_transform(X, y)
```
```
# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="503" height="43" alt="image" src="https://github.com/user-attachments/assets/b44d59f1-e4d5-4f18-8aae-f7988412c3cb" />

```
from sklearn.feature_selection import SelectPercentile, chi2

selector = SelectPercentile(score_func=chi2, percentile=10) # 10% of the features
X_new = selector.fit_transform(X, y)
```
```
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
```

```
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

<img width="556" height="55" alt="image" src="https://github.com/user-attachments/assets/57676fda-46e7-4dbe-a926-86750f67ae59" />

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
<img width="554" height="43" alt="image" src="https://github.com/user-attachments/assets/2c8094a0-a271-4294-956b-e104a630ba42" />

```
df=pd.read_csv('/content/titanic_dataset (1).csv')

df.columns
```

<img width="554" height="62" alt="image" src="https://github.com/user-attachments/assets/56074191-264c-4db0-a4ec-f615651c39cd" />

```
df
```
<img width="798" height="467" alt="image" src="https://github.com/user-attachments/assets/d070e89b-2b15-4b95-8b14-ba3586933dce" />

```
df=df.dropna()
```
```
df.isnull().sum()
```

<img width="139" height="383" alt="image" src="https://github.com/user-attachments/assets/749347b8-1721-4a7e-9feb-1b3e7ac9636c" />

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

<img width="479" height="51" alt="image" src="https://github.com/user-attachments/assets/fb3d1141-c2a5-46bd-bd8a-efddda4d7185" />

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

<img width="384" height="179" alt="image" src="https://github.com/user-attachments/assets/2532f909-f647-4ca5-9b2c-c0eef2936856" />

```
contingency_table = pd.crosstab(tips['sex'], tips['time'])
print(contingency_table)
```

<img width="173" height="90" alt="image" src="https://github.com/user-attachments/assets/a90202e3-5e10-419e-b234-522db78da855" />

```
chi2, p, dof, expected_freq = chi2_contingency(contingency_table)
# Display the results
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
```

<img width="287" height="42" alt="image" src="https://github.com/user-attachments/assets/d44c6af3-cc9e-49c7-8bc7-86fd8a6d35e5" />

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

<img width="294" height="58" alt="image" src="https://github.com/user-attachments/assets/d97b3131-d7a6-4b01-a87b-54ec11e120dd" />


# RESULT:
Thus,the Feature Scaling and selection Executed successfully.
