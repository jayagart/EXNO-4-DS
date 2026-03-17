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

# RESULT:
Thus,the Feature Scaling and selection Executed successfully.
