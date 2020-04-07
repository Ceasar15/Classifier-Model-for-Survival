import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
import statsmodels
import sklearn
import scipy

train = pd.read_csv('../Educba/train.csv')
test = pd.read_csv('../Educba/test.csv')

train.Survived.value_counts()

train.head()
test.head()

train.info()

train.isna().sum()

train.loc[train.Age.isna(), 'Age'] = train[-train.Age.isna()].Age.mean()
train.loc[train.Cabin.isna(),'Cabin'] = 'No Cabin'

print(train.Embarked.value_counts())
train.loc[train.Embarked.isna(), 'Embark'] = 'S'

train.isna().sum()

fig, axes = plt.subplots(1, 2, figsize=(25,8))
print(axes)

sns.boxplot(x='Pclass', y='Age', data=train, ax=axes[0])

sns.boxplot(x='Pclass', y='Fare', data=train, ax=axes[1])

plt.show()

train.loc[train.Fare > 200]


numerical_columm = ['int64', 'float64']

plt.figure(figsize=(10,10))
sns.heatmap( train.select_dtypes(include=numerical_columm).corr(),
 cmap=plt.cm.RdBu,
 vmax=1.0,
 linewidths=0.1,
 linecolor = 'blue',
 square= True,
 annot= True 
  )

#plt.figure(figsize=(10,10))
#sns.pairplot(train.select_dtypes(include=numerical_columm), hue='Survived')

train.Sex.value_counts()

from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
train['Sex'] = labelencoder.fit_transform(train['Sex'])
train.Sex.value_counts()
palette = {1:'g', 0:'r'}
sns.countplot(x='Sex', data=train, hue = 'Survived', palette=palette)


def feature_engineering(df):
    df.loc[df.Age.isna(), 'Age'] = df[-df.Age.isna()].Age.mean()
    df.loc[df.Cabin.isna(), 'Cabin'] = 'No Cabin'
    df.loc[df.Embarked.isna(), 'Embarked'] = 'S'
    df.loc['persons_abraod_size'] = (df['Parch']+df['SibSp']).astype(int)
    df['alone'] = np.where(df['Parch'] == 0,1,0)
    df['Embarked'] = df['Embarked'].map( {'S':0, 'C':1, 'Q':2, } ).astype(float)
    df['Sex'] = df['Sex'].map( {'male': 1, 'female': 2} ).astype(float)
    df['log_fare'] = df['Fare'].apply(np.log)
    df['Room'] = (df['Cabin']
                    .str.slice(1,5).str.extract('([0-9]+)', expand = False).fillna(0).astype(int))
    df['RoomBand'] = 0               
    df.loc[(df.Room > 0) & (df.Room <= 20), 'RoomBand'] = 1
    df.loc[(df.Room > 20) & (df.Room <= 40), 'RoomBand'] = 2
    df.loc[(df.Room > 40) & (df.Room <= 80), 'RoomBand'] = 3
    df.loc[df.Room > 80, 'RoomBand'] = 4
    df_id = df.PassengerId
    df = df.drop('PassengerId', axis=1)
    return df, df_id


train = pd.read_csv('../Educba/train.csv')
test = pd.read_csv('../Educba/test.csv')
train,train_id = feature_engineering(train)
test,test_id = feature_engineering(test)

train.info()

import xgboost as xgb 
from sklearn import model_selection
x_train = train.drop('Surived', axis=1).select_dtypes(include=['int32', 'int64', 'float64'])
y_train = train['Survived']
x_test = test.select_dtypes(include=['int32', 'int64', 'float64'])

xg_boost = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel =1, 
colsample_bytree = 0.65, gamma = 2, learning_rate=0.3, max_delta_step=1, max_depth = 4, 
min_child_weight=2, missing= None,n_estimators=280, n_jobs =1,nthread=None,objective='binary:logistic',
random_state= 0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent = True, subsample=1)

xg_boost.fit(x_train,y_train)

print(xg_boost.score(x_train,y_train))

scores = model_selection.cross_val_score(xg_boost,x_train,y_train, cv=5, scoring='accuracy')
print(scores)
print('K_Fold on XGBClassifier: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))
xgb.plot_importance(xg_boost)
plt.show()

Y_pred = xg_boost.predict(x_test)

submission = pd.DataFrame({
    'PassengerId': test_id,
    'SurivedId': Y_pred
})

submission.head(10)

submission.to_csv('submission.csv', index=False)
