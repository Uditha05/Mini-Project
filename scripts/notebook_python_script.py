### Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## Read Data
train_dataset_ = pd.read_feather('../input/amexfeather/train_data.ftr')
# Keep the latest statement features for each customer
train_dataset = train_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()
del train_dataset_
gc.collect()

## view first rows of read date
train_dataset.head()

## get infomation about data
train_dataset.info(max_cols=191,show_counts=True)


###--- Feature Engineering Techniques----####

### Categorical Encoding
categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
num_cols = [col for col in train_dataset.columns if col not in categorical_cols + ["target"]]
print(f'Total number of categorical features: {len(categorical_cols)}')
print(f'Total number of continuos features: {len(num_cols)}')

## Plot and visualization data

## Target visualization
sns.countplot(x = 'target', data = train_dataset)

## Categorial features visualization
plt.figure(figsize=(20, 30))
for i, k in enumerate(categorical_cols):
    plt.subplot(6, 2, i+1)
    temp_val = pd.DataFrame(train_dataset[k].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp_val.index.name = 'value'
    temp_val.reset_index(inplace=True)
    plt.bar(temp_val.index, temp_val['count'], alpha=0.5)
    plt.xlabel(k)
    plt.ylabel('frequency')
    plt.xticks(temp_val.index, temp_val.value)
plt.show()


## Continuous features visualization
for i, l in enumerate(num_cols):
    if i % 4 == 0: 
        if i > 0: plt.show()
        plt.figure(figsize=(20, 3))
    plt.subplot(1, 4, i % 4 + 1)
    plt.hist(train_dataset[l], bins=200, color='#C69C73')
    plt.xlabel(l)
plt.show()


## Visualize features depend on general categories
Delinquency = [d for d in train_dataset.columns if d.startswith('D_')]
Spend = [s for s in train_dataset.columns if s.startswith('S_')]
Payment = [p for p in train_dataset.columns if p.startswith('P_')]
Balance = [b for b in train_dataset.columns if b.startswith('B_')]
Risk = [r for r in train_dataset.columns if r.startswith('R_')]
Dict = {'Delinquency': len(Delinquency), 'Spend': len(Spend), 'Payment': len(Payment), 'Balance': len(Balance), 'Risk': len(Risk),}

plt.figure(figsize=(10,5))
sns.barplot(x=list(Dict.keys()), y=list(Dict.values()));

## Imputation
## Check null values
NaN_Val = np.array(train_dataset.isnull().sum())
NaN_prec = np.array((train_dataset.isnull().sum() * 100 / len(train_dataset)).round(2))
NaN_Col = pd.DataFrame([np.array(list(train_dataset.columns)).T,NaN_Val.T,NaN_prec.T,np.array(list(train_dataset.dtypes)).T], index=['Features','Num of Missing values','Percentage','DataType']).transpose()
pd.set_option('display.max_rows', None)


## Remove not usefull columns (column has more missing values / margin ===> 80%)
train_dataset = train_dataset.drop(['S_2','D_66','D_42','D_49','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)

## Fill null values
selected_col = np.array(['P_2','S_3','B_2','D_41','D_43','B_3','D_44','D_45','D_46','D_48','D_50','D_53','S_7','D_56','S_9','B_6','B_8','D_52','P_3','D_54','D_55','B_13','D_59','D_61','B_15','D_62','B_16','B_17','D_77','B_19','B_20','D_69','B_22','D_70','D_72','D_74','R_7','B_25','B_26','D_78','D_79','D_80','B_27','D_81','R_12','D_82','D_105','S_27','D_83','R_14','D_84','D_86','R_20','B_33','D_89','D_91','S_22','S_23','S_24','S_25','S_26','D_102','D_103','D_104','D_107','B_37','R_27','D_109','D_112','B_40','D_113','D_115','D_118','D_119','D_121','D_122','D_123','D_124','D_125','D_128','D_129','B_41','D_130','D_131','D_133','D_139','D_140','D_141','D_143','D_144','D_145'])
for col in selected_col:
    train_dataset[col] = train_dataset[col].fillna(train_dataset[col].median())

selcted_col2 = np.array(['D_68','B_30','B_38','D_64','D_114','D_116','D_117','D_120','D_126'])
for col2 in selcted_col2:
    train_dataset[col2] =  train_dataset[col2].fillna(train_dataset[col2].mode()[0])
## verify no null values
print(train_dataset.isnull().sum().to_string())
## view first rows of read date
train_dataset.head()


##--- Load Testing data-------##

test_dataset_ = pd.read_feather('../input/amexfeather/test_data.ftr')
# Keep the latest statement features for each customer
test_dataset = test_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()
del test_dataset_
gc.collect()

## view first rows of read date
test_dataset.head()

## Imputation
## Check null values
NaN_Val2 = np.array(test_dataset.isnull().sum())
NaN_prec2 = np.array((test_dataset.isnull().sum() * 100 / len(test_dataset)).round(2))
NaN_Col2 = pd.DataFrame([np.array(list(test_dataset.columns)).T,NaN_Val2.T,NaN_prec2.T,np.array(list(test_dataset.dtypes)).T], index=['Features','Num of Missing values','Percentage','DataType']).transpose()
pd.set_option('display.max_rows', None)

## Remove not usefull columns (column has more missing values / margin ===> 80%)
test_dataset = test_dataset.drop(['S_2','D_42','D_49','D_66','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)

## Fill null values
selected_column = np.array(['P_2','S_3','B_2','D_41','D_43','B_3','D_44','D_45','D_46','D_48','D_50','D_53','S_7','D_56','S_9','S_12','S_17','B_6','B_8','D_52','P_3','D_54','D_55','B_13','D_59','D_61','B_15','D_62','B_16','B_17','D_77','B_19','B_20','D_69','B_22','D_70','D_72','D_74','R_7','B_25','B_26','D_78','D_79','D_80','B_27','D_81','R_12','D_82','D_105','S_27','D_83','R_14','D_84','D_86','R_20','B_33','D_89','D_91','S_22','S_23','S_24','S_25','S_26','D_102','D_103','D_104','D_107','B_37','R_27','D_109','D_112','B_40','D_113','D_115','D_118','D_119','D_121','D_122','D_123','D_124','D_125','D_128','D_129','B_41','D_130','D_131','D_133','D_139','D_140','D_141','D_143','D_144','D_145'])
for column in selected_column:
    test_dataset[column] = test_dataset[column].fillna(test_dataset[column].median())

selected_column2 = np.array(['D_68','B_30','B_38','D_114','D_116','D_117','D_120','D_126'])
for column2 in selected_column2:
    test_dataset[column2] =  test_dataset[column2].fillna(test_dataset[column2].mode()[0])

## verify no null values
print(test_dataset.isnull().sum().to_string())
## view first rows of read date
test_dataset.head()

## Remove highly correlated features
#Remove columns if there are > 90% of correlations
train_dataset_without_target = train_dataset.drop(["target"],axis=1)
cor_matrix = train_dataset_without_target.corr()
col_core = set()
for i in range(len(cor_matrix.columns)):
    for j in range(i):
        if(cor_matrix.iloc[i, j] > 0.9):
            col_name = cor_matrix.columns[i]
            col_core.add(col_name)
train_dataset = train_dataset.drop(col_core, axis=1)
test_dataset = test_dataset.drop(col_core, axis=1)


## Extract Features and target
num_columns = [col for col in train_dataset.columns if col not in ["target"]]
X = train_dataset[num_columns]
y = train_dataset['target']
print(f"X shape is = {X.shape}" )
print(f"Y shape is = {y.shape}" )


##---- Hold out method-----###

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Split data set 70:30
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"X_train shape is = {x_train.shape}" )
print(f"Y_train shape is = {y_train.shape}" )
print(f"X_test shape is = {x_test.shape}" )
print(f"Y_test shape is = {y_test.shape}" )


### Apporoch 1 

### Make model with LightGBM 

import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train, categorical_feature = categorical_cols)
params = {'objective': 'binary','n_estimators': 1500,'metric': 'binary_logloss','boosting': 'gbdt','num_leaves': 31,'reg_lambda' : 50,'colsample_bytree': 0.19,'learning_rate': 0.05,'min_child_samples': 2400,'max_bins': 255,'seed': 32,'verbose': 0}

## trained model with 100 iterations
model = lgb.train(params, d_train, 100)

## Make Prediction
predictions = model.predict(test_dataset[num_columns])#logreg
print(predictions)

## -- Write output into csv
sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': sample_dataset.customer_ID, 'prediction': predictions})
output.to_csv('submission_new.csv', index=False)


### Apporoch 2 

### Make model with XGBClassifier(XGboost) 

from xgboost import XGBClassifier
model_xgb = XGBClassifier(
    learning_rate=0.05,
    n_estimators=30,
    objective="binary:logistic",
    nthread=3,
)

## trained the model
model_xgb.fit(x_train, y_train)

## Make Prediction
predictions = model_xgb.predict(test_dataset[num_columns])

## -- Write output into csv
sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': sample_dataset.customer_ID, 'prediction': predictions})
output.to_csv('submission_XGBClassifier.csv', index=False)

