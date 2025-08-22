import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy.stats import randint
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
RANDOM_STATE = 42


# Загрузка данных
data_train = pd.read_csv('heart_train.csv')
data_train=data_train.drop('Unnamed: 0',axis=1)

# Предобработка
binary_cols = ['Diabetes', 'Family History', 'Smoking', 'Obesity',
               'Alcohol Consumption', 'Previous Heart Problems', 'Medication Use']
data_train[binary_cols] = data_train[binary_cols].fillna(data_train[binary_cols].mode().iloc[0])
num_cols = ['Stress Level', 'Physical Activity Days Per Week']
data_train[num_cols] = data_train[num_cols].fillna(data_train[num_cols].median())
data_train[binary_cols] = data_train[binary_cols].fillna(0).astype(int)

#Разделение данных
X = data_train.drop(['Heart Attack Risk (Binary)','id'], axis=1)
y = data_train['Heart Attack Risk (Binary)']

# Тренировочные данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
numeric_features = [ "Age",
     "Cholesterol", "Heart rate", "Exercise Hours Per Week",
     "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
     "Physical Activity Days Per Week",
     "Sleep Hours Per Day", "Blood sugar", "CK-MB", "Troponin", "Diastolic blood pressure", "Stress Level", "Diet" ]

binary_features = [ "Family History", "Alcohol Consumption", "Previous Heart Problems", "Medication Use" ]

category_features = ['Gender']
ord_columns = binary_features
num_columns = numeric_features
ohe_columns = category_features

# Пайплайна
def pipe(ohe_columns, ord_columns, num_columns):
    # --- Pipeline для OHE ---
    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    # --- Pipeline для Ordinal ---
    ord_pipe = Pipeline([
        ("imputer_before", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ("imputer_after", SimpleImputer(strategy="most_frequent"))
    ])

    # --- Препроцессинг ---
    preprocessor = ColumnTransformer([
        ("ohe", ohe_pipe, ohe_columns),
        ("ord", ord_pipe, ord_columns),
        ("num", StandardScaler(), num_columns)
    ], remainder="passthrough")

    # --- Финальный пайп ---
    pipe_final = Pipeline([
        ("preprocessor", preprocessor),
        ("models", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])

    return pipe_final

# Настройка пайплайна
pipe_final = pipe(ohe_columns, ord_columns, num_columns)

param_grid = [
    #  RandomForest
    {
        'models': [RandomForestClassifier(random_state=42)],
        'models__n_estimators': randint(150, 300),
        'models__max_depth': [10, 15, 20],
        'models__min_samples_split': randint(2, 5),
        'models__min_samples_leaf': randint(1, 3),
        'models__class_weight': ['balanced', 'balanced_subsample']
    },

    # Для DecisionTree
    {
        'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(5, 15),
        'models__max_features': range(5, 15),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
    },

    # Для SVC
    {
        'models': [SVC(random_state=RANDOM_STATE)],
        'models__kernel': ['linear', 'rbf', 'poly'],
        'models__C': [0.1, 1, 10],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
    }
]

randomized_search = RandomizedSearchCV(
pipe_final,
param_grid,
cv = 5,
scoring = 'roc_auc',
random_state = RANDOM_STATE,
n_jobs = -1,
verbose = 1,
)
randomized_search.fit(X_train, y_train)

# Пример поиска по параметрам
print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели на тренировочной выборке:', round(randomized_search.best_score_, 2))

# Сохраняем модель
joblib.dump(randomized_search.best_estimator_, "model.pkl")
print("Модель сохранена в model.pkl")