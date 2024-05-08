from sklearn import *
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

import seaborn as sns
import matplotlib.pyplot as plt

def confusion_matrix_(X_test, y_test, model, title):
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    sns.heatmap(
        confusion_matrix(y_test, model.predict(X_test)),
        annot=True,
        fmt="d",
        ax=axes,
    )
    axes.set_title(title)
    plt.show()

def test_score(X_test, y_test, model, title):
    metric_df = []
    y_pred = model.predict(X_test)
    metric_df.append(
        {
            "Model": title,
            "Precision": model.score(X_test, y_test),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_pred),
        }
    )
    metric_df = pd.DataFrame(metric_df)
    print(metric_df)

def select_columns(df: pd.DataFrame):
    # Pré-processamento dos dados
    # Ignora a variável object id, uma vez que é usada apenas para enumeração dos dados
    binarizer = LabelEncoder()
    df[["X","Y","NCESID","NAME","ADDRESS","CITY","STATE","ZIP","ZIP4","TELEPHONE","TYPE","STATUS","POPULATION","COUNTY","COUNTYFIPS","COUNTRY","LATITUDE","LONGITUDE","NAICS_CODE","NAICS_DESC","SOURCE","SOURCEDATE","VAL_METHOD","VAL_DATE","WEBSITE","LEVEL_","ENROLLMENT","ST_GRADE","END_GRADE","DISTRICTID","FT_TEACHER","SHELTER_ID"]] = df[
        ["X","Y","NCESID","NAME","ADDRESS","CITY","STATE","ZIP","ZIP4","TELEPHONE","TYPE","STATUS","POPULATION","COUNTY","COUNTYFIPS","COUNTRY","LATITUDE","LONGITUDE","NAICS_CODE","NAICS_DESC","SOURCE","SOURCEDATE","VAL_METHOD","VAL_DATE","WEBSITE","LEVEL_","ENROLLMENT","ST_GRADE","END_GRADE","DISTRICTID","FT_TEACHER","SHELTER_ID"]
    ].apply(lambda x: binarizer.fit_transform(x))
    return df

def file_():
    # Carregamento dos dados
    # carrega o arquivo csv para ser utilizado
    df = pd.read_csv("Public_Schools.csv")
    df = df.dropna()
    df = df.drop(columns=["OBJECTID"])
    df = select_columns(df)
    return df

df = file_()

# Define os sets de trainamento e de teste
# Variável target: Level_
# Que se refere ao nível da escola: Adult Education (adulto), Elementary (fundamental 1), 
# High (ensino médio), Middle (fundamental 2), N/A
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["LEVEL_"]),
    df["LEVEL_"],
    test_size=0.3,
    stratify=df["LEVEL_"],
)

# Definição do algorítimo utilizado: KNC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

param_grid = {
    'clf__n_neighbors': range(1, 21),
    'clf__weights': ['uniform', 'distance'],
    'clf__p': [1, 2]
}

# Busca e imprime os melhores hiperparâmetros encontrados
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=5, cv=5, random_state=42)
random_search.fit(X_train, y_train)
print("Best hyperparameters:", random_search.best_params_)

# Imprime a acurácia do algorítimo
accuracy = random_search.score(X_test, y_test)
print("Accuracy on test set:", accuracy)