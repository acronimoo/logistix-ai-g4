from pathlib import Path
import json, hashlib, platform
import pandas as pd
import numpy as np
import joblib
import sklearn, xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
                             roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
NUM = ['unidades', 'carga_bodega_pct', 'stock_disponible']
CAT = ['prioridad']
FEATURES = NUM + CAT
TARGET = 'error_preparacion'
SEED = 42

def load_data(path):
    df = pd.read_excel(path, sheet_name='LogistiX')
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError('El dataset contiene valores nulos en variables seleccionadas.')
    if df.pedido_id.duplicated().any():
        raise ValueError('Se encontraron pedidos repetidos. Revisar antes de entrenar.')
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('La variable objetivo debe contener 0 y 1.')
    return df

def split_data(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=.4, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=.5, random_state=SEED, stratify=y_rest)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model():
    prep = ColumnTransformer([('numericas', 'passthrough', NUM),
                              ('prioridad', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)])
    model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=.05, subsample=.85,
                          colsample_bytree=.9, objective='binary:logistic', eval_metric='logloss',
                          random_state=SEED, n_jobs=2, tree_method='hist')
    return Pipeline([('preprocesamiento', prep), ('modelo', model)])

def evaluate(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {'umbral': float(threshold), 'accuracy': float(accuracy_score(y,pred)),
            'precision': float(precision_score(y,pred,zero_division=0)),
            'recall': float(recall_score(y,pred,zero_division=0)),
            'f1': float(f1_score(y,pred,zero_division=0)),
            'f2': float(fbeta_score(y,pred,beta=2,zero_division=0)),
            'roc_auc': float(roc_auc_score(y,prob)), 'average_precision': float(average_precision_score(y,prob)),
            'brier': float(brier_score_loss(y,prob)), 'matriz_confusion': confusion_matrix(y,pred,labels=[0,1]).tolist()}

def run_training(path=BASE/'LogistiX.xlsx', out=BASE/'artefactos'):
    path, out = Path(path), Path(out)
    df = load_data(path)
    X_train,X_val,X_test,y_train,y_val,y_test = split_data(df)
    pipeline = build_model()
    pipeline.fit(X_train,y_train)
    p_val = pipeline.predict_proba(X_val)[:,1]
    # Selección en validación, nunca en test: F2 da más peso a detectar errores.
    candidates = np.arange(.10,.81,.05)
    threshold = float(max(candidates, key=lambda t: (fbeta_score(y_val,p_val>=t,beta=2,zero_division=0),t)))
    threshold = round(threshold,2)
    p_test = pipeline.predict_proba(X_test)[:,1]
    metrics = {'modelo_version':'1.0.0','semilla':SEED,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'filas':len(df),'nulos':int(df.isna().sum().sum()),'duplicados_filas':int(df.duplicated().sum()),
               'clases':{str(k):int(v) for k,v in df[TARGET].value_counts().items()},
               'particiones':{'entrenamiento':len(y_train),'validacion':len(y_val),'prueba':len(y_test)},
               'variables':FEATURES,'objetivo':TARGET,'umbral':threshold,
               'validacion_umbral':evaluate(y_val,p_val,threshold),
               'prueba_umbral':evaluate(y_test,p_test,threshold),'prueba_05':evaluate(y_test,p_test,.5),
               'baseline_accuracy':float((y_test==0).mean()),'baseline_recall':0.0,
               'versiones':{'python':platform.python_version(),'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,
                             'pandas':pd.__version__,'numpy':np.__version__}}
    schema = {'variables':FEATURES,'numericas':{c:{'min':int(df[c].min()),'max':int(df[c].max()),'mediana':int(df[c].median())} for c in NUM},
              'prioridades':sorted(df.prioridad.unique().tolist()),'umbral':threshold,'version':'1.0.0'}
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipeline,out/'pipeline.joblib')
    (out/'metricas.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'esquema.json').write_text(json.dumps(schema,indent=2,ensure_ascii=False),encoding='utf-8')
    return metrics

if __name__=='__main__':
    print(json.dumps(run_training(),indent=2,ensure_ascii=False))
