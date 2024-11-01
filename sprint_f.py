import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Estilo CSS personalizado
st.markdown(
    """
    <style>
    /* Fundo da aplicação */
    .stApp {
        background-color: black;
    }
    
    /* Títulos e textos */
    h1, h2, h3, h4, h5, h6, p, label, .stText {
        color: white;
    }
    
    /* Tabelas de dados */
    table {
        color: black;
        background-color: #222;
        border: 1px solid #555;
    }
    th, td {
        background-color: #222;
        color: black;
        border: 1px solid #555;
    }
    
    /* Caixas JSON de parâmetros */
    .stMarkdown pre {
        background-color: #222 !important;
        color: white !important;
        border: 1px solid #555;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Gráficos */
    .stPlotlyChart, .stPyplot {
        background-color: black;
    }
    </style>
    """, unsafe_allow_html=True
)
df = pd.read_csv("C:/Users/gabriel/Documents/cs_de_front/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.isnull().sum()
# Transformar colunas categóricas em numéricas
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Utilizar get_dummies para converter as colunas categóricas restantes
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=['Attrition'])
y = df['Attrition']

st.title("sprint 4 de front-end")

# Função de pré-processamento
def preprocess_data(df):
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42), X

# Função para treinar o modelo e retornar métricas
def train_model(pipeline, params, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(pipeline, params, cv=StratifiedKFold(n_splits=5), scoring='recall')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    return grid.best_params_, metrics, fpr, tpr, auc_score

# Exibição de resultados
def display_results(name, best_params, metrics, fpr, tpr, auc_score):
    st.subheader(f"{name}")
    st.write("Melhores Parâmetros:", best_params)
    st.write("Relatório de Classificação:")
    st.write(pd.DataFrame(metrics).transpose().style.set_properties(**{'background-color': '#222', 'color': 'white'}))
    st.write(f"AUC: {auc_score:.3f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

# Configuração do pipeline
def get_pipeline(classifier, X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

# Processamento dos dados e treinamento dos modelos
(X_train, X_test, y_train, y_test), X = preprocess_data(df)

# Configuração dos modelos e parâmetros
models = {
    "RandomForest": {
        "pipeline": get_pipeline(RandomForestClassifier(random_state=42), X),
        "params": {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, 20],
            'classifier__min_samples_split': [5, 10],
            'classifier__min_samples_leaf': [2, 4]
        }
    },
    "XGBoost": {
        "pipeline": get_pipeline(XGBClassifier(random_state=42), X),
        "params": {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1]
        }
    },
    "SVM": {
        "pipeline": get_pipeline(SVC(probability=True, random_state=42), X),
        "params": {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    }
}

# Plota as curvas ROC
plt.figure(figsize=(8, 6))
for name, model_info in models.items():
    best_params, metrics, fpr, tpr, auc_score = train_model(
        model_info["pipeline"], model_info["params"], X_train, X_test, y_train, y_test
    )
    display_results(name, best_params, metrics, fpr, tpr, auc_score)

# Finalização da curva ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparação de Curvas ROC')
plt.legend()
st.pyplot(plt)
