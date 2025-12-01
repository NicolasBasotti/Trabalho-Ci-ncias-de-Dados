# -----------------------------------------------------------
# app.py — Trabalho A1 - Ciência de Dados - COVID-19 (Santa Catarina)
# Dataset REAL: Brasil.IO (caso_full)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO, TextIOWrapper
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

import joblib
import io

# -----------------------------------------------------------
# Configuração do Streamlit
# -----------------------------------------------------------
st.set_page_config(page_title="COVID-19 SC — Ciência de Dados", layout="wide")
sns.set(style="whitegrid")

st.title("Trabalho A1 — Ciência de Dados")
st.subheader("Análise COVID-19 — Santa Catarina (Brasil.IO)")

st.markdown("""
Aplicativo completo com:
1. **Dataset real Brasil.IO**
2. **EDA completa**
3. **Pré-processamento**
4. **Modelagem RandomForest**
5. **GridSearch**
6. **Exportação**
""")

# -----------------------------------------------------------
# 1) Download do dataset Brasil.IO — SC
# -----------------------------------------------------------
@st.cache_data(show_spinner=True)
def baixar_dataset():
    url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
    r = requests.get(url, timeout=60)

    if r.status_code != 200:
        st.error("Erro ao baixar dataset Brasil.IO")
        return None

    compressed = BytesIO(r.content)
    with gzip.open(compressed, "rb") as f:
        wrapper = TextIOWrapper(f, encoding="utf-8", errors="replace")
        df = pd.read_csv(wrapper, low_memory=False)

    return df


if st.button("Baixar dataset real (Brasil.IO) e filtrar SC"):
    with st.spinner("Baixando dataset..."):
        df_full = baixar_dataset()

        df_full["state"] = df_full["state"].astype(str).str.strip().str.upper()
        df_sc = df_full[df_full["state"] == "SC"].copy()

        colunas_irrelevantes = [
            "epidemiological_week",
            "estimated_population_2019",
            "order_for_place",
            "is_last",
            "is_repeated",
            "city_ibge_code",
            "last_available_date",
            "last_available_confirmed_per_100k_inhabitants",
            "last_available_death_rate",
        ]

        for col in colunas_irrelevantes:
            if col in df_sc.columns:
                df_sc.drop(columns=[col], inplace=True)


        st.session_state["df_sc"] = df_sc
        st.success(f"Dataset carregado! Registros SC: {df_sc.shape[0]:,}")
else:
    if "df_sc" in st.session_state:
        df_sc = st.session_state["df_sc"]
    else:
        st.warning("Clique no botão acima para carregar o dataset.")
        st.stop()

# Prévia do dataset
st.subheader("Prévia do Dataset (SC)")
st.dataframe(df_sc.head())
st.write("Shape:", df_sc.shape)

# -----------------------------------------------------------
# 2) EDA — Análise Exploratório
# -----------------------------------------------------------
st.header("2) EDA — Análise Exploratória de Dados")

if st.checkbox("Mostrar estatística descritiva"):
    st.dataframe(df_sc.describe(include="all").T)

if st.checkbox("Mostrar valores ausentes"):
    na = df_sc.isna().sum().sort_values(ascending=False)
    st.table(na[na > 0])

if st.checkbox("Mostrar matriz de correlação (compacta)"):
    col_num = ["last_available_confirmed", "new_confirmed", "last_available_deaths", "new_deaths"]
    col_existentes = [c for c in col_num if c in df_sc.columns]

    if col_existentes:
        correlacao = df_sc[col_existentes].corr().round(2)

        # Criar figura pequena
        fig, ax = plt.subplots(figsize=(6, 4))  # tamanho compacto
        sns.heatmap(
            correlacao,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            annot_kws={"size":8},
            xticklabels=True,
            yticklabels=True,
            ax=ax
        )
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        # Salvar em buffer e exibir como imagem
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.image(buf)  # mostra imagem compacta
        plt.close(fig)

        # Botão de download CSV
        csv_correlacao = correlacao.to_csv().encode("utf-8")
        st.download_button(
            label="⬇ Baixar CSV da correlação",
            data=csv_correlacao,
            file_name="correlacao_sc.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma coluna numérica relevante encontrada para correlação.")

# -----------------------------
# Gráfico agregado de casos por dia
# -----------------------------
if st.checkbox("Mostrar gráfico agregado de casos por dia (SC)"):
    # Agrupa casos confirmados por data
    col_casos = [c for c in ["last_available_confirmed", "new_confirmed"] if c in df_sc.columns]
    if col_casos:
        df_agregado = df_sc.groupby("date")[col_casos].sum().reset_index()
        df_agregado["date"] = pd.to_datetime(df_agregado["date"])
        
        st.line_chart(df_agregado.set_index("date")[col_casos])
    else:
        st.warning("Nenhuma coluna de casos encontrada para o gráfico.")

# -----------------------------------------------------------
# Gráfico interativo por cidade (NOVO)
# -----------------------------------------------------------
# -----------------------------------------------------------
# Gráficos interativos separados — Casos e Mortes
# -----------------------------------------------------------
st.subheader("📊 Evolução diária — Casos e Mortes (Gráficos Interativos Separados)")

if "city" in df_sc.columns:
    cidades = sorted(df_sc["city"].dropna().unique())
    cidade = st.selectbox("Selecione uma cidade:", cidades)

    df_city = df_sc[df_sc["city"] == cidade].copy()
    df_city["date"] = pd.to_datetime(df_city["date"])
    df_city = df_city.sort_values("date")

    # -------------------------------
    # NOVOS CASOS — GRÁFICO INTERATIVO
    # -------------------------------
    st.markdown("### 🟦 Novos Casos Confirmados")
    if "new_confirmed" in df_city.columns:
        df_casos = df_city.set_index("date")[["new_confirmed"]]
        st.line_chart(df_casos)
    else:
        st.warning("Coluna 'new_confirmed' não encontrada no dataset.")

    # -------------------------------
    # NOVAS MORTES — GRÁFICO INTERATIVO
    # -------------------------------
    st.markdown("### 🟥 Novas Mortes Registradas")
    if "new_deaths" in df_city.columns:
        df_mortes = df_city.set_index("date")[["new_deaths"]]
        st.line_chart(df_mortes)
    else:
        st.warning("Coluna 'new_deaths' não encontrada no dataset.")

# -----------------------------------------------------------
# 3) Pré-processamento
# -----------------------------------------------------------
st.header("3) Pré-processamento de Dados")

num_candidates = df_sc.select_dtypes(include=[np.number]).columns.tolist()
cat_candidates = df_sc.select_dtypes(exclude=[np.number]).columns.tolist()

num_features = st.multiselect("Selecione features numéricas:", num_candidates, default=["new_confirmed", "new_deaths"])
cat_features = st.multiselect("Selecione features categóricas:", cat_candidates, default=["city"])

# Criar target
if st.checkbox("Criar target (new_deaths > 0)"):
    df_sc["target_death"] = (df_sc["new_deaths"] > 0).astype(int)

target = st.selectbox("Selecione o TARGET (coluna a prever):", [c for c in df_sc.columns if df_sc[c].nunique() <= 20])

# Pipelines
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features)
])

# -----------------------------------------------------------
# 4) Modelagem — RandomForest
# -----------------------------------------------------------
st.header("4) Modelagem — RandomForest")

X = df_sc[num_features + cat_features]
y = df_sc[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

if st.button("Treinar Modelo"):
    with st.spinner("Treinando modelo..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Acurácia: {acc:.4f}")

        st.text("Relatório de Classificação:")
        st.text(classification_report(y_test, y_pred))

        # -----------------------------
        # Matriz de confusão compacta como imagem
        # -----------------------------
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))  # tamanho compacto
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap="Blues",
            cbar=False,  # remove barra de cores para economizar espaço
            annot_kws={"size":8},
            xticklabels=True,
            yticklabels=True,
            ax=ax
        )
        plt.xlabel("Predito", fontsize=8)
        plt.ylabel("Real", fontsize=8)
        plt.xticks(rotation=0, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()

        # Exibir heatmap como imagem para não esticar na tela
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.image(buf)
        plt.close(fig)

        st.session_state["baseline"] = model


# -----------------------------------------------------------
# 5) GridSearch — Ajuste de Hiperparâmetros
# -----------------------------------------------------------
st.header("5) GridSearch — Ajuste de Hiperparâmetros")

if st.checkbox("Ativar GridSearch"):
    params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }

    gs = GridSearchCV(model, params, cv=3, scoring="accuracy", n_jobs=-1)

    if st.button("Rodar GridSearch"):
        with st.spinner("Executando GridSearch..."):
            gs.fit(X_train, y_train)
            st.success("GridSearch concluído!")

            st.write("Melhores parâmetros:", gs.best_params_)
            st.write("Melhor score (CV):", gs.best_score_)

            y_pred = gs.best_estimator_.predict(X_test)
            st.write("Acurácia teste:", accuracy_score(y_test, y_pred))

            st.session_state["best_model"] = gs.best_estimator_

# -----------------------------------------------------------
# 6) Exportação
# -----------------------------------------------------------
st.header("6) Exportação dos Resultados")

if "best_model" in st.session_state:
    final_model = st.session_state["best_model"]
elif "baseline" in st.session_state:
    final_model = st.session_state["baseline"]
else:
    final_model = None

if final_model:
    joblib.dump(final_model, "modelo_covid_sc.joblib")
    with open("modelo_covid_sc.joblib", "rb") as f:
        st.download_button("⬇ Download do Modelo Treinado", f, file_name="modelo_covid_sc.joblib")

    csv_buffer = io.StringIO()
    df_sc.to_csv(csv_buffer, index=False)
    st.download_button("⬇ Download do Dataset SC", csv_buffer.getvalue(), file_name="dataset_SC.csv")

else:
    st.info("Treine o modelo antes de exportar.")
