import requests
import gzip
import pandas as pd
from io import BytesIO, TextIOWrapper

# URL oficial do dataset
url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"

print("Baixando dataset completo...")
response = requests.get(url)

if response.status_code != 200:
    raise Exception("Erro ao baixar dataset do Brasil.IO")

# Descompactar o arquivo .gz
compressed = BytesIO(response.content)
with gzip.open(compressed, "rb") as f:
    wrapper = TextIOWrapper(f, encoding="utf-8", errors="replace")
    df = pd.read_csv(wrapper, low_memory=False)

print("Linhas totais:", df.shape)

# -----------------------------------------
# FILTRAR APENAS SANTA CATARINA (SC)
# -----------------------------------------
df["state"] = df["state"].astype(str).str.upper().str.strip()
df_sc = df[df["state"] == "SC"].copy()

print("Linhas apenas SC:", df_sc.shape)

# -----------------------------------------
# REMOVER COLUNAS IRRELEVANTES
# -----------------------------------------
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

# Remove apenas se existir no dataset
colunas_remover = [c for c in colunas_irrelevantes if c in df_sc.columns]

df_sc = df_sc.drop(columns=colunas_remover)

print("Colunas removidas:", colunas_remover)
print("Formato final:", df_sc.shape)

# -----------------------------------------
# SALVAR CSV FINAL
# -----------------------------------------
df_sc.to_csv("covid_sc_limpo.csv", index=False, encoding="utf-8")

print("Arquivo salvo: covid_sc_limpo.csv")
