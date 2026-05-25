import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
import os
import holidays
import joblib

MODELING_DIR = os.path.dirname(__file__)

SRC_DIR = os.path.dirname(MODELING_DIR)

CHALLENGE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(CHALLENGE_DIR, "data")
DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, "incidentes_por_dia.csv"))

df["total_dia"] = df[["qtd_p1", "qtd_p2", "qtd_p3", "qtd_p4", "qtd_p5"]].sum(axis=1)

df_model = df[["data_abertura", "total_dia"]].copy()
df_model["data_abertura"] = pd.to_datetime(df_model["data_abertura"])
df_model = df_model.sort_values("data_abertura").reset_index(drop=True)

# Criando os Lags (Atrasos)
df_model["vol_D1"] = df_model["total_dia"].shift(1)
df_model["vol_D7"] = df_model["total_dia"].shift(7)
df_model["vol_D14"] = df_model["total_dia"].shift(14)

# Features Temporais
df_model["dia_semana"] = df_model["data_abertura"].dt.dayofweek

# OneHot dos dias da semana (Garantindo que todos de 0 a 6 existam)
# O Categorical garante que todas as colunas de dia_0 a dia_6 sejam criadas
df_model["dia_semana_cat"] = pd.Categorical(
    df_model["dia_semana"], categories=list(range(7))
)
dias_dummies = pd.get_dummies(df_model["dia_semana_cat"], prefix="dia").astype(int)
df_model = pd.concat([df_model, dias_dummies], axis=1)

# Calculando a média movel de incieentes nos últimos 7 e 14 dias.
df_model["media_movel_7d"] = df_model["total_dia"].shift(1).rolling(window=7).mean()
df_model["media_movel_14d"] = df_model["total_dia"].shift(1).rolling(window=14).mean()

# Carregando os feriados do brasil
feriados_br = holidays.Brazil(years=[2024, 2025, 2026], subdiv="SP")

# Criando a flag de feriado
df_model["is_feriado"] = df_model["data_abertura"].apply(
    lambda x: 1 if x in feriados_br else 0
)

# Removemos as linhas vazias
df_model = df_model.dropna().reset_index(drop=True)

target = "total_dia"
# Pegamos as features do modelo de forma dinâmica, removendo o que não serve pro LGBM
features = df_model.columns.tolist()
colunas_para_remover = ["data_abertura", "total_dia", "dia_semana", "dia_semana_cat"]
for col in colunas_para_remover:
    if col in features:
        features.remove(col)

X = df_model[features]
y = df_model[target]

# Fazendo o split dos dados, 70% para treino e 30% para testar
split_idx = int(len(df_model) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==========================================
# 3. TREINAMENTO DOS MODELOS
# ==========================================

# LGBM para Prever a quantidade de incidentes
modelo_lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42)
modelo_lgbm.fit(X_train, y_train)

# Treinando o Isolation Forest para definir o que é anomalia
features_iso = ["total_dia", "dia_semana"]
X_iso_train = df_model.loc[: split_idx - 1, features_iso]

modelo_iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
modelo_iso.fit(X_iso_train)

# ==========================================
# 4. O PIPELINE EM AÇÃO (Simulando o D+1)
# ==========================================
ultimo_dia_conhecido = df_model.iloc[-1]
data_hoje = ultimo_dia_conhecido["data_abertura"]
data_amanha = data_hoje + pd.Timedelta(days=1)
dia_semana_amanha = data_amanha.dayofweek

amanha_is_feriado = 1 if data_amanha in feriados_br else 0

# Construindo o dicionário
features_amanha_dict = {
    "vol_D1": ultimo_dia_conhecido["total_dia"],
    "vol_D7": df_model.iloc[-7]["total_dia"],
    "vol_D14": df_model.iloc[-14]["total_dia"],
    "dia_0": 0,
    "dia_1": 0,
    "dia_2": 0,
    "dia_3": 0,
    "dia_4": 0,
    "dia_5": 0,
    "dia_6": 0,
    "media_movel_7d": df_model.tail(7)["total_dia"].mean(),
    "media_movel_14d": df_model.tail(14)["total_dia"].mean(),
    "is_feriado": amanha_is_feriado,
}

features_amanha_dict[f"dia_{dia_semana_amanha}"] = 1

# Usando a variável 'features' definida lá em cima para garantir a ordem exata do treino
features_amanha = pd.DataFrame([features_amanha_dict])[features]

# Passo 1: LightGBM faz a previsão do D+1
previsao_D1 = modelo_lgbm.predict(features_amanha)[0]
previsao_D1 = int(round(previsao_D1))

# Passo 2 e 3: Isolation Forest emite o veredito
dados_auditoria = pd.DataFrame(
    [{"total_dia": previsao_D1, "dia_semana": dia_semana_amanha}]
)

veredito_anomalia = modelo_iso.predict(dados_auditoria)[0]
score_risco = modelo_iso.decision_function(dados_auditoria)[0]

# ==========================================
# 5. RESULTADO FINAL
# ==========================================
print(
    f"🔮 Previsão (LGBM) para D+1 ({data_amanha.strftime('%d/%m/%Y')}): {previsao_D1} incidentes."
)
print("-" * 50)

if veredito_anomalia == -1:
    print(f"🚨 ALERTA DE ISOLATION FOREST: Anomalia Detectada!")
    print(
        f"Risco operacional alto. Esse volume para este dia da semana foge do padrão histórico (Score: {score_risco:.3f})."
    )
    print("Recomendação: Avaliar risco de perda de OLA e acionar plano de mitigação.")
else:
    print(f"✅ Status (Isolation Forest): Operação Normal.")
    print(
        f"O volume previsto está dentro da capacidade histórica para este dia da semana (Score: {score_risco:.3f})."
    )

# ==========================================
# 6. EXPORTAÇÃO DOS MODELOS (Persistência)
# ==========================================

caminho_lgbm = os.path.join(MODELING_DIR + "/modelo_lgbm_locaweb.pkl")
caminho_iso = os.path.join(MODELING_DIR + "/modelo_iso_locaweb.pkl")

# Salvando os modelos em disco
joblib.dump(modelo_lgbm, caminho_lgbm)
joblib.dump(modelo_iso, caminho_iso)

print(f"✅ Modelos salvos com sucesso na pasta: {MODELING_DIR}")
