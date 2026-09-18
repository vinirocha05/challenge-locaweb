import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
import joblib
import holidays

load_dotenv()


DASHBOARDS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DASHBOARDS_DIR)
MODELING_DIR = os.path.join(SRC_DIR + "/modeling")

CHALLENGE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(CHALLENGE_DIR + "/data")
logo_locaweb = os.path.join(DASHBOARDS_DIR, "locaweb_logo.png")
logo_datatrust = os.path.join(DASHBOARDS_DIR, "logo_datatrust.png")
# -----------------------------------------------------------------------------
# 1. Título e Descrição da Aplicação
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")

st.sidebar.header("Challenge Locaweb 2026")
st.sidebar.subheader("Equipe: DataTrust")

st.sidebar.image(logo_locaweb, width=150, use_container_width=True)
st.sidebar.image(logo_datatrust, width=100)


@st.cache_resource
def carregar_modelos():
    caminho_lgbm = os.path.join(MODELING_DIR, "modelo_lgbm_locaweb.pkl")
    caminho_iso = os.path.join(MODELING_DIR, "modelo_iso_locaweb.pkl")

    modelo_lgbm_carregado = joblib.load(caminho_lgbm)
    modelo_iso_carregado = joblib.load(caminho_iso)

    return modelo_lgbm_carregado, modelo_iso_carregado


# Inicializando os modelos na sua aplicação
modelo_lgbm, modelo_iso = carregar_modelos()

# Instanciando os feriados uma única vez para não pesar o app
feriados_br = holidays.Brazil(years=[2024, 2025, 2026], subdiv="SP")

# IMPORTANTE: Cole exatamente a lista de features que você usou no treino
FEATURES_TREINO = [
    "vol_D1",
    "vol_D7",
    "vol_D14",
    "dia_0",
    "dia_1",
    "dia_2",
    "dia_3",
    "dia_4",
    "dia_5",
    "dia_6",
    "media_movel_7d",
    "media_movel_14d",
    "is_feriado",
]


# 2. FUNÇÃO GERADORA DE FEATURES PARA D+1
def prever_proximo_dia(df_historico, data_referencia):
    """
    df_historico: DataFrame com todos os dados até o D0 selecionado no filtro.
    data_referencia: O dia de "Hoje" (D0) escolhido pelo usuário.
    """
    # Garantir que temos histórico suficiente para calcular lags e médias
    if len(df_historico) < 14:
        return None, None, None  # Retorna nulo se não houver dados suficientes

    data_referencia = pd.to_datetime(data_referencia)
    data_amanha = data_referencia + pd.Timedelta(days=1)
    dia_semana_amanha = data_amanha.dayofweek
    amanha_is_feriado = 1 if data_amanha in feriados_br else 0

    # Construindo o dicionário D+1 ancorado no histórico
    features_dict = {
        "vol_D1": df_historico.iloc[-1]["total_dia"],
        "vol_D7": df_historico.iloc[-7]["total_dia"],
        "vol_D14": df_historico.iloc[-14]["total_dia"],
        "dia_0": 0,
        "dia_1": 0,
        "dia_2": 0,
        "dia_3": 0,
        "dia_4": 0,
        "dia_5": 0,
        "dia_6": 0,
        "media_movel_7d": df_historico.tail(7)["total_dia"].mean(),
        "media_movel_14d": df_historico.tail(14)["total_dia"].mean(),
        "is_feriado": amanha_is_feriado,
    }

    # Ativa a flag do dia correto
    features_dict[f"dia_{dia_semana_amanha}"] = 1

    # Cria o DataFrame com a ordem exata das colunas
    df_features_amanha = pd.DataFrame([features_dict])[FEATURES_TREINO]

    # Previsão de Volume (LightGBM)
    previsao_vol = int(round(modelo_lgbm.predict(df_features_amanha)[0]))

    # Análise de Risco (Isolation Forest)
    df_auditoria = pd.DataFrame(
        [{"total_dia": previsao_vol, "dia_semana": dia_semana_amanha}]
    )
    status_anomalia = modelo_iso.predict(df_auditoria)[0]
    score_risco = modelo_iso.decision_function(df_auditoria)[0]

    return previsao_vol, status_anomalia, score_risco


@st.cache_data
def carregar_dados():
    df = pd.read_csv(os.path.join(DATA_DIR, "incidentes_por_dia.csv"))

    return df


df = carregar_dados()
df["data_abertura"] = pd.to_datetime(df["data_abertura"])
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


opcoes_analise = [
    "Homepage",
    "Daily",
    "Estatística",
    "Lowa",
]

modo_de_analise = st.sidebar.selectbox("Escolha a análise:", opcoes_analise)

if modo_de_analise == "Homepage":
    # --- PÁGINA DE BOAS-VINDAS ---
    st.header("Bem-vindo(a) à Central de Data Science de equipe Data Trust e Locaweb!")
    st.markdown("---")
    st.subheader(
        "Esta aplicação unifica os três principais desafios do Challenge 2025:"
    )

    st.markdown("""
    1.  **Daily:** Análise diária da operação.
    2.  **Estatística:** Incidadores mais apronfundamos sobre a operação.
    3.  **Lowa:** Nossa IA que ajuda tomadores de decisão.
    """)

    st.info(
        "👈 **Para começar, selecione um modo de análise na barra lateral à esquerda.**"
    )


elif modo_de_analise == "Daily":
    st.title("Daily ")

    # Pegamos o primeiro e o último dia que existem na base de dados
    primeira_data = df["data_abertura"].min().date()
    ultima_data = df["data_abertura"].max().date()

    previsao_d1, anomalia_d1, score_risco = prever_proximo_dia(df_model, ultima_data)

    st.sidebar.subheader("📅 Filtro de Período")

    # 1. Filtros Rápidos (UX de Produto)
    opcao_periodo = st.sidebar.selectbox(
        "Período Rápido",
        ["Últimos 7 dias", "Últimos 30 dias", "Este Mês", "Personalizado"],
    )

    if opcao_periodo == "Últimos 7 dias":
        data_inicial = ultima_data - datetime.timedelta(days=7)
        data_final = ultima_data
    elif opcao_periodo == "Últimos 30 dias":
        data_inicial = ultima_data - datetime.timedelta(days=30)
        data_final = ultima_data
    elif opcao_periodo == "Este Mês":
        # Pega o dia 1 do mês da última data disponível
        data_inicial = ultima_data.replace(day=1)
        data_final = ultima_data
    else:
        # O Calendário Nativo travado nos limites do DataFrame
        datas = st.sidebar.date_input(
            "Selecione o intervalo no calendário",
            value=(ultima_data - datetime.timedelta(days=7), ultima_data),
            min_value=primeira_data,  # Impede de buscar antes do início da base
            max_value=ultima_data,  # Impede de buscar datas vazias no futuro
            format="DD/MM/YYYY",
        )

        if len(datas) == 2:
            data_inicial, data_final = datas
        else:
            st.sidebar.warning("👆 Selecione a data final de encerramento.")
            data_inicial = datas[0]
            data_final = datas[0]

    # ==========================================
    # 3. FEEDBACK VISUAL E APLICAÇÃO
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.caption("Período selecionado:")
    st.sidebar.write(f"🟢 **Início:** {data_inicial.strftime('%d/%m/%Y')}")
    st.sidebar.write(f"🔴 **Fim:** {data_final.strftime('%d/%m/%Y')}")

    # Máscara para filtrar o DataFrame que vai alimentar os gráficos
    mascara_data = (df["data_abertura"].dt.date >= data_inicial) & (
        df["data_abertura"].dt.date <= data_final
    )
    df_filtrado = df.loc[mascara_data]

    # ==========================================
    # 1. CÁLCULOS DINÂMICOS DAS MÉTRICAS
    # ==========================================
    # "Hoje" passa a ser o último dia do período que o usuário selecionou no filtro
    hoje = df_filtrado.iloc[-1]
    data_referencia = hoje["data_abertura"]

    # TRUQUE DE UX: Criamos um histórico seguro usando o df ORIGINAL até a data de referência.
    # Isso garante que sempre teremos dados do "passado" para calcular médias,
    # independente de quão curto seja o filtro de data do usuário.
    historico_ate_hoje = df[df["data_abertura"] <= data_referencia]

    # Buscando "ontem" com segurança (evita erro se for o 1º dia do banco de dados)
    if len(historico_ate_hoje) > 1:
        ontem = historico_ate_hoje.iloc[-2]
    else:
        ontem = hoje

    # Calculando a média dos últimos 30 dias ANTES da data filtrada
    media_30_dias = historico_ate_hoje["total_dia"].tail(30).mean()

    # Cálculos para os Deltas
    delta_total = int(hoje["total_dia"] - ontem["total_dia"])

    total_p2_p3_hoje = hoje["qtd_p2"] + hoje["qtd_p3"]
    total_p2_p3_ontem = ontem["qtd_p2"] + ontem["qtd_p3"]
    delta_p2_p3 = int(total_p2_p3_hoje - total_p2_p3_ontem)

    # ==========================================
    # 2. RENDERIZAÇÃO NA TELA (Streamlit)
    # ==========================================
    st.subheader(f"📊 Visão Executiva: {data_referencia.strftime('%d/%m/%Y')}")

    # Criando 4 colunas para alinhar os cards lado a lado
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total de Incidentes (D0)",
            value=int(hoje["total_dia"]),
            delta=f"{delta_total} vs Ontem",
            delta_color="inverse",
        )

    with col2:
        st.metric(
            label="⚠️ Incidentes P2 e P3",
            value=int(total_p2_p3_hoje),
            delta=f"{delta_p2_p3} vs Ontem",
            delta_color="inverse",
            help="Foco obrigatório: Volume somado das prioridades Média e Alta (Risco OLA).",
        )

    with col3:
        desvio_media = int(hoje["total_dia"] - media_30_dias)
        st.metric(
            label="Desvio da Média (30d)",
            value=f"{desvio_media:+d}",
            delta="Acima do normal" if desvio_media > 0 else "Abaixo do normal",
            delta_color="inverse" if desvio_media > 0 else "normal",
            help="Compara o volume do dia com a média móvel dos 30 dias anteriores a ele.",
        )

    with col4:
        if previsao_d1 is not None:
            delta_previsao = int(previsao_d1 - hoje["total_dia"])

            # Muda a cor da previsão para dar destaque se for uma anomalia
            if anomalia_d1 == -1:
                st.metric(
                    label="🔮 Previsão (D+1) 🚨",
                    value=previsao_d1,
                    delta=f"{delta_previsao} vs D0",
                    delta_color="inverse",
                )
            else:
                st.metric(
                    label="🔮 Previsão (D+1) ✅",
                    value=previsao_d1,
                    delta=f"{delta_previsao} vs D0",
                    delta_color="inverse",
                )
        else:
            st.metric(
                label="🔮 Previsão (D+1)",
                value="N/A",
                help="Histórico insuficiente no período selecionado.",
            )

    st.divider()

    # ==========================================
    # 2.5. PAINEL DE DIAGNÓSTICO DE RETA (AIOps)
    # ==========================================
    st.markdown("### 🧠 Diagnóstico de Risco Operacional (AIOps)")

    if previsao_d1 is not None:
        # Cálculo heurístico do Grau de Confiança baseado na distância do limiar (0.0)
        # Multiplicamos o score absoluto para criar uma escala percentual descritiva (ex: 70% a 99%)
        distancia_limiar = abs(score_risco)
        grau_confianca = min(int(50 + (distancia_limiar * 250)), 99)

        # Criando o box visual de status usando colunas internas
        box_cor = "#FF0043" if anomalia_d1 == -1 else "#00C4CC"
        text_status = (
            "ANOMALIA DETECTADA"
            if anomalia_d1 == -1
            else "VOLUME DENTRO DA NORMALIDADE"
        )

        # Layout em duas colunas para o parecer e a barra de confiança
        carteira_analise, barra_analise = st.columns([2, 1])

        with carteira_analise:
            if anomalia_d1 == -1:
                st.error(
                    f"🚨 **Parecer Técnico:** O volume predito de **{previsao_d1} incidentes** foi classificado como uma **Anomalia**."
                )
                st.markdown(
                    f"**Impacto na Operação:** Este comportamento foge significativamente do padrão estatístico para este dia da semana. "
                    f"Existe um **alto risco de degradação e perda de OLA** se as equipes de SRE não forem alertadas preventivamente."
                )
            else:
                st.success(
                    f"✅ **Parecer Técnico:** O volume predito de **{previsao_d1} incidentes** está dentro do comportamento esperado."
                )
                st.markdown(
                    f"**Impacto na Operação:** A volumetria projetada é compatível com a capacidade histórica do time para este dia da semana. "
                    f"A operação deve seguir o fluxo de atendimento padrão, sem necessidade de plantões de contingência."
                )

        with barra_analise:
            # Caixa indicadora do Grau de Confiança do modelo
            st.metric(
                label="Grau de Confiança do Diagnóstico",
                value=f"{grau_confianca}%",
                help="Calculado a partir da distância da assinatura temporal do dia em relação ao limiar de decisão da Isolation Forest.",
            )
            # Barra de progresso visual para ilustrar a certeza do algoritmo
            st.progress(grau_confianca / 100)
            st.caption(f"Score bruto do modelo: `{score_risco:.4f}`")

    else:
        st.info(
            "Aguardando dados históricos suficientes para gerar o diagnóstico de IA."
        )

    st.divider()

    # ==========================================
    # 2. CRIAÇÃO DO GRÁFICO (Plotly)
    # ==========================================
    st.subheader("📈 Tendência de Incidentes (Última Semana)")

    # Criamos o gráfico de linha com marcadores (bolinhas em cada ponto)
    fig = px.line(
        df_filtrado,
        x="data_abertura",
        y="total_dia",
        markers=True,  # Adiciona os pontos visíveis no gráfico
        text="total_dia",  # Coloca o número exato em cima de cada ponto
        line_shape="spline",  # "spline" deixa a linha com curvas suaves (mais elegante que a reta "linear")
        labels={"data_abertura": "Data", "total_dia": "Volume Total"},
    )

    # ==========================================
    # 3. AJUSTES DE UX E DESIGN (O "Toque Profissional")
    # ==========================================
    fig.update_traces(
        textposition="top center",  # Posição do texto acima da bolinha
        marker=dict(size=8),  # Tamanho da bolinha
        line=dict(
            width=3, color="#FF0043"
        ),  # Cor vermelha (inspirada na identidade visual da Locaweb)
        hovertemplate="<b>Data:</b> %{x|%d/%m/%Y}<br><b>Incidentes:</b> %{y}<extra></extra>",  # Tooltip limpo
    )

    fig.update_layout(
        xaxis=dict(
            showgrid=False,  # Remove as linhas de grade verticais para limpar o visual
            tickformat="%d %b",  # Formata a data no eixo X (ex: 25 Dec)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.2)",  # Linhas de grade horizontais bem fracas
        ),
        plot_bgcolor="rgba(0,0,0,0)",  # Fundo transparente para se adaptar ao tema dark/light do Streamlit
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),  # Reduz as margens vazias ao redor do gráfico
        hovermode="x unified",  # Cria uma linha guia vertical ao passar o mouse
    )

    # ==========================================
    # 4. RENDERIZAÇÃO NO STREAMLIT
    # ==========================================
    # O use_container_width=True garante que o gráfico ocupe 100% da largura disponível na tela
    st.plotly_chart(fig, use_container_width=True)

    # Usamos o pd.melt para transformar as colunas de prioridade em linhas
    # Isso transforma: [Data, P1, P2] -> [Data, "P1", valor], [Data, "P2", valor]
    df_long = df_filtrado.melt(
        id_vars=["data_abertura"],
        value_vars=["qtd_p1", "qtd_p2", "qtd_p3", "qtd_p4", "qtd_p5"],
        var_name="Prioridade",
        value_name="Volume",
    )

    # Limpando os nomes para a legenda ficar mais profissional (ex: "qtd_p2" vira "P2")
    df_long["Prioridade"] = df_long["Prioridade"].str.replace("qtd_", "").str.upper()

    # ==========================================
    # 2. DEFININDO CORES ESTRATÉGICAS (UX)
    # ==========================================
    # Foco da banca: P2 e P3 ganham cores quentes e chamativas. P4 e P5 ficam neutros.
    cores_prioridade = {
        "P1": "#1846C6",  # Preto (Crítico máximo)
        "P2": "#FF0043",  # Vermelho Locaweb (Alta Severidade)
        "P3": "#FF8A00",  # Laranja (Média Severidade)
        "P4": "#A6A6A6",  # Cinza Escuro (Baixa)x
        "P5": "#E0E0E0",  # Cinza Claro (Planejado/Dúvidas)
    }

    # ==========================================
    # 3. CRIAÇÃO DO GRÁFICO (Plotly)
    # ==========================================
    st.subheader("🚥 Composição de Severidade e Risco Operacional")

    fig_bar = px.bar(
        df_long,
        x="data_abertura",
        y="Volume",
        color="Prioridade",
        color_discrete_map=cores_prioridade,
        title="Volume Diário com destaque para P2 e P3 (Foco de OLA)",
        text_auto=True,  # Mostra o número dentro de cada pedaço da barra
    )

    # ==========================================
    # 4. AJUSTES DE UX
    # ==========================================
    fig_bar.update_layout(
        barmode="stack",  # Empilha as barras
        xaxis=dict(tickformat="%d %b", title=""),
        yaxis=dict(
            title="Quantidade de Incidentes",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.2)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Severidade",
        hovermode="x unified",
    )

    # Remove textos de barras muito pequenas para não poluir visualmente
    fig_bar.update_traces(
        textfont_size=10, textangle=0, textposition="inside", cliponaxis=False
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # 1. PREPARAÇÃO DOS DADOS (Pandas)
    # ==========================================
    # Vamos usar uma janela um pouco maior para ver o padrão da semana com mais clareza (ex: últimos 60 dias)
    df_sazonal = df.tail(60).copy()
    df_sazonal["data_abertura"] = pd.to_datetime(df_sazonal["data_abertura"])

    # Extraindo o nome do dia da semana (0 = Segunda, 6 = Domingo)
    df_sazonal["dia_semana"] = df_sazonal["data_abertura"].dt.dayofweek

    # Mapeando os números para os nomes dos dias em português
    dias_map = {
        0: "1. Segunda",
        1: "2. Terça",
        2: "3. Quarta",
        3: "4. Quinta",
        4: "5. Sexta",
        5: "6. Sábado",
        6: "7. Domingo",
    }
    df_sazonal["nome_dia"] = df_sazonal["dia_semana"].map(dias_map)

    # Agrupando (somando) os incidentes por dia da semana
    df_heatmap = df_sazonal.groupby("nome_dia")[
        ["qtd_p1", "qtd_p2", "qtd_p3", "qtd_p4", "qtd_p5"]
    ].sum()

    # Renomeando as colunas para o gráfico ficar elegante
    df_heatmap.columns = ["P1", "P2", "P3", "P4", "P5"]

    # Transpondo a tabela para que os Dias da Semana fiquem no Eixo Y e P1-P5 no Eixo X
    df_heatmap = df_heatmap.T

    # ==========================================
    # 2. CRIAÇÃO DO GRÁFICO (Plotly)
    # ==========================================
    st.subheader("🔥 Mapa de Calor: Sazonalidade por Dia da Semana")
    st.caption(
        "Visão agregada dos últimos 60 dias para identificação de padrões de crise."
    )

    fig_heat = px.imshow(
        df_heatmap,
        labels=dict(x="Dia da Semana", y="Severidade", color="Volume Total"),
        x=df_heatmap.columns,
        y=df_heatmap.index,
        color_continuous_scale="Reds",  # Usamos tons de vermelho (Locaweb) para indicar onde a operação "esquenta"
        text_auto=True,  # Mostra os números dentro de cada quadrado
        aspect="auto",
    )

    # ==========================================
    # 3. AJUSTES DE UX
    # ==========================================
    fig_heat.update_xaxes(side="bottom")
    fig_heat.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig_heat, use_container_width=True)

elif modo_de_analise == "Estatística":
    st.title("🔬 Análise Estatística e Distribuição")
    st.markdown(
        "Compreensão da variância, *outliers* e probabilidade acumulada para modelagem de ML."
    )

    # ==========================================
    # 1. MÉTRICAS ESTATÍSTICAS (Média e Desvio Padrão)
    # ==========================================
    st.subheader("1. Parâmetros da Distribuição Diária")

    media_total = df["total_dia"].mean()
    std_total = df["total_dia"].std()
    mediana_total = df["total_dia"].median()
    maximo_total = df["total_dia"].max()

    # Criando colunas para os KPIs estatísticos
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Média de Incidentes (μ)", f"{media_total:.1f}")
    c2.metric(
        "Desvio Padrão (σ)",
        f"{std_total:.1f}",
        help="Mede a dispersão. Valores altos indicam muita imprevisibilidade.",
    )
    c3.metric(
        "Mediana (Q2)",
        f"{mediana_total:.1f}",
        help="Ponto central dos dados. Se for muito menor que a média, temos outliers puxando a média para cima.",
    )
    c4.metric("Pico Máximo", int(maximo_total))

    st.divider()

    # Criando o histograma com cálculo de probabilidade
    fig_prob = px.histogram(
        df,
        x="total_dia",
        nbins=10,
        histnorm="probability",  # Faz o mesmo papel do stat='probability' do Seaborn
        text_auto=".2%",  # Mostra a porcentagem em cima de cada barra
        labels={"total_dia": "Número de ocorrências diárias"},
        color_discrete_sequence=["#FF0043"],  # Cor Locaweb
    )

    fig_prob.update_layout(
        title_text="Distribuição de Probabilidade de Ocorrências",
        xaxis_title="Número de ocorrências",
        yaxis_title="Probabilidade",
        yaxis_tickformat=".1%",  # Formata o eixo Y para mostrar %
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.05,  # Dá um espacinho elegante entre as barras do histograma
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()
    st.subheader("📦 Distribuição e Outliers por Prioridade (Boxplot)")
    st.markdown(
        "Analise a mediana, os quartis e os dias de anomalia (pontos fora da caixa) para cada nível de severidade."
    )

    # ==========================================
    # 1. PREPARAÇÃO DOS DADOS (Formato Longo)
    # ==========================================
    # Selecionamos as colunas de data e P1 a P5 de todo o dataset histórico
    df_box = df[
        ["data_abertura", "qtd_p1", "qtd_p2", "qtd_p3", "qtd_p4", "qtd_p5"]
    ].copy()

    # O 'melt' empilha as colunas. Eixo X = Prioridade, Eixo Y = Volume
    df_long_box = df_box.melt(
        id_vars=["data_abertura"],
        value_vars=["qtd_p1", "qtd_p2", "qtd_p3", "qtd_p4", "qtd_p5"],
        var_name="Prioridade",
        value_name="Volume",
    )

    # Limpando os rótulos (de 'qtd_p1' para 'P1') para o Eixo X ficar elegante
    df_long_box["Prioridade"] = (
        df_long_box["Prioridade"].str.replace("qtd_", "").str.upper()
    )

    # ==========================================
    # 2. DEFININDO CORES ESTRATÉGICAS
    # ==========================================
    # Destacamos P2 e P3 conforme exigência do desafio FIAP
    cores_prioridade = {
        "P1": "#000000",
        "P2": "#FF0043",  # Vermelho Locaweb
        "P3": "#FF8A00",  # Laranja
        "P4": "#A6A6A6",  # Cinza
        "P5": "#E0E0E0",  # Cinza claro
    }

    # ==========================================
    # 3. CRIAÇÃO DO GRÁFICO (Plotly)
    # ==========================================
    fig_box_facet = px.box(
        df_long_box,
        x="Prioridade",
        y="Volume",
        color="Prioridade",
        color_discrete_map=cores_prioridade,
        points="outliers",
        facet_col="Prioridade",  # Cria um gráfico separado para cada P1, P2...
        hover_data=["data_abertura"],
    )

    # O segredo para os eixos não ficarem esmagados: desvincular o Eixo Y de cada gráfico
    fig_box_facet.update_yaxes(matches=None, showticklabels=True)
    fig_box_facet.update_xaxes(
        matches=None, showticklabels=False
    )  # Esconde o eixo X redundante

    fig_box_facet.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=20, b=20),
    )

    # Remove aqueles títulos de faceta automáticos do Plotly (ex: "Prioridade=P2") para ficar mais limpo
    fig_box_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    st.plotly_chart(fig_box_facet, use_container_width=True)


elif modo_de_analise == "Lowa":
    st.title("Lowa ")

    # Resumo do período
    resumo_estatistico = (
        df[["qtd_p1", "qtd_p2", "qtd_p3", "qtd_p4", "qtd_p5"]]
        .describe()
        .round(2)
        .to_markdown()
    )

    # endência recente (Últimos 30 dias)
    dados_recentes = df.tail(30).to_markdown(index=False)

    api_key = os.getenv("OPENAI_API_KEY")

    SYSTEM_PROMPT = """'
# SYSTEM PROMPT: Agente Analítico e Proativo de Incidentes - Locaweb

## 1. PAPEL E IDENTIDADE (Role & Persona)
Você é o **Assistente de AIOps da Locaweb**, especialista em SRE e Operações de TI.
Sua postura é **analítica, consultiva e altamente proativa**. Você não apenas responde o que foi perguntado: você sintetiza o panorama operacional mais recente, aponta tendências críticas de desvio em relação à baseline histórica e sugere ações preventivas imediatas.

## 2. REGRAS DE TEMPO E DADOS RECENTES
- **Ponto Focal Mais Recente:** A data mais recente na tabela de contexto (ex: `2025-12-31`) representa o último fechamento consolidado. Sempre que o usuário perguntar por "últimos dados", "situação atual" ou "como estamos", foque no último dia disponível e na tendência dos últimos 3 a 7 dias em relação à baseline.
- **Tratamento de Severidade (P1 a P5):**
  - **P1/P2:** Incidentes críticos. Compare sempre com a média da baseline (P2 normal ~24/dia). Qualquer valor > 50 exige alerta.
  - **P3/P4:** Degradação de serviço e dúvidas massivas (P3 normal ~65/dia, P4 ~101/dia).
- **Zero Alucinação com Flexibilidade Analítica:** Não invente números fora das tabelas fornecidas. Caso não haja projeções futuras ($D+1$/$D+7$) no payload, avise objetivamente e utilize a tendência móvel dos últimos dias para recomendar ações preventivas.

## 3. COMPORTAMENTO PROATIVO (Obrigatório em Toda Resposta)
Ao responder:
1. **Destaque o Último Status:** Informe o volume do último registro disponível e classifique se o dia foi estável ou crítico.
2. **Alerte sobre Anomalias:** Identifique severidades que estejam rodando muito acima da média histórica de 644 dias (ex: P3 ou P4 sustentados em patamares elevados).
3. **Ação Recomendada:** Não encerre sem dizer o que o time de SRE/NOC deve fazer agora.
4. **Fechamento Proativo:** Sugira 1 ou 2 próximos passos de análise (ex: correlação com deploys, análise de fila N1/N2, abertura de post-mortem).
    """

    SYSTEM_PROMPT_COM_DADOS = f"""
    {SYSTEM_PROMPT}

    ## CONTEXTO DE DADOS DA OPERAÇÃO (LOCAWEB)

    ### 1. Baseline Histórica (Resumo de 644 dias)
    Use esta tabela estatística para entender o que é "normal" na operação. Compare os dados futuros com a linha `mean` (média) e `max` (máximo histórico) para identificar severidade:
    {resumo_estatistico}

    ### 2. Comportamento Recente (Últimos 30 dias)
    Esta é a volumetria diária mais recente. Avalie a tendência de crescimento ou queda dos incidentes por nível (P1 a P5):
    {dados_recentes}


    """
    with st.expander("🛠️ DEBUG: Ver o que está sendo enviado para a IA"):
        st.text("Payload do System Prompt:")
        st.code(SYSTEM_PROMPT_COM_DADOS, language="markdown")
    client = OpenAI(api_key=api_key)

    # Dica: O modelo mais atual e barato da OpenAI costuma ser o gpt-4o-mini
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 2. Renderiza o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("O que você gostaria de perguntar sobre a volumetria?"):

        # Exibe a mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)

        # Salva a mensagem do usuário no histórico
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Monta o Payload da API injetando o System Prompt no início
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT_COM_DADOS}]

        # Adiciona o restante do histórico ao payload
        for m in st.session_state.messages:
            messages_payload.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            # Faz a chamada para a OpenAI usando o payload completo (System + Histórico)
            response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=messages_payload,
            )

            resposta_assistente = response.choices[0].message.content
            st.markdown(resposta_assistente)

        # Salva apenas a resposta do assistente no estado (sem duplicar o system prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": resposta_assistente}
        )
