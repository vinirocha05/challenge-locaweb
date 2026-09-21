# Locaweb Data Trust

Projeto desenvolvido para o desafio da FIAP em parceria com a Locaweb. A solução utiliza dados, análises e recursos de inteligência artificial para apoiar a identificação de padrões e a geração de insights relacionados à confiança e segurança de dados.

## Acesso à solução

Para ter a experiência completa da aplicação, acesse:

[https://locaweb-datatrust.streamlit.app](https://locaweb-datatrust.streamlit.app)

> **Importante:** a execução local da solução não funcionará completamente sem uma chave válida da API da OpenAI. Essa chave é necessária para os recursos que utilizam inteligência artificial.

## Funcionalidades desenvolvidas

- Aplicação web construída com Streamlit.
- Análise e visualização dos dados.
- Uso de modelos de machine learning para apoiar as análises.
- Integração com a API da OpenAI.
- Geração de respostas e insights com base nos dados fornecidos.
- Interface interativa para facilitar a exploração dos resultados.

## Execução local

Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

Configure a variável de ambiente com sua chave da OpenAI:

```bash
export OPENAI_API_KEY="sua-chave-da-openai"
```

Em seguida, execute a aplicação:

```bash
streamlit run app.py
```

Caso o arquivo principal possua outro nome, substitua `/src/dashboard/main.py` pelo arquivo correspondente.

## Tecnologias utilizadas

- Python
- Streamlit
- OpenAI API
- Pandas
- Scikit-learn
- LightGBM
- XGBoost
- Plotly
- Matplotlib
- Seaborn
