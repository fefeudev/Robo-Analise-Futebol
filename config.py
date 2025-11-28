# config.py
# NOVO ARQUIVO DE CONFIGURAÇÃO (Para o "Modo Nuvem")

import streamlit as st
import os # Importa a biblioteca para ler variáveis do sistema

# --- CONFIGURAÇÃO DA API (football-data.org) ---
# Tenta ler a chave do "Cofre de Segredos" (Secrets) do Streamlit
API_KEY = st.secrets.get("FOOTBALL_DATA_TOKEN", os.environ.get("FOOTBALL_DATA_TOKEN"))
API_BASE_URL = "https://api.football-data.org/v4/"

# Temporada para analisar (podemos deixar fixo ou adicionar aos Secrets)
TEMPORADA_PARA_ANALISAR = 2025 

# Para o cálculo de Poisson
MAX_GOLS_CALCULO = 6

# --- CONFIGURAÇÃO DO TELEGRAM ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.environ.get("TELEGRAM_TOKEN"))

TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID"))

# Adicione no final do config.py
THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", "SUA_CHAVE_DA_API_AQUI")
