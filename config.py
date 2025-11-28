# config.py
# CONFIGURAÇÃO GERAL DO ROBÔ (Versão 8.0)

import streamlit as st
import os # Importa a biblioteca para ler variáveis do sistema

# --- CONFIGURAÇÃO DA API DE DADOS (football-data.org) ---
# Tenta ler a chave do "Cofre de Segredos" (Secrets) do Streamlit ou variáveis de ambiente
API_KEY = st.secrets.get("FOOTBALL_DATA_TOKEN", os.environ.get("FOOTBALL_DATA_TOKEN"))
API_BASE_URL = "https://api.football-data.org/v4/"

# Temporada para analisar (2025 é o padrão atual)
TEMPORADA_PARA_ANALISAR = 2025 

# Para o cálculo de Poisson (Máximo de gols a simular por time)
MAX_GOLS_CALCULO = 6

# --- CONFIGURAÇÃO DA API DE ODDS (The-Odds-API) ---
# NOVO: Chave para buscar odds automáticas
THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", os.environ.get("THE_ODDS_API_KEY"))

# --- CONFIGURAÇÃO DO TELEGRAM ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.environ.get("TELEGRAM_TOKEN"))
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID"))
