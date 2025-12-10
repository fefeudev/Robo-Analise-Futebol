# app.py
# Versão 8.6 - CLEAN (Apenas H2H + Chance Dupla Calculada)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats 
import config 
import time
from datetime import datetime, timedelta
import json
from difflib import get_close_matches

# --- IMPORTS DO BANCO DE DADOS ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Robô Híbrido Clean",
    page_icon="🎯", 
    layout="wide"
)

# --- CSS CUSTOMIZADO ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    h1, h2, h3 { color: #E6EDF3; }
    [data-testid="stMetric"] { background-color: #21262D; border: 1px solid #30363D; border-radius: 8px; padding: 10px; }
    [data-testid="stMetricLabel"] { color: #8B949E; }
    [data-testid="stMetricValue"] { color: #E6EDF3; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🌐 INTEGRAÇÃO COM THE ODDS API
# ==============================================================================

MAPA_LIGAS_ODDS = {
    "CL": "soccer_uefa_champs_league",
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "BSA": "soccer_brazil_campeonato",
    "PPL": "soccer_portugal_primeira_liga",
    "DED": "soccer_netherlands_eredivisie"
}

@st.cache_data(ttl=3600)
def buscar_odds_mercado(codigo_liga_app):
    try:
        api_key = st.secrets["THE_ODDS_API_KEY"]
    except:
        return None 
        
    sport_key = MAPA_LIGAS_ODDS.get(codigo_liga_app)
    if not sport_key: return None
    
    # LIMPEZA: Pedindo APENAS 'h2h' (Vencedor)
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'apiKey': api_key, 'regions': 'eu,uk', 'markets': 'h2h', 'oddsFormat': 'decimal'}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200: return r.json()
    except: pass
    return []

def encontrar_jogo_fuzzy(lista_odds, time_casa_robo, time_fora_robo):
    if not lista_odds: return None
    times_api = [j['home_team'] for j in lista_odds]
    match = get_close_matches(time_casa_robo, times_api, n=1, cutoff=0.4)
    if match:
        nome_real = match[0]
        for jogo in lista_odds:
            if jogo['home_team'] == nome_real: return jogo
    return None

def calcular_odds_chance_dupla(odd_1, odd_x, odd_2):
    """Calcula matematicamente a odd da Chance Dupla baseada nas odds 1x2"""
    try:
        if odd_1 > 0 and odd_x > 0 and odd_2 > 0:
            odd_1x = (odd_1 * odd_x) / (odd_1 + odd_x)
            odd_x2 = (odd_x * odd_2) / (odd_x + odd_2)
            odd_12 = (odd_1 * odd_2) / (odd_1 + odd_2)
            return odd_1x, odd_x2, odd_12
    except: pass
    return 0, 0, 0

# ==============================================================================
# 🧠 FUNÇÕES DO BANCO DE DADOS & UTILITÁRIOS
# ==============================================================================

@st.cache_resource 
def conectar_ao_banco_de_dados():
    try:
        creds_dict = dict(st.secrets.google_creds)
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(st.secrets.GOOGLE_SHEET_URL).sheet1
        return sheet
    except: return None

def salvar_analise_no_banco(sheet, data, liga, jogo, mercado, odd, prob_robo, valor):
    if sheet:
        try:
            sheet.append_row([data, liga, jogo, mercado, float(odd), float(prob_robo)/100, float(valor)/100, "Aguardando ⏳"], value_input_option='USER_ENTERED')
        except: pass

@st.cache_data(ttl=60) 
def carregar_historico_do_banco(_sheet):
    try:
        data = _sheet.get_all_values()
        if len(data) < 2: return pd.DataFrame(), 0, 0
        df = pd.DataFrame(data[1:], columns=data[0])
        cols = ['Odd', 'Probabilidade', 'Valor']
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(',', '.')
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=cols)
        greens = df['Status'].value_counts().get('Green ✅', 0) if 'Status' in df.columns else 0
        reds = df['Status'].value_counts().get('Red ❌', 0) if 'Status' in df.columns else 0
        return df, greens, reds
    except: return pd.DataFrame(), 0, 0

@st.cache_data 
def criar_headers_api(): return {"X-Auth-Token": config.API_KEY}

def fazer_requisicao_api(endpoint, params):
    try:
        return requests.get(config.API_BASE_URL + endpoint, headers=criar_headers_api(), params=params).json()
    except: return None

# ==============================================================================
# 🧠 CÉREBRO DIXON-COLES
# ==============================================================================

@st.cache_data
def carregar_cerebro_dixon_coles(id_liga):
    try:
        with open(f"dc_params_{id_liga}.json", 'r') as f: return json.load(f)
    except: return None

def prever_jogo_dixon_coles(dados, t1, t2):
    try:
        f = dados['forcas']
        adv = dados['vantagem_casa']
        rho = dados.get('rho', 0.0)
        if t1 not in f or t2 not in f: return None, None

        l = np.exp(f[t1]['ataque'] + f[t2]['defesa'] + adv)
        m = np.exp(f[t2]['ataque'] + f[t1]['defesa'])
        
        def tau(x, y, l, m, r):
            if r == 0: return 1
            if x==0 and y==0: return 1 - (l*m*r)
            if x==1 and y==0: return 1 + (l*r)
            if x==0 and y==1: return 1 + (m*r)
            if x==1 and y==1: return 1 - r
            return 1

        probs = np.zeros((7,7))
        for i in range(7):
            for j in range(7):
                probs[i,j] = stats.poisson.pmf(i, l) * stats.poisson.pmf(j, m) * tau(i,j,l,m,rho)
        
        home = np.sum(np.tril(probs, -1))
        draw = np.sum(np.diag(probs))
        away = np.sum(np.triu(probs, 1))
        
        total = home+draw+away
        
        # LIMPEZA: Retornando apenas H2H e Chance Dupla
        res = {
            'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
            'chance_dupla_1X': (home+draw)/total, 'chance_dupla_X2': (draw+away)/total, 'chance_dupla_12': (home+away)/total
        }
        return res, (l, m)
    except: return None, None

@st.cache_data
def buscar_jogos_por_data(id_liga, data_str):
    d = fazer_requisicao_api(f"competitions/{id_liga}/matches", {"dateFrom": data_str, "dateTo": data_str})
    if not d or 'matches' not in d: return []
    return [{'time_casa': m['homeTeam']['name'], 'time_visitante': m['awayTeam']['name'], 'data': m['utcDate']} for m in d['matches']]

def enviar_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        requests.get(url, params={'chat_id': config.TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'})
        st.toast("Enviado ao Telegram!", icon="✅")
    except: pass

# ==============================================================================
# 🖥️ INTERFACE (STREAMLIT)
# ==============================================================================

LIGAS = {"Champions League": "CL", "Premier League": "PL", "Brasileirão": "BSA", "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1"}
EMOJIS = {"CL": "🇪🇺", "PL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "BSA": "🇧🇷", "PD": "🇪🇸", "SA": "🇮🇹", "BL1": "🇩🇪"}

with st.sidebar:
    st.title("🎯 Robô H2H + DC")
    l_nome = st.selectbox("Liga:", LIGAS.keys())
    LIGA_ATUAL = LIGAS[l_nome]
    
    col1, col2 = st.columns(2)
    if 'data' not in st.session_state: st.session_state.data = datetime.now()
    if col1.button("< Ontem"): st.session_state.data -= timedelta(days=1)
    if col2.button("Amanhã >"): st.session_state.data += timedelta(days=1)
    data_sel = st.date_input("Data:", st.session_state.data)

    st.divider()
    prob_min = st.slider("Probabilidade Mínima %", 50, 90, 60) / 100

# --- CARREGA CÉREBRO ---
dados_dc = carregar_cerebro_dixon_coles(LIGA_ATUAL)

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Jogos (H2H + DC)", "📈 Histórico"])

with tab1:
    st.header(f"{EMOJIS.get(LIGA_ATUAL,'')} Jogos: {data_sel.strftime('%d/%m/%Y')}")
    
    jogos = buscar_jogos_por_data(LIGA_ATUAL, data_sel.strftime('%Y-%m-%d'))
    
    if not jogos:
        st.warning("Nenhum jogo encontrado nesta data.")
    else:
        odds_api = buscar_odds_mercado(LIGA_ATUAL)
        
        for i, jogo in enumerate(jogos):
            with st.expander(f"⚽ {jogo['time_casa']} x {jogo['time_visitante']}"):
                
                jogo_odds = encontrar_jogo_fuzzy(odds_api, jogo['time_casa'], jogo['time_visitante'])
                
                col_res1, col_res2 = st.columns([1.2, 1.8])
                
                odds_reais = {}
                
                with col_res1:
                    st.markdown("#### 🏦 Odds (Mercado)")
                    if jogo_odds:
                        # Prioridade: Pinnacle > Bet365 > 1xBet
                        bookie = next((b for b in jogo_odds['bookmakers'] if b['key'] == 'pinnacle'), None)
                        if not bookie: bookie = next((b for b in jogo_odds['bookmakers'] if b['key'] in ['bet365', 'onexbet']), None)
                        
                        if bookie:
                            st.caption(f"Fonte: {bookie['title']}")
                            
                            # --- 1. VENCEDOR (1x2) ---
                            outs_h2h = next((m['outcomes'] for m in bookie['markets'] if m['key'] == 'h2h'), [])
                            dict_h2h = {o['name']: o['price'] for o in outs_h2h}
                            
                            h = dict_h2h.get(jogo_odds['home_team']) or dict_h2h.get('Home', 0)
                            d = dict_h2h.get('Draw', 0)
                            a = dict_h2h.get(jogo_odds['away_team']) or dict_h2h.get('Away', 0)
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Casa", h)
                            c2.metric("Emp", d)
                            c3.metric("Fora", a)
                            
                            # --- 2. CHANCE DUPLA CALCULADA ---
                            if h and d and a:
                                dc_1x, dc_x2, dc_12 = calcular_odds_chance_dupla(h, d, a)
                                st.markdown("**Dupla Chance (Calc.)**")
                                c4, c5, c6 = st.columns(3)
                                c4.metric("1X", f"{dc_1x:.2f}")
                                c5.metric("X2", f"{dc_x2:.2f}")
                                c6.metric("12", f"{dc_12:.2f}")
                                
                                odds_reais = {'vitoria_casa': h, 'empate': d, 'vitoria_visitante': a, 'chance_dupla_1X': dc_1x, 'chance_dupla_X2': dc_x2, 'chance_dupla_12': dc_12}
                        else:
                            st.warning("Odds indisponíveis na região.")
                    else:
                        st.info("Jogo não mapeado na API Odds.")
                        h = st.number_input("Casa", 1.0, key=f"h{i}")
                        d = st.number_input("Emp", 1.0, key=f"d{i}")
                        a = st.number_input("Fora", 1.0, key=f"a{i}")
                        odds_reais = {'vitoria_casa':h, 'empate':d, 'vitoria_visitante':a}

                with col_res2:
                    st.markdown("#### 🧠 Estatística & Valor")
                    if st.button("Calcular Probabilidades", key=f"b{i}"):
                        res, xg = prever_jogo_dixon_coles(dados_dc, jogo['time_casa'], jogo['time_visitante'])
                        
                        if res:
                            msg_telegram = f"🔥 <b>{LIGA_ATUAL}</b>: {jogo['time_casa']} x {jogo['time_visitante']}\n\n"
                            tem_valor = False
                            
                            st.caption("Comparação: Odd do Mercado vs Odd Justa do Robô")
                            col_p1, col_p2, col_p3 = st.columns(3)
                            
                            # Lista Exclusiva: H2H e Chance Dupla
                            mercados_analise = [
                                ('vitoria_casa', 'Casa', 0), ('empate', 'Empate', 1), ('vitoria_visitante', 'Fora', 2),
                                ('chance_dupla_1X', 'DC 1X', 0), ('chance_dupla_X2', 'DC X2', 2), ('chance_dupla_12', 'DC 12', 1)
                            ]
                            
                            for ch, nome, c_idx in mercados_analise:
                                prob = res.get(ch, 0)*100
                                odd = odds_reais.get(ch, 0)
                                delta = ""
                                color = "off"
                                
                                if odd > 1:
                                    justa = 100/prob if prob > 0 else 99
                                    ev = ((odd/justa)-1)*100
                                    
                                    if ev > 3.0 and prob > (prob_min*100):
                                        delta = f"+{ev:.1f}% EV"
                                        color = "normal"
                                        msg_telegram += f"✅ <b>{nome}</b>: Odd {odd:.2f} (EV +{ev:.1f}%)\n"
                                        tem_valor = True
                                        
                                        # Salva no Banco
                                        db = conectar_ao_banco_de_dados()
                                        salvar_analise_no_banco(db, data_sel.strftime('%Y-%m-%d'), LIGA_ATUAL, f"{jogo['time_casa']}x{jogo['time_visitante']}", nome, odd, prob, ev)
                                    elif ev > 0: delta = f"+{ev:.1f}%"
                                    else: 
                                        delta = f"{ev:.1f}%"
                                        color = "inverse"
                                
                                if c_idx==0: col_p1.metric(nome, f"{prob:.1f}%", delta, delta_color=color)
                                elif c_idx==1: col_p2.metric(nome, f"{prob:.1f}%", delta, delta_color=color)
                                else: col_p3.metric(nome, f"{prob:.1f}%", delta, delta_color=color)

                            if tem_valor:
                                if st.button("Enviar Telegram 📱", key=f"tg{i}"):
                                    enviar_telegram(msg_telegram)
                        else:
                            st.error("Sem dados estatísticos para este jogo.")

with tab2:
    st.header("Histórico")
    db = conectar_ao_banco_de_dados()
    if db:
        df, g, r = carregar_historico_do_banco(db)
        c1, c2 = st.columns(2)
        c1.metric("Greens", g)
        c2.metric("Reds", r)
        st.dataframe(df, use_container_width=True)
