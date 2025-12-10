# app.py
# Versão 8.0 - SUPER HÍBRIDO (Estatística + Odds API + Chance Dupla Calculada)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats 
import config 
import time
from datetime import datetime, timedelta
import json
from difflib import get_close_matches # Importante para achar os nomes dos times

# --- NOVOS IMPORTS DO BANCO DE DADOS ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# -------------------------------------

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Robô Híbrido Pro",
    page_icon="🤖", 
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
    div[data-testid="column"] { background-color: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🌐 INTEGRAÇÃO COM THE ODDS API & CÁLCULOS DE ODDS
# ==============================================================================

# Mapeamento: Sua Liga (App) -> Chave da API (The Odds)
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

@st.cache_data(ttl=3600) # Cache de 1 hora
def buscar_odds_mercado(codigo_liga_app):
    """Busca odds na API externa"""
    try:
        api_key = st.secrets["THE_ODDS_API_KEY"]
    except:
        return None # Chave não configurada
        
    sport_key = MAPA_LIGAS_ODDS.get(codigo_liga_app)
    if not sport_key: return None
    
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'apiKey': api_key, 'regions': 'eu,uk', 'markets': 'h2h', 'oddsFormat': 'decimal'}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200: return r.json()
    except: pass
    return []

def encontrar_jogo_fuzzy(lista_odds, time_casa_robo, time_fora_robo):
    """Encontra o jogo na lista da API usando nomes parecidos"""
    if not lista_odds: return None
    times_api = [j['home_team'] for j in lista_odds]
    match = get_close_matches(time_casa_robo, times_api, n=1, cutoff=0.4)
    if match:
        nome_real = match[0]
        for jogo in lista_odds:
            if jogo['home_team'] == nome_real: return jogo
    return None

def calcular_odds_chance_dupla(odd_1, odd_x, odd_2):
    """
    Calcula matematicamente a odd da Chance Dupla baseada nas odds 1x2.
    Fórmula: (OddA * OddB) / (OddA + OddB)
    """
    try:
        odd_1x = (odd_1 * odd_x) / (odd_1 + odd_x)
        odd_x2 = (odd_x * odd_2) / (odd_x + odd_2)
        odd_12 = (odd_1 * odd_2) / (odd_1 + odd_2)
        return odd_1x, odd_x2, odd_12
    except:
        return 0, 0, 0

# ==============================================================================
# 🧠 FUNÇÕES DO BANCO DE DADOS & CÉREBRO (MANTIDAS)
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

def atualizar_status_no_banco(sheet, row_index, novo_status):
    try:
        sheet.update_cell(row_index + 2, 8, novo_status)
        st.cache_data.clear()
        st.rerun()
    except: pass

# --- API FOOTBALL DATA & PREVISÕES ---
@st.cache_data 
def criar_headers_api(): return {"X-Auth-Token": config.API_KEY}

def fazer_requisicao_api(endpoint, params):
    try:
        return requests.get(config.API_BASE_URL + endpoint, headers=criar_headers_api(), params=params).json()
    except: return None

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
        
        # Fallback simples se nome não bater exato no JSON (Tenta achar direto ou pula)
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
        over25 = np.sum(probs) - (probs[0,0]+probs[1,0]+probs[0,1]+probs[2,0]+probs[0,2]+probs[1,1])
        btts = 1 - (np.sum(probs[0,:]) + np.sum(probs[:,0]) - probs[0,0])
        
        total = home+draw+away
        res = {
            'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
            'over_2_5': over25/total, 'btts_sim': btts/total,
            'chance_dupla_1X': (home+draw)/total, 'chance_dupla_X2': (draw+away)/total, 'chance_dupla_12': (home+away)/total
        }
        return res, (l, m)
    except: return None, None

def prever_jogo_poisson(forcas, medias, t1, t2):
    # (Código Poisson mantido simplificado para economizar espaço, segue a mesma lógica do DC)
    try:
        fa_c = forcas[t1]['atk']/medias['mc']
        fd_c = forcas[t1]['def']/medias['mv']
        fa_v = forcas[t2]['atk']/medias['mv']
        fd_v = forcas[t2]['def']/medias['mc']
        l = fa_c * fd_v * medias['mc']
        m = fa_v * fd_c * medias['mv']
        
        # Matriz simples Poisson (sem Rho)
        probs = np.outer([stats.poisson.pmf(i,l) for i in range(7)], [stats.poisson.pmf(j,m) for j in range(7)])
        home = np.sum(np.tril(probs, -1))
        draw = np.sum(np.diag(probs))
        away = np.sum(np.triu(probs, 1))
        
        total = home+draw+away
        if total == 0: return None, None
        
        # Over/BTTS simplificado
        return {
            'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
            'over_2_5': 0.5, 'btts_sim': 0.5, # Simplificado para fallback
            'chance_dupla_1X': (home+draw)/total, 'chance_dupla_X2': (draw+away)/total, 'chance_dupla_12': (home+away)/total
        }, (l, m)
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
    st.title("🤖 Robô Híbrido Pro")
    l_nome = st.selectbox("Liga:", LIGAS.keys())
    LIGA_ATUAL = LIGAS[l_nome]
    
    col1, col2 = st.columns(2)
    if 'data' not in st.session_state: st.session_state.data = datetime.now()
    if col1.button("< Ontem"): st.session_state.data -= timedelta(days=1)
    if col2.button("Amanhã >"): st.session_state.data += timedelta(days=1)
    data_sel = st.date_input("Data:", st.session_state.data)

    st.divider()
    prob_min = st.slider("Probabilidade Mínima %", 50, 90, 60) / 100
    st.info("O Robô busca odds na API e cruza com a estatística.")

# --- CARREGA CÉREBRO ---
dados_dc = carregar_cerebro_dixon_coles(LIGA_ATUAL)
MODO = "DIXON_COLES" if dados_dc else "FALHA"

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Analisar Jogos", "📈 Histórico"])

with tab1:
    st.header(f"{EMOJIS.get(LIGA_ATUAL,'')} Jogos: {data_sel.strftime('%d/%m/%Y')}")
    
    jogos = buscar_jogos_por_data(LIGA_ATUAL, data_sel.strftime('%Y-%m-%d'))
    
    if not jogos:
        st.warning("Nenhum jogo encontrado nesta data.")
    else:
        # Carrega Odds da API (Cacheado)
        odds_api = buscar_odds_mercado(LIGA_ATUAL)
        
        for i, jogo in enumerate(jogos):
            with st.expander(f"⚽ {jogo['time_casa']} x {jogo['time_visitante']}"):
                
                # 1. TENTA ENCONTRAR ODDS REAIS
                jogo_odds = encontrar_jogo_fuzzy(odds_api, jogo['time_casa'], jogo['time_visitante'])
                
                col_res1, col_res2 = st.columns([1, 2])
                
                # --- COLUNA 1: DADOS DO MERCADO ---
                odds_reais = {}
                with col_res1:
                    st.markdown("#### 🏦 Odds (Mercado)")
                    if jogo_odds:
                        # Busca Pinnacle ou Bet365
                        bookie = next((b for b in jogo_odds['bookmakers'] if b['key'] == 'pinnacle'), None)
                        if not bookie: bookie = next((b for b in jogo_odds['bookmakers'] if b['key'] in ['bet365', 'onexbet']), None)
                        
                        if bookie:
                            outs = {o['name']: o['price'] for o in bookie['markets'][0]['outcomes']}
                            h_odd = outs.get(jogo_odds['home_team']) or outs.get('Home')
                            d_odd = outs.get('Draw')
                            a_odd = outs.get(jogo_odds['away_team']) or outs.get('Away')
                            
                            st.caption(f"Fonte: {bookie['title']}")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Casa", h_odd)
                            c2.metric("Emp", d_odd)
                            c3.metric("Fora", a_odd)
                            
                            # CÁLCULO DA CHANCE DUPLA (NOVO!)
                            dc_1x, dc_x2, dc_12 = calcular_odds_chance_dupla(h_odd, d_odd, a_odd)
                            st.markdown("**Dupla Chance (Calc.)**")
                            c4, c5, c6 = st.columns(3)
                            c4.metric("1X", f"{dc_1x:.2f}")
                            c5.metric("X2", f"{dc_x2:.2f}")
                            c6.metric("12", f"{dc_12:.2f}")
                            
                            odds_reais = {
                                'vitoria_casa': h_odd, 'empate': d_odd, 'vitoria_visitante': a_odd,
                                'chance_dupla_1X': dc_1x, 'chance_dupla_X2': dc_x2, 'chance_dupla_12': dc_12
                            }
                        else:
                            st.warning("Odds indisponíveis na Pinnacle/Bet365.")
                    else:
                        st.info("Jogo não encontrado na The Odds API.")
                        # Inputs manuais se não achar
                        h_odd = st.number_input("Odd Casa", 1.0, key=f"ho{i}")
                        d_odd = st.number_input("Odd Empate", 1.0, key=f"do{i}")
                        a_odd = st.number_input("Odd Fora", 1.0, key=f"ao{i}")
                        odds_reais = {'vitoria_casa': h_odd, 'empate': d_odd, 'vitoria_visitante': a_odd}

                # --- COLUNA 2: CÉREBRO ESTATÍSTICO ---
                with col_res2:
                    st.markdown("#### 🧠 Cérebro (Estatística)")
                    if st.button("Analisar Valor", key=f"btn{i}"):
                        res, xg = prever_jogo_dixon_coles(dados_dc, jogo['time_casa'], jogo['time_visitante'])
                        
                        if res:
                            # EXIBIÇÃO DE xG
                            st.info(f"xG Previsto: {jogo['time_casa']} ({xg[0]:.2f}) x ({xg[1]:.2f}) {jogo['time_visitante']}")
                            
                            # RADAR DE VALOR (TABELA)
                            st.markdown("##### 💎 Radar de Valor (+EV)")
                            
                            cols_radar = st.columns(3)
                            mercados_analise = [
                                ('vitoria_casa', 'Casa', 0), ('empate', 'Empate', 1), ('vitoria_visitante', 'Fora', 2),
                                ('chance_dupla_1X', 'DC 1X', 0), ('chance_dupla_X2', 'DC X2', 2), ('chance_dupla_12', 'DC 12', 1)
                            ]
                            
                            msg_telegram = f"🔥 <b>ALERTA {LIGA_ATUAL}</b>\n⚽ {jogo['time_casa']} x {jogo['time_visitante']}\n\n"
                            tem_valor = False
                            
                            for chave, nome, col_idx in mercados_analise:
                                prob_robo = res[chave] * 100
                                odd_mercado = odds_reais.get(chave, 0)
                                
                                # Análise
                                with cols_radar[col_idx]:
                                    delta_txt = ""
                                    delta_color = "off"
                                    
                                    if odd_mercado > 1.0:
                                        odd_justa = 100 / prob_robo
                                        valor = ((odd_mercado / odd_justa) - 1) * 100
                                        
                                        if valor > 3.0 and prob_robo > (prob_min*100):
                                            delta_txt = f"+{valor:.1f}% EV 💎"
                                            delta_color = "normal" # Verde
                                            msg_telegram += f"✅ <b>{nome}</b>: Odd {odd_mercado:.2f} (EV +{valor:.1f}%)\n"
                                            tem_valor = True
                                            
                                            # Salva no Banco
                                            db = conectar_ao_banco_de_dados()
                                            salvar_analise_no_banco(db, data_sel.strftime('%Y-%m-%d'), LIGA_ATUAL, f"{jogo['time_casa']}x{jogo['time_visitante']}", nome, odd_mercado, prob_robo, valor)
                                        elif valor > 0:
                                            delta_txt = f"+{valor:.1f}%"
                                        else:
                                            delta_txt = f"{valor:.1f}%"
                                            delta_color = "inverse" # Vermelho
                                    
                                    st.metric(nome, f"{prob_robo:.1f}%", delta_txt, delta_color=delta_color)
                            
                            if tem_valor:
                                if st.button("Enviar Telegram 📱", key=f"tel{i}"):
                                    enviar_telegram(msg_telegram)
                        else:
                            st.error("Não foi possível calcular (Times não encontrados no JSON).")

with tab2:
    st.header("Histórico")
    db = conectar_ao_banco_de_dados()
    if db:
        df, g, r = carregar_historico_do_banco(db)
        c1, c2 = st.columns(2)
        c1.metric("Greens", g)
        c2.metric("Reds", r)
        st.dataframe(df, use_container_width=True)
