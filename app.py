# app.py
# Versão 8.5 - SUPER HÍBRIDO + MULTI-LINES (0.5 a 4.5 Gols + Handicaps)

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

# --- NOVOS IMPORTS DO BANCO DE DADOS ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# -------------------------------------

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Robô Híbrido Elite",
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
    
    # ATUALIZADO: Pedindo Vencedor, Totais (Gols) e Spreads (Handicap)
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds'
    params = {'apiKey': api_key, 'regions': 'eu,uk', 'markets': 'h2h,totals,spreads', 'oddsFormat': 'decimal'}
    
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
        
        # PROBABILIDADES DE GOLS (0.5 a 4.5)
        # Soma todos os placares onde (i+j) > Linha
        prob_over05 = 1 - probs[0,0]
        prob_over15 = 1 - (probs[0,0] + probs[1,0] + probs[0,1])
        prob_over25 = np.sum(probs) - (probs[0,0]+probs[1,0]+probs[0,1]+probs[2,0]+probs[0,2]+probs[1,1])
        # Aproximação para 3.5 e 4.5 (soma placares baixos e subtrai de 1)
        placares_under35 = [(0,0),(1,0),(0,1),(2,0),(0,2),(1,1),(3,0),(0,3),(2,1),(1,2)]
        prob_under35 = sum(probs[i,j] for i,j in placares_under35 if i<7 and j<7)
        prob_over35 = 1 - prob_under35

        btts = 1 - (np.sum(probs[0,:]) + np.sum(probs[:,0]) - probs[0,0])
        
        total = home+draw+away
        res = {
            'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
            'over_0_5': prob_over05/total, 'over_1_5': prob_over15/total,
            'over_2_5': prob_over25/total, 'over_3_5': prob_over35/total,
            'btts_sim': btts/total,
            'chance_dupla_1X': (home+draw)/total, 'chance_dupla_X2': (draw+away)/total, 'chance_dupla_12': (home+away)/total
        }
        return res, (l, m)
    except: return None, None

def prever_jogo_poisson(forcas, medias, t1, t2):
    try:
        fa_c = forcas[t1]['atk']/medias['mc']
        fd_c = forcas[t1]['def']/medias['mv']
        fa_v = forcas[t2]['atk']/medias['mv']
        fd_v = forcas[t2]['def']/medias['mc']
        l = fa_c * fd_v * medias['mc']
        m = fa_v * fd_c * medias['mv']
        
        probs = np.outer([stats.poisson.pmf(i,l) for i in range(7)], [stats.poisson.pmf(j,m) for j in range(7)])
        home = np.sum(np.tril(probs, -1))
        draw = np.sum(np.diag(probs))
        away = np.sum(np.triu(probs, 1))
        
        total = home+draw+away
        if total == 0: return None, None
        
        return {
            'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
            'over_2_5': 0.5, 'btts_sim': 0.5, 
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
    st.title("🤖 Robô Elite 8.5")
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
tab1, tab2 = st.tabs(["📊 Analisar Jogos", "📈 Histórico"])

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
                handicaps = []
                
                with col_res1:
                    st.markdown("#### 🏦 Mercado (Odds)")
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
                            
                            if h and d and a:
                                dc_1x, dc_x2, dc_12 = calcular_odds_chance_dupla(h, d, a)
                                odds_reais.update({'vitoria_casa':h, 'empate':d, 'vitoria_visitante':a, 'chance_dupla_1X':dc_1x, 'chance_dupla_X2':dc_x2, 'chance_dupla_12':dc_12})
                            
                            # --- 2. GOLS (TOTALS) - Captura Multi-Linhas ---
                            outs_tot = next((m['outcomes'] for m in bookie['markets'] if m['key'] == 'totals'), [])
                            
                            # Dicionário para guardar as linhas que encontrarmos
                            linhas_gols = {0.5: None, 1.5: None, 2.5: None, 3.5: None, 4.5: None}
                            
                            for o in outs_tot:
                                if o['name'] == 'Over' and o['point'] in linhas_gols:
                                    linhas_gols[o['point']] = o['price']
                                    # Mapeia para o nome usado no robô estatístico
                                    key_robo = f"over_{str(o['point']).replace('.','_')}"
                                    odds_reais[key_robo] = o['price']
                            
                            # --- 3. HANDICAPS (SPREADS) ---
                            outs_spr = next((m['outcomes'] for m in bookie['markets'] if m['key'] == 'spreads'), [])
                            if outs_spr:
                                # Pega a primeira linha de handicap disponível para exibição
                                line = outs_spr[0].get('point')
                                price = outs_spr[0].get('price')
                                nome = outs_spr[0].get('name')
                                handicaps.append(f"{nome} ({line}): {price}")

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
                            # TABS PARA ORGANIZAR RESULTADOS
                            t_res, t_gols = st.tabs(["Principal (1x2)", "Gols & Linhas"])
                            
                            msg_telegram = f"🔥 <b>{LIGA_ATUAL}</b>: {jogo['time_casa']} x {jogo['time_visitante']}\n\n"
                            tem_valor = False
                            
                            # --- TAB 1: 1x2 e Dupla Chance ---
                            with t_res:
                                col_p1, col_p2, col_p3 = st.columns(3)
                                # Lista: (Chave_Robo, Nome_Display, Coluna)
                                principais = [
                                    ('vitoria_casa', 'Casa', 0), ('empate', 'Empate', 1), ('vitoria_visitante', 'Fora', 2),
                                    ('chance_dupla_1X', 'DC 1X', 0), ('chance_dupla_X2', 'DC X2', 2), ('chance_dupla_12', 'DC 12', 1)
                                ]
                                for ch, nome, c_idx in principais:
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
                                        elif ev > 0: delta = f"+{ev:.1f}%"
                                        else: 
                                            delta = f"{ev:.1f}%"
                                            color = "inverse"
                                    
                                    if c_idx==0: col_p1.metric(nome, f"{prob:.1f}%", delta, delta_color=color)
                                    elif c_idx==1: col_p2.metric(nome, f"{prob:.1f}%", delta, delta_color=color)
                                    else: col_p3.metric(nome, f"{prob:.1f}%", delta, delta_color=color)

                            # --- TAB 2: GOLS (0.5 a 3.5) ---
                            with t_gols:
                                st.caption("Probabilidades de Over Gols vs Odds Disponíveis")
                                cg1, cg2, cg3, cg4 = st.columns(4)
                                
                                # Lista de Gols para iterar
                                lista_gols = [
                                    ('over_0_5', 'Over 0.5', 0.5, cg1),
                                    ('over_1_5', 'Over 1.5', 1.5, cg2),
                                    ('over_2_5', 'Over 2.5', 2.5, cg3),
                                    ('over_3_5', 'Over 3.5', 3.5, cg4)
                                ]
                                
                                for chave, nome, ponto, col in lista_gols:
                                    prob = res.get(chave, 0)*100
                                    # Tenta pegar a odd exata daquele ponto (ex: 1.5)
                                    odd = odds_reais.get(chave, 0) 
                                    
                                    delta = ""
                                    color = "off"
                                    val_display = f"{prob:.1f}%"
                                    
                                    if odd > 1:
                                        val_display = f"{prob:.1f}% (@{odd})"
                                        justa = 100/prob if prob > 0 else 99
                                        ev = ((odd/justa)-1)*100
                                        if ev > 2.0:
                                            delta = f"+{ev:.1f}% EV"
                                            color = "normal"
                                            msg_telegram += f"⚽ <b>{nome}</b>: Odd {odd:.2f} (EV +{ev:.1f}%)\n"
                                            tem_valor = True
                                        else:
                                            delta = f"{ev:.1f}%"
                                            color = "inverse"
                                    elif 'linhas_gols' in locals() and linhas_gols.get(ponto) is None:
                                        # Se a API não mandou essa linha
                                        delta = "Sem Odd"
                                    
                                    col.metric(nome, val_display, delta, delta_color=color)
                                
                                if handicaps:
                                    st.divider()
                                    st.markdown(f"**Handicaps Disponíveis:** {', '.join(handicaps)}")

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
