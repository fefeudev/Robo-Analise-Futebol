# app.py - Robô v14.0 (Match por ID - Estratégia Definitiva)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import time, json, pytz, gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CONFIGURAÇÕES ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô v14.0 (ID Match)", page_icon="🤖", layout="wide")

st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- 2. CHAVES E LIGAS ---
try:
    # Tenta pegar a chave
    API_KEY = st.secrets.get("API_FOOTBALL_KEY") or st.secrets["google_creds"]["API_FOOTBALL_KEY"]
except:
    st.error("🚨 Configure 'API_FOOTBALL_KEY' nos Secrets!")
    st.stop()

# Mapeamento: Nome da Liga -> (ID da Liga na API-Football, Sigla do JSON)
LIGAS_MAP = {
    "Brasileirão": (71, "BSA"),
    "Champions League": (2, "CL"),
    "Premier League": (39, "PL"),
    "La Liga": (140, "PD"),
    "Serie A (Itália)": (135, "SA"),
    "Bundesliga": (78, "BL1"),
    "Ligue 1": (61, "FL1"),
    "Eredivisie": (88, "DED"),
    "Championship": (40, "ELC"),
    "Primeira Liga": (94, "PPL"),
    "Euro": (4, "EC")
}

# --- 3. FUNÇÕES DE API INTELIGENTES ---
@st.cache_data(ttl=300)
def get_dados_do_dia(date_str):
    """
    A MÁGICA: Baixa TODOS os jogos e TODAS as odds do dia em 2 chamadas.
    Depois cruza tudo usando o ID do jogo (Fixture ID).
    """
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': API_KEY}
    
    # 1. Baixar TODOS os jogos do dia (Sem filtro de liga para não bugar season)
    try:
        r_fix = requests.get("https://v3.football.api-sports.io/fixtures", headers=headers, params={"date": date_str})
        fixtures_raw = r_fix.json().get('response', [])
    except: fixtures_raw = []

    if not fixtures_raw: return {}, []

    # 2. Baixar TODAS as odds do dia
    try:
        r_odds = requests.get("https://v3.football.api-sports.io/odds", headers=headers, params={"date": date_str})
        odds_raw = r_odds.json().get('response', [])
    except: odds_raw = []

    # 3. Criar Mapa de Odds por ID {12345: {odds...}}
    odds_map = {}
    for o in odds_raw:
        fid = o['fixture']['id']
        if o['bookmakers']:
            bookie = o['bookmakers'][0]
            # Tenta achar Bet365 (8)
            for b in o['bookmakers']:
                if b['id'] == 8: bookie = b; break
            
            mkts = {}
            for m in bookie['bets']:
                if m['id'] == 1: mkts['1x2'] = {v['value']: float(v['odd']) for v in m['values']}
                elif m['id'] == 12: mkts['dc'] = {v['value']: float(v['odd']) for v in m['values']}
                elif m['id'] == 5: mkts['goals'] = {v['value']: float(v['odd']) for v in m['values']}
                elif m['id'] == 8: mkts['btts'] = {v['value']: float(v['odd']) for v in m['values']}
            odds_map[fid] = mkts

    return odds_map, fixtures_raw

def processar_jogos_da_liga(liga_id, fixtures_raw, odds_map):
    """Filtra os jogos da liga escolhida e cola as odds pelo ID"""
    jogos_finais = []
    
    for f in fixtures_raw:
        # Filtra só a liga que queremos
        if f['league']['id'] != liga_id:
            continue
            
        fid = f['fixture']['id']
        
        # Tenta pegar as odds pelo ID (Match Perfeito!)
        odds = odds_map.get(fid, {})
        status = "💰" if odds else "⚠️"
        
        jogos_finais.append({
            'id': fid,
            'hora': datetime.fromtimestamp(f['fixture']['timestamp'], FUSO).strftime('%H:%M'),
            'casa': f['teams']['home']['name'],
            'fora': f['teams']['away']['name'],
            'odds': odds,
            'status': status,
            'status_jogo': f['fixture']['status']['short'],
            'score_casa': f['goals']['home'],
            'score_fora': f['goals']['away']
        })
        
    return jogos_finais

# --- 4. CÉREBRO ---
@st.cache_data
def load_dc(sigla_liga):
    try:
        with open(f"dc_params_{sigla_liga}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

# Tradutor simplificado para o Cérebro DC (já que agora temos o nome oficial da API)
def match_name_dc(name, dc_names):
    match = difflib.get_close_matches(name, dc_names, n=1, cutoff=0.4)
    return match[0] if match else name

def calc_probs(l_casa, m_visit, rho=0.0):
    probs = np.zeros((7, 7)); max_prob, placar = 0, (0,0); p01, p23, p4p = 0,0,0
    for i in range(7):
        for j in range(7):
            tau = 1.0
            if i==0 and j==0: tau = 1 - (l_casa*m_visit*rho)
            elif i==1 and j==0: tau = 1 + (l_casa*rho)
            elif i==0 and j==1: tau = 1 + (m_visit*rho)
            elif i==1 and j==1: tau = 1 - rho
            p = stats.poisson.pmf(i, l_casa) * stats.poisson.pmf(j, m_visit) * tau
            probs[i, j] = p
            if p > max_prob: max_prob, placar = p, (i, j)
            if (i+j)<=1: p01+=p
            elif (i+j)<=3: p23+=p
            else: p4p+=p
    total = np.sum(probs)
    if total==0: return None
    home, draw, away = np.sum(np.tril(probs,-1)), np.sum(np.diag(probs)), np.sum(np.triu(probs,1))
    over, btts = 0, 0
    for i in range(7):
        for j in range(7):
            if (i+j)>2.5: over+=probs[i,j]
            if i>0 and j>0: btts+=probs[i,j]
    return {'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total, 'over_2_5': over/total, 'btts_sim': btts/total, 'placar': placar, 'f_01': p01/total, 'f_23': p23/total}

def predict(dc, home, away):
    if not dc: return None, (0,0)
    try:
        f = dc['forcas']
        # Traduz nomes da API Nova para nomes do JSON (Dixon Coles)
        h_dc = match_name_dc(home, list(f.keys()))
        a_dc = match_name_dc(away, list(f.keys()))
        
        l_c = np.exp(f[h_dc]['ataque'] + f[a_dc]['defesa'] + dc['vantagem_casa'])
        m_v = np.exp(f[a_dc]['ataque'] + f[h_dc]['defesa'])
        probs = calc_probs(l_c, m_v, dc.get('rho', 0))
        return probs, (l_c, m_v)
    except: return None, (0,0)

def calc_kelly(prob, odd, fracao, banca):
    if odd<=1 or prob<=0: return 0,0
    b = odd-1; q=1-prob
    f = (b*prob-q)/b
    stk = (f*fracao*banca) if f>0 else 0
    return stk, f*fracao*100

# --- 5. DB & UI ---
@st.cache_resource
def connect_db():
    try:
        if "google_creds" not in st.secrets: return None
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets.google_creds), scope)
        return gspread.authorize(creds).open_by_url(st.secrets.GOOGLE_SHEET_URL).sheet1
    except: return None

def salvar_db(sheet, data, liga, jogo, mercado, odd, prob, valor, stake):
    if sheet: 
        try: sheet.append_row([data, liga, jogo, mercado, float(odd), float(prob)/100, float(valor)/100, "Aguardando ⏳"], value_input_option='USER_ENTERED')
        except: pass

db = connect_db()

# SIDEBAR
with st.sidebar:
    st.title("🤖 Robô v14.0 (ID Match)")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_MAP.keys())
    ID_LIGA, SIGLA_LIGA = LIGAS_MAP[LIGA_NOME]
    
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

# CARGA
dc_data = load_dc(SIGLA_LIGA)

# LAYOUT
st.subheader(f"{LIGA_NOME} - {dt_sel.strftime('%d/%m')}")

# BUSCA DUPLA (Segurança de Fuso Horário)
dates_to_check = [dt_sel.strftime('%Y-%m-%d'), (dt_sel + timedelta(days=1)).strftime('%Y-%m-%d')]
final_matches = []

with st.spinner("Baixando Jogos e Odds (Por ID)..."):
    # Busca Hoje e Amanhã e junta tudo
    for d in dates_to_check:
        omap, fraw = get_dados_do_dia(d)
        if fraw:
            res = processar_jogos_da_liga(ID_LIGA, fraw, omap)
            final_matches.extend(res)

# Remove duplicatas (caso o jogo apareça nas duas buscas)
seen = set()
unique_matches = []
for m in final_matches:
    if m['id'] not in seen:
        unique_matches.append(m)
        seen.add(m['id'])

if not unique_matches:
    st.info("Nenhum jogo encontrado para esta liga nestas datas.")
else:
    # RADAR DE OPORTUNIDADES
    radar = []
    for m in unique_matches:
        p, x = predict(dc_data, m['casa'], m['fora'])
        if p and m['odds']:
            o = m['odds']
            check = [('Home', '1x2', 'Home', 'vitoria_casa'), ('Away', '1x2', 'Away', 'vitoria_visitante'), ('Over 2.5', 'goals', 'Over 2.5', 'over_2_5'), ('BTTS', 'btts', 'Yes', 'btts_sim')]
            for lbl, cat, sel, pk in check:
                if cat in o and sel in o[cat]:
                    odr = o[cat][sel]
                    prb = p[pk]
                    ev = (prb * odr) - 1
                    if prb > MIN_PROB and ev > 0.05:
                        radar.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': lbl, 'Odd': odr, 'Prob': prb, 'EV': ev*100})
    
    if radar:
        with st.expander(f"🔥 RADAR ({len(radar)})", expanded=True):
            st.dataframe(pd.DataFrame(radar).sort_values('EV', ascending=False), hide_index=True, use_container_width=True, column_config={"Prob": st.column_config.ProgressColumn("Conf", format="%.0f%%"), "EV": st.column_config.NumberColumn("Valor", format="%.1f%%")})

    # LISTA DE JOGOS
    if 'sel_game' not in st.session_state:
        for i, m in enumerate(unique_matches):
            p, xg = predict(dc_data, m['casa'], m['fora'])
            c1, c2 = st.columns([3, 1])
            if c1.button(f"{m['status']} {m['hora']} | {m['casa']} x {m['fora']}", key=f"b{i}", use_container_width=True):
                st.session_state.sel_game = m
                st.rerun()
            c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")
    else:
        g = st.session_state.sel_game
        if st.button("⬅️ Voltar"): del st.session_state.sel_game; st.rerun()
        st.markdown(f"### {g['casa']} vs {g['fora']}")
        
        p, xg = predict(dc_data, g['casa'], g['fora'])
        
        with st.form("auto"):
            col_o = st.columns(2)
            o = g['odds']
            od_h = o.get('1x2', {}).get('Home', 1.0)
            od_d = o.get('1x2', {}).get('Draw', 1.0)
            od_a = o.get('1x2', {}).get('Away', 1.0)
            od_ov = o.get('goals', {}).get('Over 2.5', 1.0)
            od_bt = o.get('btts', {}).get('Yes', 1.0)
            od_1x = o.get('dc', {}).get('Home/Draw', 1.0)
            od_x2 = o.get('dc', {}).get('Draw/Away', 1.0)
            od_12 = o.get('dc', {}).get('Home/Away', 1.0)
            
            with col_o[0]:
                uh = st.number_input("Casa", value=float(od_h))
                ud = st.number_input("Empate", value=float(od_d))
                ua = st.number_input("Fora", value=float(od_a))
                uo = st.number_input("Over 2.5", value=float(od_ov))
            with col_o[1]:
                ub = st.number_input("BTTS", value=float(od_bt))
                u1x = st.number_input("1X", value=float(od_1x))
                ux2 = st.number_input("X2", value=float(od_x2))
                u12 = st.number_input("12", value=float(od_12))
            
            if st.form_submit_button("Analisar"):
                if p:
                    st.info(f"🔮 Placar: **{p['placar'][0]}x{p['placar'][1]}**")
                    cols = st.columns(3)
                    def show(lbl, prob, odd, idx):
                        ev = (prob*odd)-1
                        cor = "normal" if (ev>0.05 and prob>MIN_PROB) else "inverse"
                        stk, _ = calc_kelly(prob, odd, KELLY, BANCA)
                        l = f"{prob:.1%}" + (f" (R${stk:.0f})" if stk>0 else "")
                        cols[idx].metric(lbl, l, f"{ev*100:.1f}% EV", delta_color=cor)
                        if stk>0 and db: salvar_db(db, g['hora'], LIGA_NOME, f"{g['casa']}x{g['fora']}", lbl, odd, prob*100, ev*100, stk)
                    
                    if uh>1: show("Casa", p['vitoria_casa'], uh, 0)
                    if ua>1: show("Fora", p['vitoria_visitante'], ua, 1)
                    if uo>1: show("Over", p['over_2_5'], uo, 2)
                    if ub>1: show("BTTS", p['btts_sim'], ub, 0)
                    if u1x>1: show("1X", p['chance_dupla_1X'], u1x, 1)
                    if db: st.success("Salvo!")
