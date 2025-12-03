# app.py - Robô de Valor (v12.2 - Híbrido Estável: Jogos FD + Odds AF)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import time, json, pytz, gspread, difflib
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CONFIGURAÇÕES INICIAIS ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô Híbrido v12.2", page_icon="🤖", layout="wide")

st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- 2. VERIFICAÇÃO DE CHAVES ---
try:
    KEY_JOGOS = st.secrets["FOOTBALL_DATA_TOKEN"]
    KEY_ODDS = st.secrets["API_FOOTBALL_KEY"]
except:
    try:
        KEY_JOGOS = st.secrets["google_creds"]["FOOTBALL_DATA_TOKEN"]
        KEY_ODDS = st.secrets["google_creds"]["API_FOOTBALL_KEY"]
    except:
        st.error("🚨 ERRO: Chaves de API não encontradas nos Secrets!")
        st.stop()

LIGAS_FD = {
    "Brasileirão": "BSA", "Champions League": "CL", "Premier League": "PL", 
    "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", 
    "Eredivisie": "DED", "Championship": "ELC", "Primeira Liga": "PPL", "Euro": "EC"
}

# --- 3. FUNÇÕES DE BANCO DE DADOS (AGORA NO TOPO) ---
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

@st.cache_data(ttl=60)
def load_db(_sheet):
    try:
        vals = _sheet.get_all_values()
        if len(vals) < 2: return pd.DataFrame(), 0, 0
        df = pd.DataFrame(vals[1:], columns=vals[0])
        for c in ['Odd', 'Probabilidade', 'Valor']:
            if c in df.columns: df[c] = pd.to_numeric(df[c].str.replace(',', '.'), errors='coerce')
        return df, df['Status'].value_counts().get('Green ✅', 0), df['Status'].value_counts().get('Red ❌', 0)
    except: return pd.DataFrame(), 0, 0

# --- 4. FUNÇÕES DE API ---
@st.cache_data(ttl=300)
def get_jogos_fd(liga_code, date_str):
    url = f"https://api.football-data.org/v4/competitions/{liga_code}/matches"
    headers = {"X-Auth-Token": KEY_JOGOS}
    params = {"dateFrom": date_str, "dateTo": date_str, "status": "SCHEDULED"}
    try:
        r = requests.get(url, headers=headers, params=params)
        return r.json().get('matches', [])
    except: return []

@st.cache_data(ttl=300)
def get_odds_e_nomes_af(date_str):
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': KEY_ODDS}
    try:
        # Fixtures (Nomes)
        r_fix = requests.get("https://v3.football.api-sports.io/fixtures", headers=headers, params={"date": date_str})
        fixtures = r_fix.json().get('response', [])
        if not fixtures: return {}
        id_to_name = {f['fixture']['id']: f['teams']['home']['name'] for f in fixtures}

        # Odds
        r_odds = requests.get("https://v3.football.api-sports.io/odds", headers=headers, params={"date": date_str})
        odds_resp = r_odds.json().get('response', [])
        
        final_map = {}
        for o in odds_resp:
            fid = o['fixture']['id']
            nome_casa = id_to_name.get(fid)
            if nome_casa and o['bookmakers']:
                bookie = o['bookmakers'][0] # Pega 10Bet/Bwin/Bet365
                markets = {}
                for m in bookie['bets']:
                    if m['id'] == 1: markets['1x2'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                    elif m['id'] == 5: markets['goals'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                    elif m['id'] == 8: markets['btts'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                final_map[nome_casa] = markets
        return final_map
    except: return {}

def fundir_dados(jogos_fd, mapa_odds_af):
    jogos_finais = []
    nomes_af = list(mapa_odds_af.keys())
    
    for jogo in jogos_fd:
        tc = jogo['homeTeam']['name']
        tf = jogo['awayTeam']['name']
        utc = datetime.strptime(jogo['utcDate'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
        hora = utc.astimezone(FUSO).strftime('%H:%M')
        
        # Match Nome
        match = difflib.get_close_matches(tc, nomes_af, n=1, cutoff=0.4)
        odds, status = {}, "⚠️"
        if match:
            odds = mapa_odds_af[match[0]]
            status = "💰"
            
        jogos_finais.append({'hora': hora, 'casa': tc, 'fora': tf, 'odds': odds, 'status': status})
    return jogos_finais

# --- 5. LÓGICA DO ROBÔ ---
@st.cache_data
def load_dc(liga_nome):
    try:
        code = LIGAS_FD[liga_nome]
        with open(f"dc_params_{code}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data
def get_historico_fd(liga_code, season):
    url = f"https://api.football-data.org/v4/competitions/{liga_code}/matches"
    headers = {"X-Auth-Token": KEY_JOGOS}
    try:
        r = requests.get(url, headers=headers, params={"season": season, "status": "FINISHED"})
        data = r.json().get('matches', [])
        lista = [{'data_jogo': m['utcDate'][:10], 'TimeCasa': m['homeTeam']['name'], 'TimeVisitante': m['awayTeam']['name'], 'GolsCasa': int(m['score']['fullTime']['home']), 'GolsVisitante': int(m['score']['fullTime']['away'])} for m in data if m['score']['fullTime']['home'] is not None]
        df = pd.DataFrame(lista)
        if not df.empty: df['data_jogo'] = pd.to_datetime(df['data_jogo'])
        return df, {'media_gols_casa': df['GolsCasa'].mean(), 'media_gols_visitante': df['GolsVisitante'].mean()} if not df.empty else None
    except: return None, None

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

def predict(mode, dc, df_poi, avg_poi, home, away):
    xg = (0,0)
    try:
        if mode == "DIXON_COLES":
            f = dc['forcas']
            if home not in f or away not in f: return None, None
            l_c = np.exp(f[home]['ataque'] + f[away]['defesa'] + dc['vantagem_casa'])
            m_v = np.exp(f[away]['ataque'] + f[home]['defesa'])
            probs = calc_probs(l_c, m_v, dc.get('rho', 0))
            xg = (l_c, m_v)
        else:
            hc = df_poi[df_poi['TimeCasa']==home].tail(6)
            hv = df_poi[df_poi['TimeVisitante']==away].tail(6)
            if len(hc)<1 or len(hv)<1: return None, None
            fa_c = hc['GolsCasa'].mean()/avg_poi['media_gols_casa']
            fd_c = hc['GolsVisitante'].mean()/avg_poi['media_gols_visitante']
            fa_v = hv['GolsVisitante'].mean()/avg_poi['media_gols_visitante']
            fd_v = hv['GolsCasa'].mean()/avg_poi['media_gols_casa']
            l_c, m_v = fa_c*fd_v*avg_poi['media_gols_casa'], fa_v*fd_c*avg_poi['media_gols_visitante']
            probs = calc_probs(l_c, m_v)
            xg = (l_c, m_v)
        return probs, xg
    except: return None, None

def get_form(team, df):
    if df is None or df.empty: return ""
    m = df[(df['TimeCasa']==team)|(df['TimeVisitante']==team)].sort_values('data_jogo').tail(5)
    r = ""
    for _, g in m.iterrows():
        if g['TimeCasa']==team: r += "✅" if g['GolsCasa']>g['GolsVisitante'] else ("➖" if g['GolsCasa']==g['GolsVisitante'] else "❌")
        else: r += "✅" if g['GolsVisitante']>g['GolsCasa'] else ("➖" if g['GolsVisitante']==g['GolsCasa'] else "❌")
    return f"({r})"

def calc_kelly(prob, odd, fracao, banca):
    if odd<=1 or prob<=0: return 0,0
    b = odd-1; q=1-prob
    f = (b*prob-q)/b
    stk = (f*fracao*banca) if f>0 else 0
    return stk, f*fracao*100

# --- 6. INTERFACE GRAFICA ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô Híbrido v12.2")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_FD.keys())
    LIGA_CODE = LIGAS_FD[LIGA_NOME]
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

dc_data = load_dc(LIGA_NOME)
df_hist, avg_hist = get_historico_fd(LIGA_CODE, 2025)
MODE = "DIXON_COLES" if dc_data else ("POISSON" if df_hist is not None else "FALHA")

t_jogos, t_hist = st.tabs(["Jogos & Radar", "Histórico"])

with t_jogos:
    st.subheader(f"{LIGA_NOME} ({MODE}) - {dt_sel.strftime('%d/%m')}")
    
    with st.spinner("🔄 Cruzando bases de dados..."):
        jogos_fd = get_jogos_fd(LIGA_CODE, dt_sel.strftime('%Y-%m-%d'))
        mapa_odds = {}
        if jogos_fd:
            mapa_odds = get_odds_e_nomes_af(dt_sel.strftime('%Y-%m-%d'))
            matches = fundir_dados(jogos_fd, mapa_odds)
        else: matches = []

    if not matches:
        st.info("Nenhum jogo encontrado.")
    else:
        # RADAR
        radar = []
        for m in matches:
            p, x = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
            if p and m['odds']:
                o = m['odds']
                check_mkts = [('Home', '1x2', 'Home', 'vitoria_casa'), ('Away', '1x2', 'Away', 'vitoria_visitante'), ('Over 2.5', 'goals', 'Over 2.5', 'over_2_5'), ('BTTS', 'btts', 'Yes', 'btts_sim')]
                for lbl, cat, sel, prob_key in check_mkts:
                    if cat in o and sel in o[cat]:
                        odd_real = o[cat][sel]
                        prob_robo = p[prob_key]
                        ev = (prob_robo * odd_real) - 1
                        if prob_robo > MIN_PROB and ev > 0.05:
                            radar.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': lbl, 'Odd Real': odd_real, 'Prob': prob_robo, 'EV': ev*100})
        
        if radar:
            with st.expander(f"🔥 RADAR DE VALOR ({len(radar)})", expanded=True):
                st.dataframe(pd.DataFrame(radar).sort_values('EV', ascending=False), hide_index=True, use_container_width=True, column_config={"Prob": st.column_config.ProgressColumn("Conf", format="%.0f%%"), "EV": st.column_config.NumberColumn("Valor", format="%.1f%%")})

        # LISTA
        if 'sel_game' not in st.session_state:
            for i, m in enumerate(matches):
                p, xg = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
                f_c = get_form(m['casa'], df_hist)
                f_f = get_form(m['fora'], df_hist)
                
                c1, c2 = st.columns([3, 1])
                status = m['status'] 
                if c1.button(f"{status} {m['hora']} | {m['casa']} {f_c} x {f_f} {m['fora']}", key=f"b{i}", use_container_width=True):
                    st.session_state.sel_game = m
                    st.rerun()
                c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")
        else:
            g = st.session_state.sel_game
            if st.button("⬅️ Voltar"): del st.session_state.sel_game; st.rerun()
            st.markdown(f"### {g['casa']} vs {g['fora']}")
            
            p, xg = predict(MODE, dc_data, df_hist, avg_hist, g['casa'], g['fora'])
            
            with st.form("auto"):
                col_o = st.columns(4)
                o = g['odds']
                od_h = o.get('1x2', {}).get('Home', 1.0)
                od_d = o.get('1x2', {}).get('Draw', 1.0)
                od_a = o.get('1x2', {}).get('Away', 1.0)
                od_ov = o.get('goals', {}).get('Over 2.5', 1.0)
                
                uh = col_o[0].number_input("Casa", value=float(od_h))
                ud = col_o[1].number_input("Empate", value=float(od_d))
                ua = col_o[2].number_input("Fora", value=float(od_a))
                uo = col_o[3].number_input("Over 2.5", value=float(od_ov))
                
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
                        if db: st.success("Salvo!")

with t_hist:
    if db:
        df_h, g, r = load_db(db)
        c1,c2 = st.columns(2); c1.metric("Greens", g); c2.metric("Reds", r)
        st.dataframe(df_h, use_container_width=True)
