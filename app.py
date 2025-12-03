# app.py - Robô de Valor (v11.3 - Blindado: Auto-Season + Key Finder)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import config, time, json, pytz, gspread, difflib
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô de Valor", page_icon="🤖", layout="wide")

# CSS
st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- 1. CAÇADOR DE CHAVES (RESOLVE PROBLEMA DE SECRETS) ---
API_KEY = None
# Tenta achar na raiz
if "API_FOOTBALL_KEY" in st.secrets:
    API_KEY = st.secrets["API_FOOTBALL_KEY"]
# Se não achar, tenta achar dentro de google_creds (caso tenha ficado indentado)
elif "google_creds" in st.secrets and "API_FOOTBALL_KEY" in st.secrets["google_creds"]:
    API_KEY = st.secrets["google_creds"]["API_FOOTBALL_KEY"]

API_BASE_URL = "https://v3.football.api-sports.io/"

LIGAS_IDS = {
    "Brasileirão": 71, "Champions League": 2, "Premier League": 39, 
    "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61, 
    "Eredivisie": 88, "Championship": 40, "Primeira Liga": 94, "Euro": 4
}
LIGAS_JSON = {
    "Brasileirão": "BSA", "Champions League": "CL", "Premier League": "PL", 
    "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", 
    "Eredivisie": "DED", "Championship": "ELC", "Primeira Liga": "PPL", "Euro": "EC"
}

# --- FUNÇÕES DE API ---
@st.cache_data(ttl=300)
def get_api_data(endpoint, params):
    if not API_KEY: return {'errors': {'config': 'API Key não encontrada'}}
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': API_KEY}
    try:
        r = requests.get(API_BASE_URL + endpoint, headers=headers, params=params)
        return r.json()
    except Exception as e: return {'errors': {'connection': str(e)}}

@st.cache_resource
def connect_db():
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets.google_creds), scope)
        return gspread.authorize(creds).open_by_url(st.secrets.GOOGLE_SHEET_URL).sheet1
    except: return None

def salvar_db(sheet, data, liga, jogo, mercado, odd, prob, valor, stake):
    if sheet: sheet.append_row([data, liga, jogo, mercado, float(odd), float(prob)/100, float(valor)/100, "Aguardando ⏳"], value_input_option='USER_ENTERED')

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

def match_team_name(name, name_list):
    matches = difflib.get_close_matches(name, name_list, n=1, cutoff=0.6)
    return matches[0] if matches else name

# --- LÓGICA DE DADOS & ODDS ---
@st.cache_data
def load_dc(liga_nome):
    try:
        code = LIGAS_JSON[liga_nome]
        with open(f"dc_params_{code}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data(ttl=300)
def fetch_league_data(league_id, target_season, date_str):
    """
    CORREÇÃO V11.3: Tenta múltiplas temporadas (2025, 2024, 2026) 
    até encontrar os jogos. Resolve o problema de 'Nenhum jogo encontrado'.
    """
    fixtures = []
    # Tenta a temporada alvo, depois a anterior, depois a próxima
    seasons_to_try = [target_season, target_season - 1, target_season + 1]
    
    used_season = None
    
    for s in seasons_to_try:
        resp = get_api_data("fixtures", {"league": league_id, "season": s, "date": date_str})
        data = resp.get('response', [])
        if data:
            fixtures = data
            used_season = s
            break # Achou! Para de procurar.
    
    if not fixtures: return []

    # Busca Odds usando a temporada que funcionou
    odds_data = []
    resp_odds = get_api_data("odds", {"league": league_id, "season": used_season, "date": date_str, "bookmaker": "8"})
    if resp_odds.get('response'):
        odds_data = resp_odds['response']
    else:
        # Tenta fallback sem bookmaker
        resp_odds = get_api_data("odds", {"league": league_id, "season": used_season, "date": date_str})
        odds_data = resp_odds.get('response', [])

    odds_map = {}
    for o in odds_data:
        fid = o['fixture']['id']
        bookie = o['bookmakers'][0]
        markets = {}
        for m in bookie['bets']:
            if m['id'] == 1: markets['1x2'] = {Op['value']: float(Op['odd']) for Op in m['values']}
            elif m['id'] == 12: markets['dc'] = {Op['value']: float(Op['odd']) for Op in m['values']}
            elif m['id'] == 5: markets['goals'] = {Op['value']: float(Op['odd']) for Op in m['values']}
            elif m['id'] == 8: markets['btts'] = {Op['value']: float(Op['odd']) for Op in m['values']}
        odds_map[fid] = markets

    final_data = []
    for f in fixtures:
        fid = f['fixture']['id']
        match = {
            'id': fid,
            'hora': datetime.fromtimestamp(f['fixture']['timestamp'], FUSO).strftime('%H:%M'),
            'casa': f['teams']['home']['name'],
            'fora': f['teams']['away']['name'],
            'status': f['fixture']['status']['short'],
            'score_casa': f['goals']['home'],
            'score_fora': f['goals']['away'],
            'odds': odds_map.get(fid, {})
        }
        final_data.append(match)
    return final_data

@st.cache_data
def get_historical_for_poisson(league_id, season):
    # Tenta buscar historico, se falhar season 2025 tenta 2024
    resp = get_api_data("fixtures", {"league": league_id, "season": season, "status": "FT", "last": 50})
    if not resp.get('response'):
         resp = get_api_data("fixtures", {"league": league_id, "season": season-1, "status": "FT", "last": 50})
         
    matches = [{'data_jogo': m['fixture']['date'][:10], 'TimeCasa': m['teams']['home']['name'], 'TimeVisitante': m['teams']['away']['name'], 'GolsCasa': m['goals']['home'], 'GolsVisitante': m['goals']['away']} for m in resp.get('response', [])]
    df = pd.DataFrame(matches)
    if not df.empty: df['data_jogo'] = pd.to_datetime(df['data_jogo'])
    return df, {'media_gols_casa': df['GolsCasa'].mean(), 'media_gols_visitante': df['GolsVisitante'].mean()} if not df.empty else None

# --- PREVISÃO ---
def calc_probs(l_casa, m_visit, rho=0.0):
    probs = np.zeros((7, 7))
    max_prob, placar = 0, (0,0)
    p01, p23, p4p = 0,0,0
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
    return {'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total, 'over_2_5': over/total, 'btts_sim': btts/total, 'placar': placar, 'f_01': p01/total, 'f_23': p23/total, 'f_4p': p4p/total}

def predict(mode, dc, df_poi, avg_poi, home, away):
    real_home, real_away = home, away
    if mode == "DIXON_COLES":
        teams_dc = list(dc['forcas'].keys())
        real_home = match_team_name(home, teams_dc)
        real_away = match_team_name(away, teams_dc)
    xg = (0,0)
    try:
        if mode == "DIXON_COLES":
            f = dc['forcas']
            l_c = np.exp(f[real_home]['ataque'] + f[real_away]['defesa'] + dc['vantagem_casa'])
            m_v = np.exp(f[real_away]['ataque'] + f[real_home]['defesa'])
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

def check_result(probs, sh, sa, min_p):
    res, win = [], False
    w = "H" if sh>sa else ("A" if sa>sh else "D")
    if probs['vitoria_casa']>min_p: 
        stt = "✅" if w=="H" else "❌"; res.append(f"{stt} Casa"); win = (stt=="✅")
    elif probs['vitoria_visitante']>min_p: 
        stt = "✅" if w=="A" else "❌"; res.append(f"{stt} Fora"); win = (stt=="✅")
    if probs['over_2_5']>min_p: 
        stt = "✅" if (sh+sa)>2.5 else "❌"; res.append(f"{stt} Over"); win = (stt=="✅")
    return res, win

# --- INTERFACE ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô v11.3")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_IDS.keys())
    LIGA_ID = LIGAS_IDS[LIGA_NOME]
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    MODO_BACKTEST = st.toggle("🔙 Backtest", False)
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0
    
    # --- BOTÃO DE DIAGNÓSTICO (DEBUG) ---
    if st.button("🛠️ Diagnóstico de API"):
        tst = get_api_data("status", {})
        st.code(json.dumps(tst, indent=2))
        tst_fix = get_api_data("fixtures", {"league": 39, "date": "2025-12-03"}) # Teste fixo PL
        st.write("Teste Premier League 03/12:")
        st.code(json.dumps(tst_fix, indent=2))

    if not API_KEY: st.error("⚠️ API Key não encontrada nos Secrets!")

dc_data = load_dc(LIGA_NOME)
df_hist, avg_hist = get_historical_for_poisson(LIGA_ID, config.TEMPORADA_PARA_ANALISAR)
MODE = "DIXON_COLES" if dc_data else ("POISSON" if df_hist is not None else "FALHA")

t_jogos, t_hist = st.tabs(["Jogos & Radar", "Histórico"])

with t_jogos:
    st.subheader(f"{LIGA_NOME} - {MODE} - {dt_sel.strftime('%d/%m')}")
    
    with st.spinner("Buscando jogos (tentando várias temporadas)..."):
        matches = fetch_league_data(LIGA_ID, config.TEMPORADA_PARA_ANALISAR, dt_sel.strftime('%Y-%m-%d'))
        
        # Filtro de Backtest
        if MODO_BACKTEST:
            status_req = ["FT", "AET", "PEN"]
            matches = [m for m in matches if m['status'] in status_req]
        else:
            status_req = ["NS", "TBD"]
            matches = [m for m in matches if m['status'] in status_req]

    if not matches: 
        st.info(f"Nenhum jogo encontrado para {dt_sel} (Modo: {'Backtest' if MODO_BACKTEST else 'Futuro'}).")
    else:
        # RADAR / BACKTEST SUMMARY
        if MODO_BACKTEST:
            tg, tr = 0, 0
            for m in matches:
                p, _ = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
                if p:
                    res, is_g = check_result(p, m['score_casa'], m['score_fora'], MIN_PROB)
                    if res: 
                        if is_g: tg+=1 
                        else: tr+=1
            if (tg+tr)>0: st.success(f"📊 Resultado: {tg} Greens ✅ | {tr} Reds ❌")
        
        elif not MODO_BACKTEST:
            # RADAR REAL TIME
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
                            if prob_robo > MIN_PROB:
                                ev = (prob_robo * odd_real) - 1
                                if ev > 0.05:
                                    radar.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': lbl, 'Odd Real': odd_real, 'Prob': prob_robo, 'EV': ev*100})
            if radar:
                with st.expander("🔥 Radar de Oportunidades", expanded=True):
                    st.dataframe(pd.DataFrame(radar).sort_values('EV', ascending=False), hide_index=True, use_container_width=True, column_config={"Prob": st.column_config.ProgressColumn("Confiança", format="%.0f%%"), "EV": st.column_config.NumberColumn("Valor", format="%.1f%%")})

        # LISTA DE JOGOS
        if 'sel_game' not in st.session_state:
            for i, m in enumerate(matches):
                p, xg = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
                f_c, f_f = get_form(m['casa'], df_hist), get_form(m['fora'], df_hist)
                c1, c2 = st.columns([3, 1])
                icon = "⚽" if not MODO_BACKTEST else "🏁"
                score = f"({m['score_casa']}-{m['score_fora']})" if MODO_BACKTEST else ""
                
                if c1.button(f"{icon} {m['hora']} | {m['casa']} {f_c} x {f_f} {m['fora']} {score}", key=f"b{i}", use_container_width=True):
                    if not MODO_BACKTEST: st.session_state.sel_game = m; st.rerun()
                
                if MODO_BACKTEST and p:
                    res, _ = check_result(p, m['score_casa'], m['score_fora'], MIN_PROB)
                    c2.write(" | ".join(res)) if res else c2.caption("-")
                else: c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")
        else:
            g = st.session_state.sel_game
            if st.button("⬅️ Voltar"): del st.session_state.sel_game; st.rerun()
            st.markdown(f"### {g['casa']} vs {g['fora']}")
            if not g['odds']: st.warning("⚠️ Odds não disponíveis.")
            
            p, xg = predict(MODE, dc_data, df_hist, avg_hist, g['casa'], g['fora'])
            with st.form("auto"):
                col_o = st.columns(4)
                o = g['odds']
                od_h, od_d, od_a, od_ov = o.get('1x2', {}).get('Home', 1.0), o.get('1x2', {}).get('Draw', 1.0), o.get('1x2', {}).get('Away', 1.0), o.get('goals', {}).get('Over 2.5', 1.0)
                uh = col_o[0].number_input("Casa", value=float(od_h))
                ud = col_o[1].number_input("Empate", value=float(od_d))
                ua = col_o[2].number_input("Fora", value=float(od_a))
                uo = col_o[3].number_input("Over 2.5", value=float(od_ov))
                
                if st.form_submit_button("Analisar"):
                    if p:
                        st.info(f"🔮 Placar: **{p['placar'][0]}x{p['placar'][1]}** | 2-3 Gols: {p['f_23']:.1%}")
                        c_mg1, c_mg2, c_mg3 = st.columns(3)
                        c_mg1.metric("0-1", f"{p['f_01']:.1%}"); c_mg2.metric("2-3", f"{p['f_23']:.1%}"); c_mg3.metric("4+", f"{p['f_4p']:.1%}")
                        
                        cols = st.columns(3)
                        def show(lbl, prob, odd, idx):
                            ev = (prob*odd)-1
                            cor = "normal" if (ev>0.05 and prob>MIN_PROB) else "inverse"
                            b, q = odd-1, 1-prob
                            f = (b*prob-q)/b if b>0 else 0
                            stk = (f*KELLY*BANCA) if f>0 and cor=="normal" else 0
                            l = f"{prob:.1%}" + (f" (R${stk:.0f})" if stk>0 else "")
                            cols[idx].metric(lbl, l, f"{ev*100:.1f}% EV", delta_color=cor)
                            if stk>0 and db: salvar_db(db, g['hora'], LIGA_NOME, f"{g['casa']}x{g['fora']}", lbl, odd, prob*100, ev*100, stk)
                        
                        if uh>1: show("Casa", p['vitoria_casa'], uh, 0)
                        if ua>1: show("Fora", p['vitoria_visitante'], ua, 1)
                        if uo>1: show("Over", p['over_2_5'], uo, 2)
                        if db: st.success("✅ Salvo!")

with t_hist:
    if db:
        df_h, g, r = load_db(db)
        c1,c2 = st.columns(2)
        c1.metric("Greens", g); c2.metric("Reds", r)
        st.dataframe(df_h, use_container_width=True)
