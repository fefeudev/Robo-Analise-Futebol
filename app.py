# app.py - Robô de Valor (v11.0 - Seguro para Nuvem)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import config, time, json, pytz, gspread, difflib
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÃO DE SEGURANÇA (SECRETS) ---
# O robô tentará pegar a chave dos "Segredos" do Streamlit Cloud
try:
    API_KEY = st.secrets["API_FOOTBALL_KEY"]
except:
    # Fallback para caso esqueça de configurar (não quebra o app, só avisa)
    API_KEY = "ERRO_CONFIGURE_NOS_SECRETS"

API_BASE_URL = "https://v3.football.api-sports.io/"
FUSO = pytz.timezone('America/Manaus')

# Mapeamento de IDs da API-Football
LIGAS_IDS = {
    "Brasileirão": 71, "Champions League": 2, "Premier League": 39, 
    "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61, 
    "Eredivisie": 88, "Championship": 40, "Primeira Liga": 94, "Euro": 4
}
# Mapeamento Reverso para Cérebro (JSONs)
LIGAS_JSON = {
    "Brasileirão": "BSA", "Champions League": "CL", "Premier League": "PL", 
    "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", 
    "Eredivisie": "DED", "Championship": "ELC", "Primeira Liga": "PPL", "Euro": "EC"
}

st.set_page_config(page_title="Robô v11 (Auto)", page_icon="🤖", layout="wide")
st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- FUNÇÕES DE API ---
@st.cache_data(ttl=300)
def get_api_data(endpoint, params):
    if API_KEY == "ERRO_CONFIGURE_NOS_SECRETS": return []
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': API_KEY}
    try:
        r = requests.get(API_BASE_URL + endpoint, headers=headers, params=params)
        return r.json().get('response', [])
    except: return []

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

@st.cache_data(ttl=3600)
def fetch_league_data(league_id, season, date_str):
    # 1. Busca Jogos
    fixtures = get_api_data("fixtures", {"league": league_id, "season": season, "date": date_str})
    if not fixtures: return []
    
    # 2. Busca Odds (Bulk)
    odds_data = get_api_data("odds", {"league": league_id, "season": season, "date": date_str, "bookmaker": "8"}) 
    if not odds_data: odds_data = get_api_data("odds", {"league": league_id, "season": season, "date": date_str})

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
    resp = get_api_data("fixtures", {"league": league_id, "season": season, "status": "FT", "last": 50})
    matches = [{'data_jogo': m['fixture']['date'][:10], 'TimeCasa': m['teams']['home']['name'], 'TimeVisitante': m['teams']['away']['name'], 'GolsCasa': m['goals']['home'], 'GolsVisitante': m['goals']['away']} for m in resp]
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
            
    return {'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total, 'over_2_5': over/total, 'btts_sim': btts/total, 'placar': placar, 'f_01': p01/total, 'f_23': p23/total}

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

# --- INTERFACE ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô v11.0")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_IDS.keys())
    LIGA_ID = LIGAS_IDS[LIGA_NOME]
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0
    
    if API_KEY == "ERRO_CONFIGURE_NOS_SECRETS":
        st.error("⚠️ Configure a API Key nos Secrets do Streamlit!")

dc_data = load_dc(LIGA_NOME)
df_hist, avg_hist = get_historical_for_poisson(LIGA_ID, config.TEMPORADA_PARA_ANALISAR)
MODE = "DIXON_COLES" if dc_data else ("POISSON" if df_hist is not None else "FALHA")

t_jogos, t_hist = st.tabs(["Jogos & Radar", "Histórico"])

with t_jogos:
    st.subheader(f"{LIGA_NOME} - {MODE} - {dt_sel.strftime('%d/%m')}")
    with st.spinner(f"Baixando Odds e Jogos da {LIGA_NOME}..."):
        matches = fetch_league_data(LIGA_ID, config.TEMPORADA_PARA_ANALISAR, dt_sel.strftime('%Y-%m-%d'))
    
    if not matches: st.info("Nenhum jogo encontrado.")
    else:
        radar = []
        for m in matches:
            p, xg = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
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
            with st.expander(f"🔥 RADAR DE VALOR ({len(radar)} Oportunidades)", expanded=True):
                st.dataframe(pd.DataFrame(radar).sort_values('EV', ascending=False), hide_index=True, use_container_width=True, column_config={"Prob": st.column_config.ProgressColumn("Confiança", format="%.0f%%", min_value=0, max_value=1), "EV": st.column_config.NumberColumn("Valor Esperado", format="%.1f%%")})

        if 'sel_game' not in st.session_state:
            for i, m in enumerate(matches):
                p, xg = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
                f_casa = get_form(m['casa'], df_hist)
                f_fora = get_form(m['fora'], df_hist)
                c1, c2 = st.columns([3, 1])
                has_odds = "💰" if m['odds'] else "⚠️"
                if c1.button(f"{has_odds} {m['hora']} | {m['casa']} {f_casa} x {f_fora} {m['fora']}", key=f"b{i}", use_container_width=True): st.session_state.sel_game = m; st.rerun()
                c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")
        else:
            g = st.session_state.sel_game
            if st.button("⬅️ Voltar"): del st.session_state.sel_game; st.rerun()
            st.markdown(f"### {g['casa']} vs {g['fora']}")
            p, xg = predict(MODE, dc_data, df_hist, avg_hist, g['casa'], g['fora'])
            if not g['odds']: st.warning("⚠️ Odds não disponíveis.")
            
            with st.form("auto_analise"):
                col_o = st.columns(4)
                o = g['odds']
                od_h, od_d, od_a, od_ov = o.get('1x2', {}).get('Home', 1.0), o.get('1x2', {}).get('Draw', 1.0), o.get('1x2', {}).get('Away', 1.0), o.get('goals', {}).get('Over 2.5', 1.0)
                uh = col_o[0].number_input("Casa", value=float(od_h))
                ud = col_o[1].number_input("Empate", value=float(od_d))
                ua = col_o[2].number_input("Fora", value=float(od_a))
                uo = col_o[3].number_input("Over 2.5", value=float(od_ov))
                
                if st.form_submit_button("Analisar (Dados Automáticos)"):
                    if p:
                        st.info(f"🔮 Placar: **{p['placar'][0]}x{p['placar'][1]}** | Faixa 2-3: {p['f_23']:.1%}")
                        cols = st.columns(3)
                        def show_met(label, prob, odd_user, col_idx):
                            ev = (prob * odd_user) - 1
                            cor = "normal" if (ev > 0.05 and prob > MIN_PROB) else "inverse"
                            b, q = odd_user - 1, 1 - prob
                            f = (b*prob - q)/b if b > 0 else 0
                            stake = (f * KELLY * BANCA) if f > 0 and cor == "normal" else 0
                            lbl = f"{prob:.1%}" + (f" (R${stake:.0f})" if stake > 0 else "")
                            cols[col_idx].metric(label, lbl, f"{ev*100:.1f}% EV", delta_color=cor)
                            if stake > 0 and db: salvar_db(db, g['hora'], LIGA_NOME, f"{g['casa']}x{g['fora']}", label, odd_user, prob*100, ev*100, stake)

                        if uh > 1: show_met("Casa", p['vitoria_casa'], uh, 0)
                        if ua > 1: show_met("Fora", p['vitoria_visitante'], ua, 1)
                        if uo > 1: show_met("Over 2.5", p['over_2_5'], uo, 2)
                        if db: st.success("✅ Salvo!")

with t_hist:
    if db:
        df_h, g, r = load_db(db)
        c1,c2 = st.columns(2)
        c1.metric("Greens", g); c2.metric("Reds", r)
        st.dataframe(df_h, use_container_width=True)
