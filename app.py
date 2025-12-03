# app.py - Robô de Valor (v12.1 - Híbrido Estável: Jogos FD + Odds AF)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import time, json, pytz, gspread, difflib
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES GERAIS ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô Híbrido v12.1", page_icon="🤖", layout="wide")

# CSS
st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- VERIFICAÇÃO DE CHAVES ---
try:
    # Tenta pegar dos secrets
    KEY_JOGOS = st.secrets["FOOTBALL_DATA_TOKEN"] # API Antiga
    KEY_ODDS = st.secrets["API_FOOTBALL_KEY"]     # API Nova
except KeyError:
    # Se falhar, tenta buscar dentro do bloco google_creds (erro comum de formatação)
    try:
        KEY_JOGOS = st.secrets["google_creds"]["FOOTBALL_DATA_TOKEN"]
        KEY_ODDS = st.secrets["google_creds"]["API_FOOTBALL_KEY"]
    except:
        st.error("🚨 ERRO CRÍTICO: Chaves de API não encontradas nos Secrets.")
        st.info("Verifique se 'FOOTBALL_DATA_TOKEN' e 'API_FOOTBALL_KEY' estão no secrets.toml")
        st.stop()

# Mapas de Ligas (Para a API ANTIGA - Football Data)
LIGAS_FD = {
    "Brasileirão": "BSA", "Champions League": "CL", "Premier League": "PL", 
    "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", 
    "Eredivisie": "DED", "Championship": "ELC", "Primeira Liga": "PPL", "Euro": "EC"
}

# --- FUNÇÕES DE BANCO DE DADOS (DEFINIDAS ANTES DO USO) ---
@st.cache_resource
def connect_db():
    try:
        if "google_creds" not in st.secrets: return None
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets.google_creds), scope)
        return gspread.authorize(creds).open_by_url(st.secrets.GOOGLE_SHEET_URL).sheet1
    except Exception as e:
        print(f"Erro DB: {e}")
        return None

def salvar_db(sheet, data, liga, jogo, mercado, odd, prob, valor, stake):
    if sheet: 
        try:
            sheet.append_row([data, liga, jogo, mercado, float(odd), float(prob)/100, float(valor)/100, "Aguardando ⏳"], value_input_option='USER_ENTERED')
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

# --- FUNÇÕES DE API ---

@st.cache_data(ttl=300)
def get_jogos_fd(liga_code, date_str):
    """Busca jogos na API Estável (Football-Data.org)"""
    url = f"https://api.football-data.org/v4/competitions/{liga_code}/matches"
    headers = {"X-Auth-Token": KEY_JOGOS}
    params = {"dateFrom": date_str, "dateTo": date_str, "status": "SCHEDULED"}
    try:
        r = requests.get(url, headers=headers, params=params)
        return r.json().get('matches', [])
    except: return []

@st.cache_data(ttl=300)
def get_odds_e_nomes_af(date_str):
    """
    Busca na API Nova (API-Football):
    1. Lista de Jogos (Fixtures) para pegar ID -> Nome
    2. Odds (Odds) para pegar ID -> Odds
    Retorna um dicionário unificado: { "Nome do Time Casa": {ODDS...} }
    """
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': KEY_ODDS}
    
    # 1. Baixar Fixtures (Nomes)
    try:
        r_fix = requests.get("https://v3.football.api-sports.io/fixtures", headers=headers, params={"date": date_str})
        fixtures = r_fix.json().get('response', [])
    except: return {}

    if not fixtures: return {}

    # Mapeia ID -> Nome do Mandante
    id_to_name = {f['fixture']['id']: f['teams']['home']['name'] for f in fixtures}

    # 2. Baixar Odds
    try:
        r_odds = requests.get("https://v3.football.api-sports.io/odds", headers=headers, params={"date": date_str, "bookmaker": "8"}) # Bet365
        odds_resp = r_odds.json().get('response', [])
        # Fallback se Bet365 falhar
        if not odds_resp:
             r_odds = requests.get("https://v3.football.api-sports.io/odds", headers=headers, params={"date": date_str})
             odds_resp = r_odds.json().get('response', [])
    except: return {}

    # 3. Unifica
    final_map = {}
    for o in odds_resp:
        fid = o['fixture']['id']
        nome_casa = id_to_name.get(fid)
        
        if nome_casa:
            bookie = o['bookmakers'][0]
            markets = {}
            for m in bookie['bets']:
                if m['id'] == 1: markets['1x2'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                elif m['id'] == 5: markets['goals'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                elif m['id'] == 8: markets['btts'] = {Op['value']: float(Op['odd']) for Op in m['values']}
            
            final_map[nome_casa] = markets
            
    return final_map

def fundir_dados(jogos_fd, mapa_odds_af):
    """Cruza os dados das duas APIs usando o nome do time"""
    jogos_finais = []
    nomes_disponiveis_af = list(mapa_odds_af.keys())
    
    for jogo in jogos_fd:
        time_casa_fd = jogo['homeTeam']['name']
        time_fora_fd = jogo['awayTeam']['name']
        
        # Converte hora UTC para Manaus
        utc_time = datetime.strptime(jogo['utcDate'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
        hora = utc_time.astimezone(FUSO).strftime('%H:%M')
        
        # Tenta achar o nome na lista da API Nova
        # cutoff=0.6 significa 60% de similaridade mínima
        match = difflib.get_close_matches(time_casa_fd, nomes_disponiveis_af, n=1, cutoff=0.5)
        
        odds = {}
        status = "⚠️"
        
        if match:
            nome_na_api_nova = match[0]
            odds = mapa_odds_af[nome_na_api_nova]
            status = "💰"
            
        jogos_finais.append({
            'hora': hora,
            'casa': time_casa_fd,
            'fora': time_fora_fd,
            'odds': odds,
            'status': status
        })
    return jogos_finais

# --- CÉREBRO (DIXON/POISSON) ---
@st.cache_data
def load_dc(liga_nome):
    try:
        code = LIGAS_FD[liga_nome]
        with open(f"dc_params_{code}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data
def get_historico_fd(liga_code, season):
    # Usa API Antiga para histórico (é mais consistente)
    url = f"https://api.football-data.org/v4/competitions/{liga_code}/matches"
    headers = {"X-Auth-Token": KEY_JOGOS}
    params = {"season": season, "status": "FINISHED"}
    try:
        r = requests.get(url, headers=headers, params=params)
        data = r.json().get('matches', [])
        lista = [{'data_jogo': m['utcDate'][:10], 'TimeCasa': m['homeTeam']['name'], 'TimeVisitante': m['awayTeam']['name'], 'GolsCasa': int(m['score']['fullTime']['home']), 'GolsVisitante': int(m['score']['fullTime']['away'])} for m in data if m['score']['fullTime']['home'] is not None]
        df = pd.DataFrame(lista)
        if not df.empty: df['data_jogo'] = pd.to_datetime(df['data_jogo'])
        return df, {'media_gols_casa': df['GolsCasa'].mean(), 'media_gols_visitante': df['GolsVisitante'].mean()} if not df.empty else None
    except: return None, None

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
    xg = (0,0)
    try:
        if mode == "DIXON_COLES":
            # Não precisa traduzir pois FD alimenta FD
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

def calc_kelly(prob, odd, fracao, banca):
    if odd<=1 or prob<=0: return 0,0
    b = odd-1; q=1-prob
    f = (b*prob-q)/b
    stk = (f*fracao*banca) if f>0 else 0
    return stk, f*fracao*100

# --- INTERFACE ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô Híbrido v12.1")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_FD.keys())
    LIGA_CODE = LIGAS_FD[LIGA_NOME]
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

dc_data = load_dc(LIGA_NOME)
df_hist, avg_hist = get_historico_fd(LIGA_CODE, 2025) # Hardcoded 2025 para FD
MODE = "DIXON_COLES" if dc_data else ("POISSON" if df_hist is not None else "FALHA")

t_jogos, t_hist = st.tabs(["Jogos", "Histórico"])

with t_jogos:
    st.subheader(f"{LIGA_NOME} ({MODE}) - {dt_sel.strftime('%d/%m')}")
    
    with st.spinner("🔄 Buscando Jogos (API Antiga) e Odds (API Nova)..."):
        # 1. Busca Jogos
        jogos = get_jogos_fd(LIGA_CODE, dt_sel.strftime('%Y-%m-%d'))
        
        # 2. Busca Odds do Dia (Somente se achou jogos)
        mapa_odds = {}
        if jogos:
            mapa_odds = get_odds_e_nomes_af(dt_sel.strftime('%Y-%m-%d'))
            
        # 3. Funde
        matches = fundir_dados(jogos, mapa_odds)
        
    if not matches:
        st.info("Nenhum jogo encontrado na Football-Data.org para esta data.")
    else:
        if 'sel_game' not in st.session_state:
            for i, m in enumerate(matches):
                p, xg = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
                c1, c2 = st.columns([3, 1])
                btn_lbl = f"{m['status']} {m['hora']} | {m['casa']} x {m['fora']}"
                if c1.button(btn_lbl, key=f"b{i}", use_container_width=True):
                    st.session_state.sel_game = m
                    st.rerun()
                c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")
        else:
            g = st.session_state.sel_game
            if st.button("⬅️ Voltar"): del st.session_state.sel_game; st.rerun()
            st.markdown(f"### {g['casa']} vs {g['fora']}")
            
            p, xg = predict(MODE, dc_data, df_hist, avg_hist, g['casa'], g['fora'])
            
            with st.form("analise"):
                col_o = st.columns(4)
                # Pega odds automáticas ou 1.0
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
                        st.info(f"🔮 Placar: **{p['placar'][0]}x{p['placar'][1]}** | Faixa 2-3: {p['f_23']:.1%}")
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
