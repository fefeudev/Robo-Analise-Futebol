# app.py - Robô de Valor (v12.0 - Híbrido: Jogos da FD + Odds da AF)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import config, time, json, pytz, gspread, difflib
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô Híbrido v12", page_icon="🤖", layout="wide")

# CSS
st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- CHAVES ---
try:
    KEY_JOGOS = st.secrets["FOOTBALL_DATA_TOKEN"] # Football-Data.org
    KEY_ODDS = st.secrets["API_FOOTBALL_KEY"]     # API-Football
except:
    st.error("⚠️ Configure 'FOOTBALL_DATA_TOKEN' e 'API_FOOTBALL_KEY' nos Secrets!")
    st.stop()

# --- MAPA DE LIGAS (PARA A API ANTIGA) ---
LIGAS_FD = {
    "Brasileirão": "BSA", "Champions League": "CL", "Premier League": "PL", 
    "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", 
    "Eredivisie": "DED", "Championship": "ELC", "Primeira Liga": "PPL", "Euro": "EC"
}

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
def get_odds_af(date_str):
    """Busca TODAS as odds do dia na API Nova (API-Football)"""
    url = "https://v3.football.api-sports.io/odds"
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': KEY_ODDS}
    # Tenta Bet365 (8), se falhar tenta geral
    params = {"date": date_str, "bookmaker": "8"} 
    try:
        r = requests.get(url, headers=headers, params=params)
        resp = r.json().get('response', [])
        if not resp: # Fallback sem bookmaker específico
             r = requests.get(url, headers=headers, params={"date": date_str})
             resp = r.json().get('response', [])
        return resp
    except: return []

# --- INTEGRAÇÃO (O CÉREBRO DA FUSÃO) ---

def encontrar_odd_correspondente(time_casa_fd, odds_af_list):
    """
    Usa Fuzzy Logic para encontrar o time da API Nova que tem 
    o nome mais parecido com o time da API Antiga.
    """
    if not odds_af_list: return None
    
    # Cria lista de todos os times mandantes nas odds baixadas
    times_odds = [o['fixture']['id'] for o in odds_af_list] # ID não ajuda, precisamos do nome
    # A API de Odds não retorna nome do time fácil na lista de odds, 
    # mas retorna a LEAGUE. Vamos filtrar por lógica.
    
    # SOLUÇÃO MELHOR: 
    # A API Football Odds retorna: league, fixture(id), bookmakers.
    # NÃO RETORNA O NOME DO TIME DIRETAMENTE NO ENDPOINT DE ODDS :(
    # Mas retorna o ID.
    
    # RECUO ESTRATÉGICO: 
    # Se a API de Odds não dá o nome, não conseguimos cruzar por nome.
    # POREM, seu teste mostrou "Casa: 10Bet".
    # O endpoint /odds RETORNA SIM os nomes? Vamos conferir seu print.
    # Seu print mostrou: "Jogo 1 (ID...)" e depois "Casa: 10Bet". 
    # NÃO MOSTROU O NOME DO TIME.
    
    # CORREÇÃO CRÍTICA:
    # Para o método híbrido funcionar, precisamos baixar as FIXTURES da API Nova também
    # só para ter o mapa "ID -> NOME".
    return None

# --- CORREÇÃO DA LÓGICA HÍBRIDA ---
# Como a API de Odds só dá IDs, precisamos de um dicionário.

@st.cache_data(ttl=3600)
def criar_mapa_id_nome_af(date_str):
    """Baixa jogos da API Nova só para saber ID -> Nome"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': KEY_ODDS}
    try:
        # Baixa TODOS os jogos do dia (leve e rápido)
        r = requests.get(url, headers=headers, params={"date": date_str})
        data = r.json().get('response', [])
        mapa = {}
        for jogo in data:
            fid = jogo['fixture']['id']
            home_name = jogo['teams']['home']['name']
            mapa[fid] = home_name
        return mapa
    except: return {}

def fundir_dados(jogos_fd, odds_af, mapa_nomes_af):
    """
    Pega os jogos da API Antiga e procura a Odd na API Nova 
    comparando os nomes dos times.
    """
    jogos_finais = []
    
    # Prepara as odds num formato mais fácil: { "Nome Time Casa": {odds...} }
    odds_por_nome = {}
    for o in odds_af:
        fid = o['fixture']['id']
        # Descobre o nome do time usando o mapa
        nome_time_af = mapa_nomes_af.get(fid)
        
        if nome_time_af:
            # Extrai as odds
            bookie = o['bookmakers'][0]
            markets = {}
            for m in bookie['bets']:
                if m['id'] == 1: markets['1x2'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                elif m['id'] == 5: markets['goals'] = {Op['value']: float(Op['odd']) for Op in m['values']}
                elif m['id'] == 8: markets['btts'] = {Op['value']: float(Op['odd']) for Op in m['values']}
            
            odds_por_nome[nome_time_af] = markets

    # Agora percorre os jogos da API Antiga e tenta achar match
    nomes_disponiveis_af = list(odds_por_nome.keys())
    
    for jogo in jogos_fd:
        time_casa_fd = jogo['homeTeam']['name']
        time_fora_fd = jogo['awayTeam']['name']
        hora = datetime.strptime(jogo['utcDate'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc).astimezone(FUSO).strftime('%H:%M')
        
        # FUZZY MATCH: Procura o nome mais parecido na lista da API Nova
        match = difflib.get_close_matches(time_casa_fd, nomes_disponiveis_af, n=1, cutoff=0.5) # 0.5 = 50% similaridade
        
        odds_encontradas = {}
        match_info = "Sem Odd"
        
        if match:
            nome_encontrado = match[0]
            odds_encontradas = odds_por_nome[nome_encontrado]
            match_info = "💰 Odd Auto"
        
        jogos_finais.append({
            'hora': hora,
            'casa': time_casa_fd,
            'fora': time_fora_fd,
            'odds': odds_encontradas,
            'status_odd': match_info
        })
        
    return jogos_finais

# --- LÓGICA DE CÁLCULO (MANTIDA) ---
@st.cache_data
def load_dc(liga_nome):
    try:
        code = LIGAS_FD[liga_nome] # Usa código da FD (ex: PL)
        with open(f"dc_params_{code}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data
def get_historico_fd(liga_code, season):
    # Usa a API antiga para histórico (pois já funcionava)
    url = f"https://api.football-data.org/v4/competitions/{liga_code}/matches"
    headers = {"X-Auth-Token": KEY_JOGOS}
    params = {"season": season, "status": "FINISHED"}
    try:
        r = requests.get(url, headers=headers, params=params)
        data = r.json().get('matches', [])
        lista = []
        for m in data:
            if m['score']['fullTime']['home'] is not None:
                lista.append({
                    'data_jogo': m['utcDate'][:10],
                    'TimeCasa': m['homeTeam']['name'],
                    'TimeVisitante': m['awayTeam']['name'],
                    'GolsCasa': int(m['score']['fullTime']['home']),
                    'GolsVisitante': int(m['score']['fullTime']['away'])
                })
        df = pd.DataFrame(lista)
        if not df.empty: df['data_jogo'] = pd.to_datetime(df['data_jogo'])
        return df, {'media_gols_casa': df['GolsCasa'].mean(), 'media_gols_visitante': df['GolsVisitante'].mean()} if not df.empty else None
    except: return None, None

def calc_probs(l_casa, m_visit, rho=0.0):
    probs = np.zeros((7, 7))
    max_prob, placar = 0, (0,0)
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
    total = np.sum(probs)
    if total==0: return None
    home, draw, away = np.sum(np.tril(probs,-1)), np.sum(np.diag(probs)), np.sum(np.triu(probs,1))
    over, btts = 0, 0
    for i in range(7):
        for j in range(7):
            if (i+j)>2.5: over+=probs[i,j]
            if i>0 and j>0: btts+=probs[i,j]
    return {'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total, 'over_2_5': over/total, 'btts_sim': btts/total, 'placar': placar}

def predict(mode, dc, df_poi, avg_poi, home, away):
    xg = (0,0)
    try:
        if mode == "DIXON_COLES":
            # Aqui não precisa traduzir pq estamos usando a mesma API (Football Data) que gerou o JSON!
            f = dc['forcas']
            # Verifica se o time existe no JSON
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

# --- INTERFACE ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô Híbrido v12")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_FD.keys())
    LIGA_CODE = LIGAS_FD[LIGA_NOME]
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

# CARGA DE DADOS
dc_data = load_dc(LIGA_NOME)
df_hist, avg_hist = get_historico_fd(LIGA_CODE, config.TEMPORADA_PARA_ANALISAR)
MODE = "DIXON_COLES" if dc_data else ("POISSON" if df_hist is not None else "FALHA")

t_jogos, t_hist = st.tabs(["Jogos & Radar", "Histórico"])

with t_jogos:
    st.subheader(f"{LIGA_NOME} ({MODE}) - {dt_sel.strftime('%d/%m')}")
    
    jogos_fd = []
    matches = []
    
    with st.spinner("1. Buscando jogos na Football-Data..."):
        jogos_fd = get_jogos_fd(LIGA_CODE, dt_sel.strftime('%Y-%m-%d'))
        
    if not jogos_fd:
        st.info("Nenhum jogo encontrado na API Football-Data.")
    else:
        with st.spinner("2. Buscando Odds na API-Football e fundindo..."):
            # Baixa odds e mapa de nomes do dia
            odds_af = get_odds_af(dt_sel.strftime('%Y-%m-%d'))
            mapa_af = criar_mapa_id_nome_af(dt_sel.strftime('%Y-%m-%d'))
            
            # Funde tudo
            matches = fundir_dados(jogos_fd, odds_af, mapa_af)
            
        # --- RADAR DE OPORTUNIDADES ---
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

        # --- LISTA DE JOGOS ---
        if 'sel_game' not in st.session_state:
            for i, m in enumerate(matches):
                p, xg = predict(MODE, dc_data, df_hist, avg_hist, m['casa'], m['fora'])
                f_c = get_form(m['casa'], df_hist)
                f_f = get_form(m['fora'], df_hist)
                
                c1, c2 = st.columns([3, 1])
                status = "💰" if m['odds'] else "📝" # Mostra saco de dinheiro se achou odd
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
                # Tenta pegar odds (pode vir vazio se não deu match)
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
                            b, q = odd-1, 1-prob
                            f = (b*prob-q)/b if b>0 else 0
                            stk = (f*KELLY*BANCA) if f>0 and cor=="normal" else 0
                            l = f"{prob:.1%}" + (f" (R${stk:.0f})" if stk>0 else "")
                            cols[idx].metric(lbl, l, f"{ev*100:.1f}% EV", delta_color=cor)
                        if uh>1: show("Casa", p['vitoria_casa'], uh, 0)
                        if ua>1: show("Fora", p['vitoria_visitante'], ua, 1)
                        if uo>1: show("Over", p['over_2_5'], uo, 2)
