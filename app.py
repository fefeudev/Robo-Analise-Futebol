# app.py - Robô v16.0 (A Solução Definitiva: Ordem Correta + Fluxo Reverso)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import time, json, pytz, gspread, difflib
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CONFIGURAÇÕES INICIAIS ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô v16.0", page_icon="🤖", layout="wide")

st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- 2. DEFINIÇÃO DE TODAS AS FUNÇÕES (AQUI NO TOPO PARA NÃO DAR ERRO) ---

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

@st.cache_data(ttl=300)
def buscar_todas_odds_range(api_key, date_obj):
    # Busca odds de Hoje e Amanhã para garantir fuso horário
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': api_key}
    url = "https://v3.football.api-sports.io/odds"
    
    # Buscamos hoje e amanhã
    dates = [date_obj.strftime('%Y-%m-%d'), (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')]
    todas_odds = []
    
    for d in dates:
        try:
            # Tenta Bet365 (8)
            r = requests.get(url, headers=headers, params={"date": d, "bookmaker": "8"})
            data = r.json().get('response', [])
            
            # Se não vier nada, tenta SEM filtro de bookmaker (Pega qualquer casa)
            if not data:
                r = requests.get(url, headers=headers, params={"date": d})
                data = r.json().get('response', [])
            
            if data:
                todas_odds.extend(data)
        except: continue
        
    return todas_odds

@st.cache_data(ttl=300)
def buscar_detalhes_jogos(api_key, lista_ids):
    if not lista_ids: return []
    
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': api_key}
    url = "https://v3.football.api-sports.io/fixtures"
    jogos = []
    
    # Paginação de 20 em 20 (limite da URL)
    chunk_size = 20
    for i in range(0, len(lista_ids), chunk_size):
        chunk = lista_ids[i:i + chunk_size]
        ids_str = "-".join(map(str, chunk))
        try:
            r = requests.get(url, headers=headers, params={"ids": ids_str})
            jogos.extend(r.json().get('response', []))
        except: continue
    return jogos

@st.cache_data
def load_dc(sigla):
    try:
        with open(f"dc_params_{sigla}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

def match_name_dc(name, dc_names):
    if name in dc_names: return name
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

# --- 3. EXECUÇÃO PRINCIPAL ---

# Tenta pegar a chave
try:
    if "API_FOOTBALL_KEY" in st.secrets:
        API_KEY = st.secrets["API_FOOTBALL_KEY"]
    elif "google_creds" in st.secrets and "API_FOOTBALL_KEY" in st.secrets["google_creds"]:
        API_KEY = st.secrets["google_creds"]["API_FOOTBALL_KEY"]
    else:
        API_KEY = None
except: API_KEY = None

if not API_KEY:
    st.error("🚨 Chave API_FOOTBALL_KEY não encontrada nos Secrets!")
    st.stop()

# Mapeamento de Ligas
LIGAS_MAP = {
    "Brasileirão": (71, "BSA"), "Champions League": (2, "CL"), "Premier League": (39, "PL"),
    "La Liga": (140, "PD"), "Serie A (Itália)": (135, "SA"), "Bundesliga": (78, "BL1"),
    "Ligue 1": (61, "FL1"), "Eredivisie": (88, "DED"), "Championship": (40, "ELC"),
    "Primeira Liga": (94, "PPL"), "Euro": (4, "EC")
}

MERCADOS_INFO = {'vitoria_casa': 'Casa', 'empate': 'Empate', 'vitoria_visitante': 'Fora', 'over_2_5': 'Over 2.5', 'btts_sim': 'BTTS Sim'}

# Conecta DB (Sem erro, pois a função já foi criada lá em cima)
db = connect_db()

with st.sidebar:
    st.title("🤖 Robô v16.0")
    st.info("Modo: Odds -> Jogos")
    LIGA_NOME = st.selectbox("Liga:", LIGAS_MAP.keys())
    ID_LIGA, SIGLA_LIGA = LIGAS_MAP[LIGA_NOME]
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

# --- PROCESSAMENTO ---
dc_data = load_dc(SIGLA_LIGA)
st.subheader(f"{LIGA_NOME} - {dt_sel.strftime('%d/%m')}")

lista_jogos = []
msg_status = "Aguardando..."

with st.spinner("Buscando Odds (Hoje e Amanhã) e filtrando sua liga..."):
    # 1. Busca Odds GERAIS (Sem filtro de liga para evitar erro de season)
    todas_odds = buscar_todas_odds_range(API_KEY, dt_sel)
    
    if not todas_odds:
        msg_status = "Zero odds encontradas na API para hoje."
    else:
        # 2. Filtra IDs da Liga Selecionada
        ids_para_buscar = []
        mapa_odds = {}
        
        for item in todas_odds:
            if item['league']['id'] == ID_LIGA:
                fid = item['fixture']['id']
                ids_para_buscar.append(fid)
                
                # Processa mercados
                if item['bookmakers']:
                    bk = item['bookmakers'][0]
                    mkts = {}
                    for m in bk['bets']:
                        if m['id'] == 1: mkts['1x2'] = {v['value']: float(v['odd']) for v in m['values']}
                        elif m['id'] == 12: mkts['dc'] = {v['value']: float(v['odd']) for v in m['values']}
                        elif m['id'] == 5: mkts['goals'] = {v['value']: float(v['odd']) for v in m['values']}
                        elif m['id'] == 8: mkts['btts'] = {v['value']: float(v['odd']) for v in m['values']}
                    mapa_odds[fid] = mkts

        if not ids_para_buscar:
            msg_status = f"API retornou {len(todas_odds)} odds, mas NENHUMA para a liga {LIGA_NOME}."
        else:
            # 3. Busca nomes dos times (Batch)
            ids_unicos = list(set(ids_para_buscar))
            detalhes = buscar_detalhes_jogos(API_KEY, ids_unicos)
            
            for d in detalhes:
                fid = d['fixture']['id']
                if fid in mapa_odds:
                    lista_jogos.append({
                        'id': fid,
                        'hora': datetime.fromtimestamp(d['fixture']['timestamp'], FUSO).strftime('%H:%M'),
                        'casa': d['teams']['home']['name'],
                        'fora': d['teams']['away']['name'],
                        'odds': mapa_odds[fid],
                        'status': "💰"
                    })

if not lista_jogos:
    st.warning(msg_status)
    if "API retornou" in msg_status:
        st.caption("Isso significa que a API está funcionando, mas a liga selecionada não tem jogos ou odds liberadas hoje.")
else:
    # RADAR AUTOMÁTICO
    radar = []
    for m in lista_jogos:
        p, x = predict(dc_data, m['casa'], m['fora'])
        if p and m['odds']:
            o = m['odds']
            check = [('Home', '1x2', 'Home', 'vitoria_casa'), ('Away', '1x2', 'Away', 'vitoria_visitante'), ('Over 2.5', 'goals', 'Over 2.5', 'over_2_5')]
            for lbl, cat, sel, pk in check:
                if cat in o and sel in o[cat]:
                    odd = o[cat][sel]
                    prob = p[pk]
                    ev = (prob * odd) - 1
                    if prob > MIN_PROB and ev > 0.05:
                        radar.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': lbl, 'Odd': odd, 'Prob': prob, 'EV': ev*100})

    if radar:
        with st.expander("🔥 RADAR DE OPORTUNIDADES", expanded=True):
            st.dataframe(pd.DataFrame(radar).sort_values('EV', ascending=False), hide_index=True, use_container_width=True)

    # LISTA DE JOGOS
    for m in lista_jogos:
        p, xg = predict(dc_data, m['casa'], m['fora'])
        c1, c2 = st.columns([3, 1])
        if c1.button(f"💰 {m['hora']} | {m['casa']} x {m['fora']}", key=f"b_{m['id']}", use_container_width=True):
            st.session_state.sel_game = m.to_dict()
            st.session_state.sel_p = p
            st.rerun()
        c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")

# ÁREA DE ANÁLISE
if 'sel_game' in st.session_state:
    st.divider()
    g = st.session_state.sel_game
    p = st.session_state.sel_p
    
    st.markdown(f"### {g['casa']} x {g['fora']}")
    
    with st.form("analise_form"):
        c_odds = st.columns(2)
        o = g['odds']
        def go(c, k): return o.get(c, {}).get(k, 1.0)
        
        with c_odds[0]:
            uh = st.number_input("Casa", value=go('1x2', 'Home'))
            ud = st.number_input("Empate", value=go('1x2', 'Draw'))
            ua = st.number_input("Fora", value=go('1x2', 'Away'))
            uo = st.number_input("Over 2.5", value=go('goals', 'Over 2.5'))
        with c_odds[1]:
            ub = st.number_input("BTTS", value=go('btts', 'Yes'))
            u1x = st.number_input("1X", value=go('dc', 'Home/Draw'))
            ux2 = st.number_input("X2", value=go('dc', 'Draw/Away'))
            u12 = st.number_input("12", value=go('dc', 'Home/Away'))
            
        if st.form_submit_button("Calcular"):
            if p:
                st.info(f"Placar: {p['placar'][0]}x{p['placar'][1]}")
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
