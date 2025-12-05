# app.py - Robô v15.0 (Fluxo Reverso: Odds -> IDs -> Nomes)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import time, json, pytz, gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CONFIGURAÇÕES ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô v15.0 (Reverso)", page_icon="🔄", layout="wide")

st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- 2. CHAVES ---
try:
    if "API_FOOTBALL_KEY" in st.secrets:
        API_KEY = st.secrets["API_FOOTBALL_KEY"]
    elif "google_creds" in st.secrets and "API_FOOTBALL_KEY" in st.secrets["google_creds"]:
        API_KEY = st.secrets["google_creds"]["API_FOOTBALL_KEY"]
    else:
        st.error("🚨 Chave API_FOOTBALL_KEY não encontrada!")
        st.stop()
except:
    st.error("🚨 Erro nos Secrets.")
    st.stop()

# IDs das Ligas que nos interessam (Filtro Pós-Busca)
LIGAS_INTERESSE = {
    71: "Brasileirão",
    2: "Champions League",
    39: "Premier League",
    140: "La Liga",
    135: "Serie A (Itália)",
    78: "Bundesliga",
    61: "Ligue 1",
    88: "Eredivisie",
    40: "Championship",
    94: "Primeira Liga",
    4: "Euro"
}
# Mapeamento Reverso para carregar o JSON do Cérebro
MAPA_JSON = {
    71: "BSA", 2: "CL", 39: "PL", 140: "PD", 135: "SA", 
    78: "BL1", 61: "FL1", 88: "DED", 40: "ELC", 94: "PPL", 4: "EC"
}

# --- 3. FUNÇÕES DO FLUXO REVERSO ---

@st.cache_data(ttl=300)
def buscar_todas_odds_do_dia(date_str):
    """
    Passo 1: Busca TODAS as odds disponíveis no dia.
    Não filtra liga na API para evitar bugs. Filtra no Python.
    """
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': API_KEY}
    url = "https://v3.football.api-sports.io/odds"
    
    try:
        # Tenta Bet365 primeiro (bookmaker 8)
        r = requests.get(url, headers=headers, params={"date": date_str, "bookmaker": "8"})
        data = r.json().get('response', [])
        
        # Se vazio, tenta genérico (qualquer casa)
        if not data:
            r = requests.get(url, headers=headers, params={"date": date_str})
            data = r.json().get('response', [])
            
        return data
    except: return []

@st.cache_data(ttl=300)
def buscar_detalhes_jogos(lista_ids):
    """
    Passo 2: Pega os IDs dos jogos que têm odds e busca os nomes.
    Usa o endpoint 'ids' que não depende de temporada.
    """
    if not lista_ids: return []
    
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': API_KEY}
    url = "https://v3.football.api-sports.io/fixtures"
    
    jogos_detalhados = []
    
    # A API aceita no máx 20 IDs por vez. Vamos paginar.
    chunk_size = 20
    for i in range(0, len(lista_ids), chunk_size):
        chunk = lista_ids[i:i + chunk_size]
        ids_str = "-".join(map(str, chunk))
        
        try:
            r = requests.get(url, headers=headers, params={"ids": ids_str})
            data = r.json().get('response', [])
            jogos_detalhados.extend(data)
        except: continue
        
    return jogos_detalhados

def processar_dados_reversos(date_str):
    # 1. Pega o sacolão de odds
    todas_odds = buscar_todas_odds_do_dia(date_str)
    
    if not todas_odds:
        return [], "A API de Odds não retornou nada para hoje."

    # 2. Filtra só as nossas ligas e guarda os IDs
    odds_filtradas = {}
    ids_para_buscar = []
    
    for item in todas_odds:
        lid = item['league']['id']
        fid = item['fixture']['id']
        
        if lid in LIGAS_INTERESSE:
            ids_para_buscar.append(fid)
            
            # Processa mercados
            bookie = item['bookmakers'][0]
            mkts = {}
            for m in bookie['bets']:
                if m['id'] == 1: mkts['1x2'] = {v['value']: float(v['odd']) for v in m['values']}
                elif m['id'] == 12: mkts['dc'] = {v['value']: float(v['odd']) for v in m['values']}
                elif m['id'] == 5: mkts['goals'] = {v['value']: float(v['odd']) for v in m['values']}
                elif m['id'] == 8: mkts['btts'] = {v['value']: float(v['odd']) for v in m['values']}
            
            odds_filtradas[fid] = {
                'liga_id': lid,
                'markets': mkts
            }
            
    if not ids_para_buscar:
        return [], "Odds encontradas, mas nenhuma das suas ligas favoritas tem jogo hoje."

    # 3. Busca os nomes desses IDs
    detalhes = buscar_detalhes_jogos(ids_para_buscar)
    
    # 4. Monta o objeto final
    lista_final = []
    for d in detalhes:
        fid = d['fixture']['id']
        lid = d['league']['id']
        
        if fid in odds_filtradas:
            lista_final.append({
                'id': fid,
                'liga_nome': LIGAS_INTERESSE.get(lid, "Desconhecida"),
                'liga_code': MAPA_JSON.get(lid, "GEN"), # Para carregar o cérebro certo
                'hora': datetime.fromtimestamp(d['fixture']['timestamp'], FUSO).strftime('%H:%M'),
                'casa': d['teams']['home']['name'],
                'fora': d['teams']['away']['name'],
                'odds': odds_filtradas[fid]['markets'],
                'status': "💰"
            })
            
    # Ordena por Liga e Hora
    lista_final.sort(key=lambda x: (x['liga_nome'], x['hora']))
    return lista_final, "Sucesso"

# --- 4. CÉREBRO (MANTIDO) ---
@st.cache_data
def load_dc(sigla):
    try:
        with open(f"dc_params_{sigla}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

# Fuzzy match para o cérebro (já que agora temos o nome oficial da API, precisamos achar no JSON antigo)
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
    st.title("🤖 Robô v15.0 (Reverso)")
    st.info("Modo: Odds -> Jogos")
    
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

# PROCESSAMENTO CENTRAL
st.header(f"Jogos com Odds Encontrados - {dt_sel.strftime('%d/%m')}")

matches = []
msg = ""

# Busca Dupla (Hoje e Amanhã) para garantir fuso
dates = [dt_sel.strftime('%Y-%m-%d'), (dt_sel + timedelta(days=1)).strftime('%Y-%m-%d')]

with st.spinner("Buscando Odds globais e filtrando suas ligas..."):
    seen_ids = set()
    for d in dates:
        res, m = processar_dados_reversos(d)
        if res:
            for jogo in res:
                if jogo['id'] not in seen_ids:
                    matches.append(jogo)
                    seen_ids.add(jogo['id'])
        else:
            if not msg: msg = m # Guarda o primeiro erro

if not matches:
    st.warning(f"Nenhum jogo das suas ligas encontrado com odds disponíveis.")
    st.caption(f"Status Técnico: {msg}")
else:
    # Agrupa por Liga para ficar bonito
    df_jogos = pd.DataFrame(matches)
    ligas_ativas = df_jogos['liga_nome'].unique()
    
    for liga in ligas_ativas:
        st.subheader(f"🏆 {liga}")
        jogos_da_liga = df_jogos[df_jogos['liga_nome'] == liga]
        
        # Carrega Cérebro específico desta liga
        sigla = jogos_da_liga.iloc[0]['liga_code']
        dc_data = load_dc(sigla)
        
        for _, m in jogos_da_liga.iterrows():
            # Previsão
            p, xg = predict(dc_data, m['casa'], m['fora'])
            
            # Botão do Jogo
            col1, col2 = st.columns([3, 1])
            if col1.button(f"💰 {m['hora']} | {m['casa']} x {m['fora']}", key=f"b_{m['id']}", use_container_width=True):
                st.session_state.sel_game = m.to_dict()
                st.session_state.sel_p = p
                st.rerun()
            
            col2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")

# ÁREA DE ANÁLISE (FIXA NO FUNDO SE TIVER SELEÇÃO)
if 'sel_game' in st.session_state:
    st.divider()
    g = st.session_state.sel_game
    p = st.session_state.sel_p
    
    st.markdown(f"## 📊 Analisando: {g['casa']} x {g['fora']}")
    
    with st.form("auto_form"):
        c_odds = st.columns(4)
        o = g['odds']
        
        # Valores padrão
        def get_o(cat, key): return o.get(cat, {}).get(key, 1.0)
        
        uh = c_odds[0].number_input("Casa", value=get_o('1x2', 'Home'))
        ud = c_odds[1].number_input("Empate", value=get_o('1x2', 'Draw'))
        ua = c_odds[2].number_input("Fora", value=get_o('1x2', 'Away'))
        uo = c_odds[3].number_input("Over 2.5", value=get_o('goals', 'Over 2.5'))
        
        if st.form_submit_button("Calcular Valor & Kelly"):
            if p:
                st.info(f"🔮 Placar Provável: {p['placar'][0]}x{p['placar'][1]}")
                cols = st.columns(3)
                
                def show(lbl, prob, odd, idx):
                    ev = (prob*odd)-1
                    cor = "normal" if (ev>0.05 and prob>MIN_PROB) else "inverse"
                    stk, _ = calc_kelly(prob, odd, KELLY, BANCA)
                    l = f"{prob:.1%}" + (f" (R${stk:.0f})" if stk>0 else "")
                    cols[idx].metric(lbl, l, f"{ev*100:.1f}% EV", delta_color=cor)
                    if stk>0 and db: salvar_db(db, g['hora'], g['liga_nome'], f"{g['casa']}x{g['fora']}", lbl, odd, prob*100, ev*100, stk)

                if uh>1: show("Casa", p['vitoria_casa'], uh, 0)
                if ua>1: show("Fora", p['vitoria_visitante'], ua, 1)
                if uo>1: show("Over", p['over_2_5'], uo, 2)
                
                if db: st.success("✅ Análise Salva!")
