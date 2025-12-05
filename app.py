# app.py - Robô v15.1 (Fluxo Reverso + Correção de Ordem + Fuso Horário)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import time, json, pytz, gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CONFIGURAÇÕES INICIAIS ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô v15.1", page_icon="🛡️", layout="wide")

st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

# --- 2. DEFINIÇÃO DE FUNÇÕES (TUDO AQUI EM CIMA PARA NÃO DAR ERRO) ---

@st.cache_resource
def connect_db():
    """Conecta ao Google Sheets de forma segura"""
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

@st.cache_data(ttl=300)
def buscar_todas_odds_range(api_key, date_obj):
    """
    Busca odds de HOJE e AMANHÃ para garantir que jogos da noite (fuso horário) apareçam.
    Não filtra por liga na requisição para evitar o erro de 'Season Required'.
    """
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': api_key}
    url = "https://v3.football.api-sports.io/odds"
    
    # Lista de datas (Hoje e Amanhã)
    dates = [date_obj.strftime('%Y-%m-%d'), (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')]
    
    todas_odds = []
    
    for d in dates:
        try:
            # Tenta Bet365 (8)
            r = requests.get(url, headers=headers, params={"date": d, "bookmaker": "8"})
            data = r.json().get('response', [])
            if not data: # Se falhar, tenta qualquer casa
                r = requests.get(url, headers=headers, params={"date": d})
                data = r.json().get('response', [])
            
            if data:
                todas_odds.extend(data)
        except: continue
        
    return todas_odds

@st.cache_data(ttl=300)
def buscar_detalhes_jogos(api_key, lista_ids):
    """Busca os nomes dos times usando os IDs encontrados nas odds"""
    if not lista_ids: return []
    
    headers = {'x-rapidapi-host': "v3.football.api-sports.io", 'x-rapidapi-key': api_key}
    url = "https://v3.football.api-sports.io/fixtures"
    
    jogos_detalhados = []
    
    # Paginação de 20 em 20 (limite da API)
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

@st.cache_data
def load_dc(sigla):
    try:
        with open(f"dc_params_{sigla}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

# Fuzzy match para o cérebro
def match_name_dc(name, dc_names):
    if name in dc_names: return name
    # Importante: difflib precisa ser importado aqui ou no topo
    import difflib
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

# --- 3. CONSTANTES E CHAVES ---
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

# IDs das Ligas e Mapeamento de JSON
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

# --- 4. EXECUÇÃO (SIDEBAR E LÓGICA) ---

# Conecta ao DB (Agora seguro, pois a função já foi lida lá em cima)
db = connect_db()

with st.sidebar:
    st.title("🤖 Robô v15.1")
    st.info("Modo: Busca Reversa (Odds -> Jogos)")
    
    dt_sel = st.date_input("Data:", datetime.now(FUSO).date())
    st.divider()
    BANCA = st.number_input("Banca (R$):", 100.0, step=50.0)
    KELLY = st.slider("Kelly:", 0.01, 0.50, 0.10)
    MIN_PROB = st.slider("Prob. Mín:", 50, 90, 60)/100.0

# PROCESSAMENTO PRINCIPAL
st.header(f"Jogos com Odds - {dt_sel.strftime('%d/%m')}")

lista_final_jogos = []
msg_status = "Aguardando busca..."

with st.spinner("Buscando Odds (Hoje + Amanhã) e filtrando..."):
    # 1. Busca Odds (Sem filtro de liga para não travar)
    todas_odds = buscar_todas_odds_range(API_KEY, dt_sel)
    
    if not todas_odds:
        msg_status = "A API de Odds retornou 0 jogos para estas datas."
    else:
        # 2. Filtra apenas nossas ligas de interesse
        ids_interesse = []
        mapa_odds = {} # ID -> Markets
        mapa_ligas = {} # ID Jogo -> Nome Liga
        
        ligas_ids_validos = [v[0] for v in LIGAS_MAP.values()]
        
        for item in todas_odds:
            lid = item['league']['id']
            if lid in ligas_ids_validos:
                fid = item['fixture']['id']
                ids_interesse.append(fid)
                
                # Descobre nome da liga
                nome_liga = next((k for k, v in LIGAS_MAP.items() if v[0] == lid), "Outra")
                mapa_ligas[fid] = nome_liga
                
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
        
        if not ids_interesse:
            msg_status = f"Odds encontradas ({len(todas_odds)}), mas nenhuma para as suas Ligas selecionadas."
        else:
            # 3. Busca detalhes dos jogos (Nomes)
            # Remove duplicatas de IDs
            ids_unicos = list(set(ids_interesse))
            detalhes = buscar_detalhes_jogos(API_KEY, ids_unicos)
            
            for d in detalhes:
                fid = d['fixture']['id']
                if fid in mapa_odds:
                    lista_final_jogos.append({
                        'id': fid,
                        'liga_nome': mapa_ligas.get(fid, "Desconhecida"),
                        'hora': datetime.fromtimestamp(d['fixture']['timestamp'], FUSO).strftime('%H:%M'),
                        'casa': d['teams']['home']['name'],
                        'fora': d['teams']['away']['name'],
                        'odds': mapa_odds[fid],
                        'status': "💰"
                    })

# Exibição
if not lista_final_jogos:
    st.warning(msg_status)
    if "Odds encontradas" in msg_status:
        st.info("Dica: Pode ser que hoje só tenha jogos de ligas menores que você não monitora.")
else:
    # Agrupa por Liga
    df = pd.DataFrame(lista_final_jogos)
    ligas_presentes = df['liga_nome'].unique()
    
    for liga in ligas_presentes:
        st.subheader(f"🏆 {liga}")
        jogos_liga = df[df['liga_nome'] == liga]
        
        # Carrega Cérebro
        sigla = LIGAS_MAP[liga][1]
        dc_data = load_dc(sigla)
        
        for _, m in jogos_liga.iterrows():
            p, xg = predict(dc_data, m['casa'], m['fora'])
            
            c1, c2 = st.columns([3, 1])
            if c1.button(f"💰 {m['hora']} | {m['casa']} x {m['fora']}", key=f"b_{m['id']}", use_container_width=True):
                st.session_state.sel_game = m.to_dict()
                st.session_state.sel_p = p
                st.rerun()
            c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")

# ANÁLISE
if 'sel_game' in st.session_state:
    st.divider()
    g = st.session_state.sel_game
    p = st.session_state.sel_p
    
    st.markdown(f"## 📊 Analisando: {g['casa']} x {g['fora']}")
    
    with st.form("auto_form"):
        c_odds = st.columns(2)
        o = g['odds']
        
        def get_o(cat, key): return o.get(cat, {}).get(key, 1.0)
        
        with c_odds[0]:
            st.caption("Principal")
            uh = st.number_input("Casa", value=get_o('1x2', 'Home'))
            ud = st.number_input("Empate", value=get_o('1x2', 'Draw'))
            ua = st.number_input("Fora", value=get_o('1x2', 'Away'))
            uo = st.number_input("Over 2.5", value=get_o('goals', 'Over 2.5'))
        
        with c_odds[1]:
            st.caption("Secundários")
            ub = st.number_input("BTTS Sim", value=get_o('btts', 'Yes'))
            u1x = st.number_input("1X", value=get_o('dc', 'Home/Draw'))
            ux2 = st.number_input("X2", value=get_o('dc', 'Draw/Away'))
        
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
                if ub>1: show("BTTS", p['btts_sim'], ub, 0)
                if u1x>1: show("1X", p['chance_dupla_1X'], u1x, 1)
                
                if db: st.success("✅ Análise Salva!")
