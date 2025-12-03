# app.py - Robô de Valor (v8.3 - Gestão de Banca Kelly)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import config, time, json, pytz, gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô de Valor", page_icon="🤖", layout="wide")

st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}</style>""", unsafe_allow_html=True)

LIGAS = {"Brasileirão": "BSA", "Champions League": "CL", "Premier League": "PL", "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", "Eredivisie": "DED", "Championship": "ELC", "Primeira Liga": "PPL", "Euro": "EC"}
EMOJIS = {"BSA": "🇧🇷", "PL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "CL": "🇪🇺", "PD": "🇪🇸", "SA": "🇮🇹", "BL1": "🇩🇪", "FL1": "🇫🇷", "DED": "🇳🇱", "ELC": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "PPL": "🇵🇹", "EC": "🇪🇺"}
MERCADOS_INFO = {'vitoria_casa': 'Casa', 'empate': 'Empate', 'vitoria_visitante': 'Fora', 'over_2_5': 'Over 2.5', 'btts_sim': 'BTTS Sim', 'chance_dupla_1X': '1X', 'chance_dupla_X2': 'X2', 'chance_dupla_12': '12'}

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def req_api(endpoint, params):
    try:
        resp = requests.get(config.API_BASE_URL + endpoint, headers={"X-Auth-Token": config.API_KEY}, params=params)
        resp.raise_for_status()
        return resp.json()
    except: return None

@st.cache_resource
def connect_db():
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets.google_creds), scope)
        return gspread.authorize(creds).open_by_url(st.secrets.GOOGLE_SHEET_URL).sheet1
    except: return None

def salvar_db(sheet, data, liga, jogo, mercado, odd, prob, valor, stake):
    # Adicionada coluna de Stake no DB se quiser usar futuramente (apenas texto por enquanto)
    if sheet: sheet.append_row([data, liga, jogo, mercado, float(odd), float(prob)/100, float(valor)/100, "Aguardando ⏳"], value_input_option='USER_ENTERED')

@st.cache_data(ttl=60)
def load_db(_sheet):
    try:
        vals = _sheet.get_all_values()
        if len(vals) < 2: return pd.DataFrame(), 0, 0
        df = pd.DataFrame(vals[1:], columns=vals[0])
        for c in ['Odd', 'Probabilidade', 'Valor']:
            if c in df.columns: df[c] = pd.to_numeric(df[c].str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['Odd'])
        counts = df['Status'].value_counts() if 'Status' in df.columns else {}
        return df, counts.get('Green ✅', 0), counts.get('Red ❌', 0)
    except: return pd.DataFrame(), 0, 0

# --- NOVO: CÁLCULO DE KELLY ---
def calc_kelly(prob_ganhar, odd_decimal, fracao_kelly, banca_total):
    """
    Calcula o valor da aposta usando o Critério de Kelly Fracionado.
    f* = (bp - q) / b
    Onde: b = odd - 1; p = probabilidade; q = 1 - p
    """
    if odd_decimal <= 1 or prob_ganhar <= 0: return 0.0, 0.0
    
    b = odd_decimal - 1
    p = prob_ganhar
    q = 1 - p
    
    f_star = (b * p - q) / b
    
    # Se f_star negativo (EV negativo), stake é 0
    if f_star <= 0: return 0.0, 0.0
    
    # Aplica a fração de segurança (Kelly Parcial)
    stake_pct = f_star * fracao_kelly
    stake_valor = banca_total * stake_pct
    
    return stake_valor, (stake_pct * 100)

# --- LÓGICA DE PREVISÃO & DADOS ---
@st.cache_data
def load_dc(liga):
    try:
        with open(f"dc_params_{liga}.json", 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

@st.cache_data(ttl=3600)
def get_standings(lid, season):
    try:
        res = req_api(f"competitions/{lid}/standings", {"season": str(season)})
        if not res or 'standings' not in res: return None
        t = res['standings'][0]['table']
        return pd.DataFrame([{'Pos': i['position'], 'Time': i['team']['name'], 'Pts': i['points'], 'J': i['playedGames'], 'SG': i['goalDifference']} for i in t])
    except: return None

@st.cache_data
def train_poisson(liga, temp):
    data = req_api(f"competitions/{liga}/matches", {"season": str(temp), "status": "FINISHED"})
    if not data or not data.get("matches"): return None, None
    matches = [{'data_jogo': m['utcDate'][:10], 'TimeCasa': m['homeTeam']['name'], 'TimeVisitante': m['awayTeam']['name'], 'GolsCasa': int(m['score']['fullTime']['home']), 'GolsVisitante': int(m['score']['fullTime']['away'])} for m in data['matches'] if m['score']['fullTime']['home'] is not None]
    df = pd.DataFrame(matches).sort_values('data_jogo')
    df['data_jogo'] = pd.to_datetime(df['data_jogo'])
    return df, {'media_gols_casa': df['GolsCasa'].mean(), 'media_gols_visitante': df['GolsVisitante'].mean()}

def calc_probs(l_casa, m_visit, rho=0.0):
    probs = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            tau = 1.0
            if i==0 and j==0: tau = 1 - (l_casa*m_visit*rho)
            elif i==1 and j==0: tau = 1 + (l_casa*rho)
            elif i==0 and j==1: tau = 1 + (m_visit*rho)
            elif i==1 and j==1: tau = 1 - rho
            probs[i, j] = stats.poisson.pmf(i, l_casa) * stats.poisson.pmf(j, m_visit) * tau
    
    total = np.sum(probs)
    if total == 0: return None
    
    home = np.sum(np.tril(probs, -1))
    draw = np.sum(np.diag(probs))
    away = np.sum(np.triu(probs, 1))
    over = np.sum(probs[np.triu_indices(7, k=0)]) - np.sum(probs[0:3, 0:3]) + probs[2,0] + probs[0,2] + probs[1,1] 
    
    over, btts = 0, 0
    for i in range(7):
        for j in range(7):
            if i+j > 2.5: over += probs[i,j]
            if i>0 and j>0: btts += probs[i,j]

    return {
        'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
        'over_2_5': over/total, 'btts_sim': btts/total,
        'chance_dupla_1X': (home+draw)/total, 'chance_dupla_X2': (draw+away)/total, 'chance_dupla_12': (home+away)/total
    }

def unified_predict(mode, dc_data, poisson_df, poisson_avg, home, away, date_game):
    xg = (0,0)
    try:
        if mode == "DIXON_COLES":
            f = dc_data['forcas']
            l_c = np.exp(f[home]['ataque'] + f[away]['defesa'] + dc_data['vantagem_casa'])
            m_v = np.exp(f[away]['ataque'] + f[home]['defesa'])
            probs = calc_probs(l_c, m_v, dc_data.get('rho', 0))
            xg = (l_c, m_v)
        elif mode == "POISSON_RECENTE":
            dt = pd.to_datetime(date_game)
            past = poisson_df[poisson_df['data_jogo'] < dt]
            hc = past[past['TimeCasa'] == home].tail(6)
            hv = past[past['TimeVisitante'] == away].tail(6)
            if len(hc) < 1 or len(hv) < 1: return None, None
            fa_c = hc['GolsCasa'].mean() / poisson_avg['media_gols_casa']
            fd_c = hc['GolsVisitante'].mean() / poisson_avg['media_gols_visitante']
            fa_v = hv['GolsVisitante'].mean() / poisson_avg['media_gols_visitante']
            fd_v = hv['GolsCasa'].mean() / poisson_avg['media_gols_casa']
            l_c = fa_c * fd_v * poisson_avg['media_gols_casa']
            m_v = fa_v * fd_c * poisson_avg['media_gols_visitante']
            probs = calc_probs(l_c, m_v)
            xg = (l_c, m_v)
        else: return None, None
        return probs, xg
    except: return None, None

# --- UI & FLUXO ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô v8.3")
    LIGA_KEY = st.selectbox("Liga:", LIGAS.keys())
    LIGA_ID = LIGAS[LIGA_KEY]
    
    agora = datetime.now(FUSO)
    if 'dt_sel' not in st.session_state: st.session_state.dt_sel = agora.date()
    c1, c2 = st.columns(2)
    if c1.button("< Ontem"): st.session_state.dt_sel -= timedelta(days=1)
    if c2.button("Amanhã >"): st.session_state.dt_sel += timedelta(days=1)
    st.session_state.dt_sel = st.date_input("Data:", st.session_state.dt_sel)
    if st.button("Hoje (Manaus)"): st.session_state.dt_sel = datetime.now(FUSO).date()
    
    st.divider()
    
    # --- NOVO: INPUTS DA BANCA ---
    st.header("💰 Gestão de Banca")
    BANCA_USUARIO = st.number_input("Banca Total (R$):", value=100.0, step=50.0)
    KELLY_FRACAO = st.slider("Fator Kelly (Risco):", 0.01, 0.50, 0.10, 0.01, help="0.10 é conservador (recomendado). 0.50 é agressivo.")
    st.caption("Sugere o valor da aposta baseado na confiança.")
    st.divider()
    # -----------------------------
    
    min_prob = st.slider("Prob. Mínima %", 0, 100, 60, 5) / 100.0
    detalhado = st.toggle("Modo Detalhado")

# Carga de Cérebro
dc_data = load_dc(LIGA_ID)
df_poi, avg_poi = (None, None)
MODE = "DIXON_COLES" if dc_data else "FALHA"
if not dc_data:
    df_poi, avg_poi = train_poisson(LIGA_ID, config.TEMPORADA_PARA_ANALISAR)
    if df_poi is not None: MODE = "POISSON_RECENTE"

# Abas
t_jogos, t_hist, t_times = st.tabs(["Jogos", "Histórico", "Times"])

with t_jogos:
    st.subheader(f"{EMOJIS.get(LIGA_ID,'')} {LIGA_KEY} - {MODE}")
    
    @st.cache_data(ttl=300)
    def get_matches(lid, dtarget):
        d1 = dtarget.strftime('%Y-%m-%d')
        d2 = (dtarget + timedelta(days=1)).strftime('%Y-%m-%d')
        res = req_api(f"competitions/{lid}/matches", {"dateFrom": d1, "dateTo": d2, "status": "SCHEDULED"})
        if not res or 'matches' not in res: return []
        final = []
        for m in res['matches']:
            dt_loc = pytz.utc.localize(datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ")).astimezone(FUSO)
            if dt_loc.date() == dtarget:
                final.append({'hora': dt_loc.strftime('%H:%M'), 'casa': m['homeTeam']['name'], 'fora': m['awayTeam']['name'], 'dt_iso': dt_loc.strftime('%Y-%m-%d')})
        return final

    matches = []
    if MODE != "FALHA": matches = get_matches(LIGA_ID, st.session_state.dt_sel)
    
    if 'sel_game' not in st.session_state:
        if not matches: st.info("Sem jogos agendados para hoje (Manaus).")
        else:
            for i, m in enumerate(matches):
                _, xg = unified_predict(MODE, dc_data, df_poi, avg_poi, m['casa'], m['fora'], m['dt_iso'])
                c1, c2 = st.columns([3, 1])
                if c1.button(f"⚽ {m['hora']} | {m['casa']} x {m['fora']}", key=f"b{i}", use_container_width=True):
                    st.session_state.sel_game = m
                    st.rerun()
                c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")
    else:
        g = st.session_state.sel_game
        if st.button("⬅️ Voltar"): 
            del st.session_state.sel_game
            st.rerun()
        
        st.markdown(f"### {g['casa']} vs {g['fora']}")
        with st.form("f1"):
            st.write("Odds:")
            oc = st.columns(4)
            in_odds = {}
            fields = [('vitoria_casa','Casa'), ('empate','X'), ('vitoria_visitante','Fora'), ('over_2_5','Over 2.5'),
                      ('btts_sim','BTTS'), ('chance_dupla_1X','1X'), ('chance_dupla_X2','X2'), ('chance_dupla_12','12')]
            for i, (k, l) in enumerate(fields):
                in_odds[k] = oc[i%4].number_input(l, min_value=1.0, step=0.01, format="%.2f")
            
            if st.form_submit_button("Analisar"):
                probs, xg = unified_predict(MODE, dc_data, df_poi, avg_poi, g['casa'], g['fora'], g['dt_iso'])
                if probs:
                    cw, cd, cl = probs['vitoria_casa'], probs['empate'], probs['vitoria_visitante']
                    res_txt = g['casa'] if cw>cd and cw>cl else (g['fora'] if cl>cw and cl>cd else "Empate")
                    st.info(f"Tendência: {res_txt} (Over 2.5: {probs['over_2_5']:.1%}) | xG: {xg[0]:.2f} - {xg[1]:.2f}")
                    
                    telegram_txt = f"🔥 <b>{LIGA_KEY}</b>: {g['casa']} x {g['fora']}\n"
                    c_res = st.columns(3)
                    idx = 0
                    has_val = False
                    
                    for k, odd_user in in_odds.items():
                        if not odd_user: continue
                        p_bot = probs[k]
                        val = (p_bot - (1/odd_user)) * 100
                        is_green = val > 5 and p_bot > min_prob
                        
                        # --- CÁLCULO DE STAKE KELLY ---
                        stake_reais, stake_pct = 0.0, 0.0
                        if is_green:
                            stake_reais, stake_pct = calc_kelly(p_bot, odd_user, KELLY_FRACAO, BANCA_USUARIO)
                        # -----------------------------
                        
                        if is_green or detalhado:
                            cor = "normal" if is_green else "inverse"
                            
                            # Label mais rico com a Stake
                            lbl_val = f"{p_bot:.1%}"
                            if stake_reais > 0:
                                lbl_val += f" (R${stake_reais:.0f})"
                                
                            c_res[idx%3].metric(f"{MERCADOS_INFO[k]}", lbl_val, f"{val:.1f}% EV", delta_color=cor)
                            idx+=1
                        
                        if is_green:
                            has_val = True
                            telegram_txt += f"✅ {MERCADOS_INFO[k]} @ {odd_user:.2f} (Prob: {p_bot:.0%})\n"
                            if stake_reais > 0:
                                telegram_txt += f"   💰 Stake: R$ {stake_reais:.2f} ({stake_pct:.1f}%)\n"
                            
                            salvar_db(db, g['dt_iso'], LIGA_KEY, f"{g['casa']} x {g['fora']}", MERCADOS_INFO[k], odd_user, p_bot*100, val, stake_reais)

                    if has_val:
                        st.success(f"Oportunidades encontradas! Baseado na banca de R$ {BANCA_USUARIO:.2f}")
                        if config.TELEGRAM_TOKEN: requests.get(f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage", params={'chat_id': config.TELEGRAM_CHAT_ID, 'text': telegram_txt, 'parse_mode': 'HTML'})
                    elif not detalhado: st.warning("Sem valor claro.")
                else: st.error("Erro no cálculo.")

with t_hist:
    if db:
        df_h, gre, red = load_db(db)
        c1, c2, c3 = st.columns(3)
        c1.metric("Greens", gre)
        c2.metric("Reds", red)
        if (gre+red)>0: c3.metric("Assert.", f"{(gre/(gre+red)*100):.1f}%")
        
        with st.expander("Atualizar"):
            pend = df_h[df_h['Status']=='Aguardando ⏳']
            sel = st.selectbox("Pendente:", [f"{i}: {r['Jogo']} - {r['Mercado']}" for i, r in pend.iterrows()])
            if sel:
                idx = int(sel.split(':')[0])
                if st.button("Green ✅"): 
                    db.update_cell(idx+2, 8, "Green ✅")
                    st.cache_data.clear(); st.rerun()
                if st.button("Red ❌"): 
                    db.update_cell(idx+2, 8, "Red ❌")
                    st.cache_data.clear(); st.rerun()
        if not df_h.empty: st.dataframe(df_h.iloc[::-1], hide_index=True, use_container_width=True)

with t_times:
    st.header("🔎 Raio-X: Matemática vs Realidade")
    c_rank1, c_rank2 = st.columns(2)
    
    with c_rank1:
        st.subheader("🤖 Ranking Robô (DC)")
        st.caption("Qualidade 'Teórica' (Ataque - Defesa)")
        if dc_data:
            lst = [{'Time':t, 'Força':v['ataque']-v['defesa']} for t,v in dc_data['forcas'].items()]
            df_t = pd.DataFrame(lst).sort_values('Força', ascending=False)
            st.dataframe(df_t, hide_index=True, use_container_width=True, height=500)
        else: st.warning("Ranking DC indisponível.")

    with c_rank2:
        st.subheader("🏆 Tabela Oficial")
        st.caption("Classificação Real (Pontos)")
        df_real = get_standings(LIGA_ID, config.TEMPORADA_PARA_ANALISAR)
        if df_real is not None:
            st.dataframe(df_real, hide_index=True, use_container_width=True, height=500)
        else: st.info("Não foi possível buscar a tabela.")
    
    st.divider()
    if dc_data:
        st.subheader("🔮 Simulador")
        tm = st.selectbox("Simular xG do Time:", df_t['Time'] if dc_data else [])
        if tm:
            f = dc_data['forcas'][tm]
            avg_a, avg_d = df_t['Força'].mean(), df_t['Força'].mean() 
            avg_a = pd.DataFrame([v['ataque'] for v in dc_data['forcas'].values()]).mean()[0]
            avg_d = pd.DataFrame([v['defesa'] for v in dc_data['forcas'].values()]).mean()[0]
            
            xg_c = np.exp(f['ataque'] + avg_d + dc_data['vantagem_casa'])
            xg_f = np.exp(avg_a + f['defesa'])
            
            sc1, sc2 = st.columns(2)
            sc1.metric(f"{tm} em CASA", f"{xg_c:.2f} xG", help="vs Time Médio")
            sc2.metric(f"{tm} como VISITANTE", f"{xg_f:.2f} xG", help="vs Time Médio")
