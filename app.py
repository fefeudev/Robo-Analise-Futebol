# app.py - Robô de Valor (v10.1 - Multigols + Lista de Compras + Backtest)
import streamlit as st
import requests, pandas as pd, numpy as np, scipy.stats as stats
import config, time, json, pytz, gspread
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURAÇÕES ---
FUSO = pytz.timezone('America/Manaus')
st.set_page_config(page_title="Robô de Valor", page_icon="🤖", layout="wide")

# CSS Otimizado
st.markdown("""<style>.stApp{background-color:#0A0A1A}[data-testid="stSidebar"]{background-color:#0F1116;border-right:1px solid #2a2a3a}h1,h2{color:#FAFAFA}h3{color:#4A90E2}[data-testid="stMetric"]{background-color:#1F202B;border:1px solid #333344;border-radius:10px}[data-testid="stButton"]>button{background-color:#4A90E2;color:#FFF;border:none}[data-testid="stExpander"]>summary{background-color:#1F202B;border:1px solid #333344}a[href]{text-decoration:none;color:white;}</style>""", unsafe_allow_html=True)

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

def calc_kelly(prob_ganhar, odd_decimal, fracao_kelly, banca_total):
    if odd_decimal <= 1 or prob_ganhar <= 0: return 0.0, 0.0
    b = odd_decimal - 1
    p = prob_ganhar
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0: return 0.0, 0.0
    stake_pct = f_star * fracao_kelly
    return banca_total * stake_pct, stake_pct * 100

# --- LÓGICA DE DADOS & FORMA ---
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
def get_historical_data(liga, temp):
    data = req_api(f"competitions/{liga}/matches", {"season": str(temp), "status": "FINISHED"})
    if not data or not data.get("matches"): return None, None
    matches = [{'data_jogo': m['utcDate'][:10], 'TimeCasa': m['homeTeam']['name'], 'TimeVisitante': m['awayTeam']['name'], 'GolsCasa': int(m['score']['fullTime']['home']), 'GolsVisitante': int(m['score']['fullTime']['away'])} for m in data['matches'] if m['score']['fullTime']['home'] is not None]
    df = pd.DataFrame(matches).sort_values('data_jogo')
    df['data_jogo'] = pd.to_datetime(df['data_jogo'])
    return df, {'media_gols_casa': df['GolsCasa'].mean(), 'media_gols_visitante': df['GolsVisitante'].mean()}

def get_form_str(team, df):
    if df is None or df.empty: return ""
    matches = df[(df['TimeCasa'] == team) | (df['TimeVisitante'] == team)].sort_values('data_jogo').tail(5)
    if matches.empty: return "(?)"
    res = ""
    for _, m in matches.iterrows():
        if m['TimeCasa'] == team:
            if m['GolsCasa'] > m['GolsVisitante']: res += "✅"
            elif m['GolsCasa'] == m['GolsVisitante']: res += "➖"
            else: res += "❌"
        else:
            if m['GolsVisitante'] > m['GolsCasa']: res += "✅"
            elif m['GolsVisitante'] == m['GolsCasa']: res += "➖"
            else: res += "❌"
    return f"({res})"

# --- CÁLCULO DE PROBABILIDADES (COM MULTIGOLS) ---
def calc_probs(l_casa, m_visit, rho=0.0):
    probs = np.zeros((7, 7))
    max_prob = 0
    placar_provavel = (0, 0)
    
    # Acumuladores de Faixa
    prob_0_1 = 0
    prob_2_3 = 0
    prob_4_plus = 0
    
    for i in range(7):
        for j in range(7):
            tau = 1.0
            if i==0 and j==0: tau = 1 - (l_casa*m_visit*rho)
            elif i==1 and j==0: tau = 1 + (l_casa*rho)
            elif i==0 and j==1: tau = 1 + (m_visit*rho)
            elif i==1 and j==1: tau = 1 - rho
            
            p = stats.poisson.pmf(i, l_casa) * stats.poisson.pmf(j, m_visit) * tau
            probs[i, j] = p
            
            # Placar mais provável
            if p > max_prob:
                max_prob = p
                placar_provavel = (i, j)
            
            # Faixas de Gols
            total_gols = i + j
            if total_gols <= 1: prob_0_1 += p
            elif total_gols <= 3: prob_2_3 += p
            else: prob_4_plus += p
                
    total = np.sum(probs)
    if total == 0: return None
    
    home = np.sum(np.tril(probs, -1))
    draw = np.sum(np.diag(probs))
    away = np.sum(np.triu(probs, 1))
    over = 0
    btts = 0
    for i in range(7):
        for j in range(7):
            if i+j > 2.5: over += probs[i,j]
            if i>0 and j>0: btts += probs[i,j]
            
    return {
        'vitoria_casa': home/total, 'empate': draw/total, 'vitoria_visitante': away/total,
        'over_2_5': over/total, 'btts_sim': btts/total,
        'chance_dupla_1X': (home+draw)/total, 'chance_dupla_X2': (draw+away)/total, 'chance_dupla_12': (home+away)/total,
        'placar_exato': placar_provavel,
        'faixa_0_1': prob_0_1/total,
        'faixa_2_3': prob_2_3/total,
        'faixa_4_plus': prob_4_plus/total
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

def check_result(probs, score_home, score_away, min_prob):
    results = []
    is_green_game = False
    winner = "Casa" if score_home > score_away else ("Fora" if score_away > score_home else "Empate")
    if probs['vitoria_casa'] > min_prob:
        status = "✅" if winner == "Casa" else "❌"
        results.append(f"{status} Casa")
        if status == "✅": is_green_game = True
    elif probs['vitoria_visitante'] > min_prob:
        status = "✅" if winner == "Fora" else "❌"
        results.append(f"{status} Fora")
        if status == "✅": is_green_game = True
    total_gols = score_home + score_away
    if probs['over_2_5'] > min_prob:
        status = "✅" if total_gols > 2.5 else "❌"
        results.append(f"{status} Over")
        if status == "✅": is_green_game = True
    return results, is_green_game

# --- UI & FLUXO ---
db = connect_db()
with st.sidebar:
    st.title("🤖 Robô v10.1")
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
    MODO_BACKTEST = st.toggle("🔙 Modo Backtesting (Passado)", value=False)
    if MODO_BACKTEST: st.warning("⚠️ Backtest ATIVO.")
    st.divider()
    st.header("💰 Gestão de Banca")
    BANCA_USUARIO = st.number_input("Banca Total (R$):", value=100.0, step=50.0)
    KELLY_FRACAO = st.slider("Fator Kelly:", 0.01, 0.50, 0.10, 0.01)
    st.divider()
    min_prob = st.slider("Prob. Mínima %", 0, 100, 60, 5) / 100.0
    detalhado = st.toggle("Modo Detalhado")

# Carga
dc_data = load_dc(LIGA_ID)
df_history, avg_history = get_historical_data(LIGA_ID, config.TEMPORADA_PARA_ANALISAR)
MODE = "DIXON_COLES" if dc_data else "FALHA"
if not dc_data and df_history is not None: MODE = "POISSON_RECENTE"

# Abas
t_jogos, t_hist, t_times = st.tabs(["Jogos", "Histórico", "Times"])

with t_jogos:
    st.subheader(f"{EMOJIS.get(LIGA_ID,'')} {LIGA_KEY} - {MODE}")
    
    @st.cache_data(ttl=300)
    def get_matches(lid, dtarget, is_backtest):
        d1 = dtarget.strftime('%Y-%m-%d')
        d2 = (dtarget + timedelta(days=1)).strftime('%Y-%m-%d')
        status_req = "FINISHED" if is_backtest else "SCHEDULED"
        res = req_api(f"competitions/{lid}/matches", {"dateFrom": d1, "dateTo": d2, "status": status_req})
        if not res or 'matches' not in res: return []
        final = []
        for m in res['matches']:
            dt_loc = pytz.utc.localize(datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ")).astimezone(FUSO)
            if dt_loc.date() == dtarget:
                item = {
                    'hora': dt_loc.strftime('%H:%M'), 
                    'casa': m['homeTeam']['name'], 
                    'fora': m['awayTeam']['name'], 
                    'dt_iso': dt_loc.strftime('%Y-%m-%d')
                }
                if is_backtest and m['score']['fullTime']['home'] is not None:
                    item['placar_casa'] = m['score']['fullTime']['home']
                    item['placar_fora'] = m['score']['fullTime']['away']
                final.append(item)
        return final

    matches = []
    if MODE != "FALHA": matches = get_matches(LIGA_ID, st.session_state.dt_sel, MODO_BACKTEST)
    
    if not matches:
        st.info(f"Sem jogos {'TERMINADOS' if MODO_BACKTEST else 'AGENDADOS'} para {st.session_state.dt_sel.strftime('%d/%m/%Y')}.")
    
    # RESUMO BACKTEST
    if MODO_BACKTEST and matches:
        total_greens, total_reds, total_entradas = 0, 0, 0
        for m in matches:
            if 'placar_casa' in m:
                p, _ = unified_predict(MODE, dc_data, df_history, avg_history, m['casa'], m['fora'], m['dt_iso'])
                if p:
                    res_check, is_green = check_result(p, m['placar_casa'], m['placar_fora'], min_prob)
                    if res_check:
                        total_entradas += 1
                        if is_green: total_greens += 1
                        else: total_reds += 1
        if total_entradas > 0:
            assertiv = (total_greens / total_entradas) * 100
            st.success(f"📊 **Simulação:** {total_greens} Greens ✅ | {total_reds} Reds ❌ | Assertividade: **{assertiv:.1f}%**")
    
    # LISTA DE COMPRAS
    if matches and not MODO_BACKTEST:
        with st.expander("🛒 Lista de Compras (Odd Justa)", expanded=True):
            radar_results = []
            telegram_radar_txt = f"🛒 **Lista de Compras - {LIGA_KEY}**\n\n"
            with st.spinner("Gerando lista..."):
                for m in matches:
                    p, x = unified_predict(MODE, dc_data, df_history, avg_history, m['casa'], m['fora'], m['dt_iso'])
                    if p:
                        if p['vitoria_casa'] > min_prob:
                            oj = 1/p['vitoria_casa']
                            radar_results.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': 'Casa', 'Confiança': p['vitoria_casa'], 'Odd Justa': f"@{oj:.2f}"})
                            telegram_radar_txt += f"⚽ {m['casa']} (Casa) | Justa: @{oj:.2f}\n"
                        if p['vitoria_visitante'] > min_prob:
                            oj = 1/p['vitoria_visitante']
                            radar_results.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': 'Fora', 'Confiança': p['vitoria_visitante'], 'Odd Justa': f"@{oj:.2f}"})
                            telegram_radar_txt += f"⚽ {m['fora']} (Fora) | Justa: @{oj:.2f}\n"
                        if p['over_2_5'] > min_prob:
                            oj = 1/p['over_2_5']
                            radar_results.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': 'Over 2.5', 'Confiança': p['over_2_5'], 'Odd Justa': f"@{oj:.2f}"})
                            telegram_radar_txt += f"⚽ Over 2.5 {m['casa']} | Justa: @{oj:.2f}\n"
                        if p['btts_sim'] > min_prob:
                            oj = 1/p['btts_sim']
                            radar_results.append({'Jogo': f"{m['casa']} x {m['fora']}", 'Aposta': 'BTTS', 'Confiança': p['btts_sim'], 'Odd Justa': f"@{oj:.2f}"})

            if radar_results:
                df_radar = pd.DataFrame(radar_results).sort_values('Confiança', ascending=False)
                st.dataframe(df_radar, hide_index=True, use_container_width=True, column_config={"Confiança": st.column_config.ProgressColumn("Confiança", format="%.0f%%", min_value=0, max_value=1)})
                if st.button("✈️ Enviar Telegram"):
                    if config.TELEGRAM_TOKEN: requests.get(f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage", params={'chat_id': config.TELEGRAM_CHAT_ID, 'text': telegram_radar_txt, 'parse_mode': 'Markdown'}); st.success("Enviado!")
            else: st.caption("Nenhuma aposta encontrada com os filtros.")

    # LISTA JOGOS
    if 'sel_game' not in st.session_state:
        for i, m in enumerate(matches):
            p, xg = unified_predict(MODE, dc_data, df_history, avg_history, m['casa'], m['fora'], m['dt_iso'])
            form_casa = get_form_str(m['casa'], df_history)
            form_fora = get_form_str(m['fora'], df_history)
            c1, c2 = st.columns([3, 1])
            label = f"⚽ {m['hora']} | {m['casa']} {form_casa} x {form_fora} {m['fora']}"
            if MODO_BACKTEST and 'placar_casa' in m: label += f" ({m['placar_casa']} - {m['placar_fora']})"
            if c1.button(label, key=f"b{i}", use_container_width=True):
                if not MODO_BACKTEST: st.session_state.sel_game = m; st.rerun()
            if MODO_BACKTEST and p and 'placar_casa' in m:
                res_check, _ = check_result(p, m['placar_casa'], m['placar_fora'], min_prob)
                if res_check: c2.success(" | ".join(res_check))
                else: c2.caption("Sem entrada")
            else: c2.metric("xG", f"{xg[0]:.2f}-{xg[1]:.2f}" if xg else "-")

    else:
        if MODO_BACKTEST: del st.session_state.sel_game; st.rerun()
        g = st.session_state.sel_game
        if st.button("⬅️ Voltar"): del st.session_state.sel_game; st.rerun()
        
        st.markdown(f"### {g['casa']} vs {g['fora']}")
        st.link_button("Apostar na Bet365 🟢", "https://www.bet365.com/#/AS/B1/", use_container_width=True)
        
        with st.form("f1"):
            st.write("Odds:")
            oc = st.columns(4)
            in_odds = {}
            fields = [('vitoria_casa','Casa'), ('empate','X'), ('vitoria_visitante','Fora'), ('over_2_5','Over 2.5'), ('btts_sim','BTTS'), ('chance_dupla_1X','1X'), ('chance_dupla_X2','X2'), ('chance_dupla_12','12')]
            for i, (k, l) in enumerate(fields): in_odds[k] = oc[i%4].number_input(l, min_value=1.0, step=0.01, format="%.2f")
            
            if st.form_submit_button("Analisar"):
                probs, xg = unified_predict(MODE, dc_data, df_history, avg_history, g['casa'], g['fora'], g['dt_iso'])
                if probs:
                    cw, cd, cl = probs['vitoria_casa'], probs['empate'], probs['vitoria_visitante']
                    
                    # --- MOSTRAR MULTIGOLS ---
                    c_mg1, c_mg2, c_mg3 = st.columns(3)
                    c_mg1.metric("0-1 Gol", f"{probs['faixa_0_1']:.1%}")
                    c_mg2.metric("2-3 Gols", f"{probs['faixa_2_3']:.1%}")
                    c_mg3.metric("4+ Gols", f"{probs['faixa_4_plus']:.1%}")
                    # -------------------------
                    
                    st.info(f"🔮 Placar Provável: **{probs['placar_exato'][0]}x{probs['placar_exato'][1]}**")
                    telegram_txt = f"🔥 <b>{LIGA_KEY}</b>: {g['casa']} x {g['fora']}\n"
                    c_res = st.columns(3)
                    idx = 0
                    has_val = False
                    for k, odd_user in in_odds.items():
                        if not odd_user: continue
                        p_bot = probs[k]
                        val = (p_bot - (1/odd_user)) * 100
                        is_green = val > 5 and p_bot > min_prob
                        stake_reais, stake_pct = 0.0, 0.0
                        if is_green: stake_reais, stake_pct = calc_kelly(p_bot, odd_user, KELLY_FRACAO, BANCA_USUARIO)
                        
                        if is_green or detalhado:
                            cor = "normal" if is_green else "inverse"
                            lbl = f"{p_bot:.1%}" + (f" (R${stake_reais:.0f})" if stake_reais > 0 else "")
                            c_res[idx%3].metric(f"{MERCADOS_INFO[k]}", lbl, f"{val:.1f}% EV", delta_color=cor)
                            idx+=1
                        
                        if is_green:
                            has_val = True
                            telegram_txt += f"✅ {MERCADOS_INFO[k]} @ {odd_user:.2f} (Prob: {p_bot:.0%})\n"
                            if stake_reais > 0: telegram_txt += f"   💰 Stake: R$ {stake_reais:.2f} ({stake_pct:.1f}%)\n"
                            salvar_db(db, g['dt_iso'], LIGA_KEY, f"{g['casa']} x {g['fora']}", MERCADOS_INFO[k], odd_user, p_bot*100, val, stake_reais)

                    if has_val:
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
                if st.button("Green ✅"): db.update_cell(idx+2, 8, "Green ✅"); st.cache_data.clear(); st.rerun()
                if st.button("Red ❌"): db.update_cell(idx+2, 8, "Red ❌"); st.cache_data.clear(); st.rerun()
        if not df_h.empty: st.dataframe(df_h.iloc[::-1], hide_index=True, use_container_width=True)

with t_times:
    st.header("🔎 Raio-X")
    c_rank1, c_rank2 = st.columns(2)
    with c_rank1:
        st.subheader("🤖 Ranking Robô")
        if dc_data:
            lst = [{'Time':t, 'Força':v['ataque']-v['defesa']} for t,v in dc_data['forcas'].items()]
            df_t = pd.DataFrame(lst).sort_values('Força', ascending=False)
            st.dataframe(df_t, hide_index=True, use_container_width=True, height=500)
        else: st.warning("Ranking DC indisponível.")
    with c_rank2:
        st.subheader("🏆 Oficial")
        df_real = get_standings(LIGA_ID, config.TEMPORADA_PARA_ANALISAR)
        if df_real is not None: st.dataframe(df_real, hide_index=True, use_container_width=True, height=500)
        else: st.info("Não foi possível buscar a tabela.")
    st.divider()
    if dc_data:
        tm = st.selectbox("Simular xG:", df_t['Time'] if dc_data else [])
        if tm:
            f = dc_data['forcas'][tm]
            avg_a = pd.DataFrame([v['ataque'] for v in dc_data['forcas'].values()]).mean()[0]
            avg_d = pd.DataFrame([v['defesa'] for v in dc_data['forcas'].values()]).mean()[0]
            xg_c = np.exp(f['ataque'] + avg_d + dc_data['vantagem_casa'])
            xg_f = np.exp(avg_a + f['defesa'])
            sc1, sc2 = st.columns(2)
            sc1.metric(f"{tm} em CASA", f"{xg_c:.2f} xG")
            sc2.metric(f"{tm} como VISITANTE", f"{xg_f:.2f} xG")
