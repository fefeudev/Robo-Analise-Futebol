# app.py
# O Robô de Análise (Versão 7.5 - Filtros no Histórico)
# UPGRADE: Adicionados filtros interativos (multiselect) na aba Histórico.

import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats 
import config 
import time
from datetime import datetime, timedelta
import json

# --- NOVOS IMPORTS DO BANCO DE DADOS ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# -------------------------------------

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Robô de Valor (BD)",
    page_icon="🤖", 
    layout="wide"
)

# --- FUNÇÕES GLOBAIS DE API (football-data.org) ---
@st.cache_data 
def criar_headers_api():
    return {"X-Auth-Token": config.API_KEY}

def fazer_requisicao_api(endpoint, params):
    url = config.API_BASE_URL + endpoint
    try:
        response = requests.get(url, headers=criar_headers_api(), params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro de API: {e}")
    return None

# --- FUNÇÕES DO BANCO DE DADOS (Google Sheets) ---

@st.cache_resource 
def conectar_ao_banco_de_dados():
    try:
        creds_dict = dict(st.secrets.google_creds)
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(st.secrets.GOOGLE_SHEET_URL).sheet1
        return sheet
    except Exception as e:
        st.error(f"Erro ao conectar ao Google Sheets: {e}") 
        st.error("Verifique seus 'Secrets' (google_creds e GOOGLE_SHEET_URL) e as permissões da API no Google Cloud.")
        return None

def salvar_analise_no_banco(sheet, data, liga, jogo, mercado, odd, prob_robo, valor):
    try:
        odd_num = float(odd)
        prob_robo_num = float(prob_robo) / 100.0
        valor_num = float(valor) / 100.0

        nova_linha = [
            data, liga, jogo, mercado, 
            odd_num, prob_robo_num, valor_num,
            "Aguardando ⏳" 
        ]
        sheet.append_row(nova_linha, value_input_option='USER_ENTERED')
        print(f"Análise salva no banco de dados: {jogo} - {mercado}")
    except Exception as e:
        st.error(f"Erro ao salvar no Google Sheets: {e}")

@st.cache_data(ttl=60) 
def carregar_historico_do_banco(_sheet):
    try:
        dados = _sheet.get_all_records() 
        df = pd.DataFrame(dados)
        
        if not df.empty:
            if 'Status' in df.columns:
                contagem_status = df['Status'].value_counts()
                greens = contagem_status.get('Green ✅', 0)
                reds = contagem_status.get('Red ❌', 0)
            else:
                greens, reds = 0, 0
        else:
            greens, reds = 0, 0
            
        return df, greens, reds
    except Exception as e:
        st.error(f"Erro ao carregar o histórico: {e}")
        return pd.DataFrame(), 0, 0

def atualizar_status_no_banco(sheet, row_index, novo_status):
    try:
        sheet.update_cell(row_index + 2, 8, novo_status) # Coluna H
        st.cache_data.clear() 
        st.rerun() 
    except Exception as e:
        st.error(f"Erro ao atualizar status: {e}")

# --- ETAPA 1 (CÉREBRO HÍBRIDO) ---
@st.cache_data
def carregar_cerebro_dixon_coles(id_liga):
    nome_arquivo = f"dc_params_{id_liga}.json"
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            dados_cerebro = json.load(f)
            return dados_cerebro
    except FileNotFoundError:
        return None 
    except Exception as e:
        st.error(f"Erro ao ler o arquivo de Cérebro: {e}")
        return None

def prever_jogo_dixon_coles(dados_cerebro, time_casa, time_visitante):
    try:
        forcas = dados_cerebro['forcas']
        vantagem_casa = dados_cerebro['vantagem_casa']
        rho = dados_cerebro.get('rho', 0.0)
        ataque_casa = forcas[time_casa]['ataque']
        defesa_casa = forcas[time_casa]['defesa']
        ataque_visitante = forcas[time_visitante]['ataque']
        defesa_visitante = forcas[time_visitante]['defesa']
        lambda_casa = np.exp(ataque_casa + defesa_visitante + vantagem_casa) 
        mu_visitante = np.exp(ataque_visitante + defesa_casa)
    except KeyError as e:
        st.warning(f"Aviso (DC): Time '{e.args[0]}' não foi encontrado no Cérebro (time novo?). Pulando.")
        return None, None
    def tau(gols_casa, gols_visitante, lambda_casa, mu_visit, rho):
        if rho == 0.0: return 1.0
        if gols_casa == 0 and gols_visitante == 0: return 1 - (lambda_casa * mu_visit * rho)
        elif gols_casa == 1 and gols_visitante == 0: return 1 + (lambda_casa * rho)
        elif gols_casa == 0 and gols_visitante == 1: return 1 + (mu_visit * rho)
        elif gols_casa == 1 and gols_visitante == 1: return 1 - rho
        else: return 1.0
    prob_vitoria_casa, prob_empate, prob_vitoria_visitante = 0.0, 0.0, 0.0
    prob_over_2_5, prob_btts_sim = 0.0, 0.0
    soma_total_probs = 0.0
    for i in range(config.MAX_GOLS_CALCULO + 1): 
        for j in range(config.MAX_GOLS_CALCULO + 1):
            prob_placar = (stats.poisson.pmf(i, lambda_casa) * stats.poisson.pmf(j, mu_visitante) * tau(i, j, lambda_casa, mu_visitante, rho))
            soma_total_probs += prob_placar
            if i > j: prob_vitoria_casa += prob_placar
            elif i == j: prob_empate += prob_placar
            elif j > i: prob_vitoria_visitante += prob_placar
            if (i + j) > 2.5: prob_over_2_5 += prob_placar
            if (i > 0 and j > 0): prob_btts_sim += prob_placar
    if soma_total_probs == 0: return None, None
    prob_dc_1x = prob_vitoria_casa + prob_empate
    prob_dc_x2 = prob_empate + prob_vitoria_visitante
    prob_dc_12 = prob_vitoria_casa + prob_vitoria_visitante
    probabilidades_mercado = {
        'vitoria_casa': prob_vitoria_casa / soma_total_probs, 'empate': prob_empate / soma_total_probs,
        'vitoria_visitante': prob_vitoria_visitante / soma_total_probs, 'over_2_5': prob_over_2_5 / soma_total_probs,
        'btts_sim': prob_btts_sim / soma_total_probs, 'chance_dupla_1X': prob_dc_1x / soma_total_probs,
        'chance_dupla_X2': prob_dc_x2 / soma_total_probs, 'chance_dupla_12': prob_dc_12 / soma_total_probs,
    }
    return (probabilidades_mercado, (lambda_casa, mu_visitante))

@st.cache_data 
def carregar_e_treinar_cerebro_poisson(id_liga, temporada):
    endpoint = f"competitions/{id_liga}/matches"
    params = {"season": str(temporada), "status": "FINISHED"}
    dados = fazer_requisicao_api(endpoint, params)
    if not dados or "matches" not in dados or not dados["matches"]:
        st.error(f"Erro (Poisson): A API não retornou 'matches' para o histórico da liga {id_liga}.")
        return None, None
    lista_jogos = []
    for match_info in dados['matches']:
        if match_info['score']['fullTime']['home'] is not None:
            jogo = {
                'data_jogo': match_info['utcDate'].split('T')[0],
                'TimeCasa': match_info['homeTeam']['name'],
                'TimeVisitante': match_info['awayTeam']['name'],
                'GolsCasa': int(match_info['score']['fullTime']['home']),
                'GolsVisitante': int(match_info['score']['fullTime']['away'])
            }
            lista_jogos.append(jogo)
    df_liga = pd.DataFrame(lista_jogos)
    df_liga['data_jogo'] = pd.to_datetime(df_liga['data_jogo'])
    df_liga = df_liga.sort_values(by='data_jogo')
    if len(df_liga) < 10:
        st.error(f"Erro de Treinamento (Poisson): A liga {id_liga} tem menos de 10 jogos no histórico.")
        return None, None
    medias_liga = {
        'media_gols_casa': df_liga['GolsCasa'].mean(),
        'media_gols_visitante': df_liga['GolsVisitante'].mean()
    }
    return df_liga, medias_liga

def calcular_forcas_recente_poisson(df_historico, time_casa, time_visitante, data_do_jogo, num_jogos=6):
    data_do_jogo_dt = pd.to_datetime(data_do_jogo)
    df_passado = df_historico[df_historico['data_jogo'] < data_do_jogo_dt]
    jogos_casa_recente = df_passado[df_passado['TimeCasa'] == time_casa].tail(num_jogos)
    jogos_visitante_recente = df_passado[df_passado['TimeVisitante'] == time_visitante].tail(num_jogos)
    if len(jogos_casa_recente) < 1 or len(jogos_visitante_recente) < 1:
        st.warning(f"Aviso (Poisson): Times novos. {time_casa} ou {time_visitante} têm menos de 1 jogo no histórico. Pulando.")
        return None
    ataque_casa_media = jogos_casa_recente['GolsCasa'].mean()
    defesa_casa_media = jogos_casa_recente['GolsVisitante'].mean()
    ataque_visitante_media = jogos_visitante_recente['GolsVisitante'].mean()
    defesa_visitante_media = jogos_visitante_recente['GolsCasa'].mean()
    forcas_times = {
        time_casa: {'ataque_casa_media': ataque_casa_media, 'defesa_casa_media': defesa_casa_media},
        time_visitante: {'ataque_visitante_media': ataque_visitante_media, 'defesa_visitante_media': defesa_visitante_media}
    }
    return forcas_times

def prever_jogo_poisson(forcas_times, medias_liga, time_casa, time_visitante):
    forca_ataque_casa = forcas_times[time_casa]['ataque_casa_media'] / medias_liga['media_gols_casa']
    forca_defesa_casa = forcas_times[time_casa]['defesa_casa_media'] / medias_liga['media_gols_visitante']
    forca_ataque_visitante = forcas_times[time_visitante]['ataque_visitante_media'] / medias_liga['media_gols_visitante']
    forca_defesa_visitante = forcas_times[time_visitante]['defesa_visitante_media'] / medias_liga['media_gols_casa']
    xg_casa = (forca_ataque_casa * forca_defesa_visitante * medias_liga['media_gols_casa'])
    xg_visitante = (forca_ataque_visitante * forca_defesa_casa * medias_liga['media_gols_visitante'])
    prob_gols_casa = [stats.poisson.pmf(i, xg_casa) for i in range(config.MAX_GOLS_CALCULO + 1)]
    prob_gols_visitante = [stats.poisson.pmf(i, xg_visitante) for i in range(config.MAX_GOLS_CALCULO + 1)]
    matriz_placar = np.outer(prob_gols_casa, prob_gols_visitante)
    prob_vitoria_casa = np.sum(np.tril(matriz_placar, -1))
    prob_empate = np.sum(np.diag(matriz_placar))
    prob_vitoria_visitante = np.sum(np.triu(matriz_placar, 1))
    prob_over_2_5 = 0
    prob_btts_sim = 0
    for i in range(config.MAX_GOLS_CALCULO + 1): 
        for j in range(config.MAX_GOLS_CALCULO + 1):
            prob_placar = matriz_placar[i, j]
            if (i + j) > 2.5: prob_over_2_5 += prob_placar
            if i > 0 and j > 0: prob_btts_sim += prob_placar
    soma_total_probs = prob_vitoria_casa + prob_empate + prob_vitoria_visitante
    if soma_total_probs == 0: return None, None
    prob_dc_1x = prob_vitoria_casa + prob_empate
    prob_dc_x2 = prob_empate + prob_vitoria_visitante
    prob_dc_12 = prob_vitoria_casa + prob_vitoria_visitante
    probabilidades_mercado = {
        'vitoria_casa': prob_vitoria_casa / soma_total_probs, 'empate': prob_empate / soma_total_probs,
        'vitoria_visitante': prob_vitoria_visitante / soma_total_probs, 'over_2_5': prob_over_2_5 / soma_total_probs,
        'btts_sim': prob_btts_sim / soma_total_probs, 'chance_dupla_1X': prob_dc_1x / soma_total_probs,
        'chance_dupla_X2': prob_dc_x2 / soma_total_probs, 'chance_dupla_12': prob_dc_12 / soma_total_probs,
    }
    return (probabilidades_mercado, (xg_casa, xg_visitante))

def encontrar_valor(probabilidades_calculadas, odds_casa, filtro_prob_minima=0.60, filtro_valor_minimo=0.05):
    oportunidades = {}
    for mercado, odd in odds_casa.items():
        if odd is None or odd == 0.0 or mercado not in probabilidades_calculadas:
            continue
        prob_casa_aposta = 1 / odd
        prob_robo = probabilidades_calculadas[mercado] 
        valor = prob_robo - prob_casa_aposta          
        temValor = valor > filtro_valor_minimo
        eProvavel = prob_robo > filtro_prob_minima
        if temValor and eProvavel:
            oportunidades[mercado] = {
                'odd_casa': odd,
                'prob_casa_aposta': prob_casa_aposta * 100,
                'prob_robo': prob_robo * 100,
                'valor_encontrado': valor * 100
            }
    return oportunidades

@st.cache_data
def buscar_jogos_por_data(id_liga, data_str):
    endpoint = f"competitions/{id_liga}/matches"
    params = {"dateFrom": data_str, "dateTo": data_str, "status": "SCHEDULED"}
    dados = fazer_requisicao_api(endpoint, params)
    if not dados or "matches" not in dados or not dados["matches"]:
        return []
    jogos_do_dia = []
    for match_info in dados['matches']:
        jogo = {
            'data_jogo': match_info['utcDate'].split('T')[0],
            'time_casa': match_info['homeTeam']['name'],
            'time_visitante': match_info['awayTeam']['name']
        }
        jogos_do_dia.append(jogo)
    return jogos_do_dia

def enviar_mensagem_telegram(mensagem):
    if not config.TELEGRAM_TOKEN or config.TELEGRAM_TOKEN == "SEU_TOKEN_DO_BOTFATHER_AQUI":
        st.warning("Token do Telegram não configurado. Pulando envio.")
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    params = {'chat_id': config.TELEGRAM_CHAT_ID, 'text': mensagem, 'parse_mode': 'HTML'}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        resultado = response.json()
        if resultado.get('ok'):
            st.success("Mensagem enviada ao Telegram com sucesso!")
        else:
            st.error(f"Erro ao enviar: {resultado.get('description')}")
    except Exception as e:
        st.error(f"Erro fatal no envio do Telegram: {e}")

# --- A INTERFACE GRÁFICA (Função Principal ATUALIZADA) ---

LIGAS_DISPONIVEIS = {
    "Brasileirão": "BSA",
    "Premier League (ING)": "PL",
    "La Liga (ESP)": "PD",
    "Serie A (ITA)": "SA",
    "Bundesliga (ALE)": "BL1",
    "Ligue 1 (FRA)": "FL1",
    "Eredivisie (HOL)": "DED",
    "Championship (ING 2)": "ELC",
    "Primeira Liga (POR)": "PPL",
    "European Championship": "EC"
}

LIGAS_EMOJI = {
    "BSA": "🇧🇷", "PL": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "CL": "🇪🇺", "PD": "🇪🇸", "SA": "🇮🇹",
    "BL1": "🇩🇪", "FL1": "🇫🇷", "DED": "🇳🇱", "ELC": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "PPL": "🇵🇹", "EC": "🇪🇺"
}

nomes_mercado = {
    'vitoria_casa': 'Casa', 'empate': 'Empate', 'vitoria_visitante': 'Fora',
    'over_2_5': 'Mais de 2.5 Gols', 'btts_sim': 'Ambas Marcam (Sim)',
    'chance_dupla_1X': 'Casa ou Empate (1X)', 'chance_dupla_X2': 'Empate ou Fora (X2)',
    'chance_dupla_12': 'Casa ou Fora (12)'
}

# --- 1. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("🤖 Robô de Valor")
    st.caption("v7.5 - Filtros no Histórico") # Versão atualizada
    
    liga_selecionada_nome = st.selectbox("1. Selecione a Liga:", LIGAS_DISPONIVEIS.keys())
    LIGA_ATUAL = LIGAS_DISPONIVEIS[liga_selecionada_nome]
    TEMPORADA_ATUAL = config.TEMPORADA_PARA_ANALISAR 
    
    st.header("3. Buscar Jogos")
    
    if 'data_selecionada' not in st.session_state:
        st.session_state.data_selecionada = datetime.now()
    def dia_anterior():
        st.session_state.data_selecionada -= timedelta(days=1)
    def proximo_dia():
        st.session_state.data_selecionada += timedelta(days=1)
    def hoje():
        st.session_state.data_selecionada = datetime.now()

    col_data1, col_data2 = st.columns([1,1])
    with col_data1:
        st.button("< Ontem", on_click=dia_anterior, use_container_width=True)
    with col_data2:
        st.button("Amanhã >", on_click=proximo_dia, use_container_width=True)
    
    data_selecionada = st.date_input(
        "Ou selecione uma data:",
        value=st.session_state.data_selecionada,
        key='data_selecionada' 
    )
    st.button("Ir para Hoje", on_click=hoje, use_container_width=True)
            
    st.header("4. Filtros de Análise")
    filtro_prob_minima_percentual = st.slider(
        "Probabilidade Mínima (Chance de Green)", 
        min_value=0, max_value=100, value=60, step=5, format="%d%%"
    )
    filtro_prob_minima = filtro_prob_minima_percentual / 100.0 
    filtro_valor_minimo = 0.05
    
    st.header("5. Modo de Análise")
    modo_detalhado = st.toggle(
        "Mostrar Análise Completa", 
        value=False,
        help="Se LIGADO, mostra a probabilidade para todos os 8 mercados, mesmo que não tenham valor."
    )
    
    with st.expander("ℹ️ Sobre este Robô"):
        st.markdown("""
            Este robô usa um **Cérebro Híbrido** para encontrar valor:
            
            1.  **Dixon-Coles (Principal):** Um modelo estatístico avançado (arquivos `.json`) que calcula forças de ataque e defesa de longo prazo.
            2.  **Poisson Recente (Fallback):** Se o cérebro DC não existir, ele usa a forma recente (últimos 6 jogos) para calcular as probabilidades.
            
            Ele se conecta a uma planilha Google Sheets para salvar e rastrear o histórico.
        """)


# --- 2. PÁGINA PRINCIPAL (COM ABAS) ---
st.title("Painel de Análise de Valor (Híbrido) 💾")

# Conecta ao nosso "banco de dados"
db_sheet = conectar_ao_banco_de_dados()

# --- TREINA O CÉREBRO (MOVIDO PARA FORA DAS ABAS) ---
MODO_CEREBRO = "FALHA" 
dados_cerebro_dc = None
df_historico_poisson = None
medias_liga_poisson = None

with st.spinner(f"Tentando carregar Cérebro Dixon-Coles para {LIGA_ATUAL}..."):
    dados_cerebro_dc = carregar_cerebro_dixon_coles(LIGA_ATUAL)

if dados_cerebro_dc is not None:
    MODO_CEREBRO = "DIXON_COLES"
else:
    with st.spinner(f"Cérebro DC não encontrado. Treinando Cérebro Poisson para {LIGA_ATUAL}..."):
        df_historico_poisson, medias_liga_poisson = carregar_e_treinar_cerebro_poisson(LIGA_ATUAL, TEMPORADA_ATUAL)
    if df_historico_poisson is not None:
        MODO_CEREBRO = "POISSON_RECENTE"

# --- FIM DO BLOCO MOVIDO ---


# Cria as três abas principais
tab_analise, tab_historico, tab_times = st.tabs([
    "📊 Analisar Jogos", 
    "📈 Histórico de Assertividade",
    "🔎 Analisar Times"
])

# --- ABA 1: ANALISAR JOGOS ---
with tab_analise:
    
    emoji_liga_selecionada = LIGAS_EMOJI.get(LIGA_ATUAL, '🏳️')
    st.subheader(f"Liga Selecionada: {emoji_liga_selecionada} {liga_selecionada_nome}")
    
    if MODO_CEREBRO == "DIXON_COLES":
        st.success(f"Cérebro Avançado (Dixon-Coles) carregado!")
        st.caption(f"Treinado em: {dados_cerebro_dc['data_treinamento'].split('T')[0]}")
    elif MODO_CEREBRO == "POISSON_RECENTE":
        st.success(f"Cérebro Poisson treinado com {len(df_historico_poisson)} jogos.")
        st.info("Usando Cérebro de 'Forma Recente' (Poisson) como fallback.")
    else:
        st.error(f"Falha ao carregar dados da {LIGA_ATUAL}. O robô não pode analisar.")
    
    st.divider() # Linha Horizontal

    # --- LÓGICA DE "TELAS" (Drill-Down) ---
    if 'jogo_selecionado' not in st.session_state:
        st.header(f"Jogos para {data_selecionada.strftime('%d/%m/%Y')}")
        st.caption(f"Usando Cérebro: {MODO_CEREBRO} | Filtro de Probabilidade: > {filtro_prob_minima_percentual}%")

        if MODO_CEREBRO != "FALHA":
            data_str = data_selecionada.strftime('%Y-%m-%d')
            jogos_do_dia = buscar_jogos_por_data(LIGA_ATUAL, data_str)

            if not jogos_do_dia:
                st.info(f"Nenhum jogo agendado encontrado para a liga {LIGA_ATUAL} na data {data_str}.")
            else:
                st.info(f"Encontrados {len(jogos_do_dia)} jogos. Clique em um jogo para analisar:")
                
                ### MELHORIA 1 - INÍCIO (Mostrar xG na Lista de Jogos) ###
                
                for i, jogo in enumerate(jogos_do_dia):
                    
                    xg_tupla = None
                    try:
                        if MODO_CEREBRO == "DIXON_COLES":
                            resultado_previsao = prever_jogo_dixon_coles(
                                dados_cerebro_dc, jogo['time_casa'], jogo['time_visitante']
                            )
                            if resultado_previsao:
                                _, xg_tupla = resultado_previsao
                        
                        elif MODO_CEREBRO == "POISSON_RECENTE":
                            forcas_times = calcular_forcas_recente_poisson(
                                df_historico_poisson, jogo['time_casa'], jogo['time_visitante'], jogo['data_jogo']
                            )
                            if forcas_times:
                                resultado_previsao = prever_jogo_poisson(
                                    forcas_times, medias_liga_poisson,
                                    jogo['time_casa'], jogo['time_visitante'] 
                                )
                                if resultado_previsao:
                                    _, xg_tupla = resultado_previsao
                    except Exception as e:
                        # Falha silenciosa para não quebrar a lista
                        print(f"Erro ao calcular xG para a lista: {e}")

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        def selecionar_jogo(jogo_clicado=jogo, indice=i):
                            st.session_state.jogo_selecionado = jogo_clicado
                            st.session_state.jogo_indice = indice

                        st.button(
                            f"⚽ **{jogo['time_casa']} vs {jogo['time_visitante']}**", 
                            on_click=selecionar_jogo,
                            use_container_width=True,
                            key=f"btn_jogo_{i}"
                        )
                    
                    with col2:
                        if xg_tupla:
                            st.metric(label="xG Previsto", 
                                      value=f"{xg_tupla[0]:.2f} - {xg_tupla[1]:.2f}",
                                      delta=f"{MODO_CEREBRO.split('_')[0]}", 
                                      delta_color="off")
                        else:
                            st.button("Analisar", on_click=selecionar_jogo, key=f"btn_analisar_basic_{i}", use_container_width=True)
                ### MELHORIA 1 - FIM ###
    
    # Se 'jogo_selecionado' ESTÁ na memória, mostra a "Tela 2" (Análise)
    else:
        jogo = st.session_state.jogo_selecionado
        i = st.session_state.jogo_indice
        
        if st.button("⬅️ Voltar para a lista de jogos"):
            del st.session_state.jogo_selecionado
            del st.session_state.jogo_indice
            st.rerun() 

        with st.form(key=f"form_jogo_{i}"):
            st.header(f"Jogo: {jogo['time_casa']} vs {jogo['time_visitante']}")
            
            ### MELHORIA 1 (DESIGN) - INÍCIO (Formulário Consolidado) ###
            st.subheader("Inserir Odds do Jogo")
            col_form1, col_form2 = st.columns(2)

            with col_form1:
                st.markdown("**📊 Resultado (1x2)**")
                odd_casa = st.number_input(f"{jogo['time_casa']} (1)", min_value=1.0, value=None, format="%.2f", key=f"casa_{i}")
                odd_empate = st.number_input("Empate (X)", min_value=1.0, value=None, format="%.2f", key=f"empate_{i}")
                odd_visitante = st.number_input(f"{jogo['time_visitante']} (2)", min_value=1.0, value=None, format="%.2f", key=f"visit_{i}")
                
                st.markdown("**⚽ Gols**")
                odd_over = st.number_input("Mais de 2.5 Gols", min_value=1.0, value=None, format="%.2f", key=f"over_{i}")
                odd_btts = st.number_input("Ambas Marcam (Sim)", min_value=1.0, value=None, format="%.2f", key=f"btts_{i}")

            with col_form2:
                st.markdown("**🤝 Chance Dupla**")
                odd_1x = st.number_input("Casa/Empate (1X)", min_value=1.0, value=None, format="%.2f", key=f"dc1x_{i}")
                odd_x2 = st.number_input("Empate/Fora (X2)", min_value=1.0, value=None, format="%.2f", key=f"dcx2_{i}")
                odd_12 = st.number_input("Casa/Fora (12)", min_value=1.0, value=None, format="%.2f", key=f"dc12_{i}")
            
            st.divider()
            ### MELHORIA 1 (DESIGN) - FIM ###

            submitted = st.form_submit_button("Analisar este Jogo")

            if submitted:
                with st.spinner("Analisando..."):
                    odds_manuais = {
                        'vitoria_casa': odd_casa, 'empate': odd_empate,
                        'vitoria_visitante': odd_visitante, 'over_2_5': odd_over, 'btts_sim': odd_btts,
                        'chance_dupla_1X': odd_1x, 'chance_dupla_X2': odd_x2, 'chance_dupla_12': odd_12
                    }
                    
                    probs_robo = None
                    xg_tupla = None 
                    
                    if MODO_CEREBRO == "DIXON_COLES":
                        resultado_previsao = prever_jogo_dixon_coles(
                            dados_cerebro_dc, jogo['time_casa'], jogo['time_visitante']
                        )
                        if resultado_previsao:
                            probs_robo, xg_tupla = resultado_previsao
                    elif MODO_CEREBRO == "POISSON_RECENTE":
                        forcas_times = calcular_forcas_recente_poisson(
                            df_historico_poisson, jogo['time_casa'], jogo['time_visitante'], jogo['data_jogo']
                        )
                        if forcas_times:
                            resultado_previsao = prever_jogo_poisson(
                                forcas_times, medias_liga_poisson,
                                jogo['time_casa'], jogo['time_visitante'] 
                            )
                            if resultado_previsao:
                                probs_robo, xg_tupla = resultado_previsao
                    
                    if probs_robo:
                        
                        ### MELHORIA 9 - INÍCIO (Termômetro de Tendência) ###
                        st.subheader("🌡️ Termômetro do Cérebro")
                        col_t1, col_t2 = st.columns(2)
                
                        # Lógica 1x2
                        prob_casa = probs_robo['vitoria_casa']
                        prob_empate = probs_robo['empate']
                        prob_fora = probs_robo['vitoria_visitante']
                
                        with col_t1:
                            st.markdown("**Resultado (1x2)**")
                            if max(prob_casa, prob_empate, prob_fora) == prob_casa:
                                st.info(f"Tendência: **{jogo['time_casa']}** é o favorito ({prob_casa:.1%})")
                            elif max(prob_casa, prob_empate, prob_fora) == prob_empate:
                                st.info(f"Tendência: **Empate** é o resultado mais provável ({prob_empate:.1%})")
                            else:
                                st.info(f"Tendência: **{jogo['time_visitante']}** é o favorito ({prob_fora:.1%})")
                        
                        # Lógica Gols
                        prob_over = probs_robo['over_2_5']
                        with col_t2:
                            st.markdown("**Gols (Mais/Menos 2.5)**")
                            if prob_over > 0.55: # 55% de chance de Over
                                st.warning(f"Tendência: Jogo para **Mais de 2.5 Gols** ({prob_over:.1%})")
                            elif prob_over < 0.45: # 45% de chance de Over (ou 55% de Under)
                                st.success(f"Tendência: Jogo para **Menos de 2.5 Gols** (Over: {prob_over:.1%})")
                            else:
                                st.info("Tendência: Mercado de Gols indefinido.")
                                
                        st.divider()
                        ### MELHORIA 9 - FIM ###
                        
                        oportunidades = encontrar_valor(
                            probs_robo, odds_manuais, 
                            filtro_prob_minima, filtro_valor_minimo
                        )
                        
                        mensagem_telegram = ""
                        if oportunidades:
                            xg_casa_str = f"{xg_tupla[0]:.2f}" if xg_tupla else "N/A"
                            xg_vis_str = f"{xg_tupla[1]:.2f}" if xg_tupla else "N/A"
                            emoji_liga = LIGAS_EMOJI.get(LIGA_ATUAL, '🏳️')
                            mensagem_telegram = f"🔥 <b>Oportunidade ({MODO_CEREBRO})</b> 🔥\n\n"
                            mensagem_telegram += f"<b>Liga:</b> {emoji_liga} {liga_selecionada_nome}\n"
                            mensagem_telegram += f"<b>Jogo:</b> ⚽️ {jogo['time_casa']} vs {jogo['time_visitante']}\n\n"
                            mensagem_telegram += f"🧠 <b>Previsão do Cérebro (xG):</b>\n"
                            mensagem_telegram += f"   <code>xG Casa: {xg_casa_str}</code>\n"
                            mensagem_telegram += f"   <code>xG Visitante: {xg_vis_str}</code>\n"
                        
                        if modo_detalhado:
                            st.subheader("Análise Completa (Todos os Mercados)")
                            col_met1, col_met2, col_met3 = st.columns(3)
                            colunas_metricas = [col_met1, col_met2, col_met3]
                            idx_coluna = 0
                            for mercado, prob_robo_pct in probs_robo.items():
                                if mercado.endswith('_nao') or mercado == 'under_2_5':
                                    continue
                                odd_manual = odds_manuais.get(mercado)
                                mercado_limpo = nomes_mercado.get(mercado, mercado)
                                col_target = colunas_metricas[idx_coluna % 3] 
                                with col_target:
                                    if (mercado in oportunidades):
                                        dados = oportunidades[mercado]
                                        st.metric(label=f"✅ {mercado_limpo}", value=f"{dados['prob_robo']:.2f}%",
                                                  delta=f"+{dados['valor_encontrado']:.2f}% Valor", delta_color="normal")
                                        st.caption(f"Odd: {dados['odd_casa']:.2f} (Casa: {dados['prob_casa_aposta']:.1f}%)")
                                    elif odd_manual:
                                        prob_robo_real = prob_robo_pct * 100
                                        prob_casa = (1 / odd_manual * 100)
                                        valor = prob_robo_real - prob_casa
                                        st.metric(label=f"❌ {mercado_limpo}", value=f"{prob_robo_real:.2f}%",
                                                  delta=f"{valor:.2f}% Valor", delta_color="inverse")
                                        st.caption(f"Odd: {odd_manual:.2f} (Casa: {prob_casa:.1f}%)")
                                    else:
                                        st.metric(label=f"⚪️ {mercado_limpo}", value=f"{(prob_robo_pct * 100):.2f}%",
                                                  delta="Sem Odd Manual", delta_color="off")
                                idx_coluna += 1
                        else:
                            if oportunidades:
                                st.success("🔥 OPORTUNIDADES DE VALOR ENCONTRADAS!")
                                for mercado, dados in oportunidades.items():
                                    mercado_limpo = nomes_mercado.get(mercado, mercado)
                                    st.subheader(f"Mercado: {mercado_limpo}")
                                    col_met1, col_met2, col_met3 = st.columns(3)
                                    with col_met1:
                                        st.metric(label="Odd (Casa %)", value=f"{dados['odd_casa']:.2f}",
                                                  delta=f"{dados['prob_casa_aposta']:.1f}% da Casa", delta_color="off")
                                    with col_met2:
                                        st.metric(label="Probabilidade", value=f"{dados['prob_robo']:.2f}%")
                                    with col_met3:
                                        st.metric(label="Valor Encontrado", value=f"+{dados['valor_encontrado']:.2f}%")
                            else:
                                st.info(f"Nenhuma oportunidade de valor (com >{filtro_prob_minima_percentual}% de prob.) encontrada.")
                        
                        if oportunidades:
                            for mercado, dados in oportunidades.items():
                                mercado_limpo = nomes_mercado.get(mercado, mercado)
                                mensagem_telegram += "------------------------------\n"
                                mensagem_telegram += f"✅ <b>Mercado: {mercado_limpo}</b>\n"
                                mensagem_telegram += f"   <code>Odd: {dados['odd_casa']:.2f} (Casa: {dados['prob_casa_aposta']:.2f}%)</code>\n"
                                mensagem_telegram += f"   <code>Probabilidade: {dados['prob_robo']:.2f}%</code>\n"
                                mensagem_telegram += f"   <code>Valor: +{dados['valor_encontrado']:.2f}%</code>\n"
                                
                                if db_sheet is not None:
                                    salvar_analise_no_banco(
                                        sheet=db_sheet,
                                        data=data_selecionada.strftime('%Y-%m-%d'), 
                                        liga=liga_selecionada_nome,
                                        jogo=f"{jogo['time_casa']} vs {jogo['time_visitante']}",
                                        mercado=mercado_limpo,
                                        odd=dados['odd_casa'],
                                        prob_robo=dados['prob_robo'],
                                        valor=dados['valor_encontrado']
                                    )
                            
                            enviar_mensagem_telegram(mensagem_telegram)
                    else:
                        st.error("Não foi possível calcular as probabilidades do robô (Times novos ou erro no Cérebro).")

# --- ABA 2: HISTÓRICO DE ASSERTIVIDADE ---
with tab_historico:
    st.header("📈 Histórico de Assertividade")
    st.caption("Aqui fica o registro de todas as análises enviadas ao Telegram.")
    
    if db_sheet is None:
        st.error("Não foi possível conectar ao Google Sheets. Verifique seus 'Secrets'.")
    else:
        # 1. Carrega os dados da planilha
        df_historico_db, greens, reds = carregar_historico_do_banco(db_sheet)
        
        # 2. Mostra os Contadores (Métricas)
        st.subheader("Desempenho Geral")
        total_analises = greens + reds
        assertividade = (greens / total_analises * 100) if total_analises > 0 else 0
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Greens ✅", f"{greens}")
        col_m2.metric("Reds ❌", f"{reds}")
        col_m3.metric("Assertividade", f"{assertividade:.1f}%")
        
        st.divider() # Linha horizontal
        
        ### MELHORIA 5 - INÍCIO (Mover "Atualizar Status" para cima) ###
        with st.expander("✏️ Atualizar Status de Análises Pendentes"):
            if 'Status' in df_historico_db.columns:
                opcoes_para_atualizar_df = df_historico_db[df_historico_db['Status'] == 'Aguardando ⏳']
            else:
                opcoes_para_atualizar_df = pd.DataFrame(columns=df_historico_db.columns) # Cria um DF vazio

            opcoes_para_atualizar_lista = [
                f"{idx}: [{row['Liga']}] {row['Jogo']} - Mercado: {row['Mercado']}" 
                for idx, row in opcoes_para_atualizar_df.iterrows()
            ]
            
            if not opcoes_para_atualizar_lista:
                st.info("Nenhuma análise 'Aguardando' para atualizar.")
            else:
                analise_selecionada = st.selectbox(
                    "Selecione a análise para atualizar:",
                    opcoes_para_atualizar_lista,
                    key="select_analise_status"
                )
                
                col_b1, col_b2 = st.columns(2)
                
                if col_b1.button("Marcar como Green ✅", use_container_width=True, key="btn_green"):
                    if analise_selecionada: # Verifica se algo foi selecionado
                        indice_real_df = int(analise_selecionada.split(':')[0])
                        atualizar_status_no_banco(db_sheet, indice_real_df, "Green ✅")
                    
                if col_b2.button("Marcar como Red ❌", use_container_width=True, key="btn_red"):
                    if analise_selecionada: # Verifica se algo foi selecionado
                        indice_real_df = int(analise_selecionada.split(':')[0])
                        atualizar_status_no_banco(db_sheet, indice_real_df, "Red ❌")
        ### MELHORIA 5 - FIM ###
        
        # (Correção para planilha vazia)
        if df_historico_db.empty:
            st.info("Nenhuma análise foi salva no banco de dados ainda. Faça sua primeira análise!")
        else:
            ### MELHORIA 2 - INÍCIO (Gráficos de Desempenho Lado a Lado) ###
            st.subheader("Desempenho Detalhado")
            
            # Filtra apenas por Green/Red
            df_resultados = df_historico_db[df_historico_db['Status'].isin(['Green ✅', 'Red ❌'])]
            
            if df_resultados.empty:
                st.info("Nenhum Green ou Red registrado para gerar gráficos.")
            else:
                col_graf1, col_graf2 = st.columns(2)
                
                with col_graf1:
                    st.markdown("**Por Mercado**")
                    desempenho_mercado = df_resultados.groupby('Mercado')['Status'].value_counts().unstack(fill_value=0)
                    st.bar_chart(desempenho_mercado, color=["#008000", "#FF4B4B"]) # Verde, Vermelho

                with col_graf2:
                    st.markdown("**Por Liga**")
                    desempenho_liga = df_resultados.groupby('Liga')['Status'].value_counts().unstack(fill_value=0)
                    st.bar_chart(desempenho_liga, color=["#008000", "#FF4B4B"]) # Verde, Vermelho
            
            st.divider() # Linha horizontal
            ### MELHORIA 2 - FIM ###
            
            # 3. Mostra a tabela de dados
            
            ### MELHORIA 2 (DESIGN) / 4 (FUNC) - INÍCIO (Filtros Interativos) ###
            st.subheader("🔎 Filtrar Histórico")
            df_para_mostrar = df_historico_db.copy() # Copia o dataframe original

            # Garante que as colunas existem antes de tentar filtrar
            colunas_presentes = df_para_mostrar.columns
            
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                if 'Liga' in colunas_presentes:
                    ligas_selecionadas = st.multiselect(
                        "Filtrar por Liga:",
                        options=df_para_mostrar['Liga'].unique(),
                        default=[]
                    )
                    if ligas_selecionadas:
                        df_para_mostrar = df_para_mostrar[df_para_mostrar['Liga'].isin(ligas_selecionadas)]
                else:
                    st.caption("Coluna 'Liga' não encontrada.")
            
            with col_f2:
                if 'Mercado' in colunas_presentes:
                    mercados_selecionados = st.multiselect(
                        "Filtrar por Mercado:",
                        options=df_para_mostrar['Mercado'].unique(),
                        default=[]
                    )
                    if mercados_selecionados:
                        df_para_mostrar = df_para_mostrar[df_para_mostrar['Mercado'].isin(mercados_selecionados)]
                else:
                    st.caption("Coluna 'Mercado' não encontrada.")

            with col_f3:
                if 'Status' in colunas_presentes:
                    status_selecionados = st.multiselect(
                        "Filtrar por Status:",
                        options=df_para_mostrar['Status'].unique(),
                        default=[]
                    )
                    if status_selecionados:
                        df_para_mostrar = df_para_mostrar[df_para_mostrar['Status'].isin(status_selecionados)]
                else:
                    st.caption("Coluna 'Status' não encontrada.")

            st.subheader("Últimas Análises (Filtrado)")
            # Removemos o st.checkbox antigo
            st.dataframe(df_para_mostrar.iloc[::-1], use_container_width=True)
            ### MELHORIA 2 (DESIGN) / 4 (FUNC) - FIM ###


### MELHORIA 7 (A, B, C) - INÍCIO (Nova Aba: Analisar Times) ###
with tab_times:
    st.header("🔎 Dashboard de Análise de Times")
    
    emoji_liga_selecionada = LIGAS_EMOJI.get(LIGA_ATUAL, '🏳️')

    # CASO 1: Cérebro DIXON-COLES está ativo
    if MODO_CEREBRO == "DIXON_COLES" and dados_cerebro_dc:
        
        try:
            # --- Bloco de dados para ambas as melhorias ---
            lista_forcas = []
            for time, forcas in dados_cerebro_dc['forcas'].items():
                lista_forcas.append({
                    'Time': time,
                    'Ataque': forcas['ataque'],
                    'Defesa': forcas['defesa']
                })

            if not lista_forcas:
                st.warning("Nenhum time encontrado nos dados do cérebro.")
            else:
                df_liga = pd.DataFrame(lista_forcas)
                df_liga['Força Geral'] = df_liga['Ataque'] - df_liga['Defesa']
                df_liga = df_liga.sort_values(by="Força Geral", ascending=False)
                
                ### MELHORIA B - INÍCIO (Simulador de xG) ###
                st.subheader(f"🤖 Simulador de xG vs. Média da Liga ({emoji_liga_selecionada} {liga_selecionada_nome})")
                st.info("Traduza os 'números de força' em Gols Esperados (xG). Selecione um time para ver qual seria o xG esperado dele contra um time 'médio' da liga.")

                # 1. Calcular o "Time Médio"
                avg_ataque = df_liga['Ataque'].mean()
                avg_defesa = df_liga['Defesa'].mean()
                vantagem_casa = dados_cerebro_dc['vantagem_casa']

                # 2. Selectbox para escolher o time (ordena a lista de times do DF)
                time_para_simular = st.selectbox(
                    "Selecione um time para simular:",
                    options=df_liga['Time'], # Usa a lista de times do DF já ordenado
                    index=0 # Pega o primeiro time (o mais forte) por padrão
                )

                if time_para_simular:
                    # 3. Pegar as forças do time selecionado
                    forcas_time = dados_cerebro_dc['forcas'][time_para_simular]
                    ataque_time = forcas_time['ataque']
                    defesa_time = forcas_time['defesa']

                    # 4. Simular Jogos
                    # Jogo 1: Time Selecionado (Casa) vs. Time Médio (Fora)
                    xg_casa_sim1 = np.exp(ataque_time + avg_defesa + vantagem_casa)
                    xg_fora_sim1 = np.exp(avg_ataque + defesa_time)

                    # Jogo 2: Time Médio (Casa) vs. Time Selecionado (Fora)
                    xg_casa_sim2 = np.exp(avg_ataque + defesa_time + vantagem_casa)
                    xg_fora_sim2 = np.exp(ataque_time + avg_defesa)

                    # 5. Exibir métricas
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"xG de {time_para_simular} jogando EM CASA",
                            value=f"{xg_casa_sim1:.2f}",
                            help=f"Expectativa de gols contra um time 'médio' da liga. (Adversário faria {xg_fora_sim1:.2f} xG)"
                        )
                    with col2:
                        st.metric(
                            label=f"xG de {time_para_simular} jogando FORA",
                            value=f"{xg_fora_sim2:.2f}",
                            help=f"Expectativa de gols contra um time 'médio' da liga. (Adversário faria {xg_casa_sim2:.2f} xG)"
                        )
                
                st.divider()
                ### MELHORIA B - FIM ###
                
                ### MELHORIA A - INÍCIO (Ranking da Liga) ###
                st.subheader(f"Ranking de Forças Completo (DC) - {emoji_liga_selecionada} {liga_selecionada_nome}")
                st.info("Clique no título de uma coluna para ordenar. 'Força Geral' é a melhor métrica de ranking.")
                
                st.dataframe(
                    df_liga.style.format({
                        'Ataque': '{:.3f}',
                        'Defesa': '{:.3f}',
                        'Força Geral': '{:.3f}'
                    }),
                    column_config={
                        "Time": "Time",
                        "Ataque": st.column_config.NumberColumn(
                            "Força de Ataque (DC)", 
                            help="Mais alto = melhor ataque"
                        ),
                        "Defesa": st.column_config.NumberColumn(
                            "Força de Defesa (DC)", 
                            help="Mais baixo (mais negativo) = melhor defesa"
                        ),
                        "Força Geral": st.column_config.ProgressColumn(
                            "Força Geral (Ataque - Defesa)", 
                            help="Métrica combinada. Mais alto = melhor time.",
                            min_value=float(df_liga['Força Geral'].min()),
                            max_value=float(df_liga['Força Geral'].max())
                        )
                    },
                    use_container_width=True,
                    hide_index=True 
                )
        
        except Exception as e:
            st.error(f"Erro ao processar o ranking do cérebro DC: {e}")
        ### MELHORIA A - FIM ###
    
    # CASO 2: Cérebro POISSON está ativo
    elif MODO_CEREBRO == "POISSON_RECENTE" and df_historico_poisson is not None:
        
        ### MELHORIA C - INÍCIO (Dashboard de Forma Recente) ###
        st.subheader(f"📈 Análise de Forma Recente (Poisson) - {emoji_liga_selecionada} {liga_selecionada_nome}")
        st.info(f"Esta liga usa o Cérebro Poisson, que se baseia nos últimos 6 jogos. Veja abaixo as estatísticas de 'forma' de cada time.")
        
        # 1. Pegar lista de times
        times_na_liga = pd.concat([df_historico_poisson['TimeCasa'], df_historico_poisson['TimeVisitante']]).unique()
        times_na_liga.sort() # Ordenar alfabeticamente
        
        if len(times_na_liga) == 0:
            st.warning("Nenhum time encontrado no histórico desta liga.")
        else:
            # 2. Selectbox para escolher o time
            time_selecionado = st.selectbox(
                "Selecione um time para analisar a forma recente:",
                options=times_na_liga,
                index=0
            )
            
            if time_selecionado:
                num_jogos_recente = 6 # Conforme usado na função de prever
                
                # 3. Filtrar os últimos 6 jogos (o DF já está ordenado por data)
                jogos_casa_recente = df_historico_poisson[df_historico_poisson['TimeCasa'] == time_selecionado].tail(num_jogos_recente)
                jogos_fora_recente = df_historico_poisson[df_historico_poisson['TimeVisitante'] == time_selecionado].tail(num_jogos_recente)
                
                # 4. Calcular médias, tratando se o DF for vazio (time novo)
                gols_marcados_casa = jogos_casa_recente['GolsCasa'].mean() if not jogos_casa_recente.empty else 0.0
                gols_sofridos_casa = jogos_casa_recente['GolsVisitante'].mean() if not jogos_casa_recente.empty else 0.0
                
                gols_marcados_fora = jogos_fora_recente['GolsVisitante'].mean() if not jogos_fora_recente.empty else 0.0
                gols_sofridos_fora = jogos_fora_recente['GolsCasa'].mean() if not jogos_fora_recente.empty else 0.0
                
                # 5. Exibir métricas
                st.markdown(f"**Análise de Forma (Baseada nos últimos {num_jogos_recente} jogos)**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Jogando EM CASA**")
                    st.metric(f"Gols Marcados (Média)", f"{gols_marcados_casa:.2f}")
                    st.metric(f"Gols Sofridos (Média)", f"{gols_sofridos_casa:.2f}")
                    
                with col2:
                    st.markdown("**Jogando FORA**")
                    st.metric(f"Gols Marcados (Média)", f"{gols_marcados_fora:.2f}")
                    st.metric(f"Gols Sofridos (Média)", f"{gols_sofridos_fora:.2f}")
                    
                st.divider()
                
                # 6. (Opcional) Mostrar os jogos usados
                with st.expander(f"Ver últimos {num_jogos_recente} jogos em casa usados no cálculo"):
                    if jogos_casa_recente.empty:
                        st.write(f"Nenhum jogo recente em casa encontrado no histórico para {time_selecionado}.")
                    else:
                        st.dataframe(jogos_casa_recente[['data_jogo', 'TimeCasa', 'TimeVisitante', 'GolsCasa', 'GolsVisitante']].sort_values(by='data_jogo', ascending=False), use_container_width=True, hide_index=True)
                
                with st.expander(f"Ver últimos {num_jogos_recente} jogos fora usados no cálculo"):
                    if jogos_fora_recente.empty:
                        st.write(f"Nenhum jogo recente fora encontrado no histórico para {time_selecionado}.")
                    else:
                        st.dataframe(jogos_fora_recente[['data_jogo', 'TimeCasa', 'TimeVisitante', 'GolsCasa', 'GolsVisitante']].sort_values(by='data_jogo', ascending=False), use_container_width=True, hide_index=True)
        ### MELHORIA C - FIM ###

    # CASO 3: Nenhum cérebro carregado
    else:
        st.error("Cérebro não carregado. Selecione uma liga válida na aba 'Analisar Jogos' primeiro.")
### MELHORIA 7 - FIM ###
