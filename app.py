# app.py
# O Robô de Análise (Versão 4.3 - Design de Abas)
# UPGRADE: Adicionado st.tabs para organizar os formulários de odds

import streamlit as st
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats 
import config 
import time
from datetime import datetime
import json

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Robô de Valor (Híbrido)",
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

# --- ETAPA 1 (CÉREBRO HÍBRIDO) ---
@st.cache_data
def carregar_cerebro_dixon_coles(id_liga):
    nome_arquivo = f"dc_params_{id_liga}.json"
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            dados_cerebro = json.load(f)
            print(f"Cérebro Dixon-Coles ({id_liga}) carregado do arquivo.")
            return dados_cerebro
    except FileNotFoundError:
        print(f"Arquivo 'dc_params_{id_liga}.json' não encontrado. Usando Cérebro Poisson.")
        return None 
    except Exception as e:
        st.error(f"Erro ao ler o arquivo de Cérebro: {e}")
        return None

def prever_jogo_dixon_coles(dados_cerebro, time_casa, time_visitante):
    # (Função idêntica, sem 'print')
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
        return None
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
    if soma_total_probs == 0: return None
    prob_dc_1x = prob_vitoria_casa + prob_empate
    prob_dc_x2 = prob_empate + prob_vitoria_visitante
    prob_dc_12 = prob_vitoria_casa + prob_vitoria_visitante
    return {
        'vitoria_casa': prob_vitoria_casa / soma_total_probs, 'empate': prob_empate / soma_total_probs,
        'vitoria_visitante': prob_vitoria_visitante / soma_total_probs, 'over_2_5': prob_over_2_5 / soma_total_probs,
        'btts_sim': prob_btts_sim / soma_total_probs, 'chance_dupla_1X': prob_dc_1x / soma_total_probs,
        'chance_dupla_X2': prob_dc_x2 / soma_total_probs, 'chance_dupla_12': prob_dc_12 / soma_total_probs,
    }

@st.cache_data 
def carregar_e_treinar_cerebro_poisson(id_liga, temporada):
    # (Função idêntica)
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
    # (Função idêntica)
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
    # (Função idêntica)
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
    if soma_total_probs == 0: return None
    prob_dc_1x = prob_vitoria_casa + prob_empate
    prob_dc_x2 = prob_empate + prob_vitoria_visitante
    prob_dc_12 = prob_vitoria_casa + prob_vitoria_visitante
    return {
        'vitoria_casa': prob_vitoria_casa / soma_total_probs, 'empate': prob_empate / soma_total_probs,
        'vitoria_visitante': prob_vitoria_visitante / soma_total_probs, 'over_2_5': prob_over_2_5 / soma_total_probs,
        'btts_sim': prob_btts_sim / soma_total_probs, 'chance_dupla_1X': prob_dc_1x / soma_total_probs,
        'chance_dupla_X2': prob_dc_x2 / soma_total_probs, 'chance_dupla_12': prob_dc_12 / soma_total_probs,
    }

def encontrar_valor(probabilidades_calculadas, odds_casa, filtro_prob_minima=0.60, filtro_valor_minimo=0.05):
    # (Função idêntica)
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
    # (Função idêntica)
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
    # (Função idêntica)
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

# Dicionário de Ligas
LIGAS_DISPONIVEIS = {
    "Brasileirão": "BSA",
    "Premier League (ING)": "PL",
    "La Liga (ESP)": "PD",
    "Serie A (ITA)": "SA",
    "Bundesliga (ALE)": "BL1",
    "Ligue 1 (FRA)": "FL1",
    "Eredivisie (HOL)": "DED",
    "Championship (ING 2)": "ELC"
}

# --- 1. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("Controles do Robô 🤖")
    
    liga_selecionada_nome = st.selectbox(
        "1. Selecione a Liga:",
        LIGAS_DISPONIVEIS.keys() 
    )
    LIGA_ATUAL = LIGAS_DISPONIVEIS[liga_selecionada_nome]
    TEMPORADA_ATUAL = config.TEMPORADA_PARA_ANALISAR 

    st.header("2. Cérebro do Robô")
    MODO_CEREBRO = "FALHA" 
    
    with st.spinner(f"Tentando carregar Cérebro Dixon-Coles para {LIGA_ATUAL}..."):
        dados_cerebro_dc = carregar_cerebro_dixon_coles(LIGA_ATUAL)
    
    if dados_cerebro_dc is not None:
        st.success(f"Cérebro Avançado (Dixon-Coles) carregado!")
        st.caption(f"Treinado em: {dados_cerebro_dc['data_treinamento'].split('T')[0]}")
        MODO_CEREBRO = "DIXON_COLES"
        df_historico_poisson = None
        medias_liga_poisson = None
    else:
        st.warning(f"Cérebro Dixon-Coles não encontrado para {LIGA_ATUAL}.")
        st.info("Usando Cérebro de 'Forma Recente' (Poisson) como fallback.")
        
        with st.spinner(f"Treinando Cérebro Poisson para {LIGA_ATUAL}..."):
            df_historico_poisson, medias_liga_poisson = carregar_e_treinar_cerebro_poisson(LIGA_ATUAL, TEMPORADA_ATUAL)
        
        if df_historico_poisson is None:
            st.error(f"Falha ao carregar dados da {LIGA_ATUAL}.")
        else:
            st.success(f"Cérebro Poisson treinado com {len(df_historico_poisson)} jogos.")
            MODO_CEREBRO = "POISSON_RECENTE"

    st.header("3. Buscar Jogos")
    data_selecionada = st.date_input(
        "Selecione a data para analisar:",
        datetime.now()
    )
    
    st.header("4. Filtros de Análise")
    filtro_prob_minima_percentual = st.slider(
        "Probabilidade Mínima (Chance de Green)", 
        min_value=0, max_value=100, value=60, step=5, format="%d%%"
    )
    filtro_prob_minima = filtro_prob_minima_percentual / 100.0 
    filtro_valor_minimo = 0.05
    
# --- 2. PÁGINA PRINCIPAL ---
st.title("Robô de Análise de Valor (Híbrido) 🧠")
st.header(f"Jogos para {data_selecionada.strftime('%d/%m/%Y')} na Liga: {LIGA_ATUAL}")
st.caption(f"Usando Cérebro: {MODO_CEREBRO} | Filtro de Probabilidade: > {filtro_prob_minima_percentual}%")

if MODO_CEREBRO != "FALHA":
    data_str = data_selecionada.strftime('%Y-%m-%d')
    jogos_do_dia = buscar_jogos_por_data(LIGA_ATUAL, data_str)

    if not jogos_do_dia:
        st.info(f"Nenhum jogo agendado encontrado para a liga {LIGA_ATUAL} na data {data_str}.")
    else:
        nomes_mercado = {
            'vitoria_casa': 'Casa', 'empate': 'Empate', 'vitoria_visitante': 'Fora',
            'over_2_5': 'Mais de 2.5 Gols', 'btts_sim': 'Ambas Marcam (Sim)',
            'chance_dupla_1X': 'Casa ou Empate (1X)', 'chance_dupla_X2': 'Empate ou Fora (X2)',
            'chance_dupla_12': 'Casa ou Fora (12)'
        }
        
        for i, jogo in enumerate(jogos_do_dia):
            with st.expander(f"Jogo: {jogo['time_casa']} vs {jogo['time_visitante']}"):
                
                # --- ESTE É O FORMULÁRIO ATUALIZADO (MELHORIA 1) ---
                with st.form(key=f"form_jogo_{i}"):
                    
                    # Cria as 3 abas
                    tab_1x2, tab_dc, tab_gols = st.tabs(["📊 Resultado (1x2)", "🤝 Chance Dupla", "⚽ Gols"])

                    with tab_1x2:
                        st.write("**Mercado 1X2**")
                        col1, col2, col3 = st.columns(3)
                        with col1: odd_casa = st.number_input(f"{jogo['time_casa']} (1)", min_value=1.0, value=None, format="%.2f", key=f"casa_{i}")
                        with col2: odd_empate = st.number_input("Empate (X)", min_value=1.0, value=None, format="%.2f", key=f"empate_{i}")
                        with col3: odd_visitante = st.number_input(f"{jogo['time_visitante']} (2)", min_value=1.0, value=None, format="%.2f", key=f"visit_{i}")
                    
                    with tab_dc:
                        st.write("**Chance Dupla**")
                        col_dc1, col_dc2, col_dc3 = st.columns(3)
                        with col_dc1: odd_1x = st.number_input("Casa/Empate (1X)", min_value=1.0, value=None, format="%.2f", key=f"dc1x_{i}")
                        with col_dc2: odd_x2 = st.number_input("Empate/Fora (X2)", min_value=1.0, value=None, format="%.2f", key=f"dcx2_{i}")
                        with col_dc3: odd_12 = st.number_input("Casa/Fora (12)", min_value=1.0, value=None, format="%.2f", key=f"dc12_{i}")
                    
                    with tab_gols:
                        st.write("**Gols**")
                        col4, col5 = st.columns(2)
                        with col4: odd_over = st.number_input("Mais de 2.5 Gols", min_value=1.0, value=None, format="%.2f", key=f"over_{i}")
                        with col5: odd_btts = st.number_input("Ambas Marcam (Sim)", min_value=1.0, value=None, format="%.2f", key=f"btts_{i}")
                    
                    # Botão de submit (fora das abas, dentro do form)
                    submitted = st.form_submit_button("Analisar este Jogo")

                    if submitted:
                        with st.spinner("Analisando..."):
                            odds_manuais = {
                                'vitoria_casa': odd_casa, 'empate': odd_empate,
                                'vitoria_visitante': odd_visitante, 'over_2_5': odd_over, 'btts_sim': odd_btts,
                                'chance_dupla_1X': odd_1x, 'chance_dupla_X2': odd_x2, 'chance_dupla_12': odd_12
                            }
                            
                            probs_robo = None
                            
                            if MODO_CEREBRO == "DIXON_COLES":
                                probs_robo = prever_jogo_dixon_coles(
                                    dados_cerebro_dc, jogo['time_casa'], jogo['time_visitante']
                                )
                            
                            elif MODO_CEREBRO == "POISSON_RECENTE":
                                forcas_times = calcular_forcas_recente_poisson(
                                    df_historico_poisson, jogo['time_casa'], jogo['time_visitante'], jogo['data_jogo']
                                )
                                if forcas_times:
                                    probs_robo = prever_jogo_poisson(
                                        forcas_times, medias_liga_poisson,
                                        jogo['time_casa'], jogo['time_visitante'] 
                                    )
                            
                            if probs_robo:
                                oportunidades = encontrar_valor(
                                    probs_robo, odds_manuais, 
                                    filtro_prob_minima, filtro_valor_minimo
                                )
                                
                                if oportunidades:
                                    st.success("🔥 OPORTUNIDADES DE VALOR ENCONTRADAS!")
                                    mensagem = f"🔥 <b>Oportunidade ({MODO_CEREBRO})</b> 🔥\n\n"
                                    mensagem += f"<b>Liga:</b> {liga_selecionada_nome}\n"
                                    mensagem += f"<b>Jogo:</b> {jogo['time_casa']} vs {jogo['time_visitante']}\n"
                                    
                                    for mercado, dados in oportunidades.items():
                                        mercado_limpo = nomes_mercado.get(mercado, mercado)
                                        st.subheader(f"Mercado: {mercado_limpo}")
                                        st.text(f"  Odd: {dados['odd_casa']:.2f} (Casa: {dados['prob_casa_aposta']:.2f}%)")
                                        st.text(f"  Probabilidade: {dados['prob_robo']:.2f}%")
*_                                        st.text(f"  Valor: +{dados['valor_encontrado']:.2f}%")
                                        
                                        mensagem += f"\n<b>Mercado: {mercado_limpo}</b>\n"
                                        mensagem += f"  Odd: {dados['odd_casa']:.2f} (Casa: {dados['prob_casa_aposta']:.2f}%)\n"
                                        mensagem += f"  <b>Probabilidade: {dados['prob_robo']:.2f}%</b>\n"
                                        mensagem += f"  <b>Valor: +{dados['valor_encontrado']:.2f}%</b>\n"
                                    
                                    enviar_mensagem_telegram(mensagem)
                                else:
                                    st.info(f"Nenhuma oportunidade de valor (com >{filtro_prob_minima_percentual}% de prob.) encontrada.")
                            else:
                                st.error("Não foi possível calcular as probabilidades do robô (Times novos ou erro no Cérebro).")
