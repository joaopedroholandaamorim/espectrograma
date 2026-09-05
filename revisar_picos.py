import os
import glob
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from espectograma import (
    multi_gaussian,
    agrupar_picos_entre_amostras,
    montar_tabela_detalhada,
    salvar_detalhe_amostra,
    PASTA_FOTOS,
    ARQUIVO_SAIDA_XLSX,
    ARQUIVO_DETALHADO_XLSX,
    TOLERANCIA_CLUSTER,
)

st.set_page_config(page_title="Revisão de Picos Raman", layout="wide")


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def refit_com_novo_pico(x_reg, y_reg, picos, xc_click):
    """
    Adiciona um pico novo centrado em xc_click (posição clicada no gráfico) e
    reajusta AMPLITUDE e SIGMA de todos os picos (existentes + o novo) com
    curve_fit, exatamente a mesma lógica usada no espectrograma.py.

    O chute inicial de amplitude do pico novo vem do resíduo (experimental
    menos a soma dos picos já existentes) no ponto clicado - assim o fit já
    começa perto do valor certo. Se o curve_fit falhar (região muito ruidosa,
    picos colidindo, etc.), o pico novo é mantido com o chute inicial, sem
    reajustar os demais, e a função avisa o usuário.
    """
    if len(picos) > 0:
        params_atuais = []
        for p in picos:
            params_atuais.extend([p["amplitude"], p["posicao"], p["sigma"]])
        y_fit_atual = multi_gaussian(x_reg, *params_atuais)
    else:
        y_fit_atual = np.zeros_like(x_reg)

    residuo = y_reg - y_fit_atual
    idx_click = int(np.argmin(np.abs(x_reg - xc_click)))
    amp_guess = max(float(residuo[idx_click]), 1.0)
    sigma_guess = 8.0  # chute inicial; o curve_fit refina a partir daqui

    novo_pico = {"amplitude": amp_guess, "posicao": float(xc_click), "sigma": sigma_guess}
    picos_tentativa = picos + [novo_pico]

    dx = float(np.mean(np.diff(x_reg))) if len(x_reg) > 1 else 1.0
    p0, lower, upper = [], [], []
    for p in picos_tentativa:
        p0.extend([p["amplitude"], p["posicao"], p["sigma"]])
        lower.extend([1e-6, p["posicao"] - 10, dx])
        upper.extend([np.inf, p["posicao"] + 10, 80])

    try:
        params_fit, _ = curve_fit(
            multi_gaussian, x_reg, y_reg, p0=p0, bounds=(lower, upper), maxfev=10000
        )
        picos_ajustados = []
        for i in range(0, len(params_fit), 3):
            picos_ajustados.append({
                "amplitude": float(params_fit[i]),
                "posicao": float(params_fit[i + 1]),
                "sigma": float(params_fit[i + 2]),
            })
        return picos_ajustados, True
    except Exception:
        return picos + [novo_pico], False


def listar_amostras():
    """
    Varre a pasta Fotos/ procurando subpastas que já foram processadas pelo
    espectrograma.py (ou seja, que têm regioes.json e dados_espectro.npz).
    """
    pastas = sorted(glob.glob(os.path.join(PASTA_FOTOS, "*")))
    amostras = []
    for pasta in pastas:
        regioes_path = os.path.join(pasta, "regioes.json")
        npz_path = os.path.join(pasta, "dados_espectro.npz")
        if os.path.exists(regioes_path) and os.path.exists(npz_path):
            with open(regioes_path, encoding="utf-8") as f:
                info = json.load(f)
            amostras.append({
                "sample_id": os.path.basename(pasta),
                "nome_amostra": info["amostra"],
                "pasta": pasta,
                "regioes": info["regioes"],
            })
    return amostras


def carregar_picos(pasta, regiao_num):
    json_path = os.path.join(pasta, f"picos_regiao_{regiao_num}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def salvar_picos(pasta, regiao_num, picos):
    json_path = os.path.join(pasta, f"picos_regiao_{regiao_num}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(picos, f, indent=2)


def picos_para_resultados(picos, regiao_num):
    """Converte a lista de picos (amplitude/posicao/sigma) para o formato
    {"regiao","posicao","amplitude","fwhm","area"} usado na consolidação."""
    resultados = []
    for p in picos:
        fwhm = 2.3548 * p["sigma"]
        area = p["amplitude"] * p["sigma"] * np.sqrt(2 * np.pi)
        resultados.append({
            "regiao": regiao_num,
            "posicao": p["posicao"],
            "amplitude": p["amplitude"],
            "fwhm": fwhm,
            "area": area,
        })
    return resultados


def gerar_planilha_consolidada(amostras):
    """
    Relê o JSON de picos de TODAS as regiões de TODAS as amostras (ou seja,
    já usando qualquer edição manual salva) e regenera:
    - o CSV de detalhe por amostra (Fotos/<id>/detalhe_picos_<amostra>.csv)
    - a planilha alinhada por pico (picos_raman.xlsx, 4 abas)
    - a planilha detalhada (picos_detalhado_todas_amostras.xlsx)
    """
    todos_resultados = {}

    for a in amostras:
        pasta = a["pasta"]
        nome_amostra = a["nome_amostra"]
        picos_amostra = []

        for r in a["regioes"]:
            picos_regiao = carregar_picos(pasta, r["regiao"])
            picos_amostra.extend(picos_para_resultados(picos_regiao, r["regiao"]))

        picos_amostra.sort(key=lambda d: d["posicao"])
        todos_resultados[nome_amostra] = picos_amostra

        # atualiza também o CSV individual daquela amostra
        salvar_detalhe_amostra(nome_amostra, picos_amostra, pasta)

    tabelas = agrupar_picos_entre_amostras(todos_resultados, TOLERANCIA_CLUSTER)
    with pd.ExcelWriter(ARQUIVO_SAIDA_XLSX, engine="openpyxl") as writer:
        for nome_aba, df in tabelas.items():
            df.to_excel(writer, sheet_name=nome_aba)

    df_detalhado = montar_tabela_detalhada(todos_resultados)
    df_detalhado.to_excel(ARQUIVO_DETALHADO_XLSX, index=False)


def renderizar_regiao(amostra_atual, regiao_info):
    pasta = amostra_atual["pasta"]
    sample_id = amostra_atual["sample_id"]
    regiao_num = regiao_info["regiao"]
    xmin, xmax = regiao_info["xmin"], regiao_info["xmax"]

    # chave única de sessão para essa combinação amostra + região
    state_key = f"picos__{sample_id}__{regiao_num}"
    chart_key = f"{state_key}_chart"
    click_sig_key = f"{state_key}_ultimo_clique"
    versao_key = f"{state_key}_versao"

    # "versão" da lista de picos: incrementada toda vez que os picos são alterados
    # programaticamente (clique+refit, adicionar, remover). Ela entra na key dos
    # sliders para forçá-los a "nascer de novo" com o valor certo -- sem isso, o
    # Streamlit reaproveita o valor antigo que o slider já tinha guardado por causa
    # da key repetida, e o resultado do curve_fit é sobrescrito silenciosamente.
    if versao_key not in st.session_state:
        st.session_state[versao_key] = 0

    # carrega o espectro (já com baseline removida) e recorta para a região
    dados = np.load(os.path.join(pasta, "dados_espectro.npz"))
    x_full, y_full = dados["x"], dados["y_corr"]
    mask = (x_full >= xmin) & (x_full <= xmax)
    x_reg, y_reg = x_full[mask], y_full[mask]

    # carrega os picos do arquivo apenas na primeira vez que essa região é aberta
    # nessa sessão; depois disso, o que está em session_state manda (permite editar
    # sem salvar a cada clique)
    if state_key not in st.session_state:
        st.session_state[state_key] = carregar_picos(pasta, regiao_num)

    picos = st.session_state[state_key]

    col_grafico, col_controles = st.columns([2, 1])

    # ---------- calcula o fit atual e o R² ao vivo (com o estado de ANTES do clique) ----------
    def calcular_fit_r2(lista_picos):
        if len(lista_picos) > 0 and len(x_reg) > 0:
            params_flat = []
            for p in lista_picos:
                params_flat.extend([p["amplitude"], p["posicao"], p["sigma"]])
            y_fit = multi_gaussian(x_reg, *params_flat)
            return y_fit, r2_score(y_reg, y_fit)
        return np.zeros_like(x_reg), float("nan")

    y_fit, r2 = calcular_fit_r2(picos)

    # ---------- gráfico interativo (Plotly), com clique para adicionar pico ----------
    with col_grafico:
        if np.isnan(r2):
            st.markdown("### R² atual: _sem picos nesta região_")
        else:
            st.markdown(f"### R² atual: `{r2:.4f}`")

        st.caption(
            "💡 Clique em cima da curva experimental (linha preta) no ponto onde deveria "
            "existir um pico — amplitude e largura são ajustadas automaticamente (curve_fit)."
        )

        fig = go.Figure()
        # experimental com marcadores (necessário para o clique ser capturado pelo Plotly);
        # markers pequenos para não distorcer visualmente a curva
        fig.add_trace(go.Scatter(
            x=x_reg, y=y_reg, mode="lines+markers", name="Experimental",
            line=dict(color="black", width=1.5), marker=dict(size=3, color="black"),
            opacity=0.7
        ))
        if len(picos) > 0:
            fig.add_trace(go.Scatter(
                x=x_reg, y=y_fit, mode="lines", name="Soma (fit)",
                line=dict(color="red", dash="dash", width=2)
            ))
            for i, p in enumerate(picos):
                comp = p["amplitude"] * np.exp(-(x_reg - p["posicao"]) ** 2 / (2 * p["sigma"] ** 2))
                fig.add_trace(go.Scatter(
                    x=x_reg, y=comp, mode="lines", name=f"Pico {i+1}",
                    line=dict(dash="dot", width=1), showlegend=False, opacity=0.6
                ))
        fig.update_layout(
            xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Intensidade (a.u.)",
            height=480, margin=dict(l=10, r=10, t=10, b=10),
            clickmode="event+select"
        )

        evento = st.plotly_chart(
            fig, key=chart_key, on_select="rerun", selection_mode=["points"],
            use_container_width=True
        )

        # ---------- processa o clique, se houver um NOVO clique não processado ainda ----------
        pontos = []
        if evento and evento.get("selection"):
            pontos = evento["selection"].get("points", [])

        if pontos:
            ponto = pontos[0]
            x_click = ponto.get("x")
            if x_click is None and "point_index" in ponto:
                idx = ponto["point_index"]
                if 0 <= idx < len(x_reg):
                    x_click = float(x_reg[idx])

            assinatura = (ponto.get("point_index"), x_click)

            # só processa se for um clique diferente do último já tratado
            if x_click is not None and st.session_state.get(click_sig_key) != assinatura:
                st.session_state[click_sig_key] = assinatura
                with st.spinner("Ajustando o pico novo (curve_fit)..."):
                    novos_picos, sucesso = refit_com_novo_pico(x_reg, y_reg, picos, x_click)
                st.session_state[state_key] = novos_picos
                st.session_state[versao_key] += 1  # invalida os sliders antigos
                if not sucesso:
                    st.warning(
                        "O ajuste automático não convergiu para essa posição; o pico foi "
                        "adicionado com um chute inicial. Ajuste manualmente pelos sliders."
                    )
                st.rerun()

    # ---------- painel de controles (sliders por pico) ----------
    with col_controles:
        st.markdown(f"**{len(picos)} pico(s) nesta região**")

        versao = st.session_state[versao_key]
        remover_idx = None
        for i, pico in enumerate(picos):
            with st.expander(f"Pico {i + 1} — {pico['posicao']:.1f} cm⁻¹", expanded=False):
                pico["amplitude"] = st.slider(
                    "Amplitude", 0.0, max(200.0, pico["amplitude"] * 2.0),
                    float(pico["amplitude"]), key=f"{state_key}_v{versao}_amp_{i}"
                )
                pico["posicao"] = st.slider(
                    "Posição (cm⁻¹)", float(xmin), float(xmax),
                    float(min(max(pico["posicao"], xmin), xmax)), key=f"{state_key}_v{versao}_pos_{i}"
                )
                pico["sigma"] = st.slider(
                    "Sigma (largura)", 0.5, 80.0,
                    float(pico["sigma"]), key=f"{state_key}_v{versao}_sig_{i}"
                )
                if st.button("🗑️ Remover este pico", key=f"{state_key}_v{versao}_rm_{i}"):
                    remover_idx = i

        if remover_idx is not None:
            picos.pop(remover_idx)
            st.session_state[versao_key] += 1  # invalida os sliders antigos (índices deslizaram)
            st.rerun()

        st.markdown("---")
        if st.button("➕ Adicionar pico (manual, sem ajuste automático)", key=f"{state_key}_add"):
            centro = (xmin + xmax) / 2
            amp_inicial = float(np.interp(centro, x_reg, y_reg)) if len(x_reg) else 1.0
            picos.append({"amplitude": max(amp_inicial, 1.0), "posicao": centro, "sigma": 10.0})
            st.session_state[versao_key] += 1  # mantém consistência com os outros pontos de mutação
            st.rerun()

        st.markdown("---")
        if st.button("💾 Salvar esta região", type="primary", key=f"{state_key}_save"):
            salvar_picos(pasta, regiao_num, picos)
            st.success("Região salva em disco.")


# =========================================================
# FLUXO PRINCIPAL DA INTERFACE
# =========================================================
st.title("Revisão manual dos picos Raman — Li4Mo5O17")
st.caption(
    "Ajuste amplitude, posição e largura de cada pico com os sliders e acompanhe o R² "
    "em tempo real. Use 'Salvar esta região' para gravar as alterações no arquivo."
)

amostras = listar_amostras()

if not amostras:
    st.error(
        f"Nenhuma amostra processada encontrada em '{PASTA_FOTOS}/'. "
        "Rode o espectrograma.py primeiro para gerar os dados."
    )
    st.stop()

# ---------- barra lateral: seleção de amostra + ações globais ----------
opcoes = {f"{a['sample_id']} — {a['nome_amostra']}": a for a in amostras}
escolha = st.sidebar.selectbox("Amostra", list(opcoes.keys()))
amostra_atual = opcoes[escolha]

st.sidebar.markdown("---")
st.sidebar.subheader("Ações globais")
st.sidebar.caption(
    "Relê o JSON de picos (já com suas edições) de todas as amostras e "
    "regenera as planilhas finais."
)
if st.sidebar.button("🔄 Gerar planilhas consolidadas"):
    with st.spinner("Consolidando todas as amostras..."):
        gerar_planilha_consolidada(amostras)
    st.sidebar.success(f"Planilhas atualizadas:\n\n- {ARQUIVO_SAIDA_XLSX}\n- {ARQUIVO_DETALHADO_XLSX}")

# ---------- abas de região da amostra selecionada ----------
regioes = amostra_atual["regioes"]
nomes_abas = [f"Região {r['regiao']} ({r['xmin']:.0f}-{r['xmax']:.0f} cm⁻¹)" for r in regioes]
abas = st.tabs(nomes_abas)

for aba, regiao_info in zip(abas, regioes):
    with aba:
        renderizar_regiao(amostra_atual, regiao_info)