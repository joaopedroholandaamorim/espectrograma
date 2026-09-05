import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem janela -> essencial para rodar em lote sem travar em plt.show()
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import ParameterGrid
from scipy.cluster.hierarchy import linkage, fcluster

# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================
PASTA_ARQUIVOS = "arquivos"
PASTA_FOTOS = "Fotos"
ARQUIVO_SAIDA_XLSX = "picos_raman.xlsx"                        # planilha final alinhada (abas: Posicao, Amplitude, FWHM, Area)
ARQUIVO_DETALHADO_XLSX = "picos_detalhado_todas_amostras.xlsx"  # todos os picos de todas as amostras, um por linha (sem alinhamento)
TOLERANCIA_CLUSTER = 10   # cm^-1 - picos mais próximos que isso, entre amostras diferentes,
                          # são considerados "o mesmo pico" e caem na mesma coluna

PARAM_GRID = {
    "distance": [6, 10],
    "width": [1, 3, 5],
    "prominence": [0.5, 1, 5, 10]
}

INITIAL_REGIONS = [
    (75, 325),
    (575, 825),
    (825, 1575)
]


# =========================================================
# FUNÇÕES DE PROCESSAMENTO (mesma lógica do script original)
# =========================================================
def baseline_als(y, lam=5e5, p=0.01, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0)
        Z = csc_matrix(W + lam * D.dot(D.T))
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        A, xc, sigma = params[i], params[i + 1], params[i + 2]
        y += A * np.exp(-(x - xc) ** 2 / (2 * sigma ** 2))
    return y


def AIC(y, y_fit, k):
    n = len(y)
    rss = np.sum((y - y_fit) ** 2)
    return n * np.log(rss / n) + 2 * k


def evaluate_params(params, x, y, dx):
    try:
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)

        peaks, props = find_peaks(
            y_smooth,
            distance=params["distance"],
            width=params["width"],
            prominence=params["prominence"]
        )

        noise = np.std(y_smooth[y_smooth < np.percentile(y_smooth, 20)])

        valid = (
            (y_smooth[peaks] > 3 * noise) &
            (y_smooth[peaks] > 0.02 * np.max(y_smooth)) &
            (peaks > 5) &
            (peaks < len(y_smooth) - 5)
        )
        peaks = peaks[valid]

        if len(peaks) == 0:
            return -np.inf, None, None, None

        x_peaks = x[peaks]
        p0, lower, upper = [], [], []

        for i, p in enumerate(peaks):
            A = max(y[p], 1e-6)
            xc = x[p]
            fwhm = props["widths"][i] * dx
            sigma = max(fwhm / 2.3548, dx)
            p0.extend([A, xc, sigma])

        for xp in x_peaks:
            lower.extend([1e-6, xp - 10, dx])
            upper.extend([np.inf, xp + 10, 80])

        params_fit, _ = curve_fit(
            multi_gaussian, x, y, p0=p0, bounds=(lower, upper), maxfev=10000
        )

        y_fit = multi_gaussian(x, *params_fit)
        k = len(params_fit)
        aic = AIC(y, y_fit, k)

        return aic, params_fit, y_fit, peaks

    except Exception:
        return -np.inf, None, None, None


def adaptive_regions_chained(x, y, initial_regions, tol, xmax_limit=1000):
    adaptive = []
    current_start = initial_regions[0][0]
    idx_max = np.argmin(np.abs(x - xmax_limit))

    for _, x_min_end in initial_regions:
        idx_start = np.argmin(np.abs(x - current_start))
        idx_end = np.argmin(np.abs(x - x_min_end))
        idx_start = min(idx_start, idx_max)
        idx_end = min(idx_end, idx_max)
        y_start = y[idx_start]

        while idx_end < idx_max:
            if abs(y[idx_end] - y_start) < tol:
                break
            idx_end += 1

        xmin, xmax = x[idx_start], x[idx_end]
        adaptive.append((xmin, xmax))
        current_start = xmax

        if idx_end >= idx_max:
            break

    return adaptive


# =========================================================
# EXTRAI O ID DA AMOSTRA A PARTIR DO NOME DO ARQUIVO
# =========================================================
def get_sample_id(filename):
    """
    Extrai o número da amostra do nome do arquivo.
    Ex: 'Li4Mo5O17_05_1800gmm 029.txt' -> '05'
    Isso é usado para salvar as fotos na pasta Fotos/05/, que já existe.
    """
    match = re.search(r'_(\d{2})_', os.path.basename(filename))
    if match:
        return match.group(1)
    # fallback: usa o nome do arquivo sem extensão, caso o padrão não bata
    return os.path.splitext(os.path.basename(filename))[0]


# =========================================================
# PROCESSA UM ÚNICO ARQUIVO
# =========================================================
def processar_espectro(caminho_arquivo, pasta_fotos_amostra):
    os.makedirs(pasta_fotos_amostra, exist_ok=True)
    nome_amostra = os.path.splitext(os.path.basename(caminho_arquivo))[0]

    data = np.loadtxt(caminho_arquivo, comments='#')
    x = data[:, 0]
    y = data[:, 1]

    # --- espectro bruto ---
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(nome_amostra)
    plt.savefig(f"{pasta_fotos_amostra}/00_espectro_bruto.png", dpi=150)
    plt.close()

    tol = 0.02 * np.std(y)
    adaptive_regions = adaptive_regions_chained(x, y, INITIAL_REGIONS, tol)

    # salva a lista de regiões encontradas -> a interface de revisão usa isso para montar as abas
    with open(os.path.join(pasta_fotos_amostra, "regioes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "amostra": nome_amostra,
            "regioes": [
                {"regiao": i + 1, "xmin": float(xmin), "xmax": float(xmax)}
                for i, (xmin, xmax) in enumerate(adaptive_regions)
            ]
        }, f, indent=2, ensure_ascii=False)

    # --- baseline global ---
    mask_global = (x >= 75) & (x <= 1000)
    x_base, y_base = x[mask_global], y[mask_global]

    best_baseline, best_std = None, np.inf
    for lam in [1e4, 5e4, 1e5]:
        b = baseline_als(y_base, lam=lam, p=0.01)
        res = y_base - b
        if np.std(res) < best_std:
            best_std = np.std(res)
            best_baseline = b

    baseline_global = np.interp(x, x_base, best_baseline)
    y_corr_global = y - baseline_global

    # salva o espectro corrigido (x, y_corr) para a interface de revisão reutilizar sem reprocessar
    np.savez(
        os.path.join(pasta_fotos_amostra, "dados_espectro.npz"),
        x=x, y_corr=y_corr_global
    )

    plt.figure()
    plt.plot(x, y, label="Original")
    plt.plot(x, baseline_global, '--', label="Baseline global")
    plt.legend()
    plt.title(nome_amostra)
    plt.savefig(f"{pasta_fotos_amostra}/01_baseline.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, y, 'k', alpha=0.5)
    for xmin, xmax in adaptive_regions:
        plt.axvspan(xmin, xmax, alpha=0.2)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Regiões adaptadas - {nome_amostra}")
    plt.savefig(f"{pasta_fotos_amostra}/02_regioes.png", dpi=150)
    plt.close()

    resultados_picos = []  # lista de dicts: {"posicao":..., "fwhm":..., "area":...}

    for region_id, (xmin, xmax) in enumerate(adaptive_regions):
        mask = (x >= xmin) & (x <= xmax)
        x_reg, y_reg = x[mask], y[mask]
        y_corr = y_corr_global[mask]
        dx = np.mean(np.diff(x_reg))

        best_score = -np.inf
        best_params = best_fit = best_yfit = best_peaks = best_aic = None

        #procura melhor r2 possivel para diferentes parametros, no final achando melhores gausianas em best_yfit
        for params in ParameterGrid(PARAM_GRID):
            aic, fit, y_fit, peaks = evaluate_params(params, x_reg, y_corr, dx)
            if y_fit is None:
                continue
            r2_local = r2_score(y_corr, y_fit)
            if r2_local > best_score:
                best_score, best_aic = r2_local, aic
                best_params, best_fit = params, fit
                best_peaks, best_yfit = peaks, y_fit

        # nenhuma combinação de parâmetros funcionou nessa região -> pula (sem quebrar o script)
        if best_fit is None:
            print(f"  [aviso] Região {region_id+1} ({xmin:.0f}-{xmax:.0f}) sem picos válidos em {nome_amostra}")
            # salva um JSON vazio: a interface de revisão ainda mostra a região, permitindo
            # que o pico seja adicionado manualmente depois
            caminho_json = os.path.join(pasta_fotos_amostra, f"picos_regiao_{region_id+1}.json")
            with open(caminho_json, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            continue

        x_peaks, y_peaks = x_reg[best_peaks], y_corr[best_peaks]

        plt.figure()
        plt.plot(x_reg, y_corr)
        plt.plot(x_peaks, y_peaks, 'ro')
        plt.title(f"{nome_amostra} - Região {region_id+1} - Picos detectados")
        plt.savefig(f"{pasta_fotos_amostra}/regiao{region_id+1}_picos.png", dpi=150)
        plt.close()

        res = y_corr - best_yfit
        plt.figure()
        plt.plot(x_reg, res)
        plt.axhline(0, color='k')
        plt.title(f"Resíduos - {nome_amostra} - Região {region_id+1}")
        plt.savefig(f"{pasta_fotos_amostra}/regiao{region_id+1}_residuos.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(x_reg, y_corr, label="Experimental")
        plt.plot(x_reg, best_yfit, 'r--', label="Fit")
        for i in range(0, len(best_fit), 3):
            A, xc, sigma = best_fit[i:i + 3]
            plt.plot(x_reg, A * np.exp(-(x_reg - xc) ** 2 / (2 * sigma ** 2)), ':')
        plt.legend()
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"{nome_amostra} - Região {region_id+1} - Fit final")
        plt.savefig(f"{pasta_fotos_amostra}/regiao{region_id+1}_fit.png", dpi=150)
        plt.close()

        print(f"  Região {region_id+1}: AIC={best_aic:.2f} | R²={best_score:.4f} | params={best_params}")

        # salva os parâmetros "crus" da gaussiana (amplitude, posição, sigma) em JSON editável ->
        # é isso que a interface de revisão lê e sobrescreve
        picos_editaveis = []
        for i in range(0, len(best_fit), 3):
            A, xc, sigma = best_fit[i:i + 3]
            picos_editaveis.append({"amplitude": float(A), "posicao": float(xc), "sigma": float(sigma)})

        caminho_json = os.path.join(pasta_fotos_amostra, f"picos_regiao_{region_id+1}.json")
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(picos_editaveis, f, indent=2)

        for i in range(0, len(best_fit), 3):
            A, xc, sigma = best_fit[i:i + 3]
            fwhm = 2.3548 * sigma
            area = A * sigma * np.sqrt(2 * np.pi)
            resultados_picos.append({
                "regiao": region_id + 1,
                "posicao": xc,
                "amplitude": A,
                "fwhm": fwhm,
                "area": area
            })
            print(f"    Pico em {xc:.1f} cm⁻¹ | Amplitude = {A:.1f} | FWHM = {fwhm:.1f} | Área = {area:.1f}")

    resultados_picos.sort(key=lambda d: d["posicao"])

    # --- salva o detalhe dos picos dessa amostra junto com as fotos ---
    salvar_detalhe_amostra(nome_amostra, resultados_picos, pasta_fotos_amostra)

    return resultados_picos


# =========================================================
# SALVA UM ARQUIVO POR AMOSTRA COM POSIÇÃO, AMPLITUDE, FWHM E ÁREA
# =========================================================
def salvar_detalhe_amostra(nome_amostra, resultados_picos, pasta_fotos_amostra):
    """
    Salva, na mesma pasta das fotos daquela amostra, um CSV com uma linha por
    pico encontrado (posição, amplitude, FWHM e área). Se nenhum pico foi
    encontrado, um arquivo vazio (só com cabeçalho) é salvo mesmo assim, para
    deixar claro que a amostra foi processada mas não rendeu picos válidos.
    """
    df_amostra = pd.DataFrame(resultados_picos)
    if df_amostra.empty:
        df_amostra = pd.DataFrame(columns=["regiao", "posicao", "amplitude", "fwhm", "area"])
    else:
        df_amostra = df_amostra[["regiao", "posicao", "amplitude", "fwhm", "area"]]

    df_amostra.rename(columns={
        "regiao": "Regiao",
        "posicao": "Posicao_cm-1",
        "amplitude": "Amplitude",
        "fwhm": "FWHM_cm-1",
        "area": "Area"
    }, inplace=True)
    df_amostra.index.name = "Pico"
    df_amostra.index = df_amostra.index + 1  # começa em 1, não em 0

    caminho_csv = os.path.join(pasta_fotos_amostra, f"detalhe_picos_{nome_amostra}.csv")
    df_amostra.to_csv(caminho_csv, sep=";", decimal=",", encoding="utf-8-sig")


# =========================================================
# AGRUPA PICOS DE TODAS AS AMOSTRAS EM "FAMÍLIAS" (COLUNAS FIXAS)
# =========================================================
def agrupar_picos_entre_amostras(todos_resultados, tolerancia):
    """
    todos_resultados: dict { nome_amostra: [ {"posicao":..,"amplitude":..,"fwhm":..,"area":..}, ... ] }

    Junta todos os picos de todas as amostras e agrupa (clustering hierárquico 1D)
    os que estão a menos de `tolerancia` cm^-1 uns dos outros, tratando-os como
    o "mesmo pico". Para cada amostra e cada família de pico, guarda o pico inteiro
    (posição, amplitude, fwhm, área) mais próximo da posição média do cluster.

    Retorna um dict: { "Posicao": df, "Amplitude": df, "FWHM": df, "Area": df }
    onde cada df tem uma linha por amostra e uma coluna por família de pico.
    """
    todas_posicoes = []
    origem = []  # (amostra, indice do pico dentro da amostra)
    for amostra, picos in todos_resultados.items():
        for idx, p in enumerate(picos):
            todas_posicoes.append(p["posicao"])
            origem.append((amostra, idx))

    propriedades = ["posicao", "amplitude", "fwhm", "area"]
    nomes_tabelas = {"posicao": "Posicao", "amplitude": "Amplitude", "fwhm": "FWHM", "area": "Area"}

    if len(todas_posicoes) == 0:
        return {nomes_tabelas[p]: pd.DataFrame() for p in propriedades}

    posicoes_arr = np.array(todas_posicoes).reshape(-1, 1)

    if len(posicoes_arr) == 1:
        clusters = np.array([1])
    else:
        Z = linkage(posicoes_arr, method="average")
        clusters = fcluster(Z, t=tolerancia, criterion="distance")

    # posição média de cada cluster -> vira o nome da coluna
    cluster_pos = {}
    for c in np.unique(clusters):
        pos_vals = posicoes_arr[clusters == c].flatten()
        cluster_pos[c] = np.mean(pos_vals)

    clusters_ordenados = sorted(cluster_pos, key=lambda c: cluster_pos[c])
    nome_coluna = {c: f"Pico_{cluster_pos[c]:.0f}cm-1" for c in clusters_ordenados}

    # tabela auxiliar: para cada amostra/coluna, guarda o PICO INTEIRO (dict) escolhido
    tabela_picos = {amostra: {} for amostra in todos_resultados}

    for (amostra, idx), cluster_id in zip(origem, clusters):
        pico = todos_resultados[amostra][idx]
        col = nome_coluna[cluster_id]
        media = cluster_pos[cluster_id]

        # se já existe pico dessa amostra nessa coluna, mantém o mais próximo da média do cluster
        if col in tabela_picos[amostra]:
            atual = tabela_picos[amostra][col]
            if abs(pico["posicao"] - media) < abs(atual["posicao"] - media):
                tabela_picos[amostra][col] = pico
        else:
            tabela_picos[amostra][col] = pico

    colunas_ordenadas = [nome_coluna[c] for c in clusters_ordenados]

    # a partir da tabela de picos escolhidos, monta uma tabela numérica por propriedade
    tabelas = {}
    for prop in propriedades:
        dados = {}
        for amostra, colunas in tabela_picos.items():
            dados[amostra] = {col: pico[prop] for col, pico in colunas.items()}
        df = pd.DataFrame.from_dict(dados, orient="index")
        df = df.reindex(columns=colunas_ordenadas)
        df.index.name = "Amostra"
        df = df.sort_index()
        tabelas[nomes_tabelas[prop]] = df

    return tabelas


# =========================================================
# MONTA A TABELA "LONGA" COM TODOS OS PICOS DE TODAS AS AMOSTRAS
# (uma linha por pico, sem alinhamento/clustering - dado bruto)
# =========================================================
def montar_tabela_detalhada(todos_resultados):
    linhas = []
    for amostra, picos in todos_resultados.items():
        for i, p in enumerate(picos, start=1):
            linhas.append({
                "Amostra": amostra,
                "Pico": i,
                "Regiao": p.get("regiao"),
                "Posicao_cm-1": p["posicao"],
                "Amplitude": p["amplitude"],
                "FWHM_cm-1": p["fwhm"],
                "Area": p["area"],
            })
    return pd.DataFrame(linhas)


# =========================================================
# LOOP PRINCIPAL - PROCESSA TODOS OS ARQUIVOS DA PASTA
# =========================================================
def main():
    arquivos = sorted(glob.glob(os.path.join(PASTA_ARQUIVOS, "*.txt")))

    if not arquivos:
        print(f"Nenhum arquivo .txt encontrado em '{PASTA_ARQUIVOS}'.")
        return

    todos_resultados = {}
    falhas = []

    for caminho in arquivos:
        nome_amostra = os.path.splitext(os.path.basename(caminho))[0]
        sample_id = get_sample_id(caminho)
        pasta_fotos_amostra = os.path.join(PASTA_FOTOS, sample_id)

        print(f"\n=== Processando: {nome_amostra} (Fotos/{sample_id}) ===")

        try:
            picos = processar_espectro(caminho, pasta_fotos_amostra)
            if len(picos) == 0:
                print(f"  [aviso] Nenhum pico válido encontrado em {nome_amostra}")
            todos_resultados[nome_amostra] = picos
        except Exception as e:
            print(f"  [ERRO] Falha ao processar {nome_amostra}: {e}")
            falhas.append(nome_amostra)

    # --- planilha final alinhada por pico: uma aba por propriedade ---
    tabelas = agrupar_picos_entre_amostras(todos_resultados, TOLERANCIA_CLUSTER)
    with pd.ExcelWriter(ARQUIVO_SAIDA_XLSX, engine="openpyxl") as writer:
        for nome_aba, df in tabelas.items():
            df.to_excel(writer, sheet_name=nome_aba)

    # --- arquivo detalhado: todos os picos de todas as amostras, um por linha ---
    df_detalhado = montar_tabela_detalhada(todos_resultados)
    df_detalhado.to_excel(ARQUIVO_DETALHADO_XLSX, index=False)

    print("\n===================================")
    print(f"Processamento concluído: {len(todos_resultados)}/{len(arquivos)} amostras processadas com sucesso.")
    if falhas:
        print(f"Falharam: {falhas}")
    print(f"Planilha alinhada (Posicao/Amplitude/FWHM/Area) salva em: {ARQUIVO_SAIDA_XLSX}")
    print(f"Planilha detalhada (todos os picos, todas as amostras) salva em: {ARQUIVO_DETALHADO_XLSX}")
    print(f"Detalhe por amostra salvo dentro de cada '{PASTA_FOTOS}/<id>/detalhe_picos_<amostra>.csv'")


if __name__ == "__main__":
    main()