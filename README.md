# Automação de Deconvolução de Espectros Raman — Li4Mo5O17

Pipeline em Python para automatizar a análise de espectros Raman (correção de
linha de base, detecção e ajuste multi-gaussiano de picos, consolidação em
planilha), com uma interface web de revisão semiautomática para corrigir o
resultado manualmente quando necessário.

## Estrutura de pastas esperada

```
projeto/
├── espectograma.py       # pipeline batch (processa todas as amostras)
├── revisar_picos.py       # interface de revisão (Streamlit)
├── arquivos/               # coloque aqui os .txt de entrada (um por amostra)
│   ├── Li4Mo5O17_01_....txt
│   ├── Li4Mo5O17_02_....txt
│   └── ...
└── Fotos/                  # gerado automaticamente pelo espectrograma.py
    ├── 01/
    ├── 02/
    └── ...
```

## Instalação

Requer Python 3.9+. Instale as dependências com:

```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl streamlit plotly
```

## Como executar

**1. Rodar o pipeline automatizado** (processa todos os `.txt` de `arquivos/`,
gera as figuras e os arquivos de dados em `Fotos/<amostra>/`, e as planilhas
consolidadas):

```bash
python espectograma.py
```

Saídas geradas:
- `Fotos/<id>/` — figuras de diagnóstico, `detalhe_picos_<amostra>.csv`, `dados_espectro.npz`, `regioes.json`, `picos_regiao_<n>.json`
- `picos_raman.xlsx` — planilha alinhada por pico (abas: Posicao, Amplitude, FWHM, Area)
- `picos_detalhado_todas_amostras.xlsx` — todos os picos de todas as amostras, um por linha

**2. Abrir a interface de revisão** (só funciona depois do passo 1, pois lê os
arquivos gerados em `Fotos/`):

```bash
streamlit run revisar_picos.py
```

Abre automaticamente no navegador (geralmente em `http://localhost:8501`).
Nela é possível navegar por amostra e região, ajustar picos com sliders,
adicionar um pico clicando no gráfico (com ajuste automático de amplitude e
largura), remover picos, salvar as alterações e regenerar as planilhas
consolidadas a partir do botão na barra lateral.

## Observações

- O `espectrograma.py` precisa rodar pelo menos uma vez antes da interface,
  pois é ele quem gera os arquivos que ela lê.
- Sempre que os picos forem editados na interface e salvos, use o botão
  **"Gerar planilhas consolidadas"** na barra lateral para atualizar
  `picos_raman.xlsx` e `picos_detalhado_todas_amostras.xlsx` com as edições.
