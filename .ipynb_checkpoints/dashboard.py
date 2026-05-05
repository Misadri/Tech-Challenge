"""
Tech Challenge - Dashboard de Resultados
Execute: python dashboard.py
Abre automaticamente no navegador em http://localhost:8000
"""

import http.server
import webbrowser
import threading
import os

# ── 1. Roda o notebook e coleta métricas ─────────────────────────────────────
print("Treinando modelos e gerando resultados...")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64, io, json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import shap

sns.set_theme(style='whitegrid')

# Dados
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df[list(data.feature_names)]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

metrics = {
    "lr": {
        "acc": accuracy_score(y_test, y_pred_lr),
        "rec": recall_score(y_test, y_pred_lr, pos_label=0),
        "f1":  f1_score(y_test, y_pred_lr, pos_label=0),
    },
    "dt": {
        "acc": accuracy_score(y_test, y_pred_dt),
        "rec": recall_score(y_test, y_pred_dt, pos_label=0),
        "f1":  f1_score(y_test, y_pred_dt, pos_label=0),
    }
}

# ── 2. Helpers ────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='none', transparent=True)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

# ── 3. Gráficos ───────────────────────────────────────────────────────────────

# 3a. Distribuição das classes
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = df['target'].map({0:'Maligno', 1:'Benigno'}).value_counts()
bars = ax.bar(counts.index, counts.values, color=['#ef4444','#22c55e'],
              edgecolor='white', linewidth=1.5, width=0.5)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v+4, str(v),
            ha='center', fontweight='bold', color='white', fontsize=11)
ax.set_facecolor('none'); fig.patch.set_alpha(0)
ax.tick_params(colors='white'); ax.yaxis.label.set_color('white')
ax.spines[:].set_color('#ffffff22')
ax.set_title('Distribuição dos Diagnósticos', color='white', fontweight='bold')
img_dist = fig_to_b64(fig)

# 3b. Matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, y_pred, title, color in zip(
    axes,
    [y_pred_lr, y_pred_dt],
    ['Regressão Logística', 'Árvore de Decisão'],
    ['Blues', 'Greens']
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['Maligno','Benigno'],
                yticklabels=['Maligno','Benigno'],
                linewidths=2, linecolor='#0f172a',
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
    ax.set_title(title, color='white', fontweight='bold')
    ax.set_xlabel('Previsto', color='#94a3b8')
    ax.set_ylabel('Real', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_facecolor('none')
fig.patch.set_alpha(0)
img_cm = fig_to_b64(fig)

# 3c. Feature importance
importances = pd.Series(dt.feature_importances_, index=data.feature_names)
top = importances.sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(top.index, top.values, color='#38bdf8', edgecolor='none')
ax.set_facecolor('none'); fig.patch.set_alpha(0)
ax.tick_params(colors='white'); ax.xaxis.label.set_color('#94a3b8')
ax.spines[:].set_color('#ffffff22')
ax.set_title('Feature Importance (Árvore de Decisão)', color='white', fontweight='bold')
img_fi = fig_to_b64(fig)

# 3d. SHAP
explainer = shap.LinearExplainer(lr, X_train_s)
shap_vals  = explainer.shap_values(X_test_s)
fig, ax = plt.subplots(figsize=(7, 4))
shap.summary_plot(shap_vals, X_test, feature_names=list(data.feature_names),
                  plot_type='bar', show=False, color='#f472b6')
ax = plt.gca()
ax.set_facecolor('none'); fig.patch.set_alpha(0)
ax.tick_params(colors='white'); ax.xaxis.label.set_color('#94a3b8')
ax.spines[:].set_color('#ffffff22')
ax.set_title('SHAP – Importância Global (Reg. Logística)', color='white', fontweight='bold')
img_shap = fig_to_b64(fig)

print("Modelos treinados e gráficos gerados!")

# ── 4. HTML ───────────────────────────────────────────────────────────────────
def pct(v): return f"{v*100:.1f}%"

HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tech Challenge – Saúde da Mulher</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #080f1e;
    --surface: #0f1f38;
    --card:    #132040;
    --border:  #1e3a5f;
    --accent1: #38bdf8;
    --accent2: #f472b6;
    --accent3: #22c55e;
    --danger:  #ef4444;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── grid background ── */
  body::before {{
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(56,189,248,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56,189,248,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }}

  .wrap {{ position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 0 24px 80px; }}

  /* ── header ── */
  header {{
    padding: 60px 0 48px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 52px;
    display: flex; flex-direction: column; gap: 12px;
    animation: fadeUp .6s ease both;
  }}
  .tag {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .15em;
    color: var(--accent1);
    text-transform: uppercase;
  }}
  h1 {{
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 700;
    line-height: 1.15;
  }}
  h1 span {{ color: var(--accent2); }}
  .subtitle {{ color: var(--muted); font-size: 1rem; font-weight: 300; max-width: 560px; line-height: 1.6; }}

  /* ── sections ── */
  section {{ margin-bottom: 56px; animation: fadeUp .5s ease both; }}
  section:nth-child(2) {{ animation-delay:.1s }}
  section:nth-child(3) {{ animation-delay:.2s }}
  section:nth-child(4) {{ animation-delay:.3s }}
  section:nth-child(5) {{ animation-delay:.4s }}

  .section-label {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: .2em;
    color: var(--accent1);
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 24px; color: var(--text); }}

  /* ── metric cards ── */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }}
  .model-block {{ display: flex; flex-direction: column; gap: 12px; }}
  .model-title {{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    letter-spacing: .1em;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: transform .2s, border-color .2s;
  }}
  .metric-card:hover {{ transform: translateY(-3px); border-color: var(--accent1); }}
  .metric-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-line, var(--accent1));
  }}
  .metric-card.pink  {{ --accent-line: var(--accent2); }}
  .metric-card.green {{ --accent-line: var(--accent3); }}
  .metric-card.red   {{ --accent-line: var(--danger); }}
  .metric-label {{ font-size: .75rem; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .08em; }}
  .metric-value {{ font-family: var(--mono); font-size: 2rem; font-weight: 700; color: var(--text); }}
  .metric-value small {{ font-size: .85rem; color: var(--muted); }}

  /* ── chart cards ── */
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
  }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
  .chart-title {{ font-size: .9rem; font-weight: 600; color: var(--muted); margin-bottom: 16px; font-family: var(--mono); letter-spacing: .06em; }}

  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media(max-width:680px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

  /* ── discussion ── */
  .discussion-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
  }}
  .disc-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px;
    transition: border-color .2s;
  }}
  .disc-card:hover {{ border-color: var(--accent1); }}
  .disc-icon {{ font-size: 1.6rem; margin-bottom: 10px; }}
  .disc-title {{ font-weight: 600; margin-bottom: 8px; font-size: .95rem; }}
  .disc-text {{ color: var(--muted); font-size: .85rem; line-height: 1.6; }}

  /* ── footer ── */
  footer {{
    border-top: 1px solid var(--border);
    padding-top: 28px;
    color: var(--muted);
    font-size: .8rem;
    font-family: var(--mono);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
  }}

  @keyframes fadeUp {{
    from {{ opacity:0; transform: translateY(20px); }}
    to   {{ opacity:1; transform: translateY(0); }}
  }}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <span class="tag">PosTech · Tech Challenge · Fase 1</span>
    <h1>Diagnóstico de <span>Câncer de Mama</span><br>com Machine Learning</h1>
    <p class="subtitle">Sistema de suporte ao diagnóstico para redes hospitalares especializadas em saúde da mulher. Dataset: Breast Cancer Wisconsin (UCI / Kaggle).</p>
  </header>

  <!-- Métricas -->
  <section>
    <div class="section-label">01 — Avaliação</div>
    <h2>Métricas dos Modelos</h2>
    <div class="metrics-grid">

      <div class="model-block">
        <div class="model-title">// Regressão Logística</div>
        <div class="metric-card">
          <div class="metric-label">Acurácia</div>
          <div class="metric-value">{pct(metrics['lr']['acc'])}</div>
        </div>
        <div class="metric-card pink">
          <div class="metric-label">Recall (Maligno)</div>
          <div class="metric-value">{pct(metrics['lr']['rec'])}</div>
        </div>
        <div class="metric-card green">
          <div class="metric-label">F1-Score (Maligno)</div>
          <div class="metric-value">{pct(metrics['lr']['f1'])}</div>
        </div>
      </div>

      <div class="model-block">
        <div class="model-title">// Árvore de Decisão</div>
        <div class="metric-card">
          <div class="metric-label">Acurácia</div>
          <div class="metric-value">{pct(metrics['dt']['acc'])}</div>
        </div>
        <div class="metric-card pink">
          <div class="metric-label">Recall (Maligno)</div>
          <div class="metric-value">{pct(metrics['dt']['rec'])}</div>
        </div>
        <div class="metric-card green">
          <div class="metric-label">F1-Score (Maligno)</div>
          <div class="metric-value">{pct(metrics['dt']['f1'])}</div>
        </div>
      </div>

      <div class="model-block">
        <div class="model-title">// Dataset</div>
        <div class="metric-card">
          <div class="metric-label">Total de amostras</div>
          <div class="metric-value">569</div>
        </div>
        <div class="metric-card red">
          <div class="metric-label">Casos Malignos</div>
          <div class="metric-value">212</div>
        </div>
        <div class="metric-card green">
          <div class="metric-label">Casos Benignos</div>
          <div class="metric-value">357</div>
        </div>
      </div>

    </div>
  </section>

  <!-- Distribuição -->
  <section>
    <div class="section-label">02 — Exploração</div>
    <h2>Distribuição dos Dados</h2>
    <div class="chart-card">
      <div class="chart-title">distribuicao_diagnosticos.png</div>
      <img src="data:image/png;base64,{img_dist}" alt="Distribuição">
    </div>
  </section>

  <!-- Matrizes de confusão -->
  <section>
    <div class="section-label">03 — Avaliação Visual</div>
    <h2>Matrizes de Confusão</h2>
    <div class="chart-card">
      <div class="chart-title">matrizes_confusao.png</div>
      <img src="data:image/png;base64,{img_cm}" alt="Matrizes de Confusão">
    </div>
  </section>

  <!-- Explicabilidade -->
  <section>
    <div class="section-label">04 — Explicabilidade</div>
    <h2>Feature Importance & SHAP</h2>
    <div class="two-col">
      <div class="chart-card">
        <div class="chart-title">feature_importance.png — Árvore de Decisão</div>
        <img src="data:image/png;base64,{img_fi}" alt="Feature Importance">
      </div>
      <div class="chart-card">
        <div class="chart-title">shap_summary.png — Regressão Logística</div>
        <img src="data:image/png;base64,{img_shap}" alt="SHAP">
      </div>
    </div>
  </section>

  <!-- Discussão -->
  <section>
    <div class="section-label">05 — Discussão Crítica</div>
    <h2>Interpretação dos Resultados</h2>
    <div class="discussion-grid">
      <div class="disc-card">
        <div class="disc-icon">🎯</div>
        <div class="disc-title">Por que Recall?</div>
        <div class="disc-text">Em diagnóstico médico, um falso negativo (maligno classificado como benigno) é muito mais perigoso que um falso positivo. Por isso o Recall da classe Maligno é a métrica prioritária.</div>
      </div>
      <div class="disc-card">
        <div class="disc-icon">🧠</div>
        <div class="disc-title">Explicabilidade</div>
        <div class="disc-text">SHAP e Feature Importance revelam que "worst concave points" e "worst perimeter" são as features mais determinantes — alinhado com o conhecimento clínico sobre morfologia tumoral.</div>
      </div>
      <div class="disc-card">
        <div class="disc-icon">🏥</div>
        <div class="disc-title">Uso na Prática</div>
        <div class="disc-text">O modelo pode ser usado como ferramenta de triagem inicial, alertando profissionais sobre casos de alto risco. O médico sempre tem a palavra final no diagnóstico.</div>
      </div>
      <div class="disc-card">
        <div class="disc-icon">⚠️</div>
        <div class="disc-title">Limitações</div>
        <div class="disc-text">Dataset pequeno (569 amostras) de uma única fonte. Precisa de validação em dados externos e auditoria regular antes de qualquer implantação clínica.</div>
      </div>
    </div>
  </section>

  <footer>
    <span>Tech Challenge · Fase 1 · PosTech</span>
    <span>Breast Cancer Wisconsin Dataset · scikit-learn · SHAP</span>
  </footer>

</div>
</body>
</html>"""

# ── 5. Salva o HTML ───────────────────────────────────────────────────────────
with open('dashboard.html', 'w', encoding='utf-8') as f:
    f.write(HTML)

# ── 6. Servidor e abre o navegador ───────────────────────────────────────────
PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

handler = http.server.SimpleHTTPRequestHandler
httpd   = http.server.HTTPServer(('', PORT), handler)

url = f'http://localhost:{PORT}/dashboard.html'
print(f'\n✅ Dashboard pronto! Abrindo em {url}')
print('   Pressione Ctrl+C para encerrar o servidor.\n')

threading.Timer(1.0, lambda: webbrowser.open(url)).start()
httpd.serve_forever()
