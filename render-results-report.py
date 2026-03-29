#!/usr/bin/env python3
import argparse
import html
import math
import pathlib
import re
from dataclasses import dataclass


@dataclass
class PositiveMetric:
    label: str
    average_score: float


@dataclass
class NegativeMetric:
    label: str
    correct: int
    total: int
    accuracy_pct: float


def _parse_positive_log(path: pathlib.Path) -> dict[str, PositiveMetric]:
    metrics: dict[str, PositiveMetric] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^== (.+) ==$", line)
        if match:
            current = match.group(1)
            continue
        match = re.match(r"^Average score: ([0-9.]+)$", line)
        if match and current:
            metrics[current] = PositiveMetric(current, float(match.group(1)))
    return metrics


def _parse_negative_log(path: pathlib.Path) -> dict[str, NegativeMetric]:
    metrics: dict[str, NegativeMetric] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^== (.+) ==$", line)
        if match:
            current = match.group(1)
            continue
        match = re.match(r"^Accuracy: (\d+)/(\d+) \(([0-9.]+)%\)$", line)
        if match and current:
            metrics[current] = NegativeMetric(
                current,
                int(match.group(1)),
                int(match.group(2)),
                float(match.group(3)),
            )
    return metrics


def _model_sort_key(label: str):
    if label == "base":
        return (-1, 0, label)
    if label.startswith("checkpoint-"):
        try:
            return (0, int(label.split("-", 1)[1]), label)
        except ValueError:
            return (0, math.inf, label)
    if label == "final":
        return (1, 0, label)
    return (2, math.inf, label)


def _stage_color(label: str) -> str:
    if label == "base":
        return "#6b7280"
    if label == "final":
        return "#16a34a"
    return "#2563eb"


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _build_bar_rows(metrics, metric_name, value_formatter, max_value):
    rows = []
    for label, value, extra in metrics:
        width_pct = 0 if max_value == 0 else (value / max_value) * 100
        rows.append(
            f"""
            <div class=\"bar-row\">
              <div class=\"bar-label\">{_escape(label)}</div>
              <div class=\"bar-track\">
                <div class=\"bar-fill\" style=\"width: {width_pct:.2f}%; background: {_stage_color(label)};\"></div>
              </div>
              <div class=\"bar-value\">{_escape(value_formatter(value, extra))}</div>
            </div>
            """
        )
    return "\n".join(rows)


def _render_scatter(positive, negative):
    width = 900
    height = 560
    pad_left = 90
    pad_right = 30
    pad_top = 30
    pad_bottom = 80
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom

    def x_to_px(x):
        return pad_left + (x * inner_w)

    def y_to_px(y):
        return pad_top + inner_h - (y / 100.0 * inner_h)

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Scatter plot comparing positive and negative performance">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = x_to_px(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{pad_top}" x2="{x:.2f}" y2="{pad_top + inner_h}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{height - 48}" text-anchor="middle" font-size="12" fill="#6b7280">{tick:.2f}</text>'
        )

    for tick in [0, 25, 50, 75, 100]:
        y = y_to_px(tick)
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.2f}" x2="{pad_left + inner_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#6b7280">{tick:.0f}%</text>'
        )

    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + inner_h}" x2="{pad_left + inner_w}" y2="{pad_top + inner_h}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + inner_h}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{pad_left + inner_w / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="14" fill="#111827">Score positif moyen</text>'
    )
    parts.append(
        f'<text transform="translate(22 {pad_top + inner_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="14" fill="#111827">Exactitude négative</text>'
    )

    for label in sorted(positive.keys(), key=_model_sort_key):
        p = positive[label].average_score
        n = negative[label].accuracy_pct
        x = x_to_px(p)
        y = y_to_px(n)
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{_stage_color(label)}" stroke="white" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + 10:.2f}" y="{y - 10:.2f}" font-size="12" fill="#111827">{_escape(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_html(positive_log: pathlib.Path, negative_log: pathlib.Path, output_name: str):
    positive = _parse_positive_log(positive_log)
    negative = _parse_negative_log(negative_log)
    labels = sorted(set(positive) & set(negative), key=_model_sort_key)
    if not labels:
        raise RuntimeError("No common model labels found between the two logs.")

    rows = []
    for label in labels:
        pos = positive[label]
        neg = negative[label]
        balanced = (pos.average_score + (neg.accuracy_pct / 100.0)) / 2.0
        rows.append(
            {
                "label": label,
                "positive": pos.average_score,
                "negative": neg.accuracy_pct,
                "balanced": balanced,
                "positive_delta": pos.average_score - positive["base"].average_score,
                "negative_delta": neg.accuracy_pct - negative["base"].accuracy_pct,
                "stage": _stage_color(label),
            }
        )

    best_positive = max(rows, key=lambda row: row["positive"])
    best_negative = max(rows, key=lambda row: row["negative"])
    best_balanced = max(rows, key=lambda row: row["balanced"])
    final_row = next(row for row in rows if row["label"] == "final")
    base_row = next(row for row in rows if row["label"] == "base")

    positive_rows = [
        (row["label"], row["positive"], None)
        for row in sorted(rows, key=lambda row: _model_sort_key(row["label"]))
    ]
    negative_rows = [
        (row["label"], row["negative"], None)
        for row in sorted(rows, key=lambda row: _model_sort_key(row["label"]))
    ]

    positive_bars = _build_bar_rows(
        positive_rows,
        "positive",
        lambda value, _extra: f"{value:.3f}",
        1.0,
    )
    negative_bars = _build_bar_rows(
        negative_rows,
        "negative",
        lambda value, _extra: f"{value:.2f}%",
        100.0,
    )

    table_rows = []
    for row in sorted(rows, key=lambda row: _model_sort_key(row["label"])):
        table_rows.append(
            f"""
            <tr>
              <td><span class=\"tag\" style=\"background:{row['stage']}\">{_escape(row['label'])}</span></td>
              <td>{row['positive']:.3f}</td>
              <td>{row['negative']:.2f}%</td>
              <td>{row['balanced']:.3f}</td>
              <td class=\"delta {'up' if row['positive_delta'] >= 0 else 'down'}\">{row['positive_delta']:+.3f}</td>
              <td class=\"delta {'up' if row['negative_delta'] >= 0 else 'down'}\">{row['negative_delta']:+.2f} pts</td>
            </tr>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Résultats FAQ MES</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #10182f;
      --panel-2: #0f1730;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --grid: #23304f;
      --border: #26324f;
      --good: #16a34a;
      --warn: #f59e0b;
      --bad: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #1f2a58 0, transparent 32%),
        radial-gradient(circle at top right, #11204f 0, transparent 28%),
        linear-gradient(180deg, #090d1a 0, #0b1020 60%, #090d1a 100%);
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 1100px;
      line-height: 1.55;
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card {{
      background: rgba(16, 24, 47, 0.88);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }}
    .card .k {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .card .v {{
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .section {{
      margin-top: 18px;
      background: rgba(16, 24, 47, 0.74);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px;
    }}
    .section h2 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .section p {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 210px 1fr 100px;
      gap: 14px;
      align-items: center;
      margin: 10px 0;
    }}
    .bar-label {{
      font-size: 14px;
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar-track {{
      width: 100%;
      height: 16px;
      border-radius: 999px;
      background: #17223d;
      border: 1px solid rgba(255, 255, 255, 0.06);
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .bar-value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--text);
      font-weight: 600;
    }}
    .svg-wrap {{
      overflow: auto;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.96);
      margin-top: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 11px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.18);
      text-align: right;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .tag {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      color: white;
    }}
    .delta.up {{ color: #86efac; }}
    .delta.down {{ color: #fca5a5; }}
    .footer {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 14px;
    }}
    @media (max-width: 760px) {{
      .bar-row {{
        grid-template-columns: 1fr;
      }}
      .bar-value {{
        text-align: left;
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Résultats comparatifs du fine-tuning</h1>
    <div class="subtitle">
      Le fichier <strong>{_escape(positive_log.name)}</strong> contient la validation positive avec juge OVH.
      Le fichier <strong>{_escape(negative_log.name)}</strong> contient la validation négative en exact-match.
      Les deux partagent les mêmes variantes: base, checkpoints intermédiaires et modèle final.
    </div>

    <div class="grid">
      <div class="card">
        <div class="k">Base positive</div>
        <div class="v">{base_row['positive']:.3f}</div>
      </div>
      <div class="card">
        <div class="k">Base négative</div>
        <div class="v">{base_row['negative']:.2f}%</div>
      </div>
      <div class="card">
        <div class="k">Meilleur positif</div>
        <div class="v">{best_positive['label']} ({best_positive['positive']:.3f})</div>
      </div>
      <div class="card">
        <div class="k">Meilleur négatif</div>
        <div class="v">{best_negative['label']} ({best_negative['negative']:.2f}%)</div>
      </div>
      <div class="card">
        <div class="k">Meilleur équilibré</div>
        <div class="v">{best_balanced['label']} ({best_balanced['balanced']:.3f})</div>
      </div>
      <div class="card">
        <div class="k">Final</div>
        <div class="v">{final_row['positive']:.3f} / {final_row['negative']:.2f}%</div>
      </div>
    </div>

    <div class="section">
      <h2>Score positif moyen</h2>
      <p>Chaque barre représente la moyenne des scores fournis par le juge OVH sur les exemples positifs.</p>
      {positive_bars}
    </div>

    <div class="section">
      <h2>Exactitude négative</h2>
      <p>Chaque barre représente le taux de réponses refusées correctement ou, plus généralement, l'exact-match sur les exemples négatifs.</p>
      {negative_bars}
    </div>

    <div class="section">
      <h2>Projection 2D</h2>
      <p>Le nuage de points compare directement la capacité à répondre positivement et à résister aux questions négatives. Le quadrant supérieur droit est le plus souhaitable.</p>
      <div class="svg-wrap">
        {_render_scatter(positive, negative)}
      </div>
    </div>

    <div class="section">
      <h2>Tableau récapitulatif</h2>
      <p>La colonne « équilibré » est la moyenne des deux métriques normalisées, utile seulement pour comparer visuellement les versions entre elles.</p>
      <table>
        <thead>
          <tr>
            <th>Modèle</th>
            <th>Positif</th>
            <th>Négatif</th>
            <th>Équilibré</th>
            <th>Δ positif vs base</th>
            <th>Δ négatif vs base</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
      <div class="footer">
        Valeurs extraites de {_escape(positive_log.name)} et {_escape(negative_log.name)}.
      </div>
    </div>
  </div>
</body>
</html>
"""
    return html_doc


def main():
    parser = argparse.ArgumentParser(
        description="Generate a visual comparison report from results logs."
    )
    parser.add_argument("--positive-log", type=pathlib.Path, default=pathlib.Path("results2.log"))
    parser.add_argument("--negative-log", type=pathlib.Path, default=pathlib.Path("results3.log"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results-comparison.html"))
    args = parser.parse_args()

    if not args.positive_log.exists():
        raise SystemExit(f"Missing positive log: {args.positive_log}")
    if not args.negative_log.exists():
        raise SystemExit(f"Missing negative log: {args.negative_log}")

    html_doc = _build_html(args.positive_log, args.negative_log, args.output.name)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
