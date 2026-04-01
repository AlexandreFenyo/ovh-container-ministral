#!/usr/bin/env python3
import argparse
import html
import pathlib
import re
from dataclasses import dataclass


@dataclass
class PositiveMetric:
    label: str
    average_score: float
    count: int
    total: int


@dataclass
class NegativeMetric:
    label: str
    correct: int
    total: int
    accuracy_pct: float


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _model_sort_key(label: str):
    if label == "base":
        return (-1, 0, label)
    if label.startswith("checkpoint-"):
        try:
            return (0, int(label.split("-", 1)[1]), label)
        except ValueError:
            return (0, 10**9, label)
    if label == "final":
        return (1, 0, label)
    return (2, 10**9, label)


def _stage_color(label: str) -> str:
    if label == "base":
        return "#64748b"
    if label == "final":
        return "#16a34a"
    return "#f59e0b"


def _parse_positive_log(path: pathlib.Path) -> dict[str, PositiveMetric]:
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\[split=.*\]\[checkpoint=([^\]]+)\] \d+/(\d+) -> ([0-9.]+)$", line)
        if not match:
            continue
        label = match.group(1)
        total = int(match.group(2))
        score = float(match.group(3))
        scores[label] = scores.get(label, 0.0) + score
        counts[label] = counts.get(label, 0) + 1
        totals[label] = total

    metrics: dict[str, PositiveMetric] = {}
    for label in scores:
        metrics[label] = PositiveMetric(
            label=label,
            average_score=scores[label] / counts[label],
            count=counts[label],
            total=totals[label],
        )
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
                label=current,
                correct=int(match.group(1)),
                total=int(match.group(2)),
                accuracy_pct=float(match.group(3)),
            )
    return metrics


def _svg_chart(title, subtitle, values, y_max, y_label, color, value_formatter):
    width = 900
    height = 480
    pad_left = 90
    pad_right = 28
    pad_top = 38
    pad_bottom = 78
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    steps = max(len(values) - 1, 1)

    def x_pos(idx):
        return pad_left + (idx / steps) * inner_w

    def y_pos(value):
        return pad_top + inner_h - (value / y_max) * inner_h

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="18" y="24" font-size="22" font-weight="700" fill="#111827">{_escape(title)}</text>')
    parts.append(f'<text x="18" y="46" font-size="13" fill="#6b7280">{_escape(subtitle)}</text>')

    for tick in [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]:
        yy = y_pos(tick)
        parts.append(
            f'<line x1="{pad_left}" y1="{yy:.2f}" x2="{pad_left + inner_w}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        label = f"{tick:.2f}" if y_max <= 1.0 else f"{tick:.0f}%"
        if y_max > 1.0:
            label = f"{tick:.0f}%"
        parts.append(
            f'<text x="{pad_left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7280">{label}</text>'
        )

    xs = []
    ys = []
    for idx, (label, value) in enumerate(values):
        xx = x_pos(idx)
        yy = y_pos(value)
        xs.append(xx)
        ys.append(yy)
        parts.append(
            f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5.2" fill="{color}" stroke="white" stroke-width="2"/>'
        )
        if label in {"base", "final"} or label.endswith("625") or label.endswith("375"):
            parts.append(
                f'<text x="{xx:.2f}" y="{height - 24}" text-anchor="middle" font-size="10" fill="#374151">{_escape(label)}</text>'
            )

    for idx in range(len(xs) - 1):
        parts.append(
            f'<line x1="{xs[idx]:.2f}" y1="{ys[idx]:.2f}" x2="{xs[idx+1]:.2f}" y2="{ys[idx+1]:.2f}" stroke="{color}" stroke-width="3.5"/>'
        )

    parts.append(f'<text x="{pad_left + inner_w / 2:.2f}" y="{height - 10}" text-anchor="middle" font-size="13" fill="#374151">{_escape(y_label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_scatter(title, subtitle, points, annotate_labels=None):
    width = 900
    height = 520
    pad_left = 84
    pad_right = 28
    pad_top = 38
    pad_bottom = 72
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom

    def x_pos(value):
        return pad_left + value * inner_w

    def y_pos(value):
        return pad_top + inner_h - (value / 100.0) * inner_h

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="18" y="24" font-size="22" font-weight="700" fill="#111827">{_escape(title)}</text>')
    parts.append(f'<text x="18" y="46" font-size="13" fill="#6b7280">{_escape(subtitle)}</text>')

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xx = x_pos(tick)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{pad_top}" x2="{xx:.2f}" y2="{pad_top + inner_h}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{height - 22}" text-anchor="middle" font-size="11" fill="#6b7280">{tick:.2f}</text>'
        )

    for tick in [0, 25, 50, 75, 100]:
        yy = y_pos(tick)
        parts.append(
            f'<line x1="{pad_left}" y1="{yy:.2f}" x2="{pad_left + inner_w}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7280">{tick:.0f}%</text>'
        )

    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + inner_h}" x2="{pad_left + inner_w}" y2="{pad_top + inner_h}" stroke="#111827" stroke-width="1.4"/>'
    )
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + inner_h}" stroke="#111827" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{pad_left + inner_w / 2:.2f}" y="{height - 6}" text-anchor="middle" font-size="13" fill="#374151">Score positif moyen</text>'
    )
    parts.append(
        f'<text transform="translate(18 {pad_top + inner_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="13" fill="#374151">Exactitude négative</text>'
    )

    annotate_labels = set(annotate_labels or ())
    for label, positive_value, negative_value in points:
        xx = x_pos(positive_value)
        yy = y_pos(negative_value)
        fill = _stage_color(label)
        parts.append(
            f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="6.2" fill="{fill}" stroke="white" stroke-width="2"/>'
        )
        if label in annotate_labels:
            parts.append(
                f'<text x="{xx + 9:.2f}" y="{yy - 10:.2f}" font-size="11" fill="#111827">{_escape(label)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_html(positive_log: pathlib.Path, negative_log: pathlib.Path, output_name: str):
    positive = _parse_positive_log(positive_log)
    negative = _parse_negative_log(negative_log)

    common = sorted(set(positive) & set(negative), key=_model_sort_key)
    if not common:
        raise RuntimeError("No common model labels found between the logs.")

    completed_positive = [
        label
        for label in common
        if positive[label].count == positive[label].total
    ]
    completed_negative = [
        label for label in common if label in negative
    ]

    rows = []
    for label in common:
        pos = positive[label]
        neg = negative[label]
        rows.append(
            {
                "label": label,
                "positive": pos.average_score,
                "negative": neg.accuracy_pct,
                "positive_count": pos.count,
                "positive_total": pos.total,
                "negative_correct": neg.correct,
                "negative_total": neg.total,
                "balanced": (pos.average_score + neg.accuracy_pct / 100.0) / 2.0,
                "positive_delta": pos.average_score - positive["base"].average_score,
                "negative_delta": neg.accuracy_pct - negative["base"].accuracy_pct,
            }
        )

    complete_rows = [row for row in rows if row["label"] in completed_positive]
    best_positive = max(complete_rows, key=lambda row: row["positive"])
    best_negative = max(complete_rows, key=lambda row: row["negative"])
    best_balanced = max(complete_rows, key=lambda row: row["balanced"])
    base_row = next(row for row in rows if row["label"] == "base")
    final_row = next((row for row in rows if row["label"] == "final"), None)
    checkpoint_750_row = positive.get("checkpoint-750")
    checkpoint_750_value = checkpoint_750_row.average_score if checkpoint_750_row else 0.0

    positive_chart = _svg_chart(
        "Score positif moyen",
        "results5.log, checkpoints complètement terminés",
        [(row["label"], row["positive"]) for row in complete_rows],
        1.0,
        "Positif moyen",
        "#f59e0b",
        lambda value: f"{value:.3f}",
    )
    negative_chart = _svg_chart(
        "Exactitude négative",
        "results4.log, mêmes checkpoints complètement terminés",
        [(row["label"], row["negative"]) for row in complete_rows],
        100.0,
        "Exact-match négatif",
        "#38bdf8",
        lambda value: f"{value:.2f}%",
    )
    scatter_chart = _svg_scatter(
        "Projection 2D des checkpoints",
        "Chaque point combine le score positif moyen et l'exactitude négative. Le coin supérieur droit est le meilleur.",
        [(row["label"], row["positive"], row["negative"]) for row in complete_rows],
        annotate_labels={base_row["label"], best_positive["label"], best_negative["label"], best_balanced["label"]},
    )

    rows_html = []
    for row in complete_rows:
        rows_html.append(
            f"""
            <tr>
              <td><span class="tag" style="background:{_stage_color(row['label'])}">{_escape(row['label'])}</span></td>
              <td>{row['positive']:.3f}</td>
              <td>{row['negative']:.2f}%</td>
              <td>{row['balanced']:.3f}</td>
              <td class="delta {'up' if row['positive_delta'] >= 0 else 'down'}">{row['positive_delta']:+.3f}</td>
              <td class="delta {'up' if row['negative_delta'] >= 0 else 'down'}">{row['negative_delta']:+.2f} pts</td>
            </tr>
            """
        )

    extra_negative_rows = []
    for row in rows:
        if row["label"] not in completed_positive and row["label"] != "base":
            extra_negative_rows.append(
                f"""
                <tr>
                  <td><span class="tag" style="background:{_stage_color(row['label'])}">{_escape(row['label'])}</span></td>
                  <td colspan="2" class="muted">pas encore comparable côté positif</td>
                  <td>{row['negative']:.2f}%</td>
                  <td colspan="2" class="muted">{row['negative_correct']}/{row['negative_total']}</td>
                </tr>
                """
            )

    partial_note = ""
    if "checkpoint-750" in positive:
        metric = positive["checkpoint-750"]
        partial_note = (
            f"<p class=\"warning\">checkpoint-750 est partiel dans { _escape(positive_log.name) } "
            f"({metric.count}/{metric.total} lignes) et n'est pas inclus dans les courbes principales.</p>"
        )

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rapport checkpoints results4/results5</title>
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
      max-width: 1520px;
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
      max-width: 1200px;
      line-height: 1.55;
      margin-bottom: 14px;
    }}
    .warning {{
      color: #fbbf24;
      margin: 8px 0 0;
      line-height: 1.5;
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
      font-size: 24px;
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
    .muted {{ color: var(--muted); text-align: left; }}
    .footer {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 14px;
    }}
    @media (max-width: 760px) {{
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Rapport croisé des checkpoints</h1>
    <div class="subtitle">
      Le score positif est extrait de <strong>{_escape(positive_log.name)}</strong> et l'exactitude négative de <strong>{_escape(negative_log.name)}</strong>.
      Le rapport se concentre sur les checkpoints complètement terminés dans les deux fichiers.
    </div>
    {partial_note}

    <div class="grid">
      <div class="card">
        <div class="k">Base</div>
        <div class="v">{base_row['positive']:.3f} / {base_row['negative']:.2f}%</div>
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
        <div class="k">Checkpoint-750</div>
        <div class="v">{checkpoint_750_value:.3f} *</div>
      </div>
    </div>

    <div class="section">
      <h2>Score positif moyen</h2>
      <p>Courbe des checkpoints complets du validateur positif. Le meilleur point terminé est mis en évidence indirectement dans le tableau.</p>
      <div class="svg-wrap">{positive_chart}</div>
    </div>

    <div class="section">
      <h2>Exactitude négative</h2>
      <p>Courbe des mêmes checkpoints, calculée depuis la validation négative. C'est ici que le meilleur point arrive très tôt.</p>
      <div class="svg-wrap">{negative_chart}</div>
    </div>

    <div class="section">
      <h2>Projection 2D</h2>
      <p>Ce nuage de points permet de voir quels checkpoints avancent dans les deux directions à la fois. Plus un point est en haut à droite, meilleur est le compromis.</p>
      <div class="svg-wrap">{scatter_chart}</div>
    </div>

    <div class="section">
      <h2>Tableau croisé</h2>
      <p>Chaque ligne compare les deux métriques sur les checkpoints complets dans les deux logs. Le score équilibré est une moyenne simple des deux axes normalisés.</p>
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
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Checkpoints négatifs supplémentaires</h2>
      <p>Ces checkpoints sont terminés côté négatif dans { _escape(negative_log.name) }, mais n'ont pas encore de score positif complet dans { _escape(positive_log.name) }.</p>
      <table>
        <thead>
          <tr>
            <th>Modèle</th>
            <th colspan="2">Statut positif</th>
            <th>Négatif</th>
            <th colspan="2">Détails négatifs</th>
          </tr>
        </thead>
        <tbody>
          {''.join(extra_negative_rows) if extra_negative_rows else '<tr><td colspan="6" class="muted">Aucun checkpoint supplémentaire terminé côté négatif.</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="footer">
      Fichier généré à partir de {_escape(positive_log.name)} et {_escape(negative_log.name)}.
    </div>
  </div>
</body>
</html>
"""
    return html_doc


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML report comparing positive and negative validation logs."
    )
    parser.add_argument("--positive-log", type=pathlib.Path, default=pathlib.Path("results5.log"))
    parser.add_argument("--negative-log", type=pathlib.Path, default=pathlib.Path("results4.log"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results4-results5-report.html"))
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
