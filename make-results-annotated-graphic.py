#!/usr/bin/env python3
import argparse
import html
import pathlib
import re
from dataclasses import dataclass
from xml.sax.saxutils import escape


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


def parse_positive_log(path: pathlib.Path) -> dict[str, PositiveMetric]:
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


def parse_negative_log(path: pathlib.Path) -> dict[str, NegativeMetric]:
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


def model_sort_key(label: str):
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


def stage_color(label: str) -> str:
    if label == "base":
        return "#64748b"
    if label == "final":
        return "#16a34a"
    return "#2563eb"


def line_color(label: str) -> str:
    if label == "base":
        return "#475569"
    if label == "final":
        return "#0f766e"
    return "#f59e0b"


def fmt_score(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def svg_rect(x, y, w, h, rx=16, fill="#ffffff", stroke="none", sw=1, opacity=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def svg_text(x, y, text, size=16, weight=400, fill="#111827", anchor="start", opacity=1.0):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" opacity="{opacity}">{escape(text)}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke="#111827", sw=2, opacity=1.0, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{dash_attr}/>'
    )


def draw_chart(x, y, w, h, title, subtitle, values, value_formatter, y_max, y_min=0.0):
    left = x + 56
    right = x + w - 20
    top = y + 54
    bottom = y + h - 44
    inner_w = right - left
    inner_h = bottom - top

    pieces = []
    pieces.append(svg_rect(x, y, w, h, rx=22, fill="#0f172a", stroke="#253046", sw=1.4))
    pieces.append(svg_text(x + 24, y + 32, title, size=21, weight=700, fill="#e5e7eb"))
    pieces.append(svg_text(x + 24, y + 50, subtitle, size=12, weight=400, fill="#94a3b8", opacity=0.95))

    # Grid
    for frac, label in [(0, y_min), (0.25, None), (0.5, None), (0.75, None), (1.0, y_max)]:
        yy = bottom - frac * inner_h
        pieces.append(svg_line(left, yy, right, yy, stroke="#23304f", sw=1))
        if label is not None:
            pieces.append(svg_text(left - 10, yy + 4, value_formatter(label), size=12, fill="#94a3b8", anchor="end"))
        else:
            value = y_min + frac * (y_max - y_min)
            pieces.append(svg_text(left - 10, yy + 4, value_formatter(value), size=12, fill="#94a3b8", anchor="end"))

    # X axis labels and data
    xs = []
    ys = []
    labels = []
    n = max(len(values) - 1, 1)
    for i, (label, value) in enumerate(values):
        xx = left + (i / n) * inner_w
        yy = bottom - ((value - y_min) / (y_max - y_min)) * inner_h
        xs.append(xx)
        ys.append(yy)
        labels.append(label)

        pieces.append(svg_text(xx, bottom + 22, label, size=11, fill="#cbd5e1", anchor="middle"))
        pieces.append(f'<circle cx="{xx}" cy="{yy}" r="4.5" fill="{line_color(label)}" stroke="#ffffff" stroke-width="2"/>')

    for i in range(len(xs) - 1):
        pieces.append(svg_line(xs[i], ys[i], xs[i + 1], ys[i + 1], stroke="#f59e0b" if "checkpoint" in labels[i + 1] else line_color(labels[i + 1]), sw=3.5))

    # annotate first / best / final
    return pieces, xs, ys


def add_callout(pieces, x, y, title, body, color="#0f172a", text_color="#e5e7eb", line_color="#94a3b8"):
    box_w = 280
    box_h = 76
    box_x = x
    box_y = y
    pieces.append(svg_rect(box_x, box_y, box_w, box_h, rx=14, fill=color, stroke="#334155", sw=1.2, opacity=0.98))
    pieces.append(svg_text(box_x + 14, box_y + 26, title, size=15, weight=700, fill=text_color))
    pieces.append(svg_text(box_x + 14, box_y + 49, body, size=12, weight=400, fill="#cbd5e1"))


def build_svg(positive, negative, out_name):
    labels = sorted(set(positive) & set(negative), key=model_sort_key)
    if not labels:
        raise RuntimeError("No common model labels found.")

    rows = []
    for label in labels:
        p = positive[label].average_score
        n = negative[label].accuracy_pct
        rows.append(
            {
                "label": label,
                "positive": p,
                "negative": n,
                "balanced": (p + n / 100.0) / 2.0,
            }
        )

    base = next(r for r in rows if r["label"] == "base")
    final = next(r for r in rows if r["label"] == "final")
    best_pos = max(rows, key=lambda r: r["positive"])
    best_neg = max(rows, key=lambda r: r["negative"])
    best_bal = max(rows, key=lambda r: r["balanced"])

    positive_points = [(r["label"], r["positive"]) for r in rows]
    negative_points = [(r["label"], r["negative"]) for r in rows]
    positive_chart, p_xs, p_ys = draw_chart(
        24,
        200,
        850,
        440,
        "Score positif moyen",
        "Juge OVH gpt-oss:120b sur les exemples positive",
        positive_points,
        fmt_score,
        1.0,
        0.0,
    )
    negative_chart, n_xs, n_ys = draw_chart(
        912,
        200,
        850,
        440,
        "Exactitude négative",
        "Exact-match des exemples negative",
        negative_points,
        fmt_pct,
        100.0,
        0.0,
    )

    # Rebuild line colors for negative chart points and labels later.
    pieces = []
    pieces.append('<?xml version="1.0" encoding="UTF-8"?>')
    pieces.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1400" viewBox="0 0 1800 1400">'
    )
    pieces.append(svg_rect(0, 0, 1800, 1400, rx=0, fill="#070b16"))
    pieces.append(
        '<defs>'
        '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#0b1020"/>'
        '<stop offset="100%" stop-color="#090d1a"/>'
        '</linearGradient>'
        '</defs>'
    )
    pieces.append(svg_rect(0, 0, 1800, 1400, rx=0, fill="url(#bg)"))

    # Header
    pieces.append(svg_text(36, 48, "Comparaison visuelle du fine-tuning", size=30, weight=800, fill="#f8fafc"))
    pieces.append(
        svg_text(
            36,
            76,
            "Le meilleur refus arrive tôt, la meilleure réponse positive arrive plus tard: un vrai compromis entre robustesse et complétude.",
            size=14,
            fill="#94a3b8",
        )
    )

    # Metric cards
    card_y = 96
    card_w = 410
    card_h = 90
    gap = 20
    card_specs = [
        ("Base", f"Positif {base['positive']:.3f} | Négatif {base['negative']:.2f}%", "#64748b"),
        ("Meilleur négatif", f"{best_neg['label']} | {best_neg['negative']:.2f}%", "#2563eb"),
        ("Meilleur positif", f"{best_pos['label']} | {best_pos['positive']:.3f}", "#f59e0b"),
        ("Compromis final", f"{final['label']} | {final['positive']:.3f} / {final['negative']:.2f}%", "#16a34a"),
    ]
    for i, (title, body, color) in enumerate(card_specs):
        x = 24 + i * (card_w + gap)
        pieces.append(svg_rect(x, card_y, card_w, card_h, rx=18, fill="#10182f", stroke="#253046", sw=1.2))
        pieces.append(svg_rect(x + 14, card_y + 14, 10, card_h - 28, rx=5, fill=color))
        pieces.append(svg_text(x + 36, card_y + 30, title, size=18, weight=700, fill="#f8fafc"))
        pieces.append(svg_text(x + 36, card_y + 58, body, size=13, fill="#cbd5e1"))

    pieces.extend(positive_chart)
    pieces.extend(negative_chart)

    # Callouts on charts
    # Positive chart callouts
    idx_base = labels.index("base")
    idx_best_pos = labels.index(best_pos["label"])
    idx_final = labels.index("final")
    pieces.append(svg_line(p_xs[idx_base], p_ys[idx_base], 220, 690, stroke="#94a3b8", sw=2, dash="6 6"))
    add_callout(
        pieces,
        44,
        692,
        "Base",
        f"{base['positive']:.3f} positif, {base['negative']:.2f}% négatif",
        color="#10182f",
    )
    pieces.append(svg_line(p_xs[idx_best_pos], p_ys[idx_best_pos], 360, 690, stroke="#f59e0b", sw=2, dash="6 6"))
    add_callout(
        pieces,
        366,
        692,
        "Pic positif",
        f"{best_pos['label']} atteint {best_pos['positive']:.3f}",
        color="#1a1520",
    )
    pieces.append(svg_line(p_xs[idx_final], p_ys[idx_final], 680, 690, stroke="#16a34a", sw=2, dash="6 6"))
    add_callout(
        pieces,
        690,
        692,
        "Final",
        f"{final['positive']:.3f} positif, {final['negative']:.2f}% négatif",
        color="#10201a",
    )

    # Negative chart callouts
    idx_best_neg = labels.index(best_neg["label"])
    pieces.append(svg_line(n_xs[idx_base], n_ys[idx_base], 1110, 690, stroke="#94a3b8", sw=2, dash="6 6"))
    add_callout(
        pieces,
        936,
        692,
        "Base",
        f"{base['negative']:.2f}% seulement sur les négatifs",
        color="#10182f",
    )
    pieces.append(svg_line(n_xs[idx_best_neg], n_ys[idx_best_neg], 1338, 690, stroke="#2563eb", sw=2, dash="6 6"))
    add_callout(
        pieces,
        1344,
        692,
        "Pic négatif",
        f"{best_neg['label']} atteint {best_neg['negative']:.2f}%",
        color="#101b2f",
    )
    pieces.append(svg_line(n_xs[idx_final], n_ys[idx_final], 1590, 690, stroke="#16a34a", sw=2, dash="6 6"))
    add_callout(
        pieces,
        1590,
        692,
        "Final",
        f"{final['negative']:.2f}% au final",
        color="#10201a",
    )

    # Middle section: training dynamics
    pieces.append(svg_rect(24, 730, 1752, 270, rx=22, fill="#0f172a", stroke="#253046", sw=1.4))
    pieces.append(svg_text(48, 766, "Ce que disent les graphes train/eval", size=22, weight=700, fill="#f8fafc"))
    pieces.append(svg_text(48, 790, "La courbe eval/loss baisse puis remonte légèrement: le meilleur point métier n'est pas forcément le dernier.", size=13, fill="#94a3b8"))

    takeaways = [
        ("Eval/loss", "Minimum vers 625-750 steps", "#f59e0b"),
        ("Train/loss", "Chute rapide vers ~0", "#2563eb"),
        ("Train/acc", "Plateau proche de 0.99", "#16a34a"),
        ("Eval/acc", "Stabilisé autour de 0.965", "#0ea5e9"),
    ]
    for i, (title, body, color) in enumerate(takeaways):
        x = 48 + i * 424
        pieces.append(svg_rect(x, 820, 392, 132, rx=18, fill="#111a31", stroke="#24324f", sw=1.1))
        pieces.append(svg_rect(x + 14, 838, 8, 96, rx=4, fill=color))
        pieces.append(svg_text(x + 34, 858, title, size=18, weight=700, fill="#f8fafc"))
        pieces.append(svg_text(x + 34, 886, body, size=14, fill="#cbd5e1"))
        if title == "Eval/loss":
            pieces.append(svg_text(x + 34, 912, "Surapprentissage léger après le creux", size=12, fill="#fcd34d"))
        elif title == "Train/loss":
            pieces.append(svg_text(x + 34, 912, "Le modèle apprend très vite", size=12, fill="#93c5fd"))
        elif title == "Train/acc":
            pieces.append(svg_text(x + 34, 912, "Capacité d'ajustement élevée", size=12, fill="#86efac"))
        else:
            pieces.append(svg_text(x + 34, 912, "Généralisation correcte mais pas parfaite", size=12, fill="#67e8f9"))

    # Bottom: scatter plot
    pieces.append(svg_rect(24, 1030, 1752, 320, rx=22, fill="#0f172a", stroke="#253046", sw=1.4))
    pieces.append(svg_text(48, 1066, "Carte de compromis des versions", size=22, weight=700, fill="#f8fafc"))
    pieces.append(svg_text(48, 1090, "Axe X = score positif moyen, Axe Y = exactitude négative. Le coin haut-droite est l'objectif.", size=13, fill="#94a3b8"))

    left = 88
    top = 1138
    width = 1620
    height = 184
    # grid
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = left + frac * width
        yy = top + height - frac * height
        pieces.append(svg_line(xx, top, xx, top + height, stroke="#23304f", sw=1))
        pieces.append(svg_line(left, yy, left + width, yy, stroke="#23304f", sw=1))
        pieces.append(svg_text(xx, top + height + 22, f"{frac:.2f}", size=11, fill="#94a3b8", anchor="middle"))
        pieces.append(svg_text(left - 10, yy + 4, f"{int(frac*100)}%", size=11, fill="#94a3b8", anchor="end"))
    pieces.append(svg_line(left, top + height, left + width, top + height, stroke="#e2e8f0", sw=1.2))
    pieces.append(svg_line(left, top, left, top + height, stroke="#e2e8f0", sw=1.2))
    pieces.append(svg_text(left + width / 2, top + height + 44, "Positif", size=13, fill="#cbd5e1", anchor="middle"))
    pieces.append(svg_text(left - 70, top + height / 2, "Négatif", size=13, fill="#cbd5e1", anchor="middle"))

    def px(p):
        return left + p * width

    def py(n):
        return top + height - (n / 100.0) * height

    key_labels = {"base", best_neg["label"], best_pos["label"], "final"}
    for row in rows:
        xx = px(row["positive"])
        yy = py(row["negative"])
        label = row["label"]
        pieces.append(
            f'<circle cx="{xx}" cy="{yy}" r="7" fill="{stage_color(label)}" stroke="#ffffff" stroke-width="2.5"/>'
        )
        if label in key_labels:
            pieces.append(svg_text(xx + 10, yy - 10, label, size=11, fill="#f8fafc"))

    # highlight arrows
    b = next(r for r in rows if r["label"] == "base")
    pieces.append(svg_line(px(b["positive"]), py(b["negative"]), px(best_neg["positive"]), py(best_neg["negative"]), stroke="#2563eb", sw=2.5, dash="8 6"))
    pieces.append(svg_line(px(best_neg["positive"]), py(best_neg["negative"]), px(best_pos["positive"]), py(best_pos["negative"]), stroke="#f59e0b", sw=2.5, dash="8 6"))
    pieces.append(svg_line(px(best_pos["positive"]), py(best_pos["negative"]), px(final["positive"]), py(final["negative"]), stroke="#16a34a", sw=2.5, dash="8 6"))

    pieces.append(svg_rect(1180, 1322, 520, 56, rx=14, fill="#111a31", stroke="#24324f", sw=1.1))
    pieces.append(svg_text(1200, 1354, "Lecture rapide: base faible, meilleur refus tôt, meilleure qualité positive plus tard, final = compromis.", size=14, fill="#e5e7eb"))

    pieces.append("</svg>")
    return "\n".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="Create an annotated SVG/PNG summary from results logs.")
    parser.add_argument("--positive-log", type=pathlib.Path, default=pathlib.Path("results2.log"))
    parser.add_argument("--negative-log", type=pathlib.Path, default=pathlib.Path("results3.log"))
    parser.add_argument("--svg", type=pathlib.Path, default=pathlib.Path("results-annotated.svg"))
    parser.add_argument("--png", type=pathlib.Path, default=pathlib.Path("results-annotated.png"))
    args = parser.parse_args()

    if not args.positive_log.exists():
        raise SystemExit(f"Missing positive log: {args.positive_log}")
    if not args.negative_log.exists():
        raise SystemExit(f"Missing negative log: {args.negative_log}")

    positive = parse_positive_log(args.positive_log)
    negative = parse_negative_log(args.negative_log)
    svg = build_svg(positive, negative, args.svg.name)
    args.svg.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.svg}")


if __name__ == "__main__":
    main()
