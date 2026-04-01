#!/usr/bin/env python3
import argparse
import html
import pathlib
import re
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class PositiveRun:
    label: str
    scores: list[float]
    questions: list[str]
    ids: list[int]


@dataclass
class NegativeRun:
    label: str
    failures: list[str]
    ids: list[int]


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


def _positive_questions(dataset_split, log_path: pathlib.Path, checkpoint: str) -> PositiveRun:
    scores = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"^\[split=validation\]\[checkpoint=([^\]]+)\] (\d+)/(\d+) -> ([0-9.]+)$",
            line,
        )
        if match and match.group(1) == checkpoint:
            scores.append(float(match.group(4)))

    if len(scores) != len(dataset_split):
        raise RuntimeError(
            f"{log_path.name}: expected {len(dataset_split)} scores for {checkpoint}, got {len(scores)}"
        )

    questions = [dataset_split[i]["user"] for i in range(len(dataset_split))]
    ids = [int(dataset_split[i]["id"]) for i in range(len(dataset_split))]
    return PositiveRun(checkpoint, scores, questions, ids)


def _negative_questions(dataset_split, log_path: pathlib.Path, checkpoint: str) -> NegativeRun:
    failures = []
    ids = []
    current = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^== (.+) ==$", line)
        if match:
            current = match.group(1)
            continue
        if current != checkpoint:
            continue
        match = re.match(r"^\[FAIL\] (\d+)/(\d+) \| (.+)$", line)
        if match:
            failures.append(match.group(3))
            ids.append(int(dataset_split[int(match.group(1)) - 1]["id"]))
    return NegativeRun(checkpoint, failures, ids)


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _render_list(items):
    if not items:
        return '<li class="empty">Aucun élément.</li>'
    return "\n".join(f"<li>{_escape(item)}</li>" for item in items)


def _render_overlap_block(title, left_label, right_label, left_set, right_set):
    shared = sorted(left_set & right_set)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)
    return f"""
    <div class="section">
      <h2>{_escape(title)}</h2>
      <p>
        {len(left_set)} éléments pour { _escape(left_label) }, {len(right_set)} pour { _escape(right_label) }.
        Intersection: {len(shared)}. Jaccard: {_jaccard(left_set, right_set):.3f}.
      </p>
      <div class="grid-3">
        <div class="panel">
          <div class="panel-title">Commun</div>
          <div class="panel-meta">{len(shared)} questions</div>
          <ul>{_render_list(shared)}</ul>
        </div>
        <div class="panel">
          <div class="panel-title">Seulement { _escape(left_label) }</div>
          <div class="panel-meta">{len(only_left)} questions</div>
          <ul>{_render_list(only_left)}</ul>
        </div>
        <div class="panel">
          <div class="panel-title">Seulement { _escape(right_label) }</div>
          <div class="panel-meta">{len(only_right)} questions</div>
          <ul>{_render_list(only_right)}</ul>
        </div>
      </div>
    </div>
    """


def _render_overlap_block_with_ids(title, left_label, right_label, left_items, right_items):
    left_map = {item["question"]: item["id"] for item in left_items}
    right_map = {item["question"]: item["id"] for item in right_items}
    left_set = set(left_map)
    right_set = set(right_map)
    shared = sorted(left_set & right_set)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)

    def fmt(question, qid):
        return f"[id={qid}] {question}"

    return f"""
    <div class="section">
      <h2>{_escape(title)}</h2>
      <p>
        {len(left_set)} éléments pour { _escape(left_label) }, {len(right_set)} pour { _escape(right_label) }.
        Intersection: {len(shared)}. Jaccard: {_jaccard(left_set, right_set):.3f}.
      </p>
      <div class="grid-3">
        <div class="panel">
          <div class="panel-title">Commun</div>
          <div class="panel-meta">{len(shared)} questions</div>
          <ul>{_render_list(fmt(q, left_map[q]) for q in shared)}</ul>
        </div>
        <div class="panel">
          <div class="panel-title">Seulement { _escape(left_label) }</div>
          <div class="panel-meta">{len(only_left)} questions</div>
          <ul>{_render_list(fmt(q, left_map[q]) for q in only_left)}</ul>
        </div>
        <div class="panel">
          <div class="panel-title">Seulement { _escape(right_label) }</div>
          <div class="panel-meta">{len(only_right)} questions</div>
          <ul>{_render_list(fmt(q, right_map[q]) for q in only_right)}</ul>
        </div>
      </div>
    </div>
    """


def _build_html(
    positive_left: PositiveRun,
    positive_right: PositiveRun,
    negative_left: NegativeRun,
    negative_right: NegativeRun,
):
    left_non_perfect = {
        q for q, score in zip(positive_left.questions, positive_left.scores) if score < 1.0
    }
    right_non_perfect = {
        q for q, score in zip(positive_right.questions, positive_right.scores) if score < 1.0
    }
    left_zero = {q for q, score in zip(positive_left.questions, positive_left.scores) if score == 0.0}
    right_zero = {q for q, score in zip(positive_right.questions, positive_right.scores) if score == 0.0}
    left_positive_items = [
        {"question": q, "id": qid}
        for q, qid, score in zip(positive_left.questions, positive_left.ids, positive_left.scores)
        if score < 1.0
    ]
    right_positive_items = [
        {"question": q, "id": qid}
        for q, qid, score in zip(positive_right.questions, positive_right.ids, positive_right.scores)
        if score < 1.0
    ]
    left_zero_items = [
        {"question": q, "id": qid}
        for q, qid, score in zip(positive_left.questions, positive_left.ids, positive_left.scores)
        if score == 0.0
    ]
    right_zero_items = [
        {"question": q, "id": qid}
        for q, qid, score in zip(positive_right.questions, positive_right.ids, positive_right.scores)
        if score == 0.0
    ]
    left_negative_items = [
        {"question": q, "id": qid}
        for q, qid in zip(negative_left.failures, negative_left.ids)
    ]
    right_negative_items = [
        {"question": q, "id": qid}
        for q, qid in zip(negative_right.failures, negative_right.ids)
    ]

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Recouvrement des checkpoints</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #10182f;
      --border: #26324f;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #f59e0b;
      --accent2: #38bdf8;
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
      max-width: 1540px;
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
      line-height: 1.55;
      margin-bottom: 18px;
      max-width: 1200px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card, .section, .panel {{
      background: rgba(16, 24, 47, 0.86);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }}
    .card {{
      padding: 16px 18px;
    }}
    .card .k {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .card .v {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.03em;
      line-height: 1.2;
    }}
    .section {{
      margin-top: 18px;
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
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .panel {{
      padding: 14px;
    }}
    .panel-title {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .panel-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      max-height: 420px;
      overflow: auto;
    }}
    li {{
      margin: 6px 0;
      line-height: 1.35;
    }}
    li.empty {{
      list-style: none;
      padding-left: 0;
      color: var(--muted);
    }}
    .note {{
      color: #fbbf24;
      margin-top: 6px;
      line-height: 1.5;
    }}
    .footer {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Recouvrement des échecs entre checkpoints</h1>
    <div class="subtitle">
      Comparaison de <strong>{_escape(positive_left.label)}</strong> contre <strong>{_escape(positive_right.label)}</strong> pour les positifs,
      puis même comparaison pour les négatifs. Les positifs sont d’abord analysés en <code>score &lt; 1</code>, puis en <code>score == 0</code>.
    </div>

    <div class="grid">
      <div class="card">
        <div class="k">Positifs non parfaits</div>
        <div class="v">{len(left_non_perfect)} vs {len(right_non_perfect)}</div>
      </div>
      <div class="card">
        <div class="k">Positifs à 0</div>
        <div class="v">{len(left_zero)} vs {len(right_zero)}</div>
      </div>
      <div class="card">
        <div class="k">Négatifs en échec</div>
        <div class="v">{len(left_negative_items)} vs {len(right_negative_items)}</div>
      </div>
      <div class="card">
        <div class="k">Checkpoint A</div>
        <div class="v">{_escape(positive_left.label)} / {_escape(negative_left.label)}</div>
      </div>
      <div class="card">
        <div class="k">Checkpoint B</div>
        <div class="v">{_escape(positive_right.label)} / {_escape(negative_right.label)}</div>
      </div>
    </div>

    {_render_overlap_block_with_ids(
        "Positifs - tous les scores non parfaits",
        f"{positive_left.label} ({len(left_non_perfect)})",
        f"{positive_right.label} ({len(right_non_perfect)})",
        left_positive_items,
        right_positive_items,
    )}

    {_render_overlap_block_with_ids(
        "Positifs - uniquement les scores nuls",
        f"{positive_left.label} ({len(left_zero)})",
        f"{positive_right.label} ({len(right_zero)})",
        left_zero_items,
        right_zero_items,
    )}

    {_render_overlap_block_with_ids(
        "Négatifs - échecs exact-match",
        f"{negative_left.label} ({len(left_negative_items)})",
        f"{negative_right.label} ({len(right_negative_items)})",
        left_negative_items,
        right_negative_items,
    )}

    <div class="section">
      <h2>Lecture rapide</h2>
      <p>
        Le recouvrement est modéré sur les positifs non parfaits, plus net sur les scores nuls, et assez faible sur les négatifs.
        En pratique, les deux checkpoints partagent un noyau de fragilités, mais leurs échecs spécifiques restent différents.
      </p>
      <div class="note">
        Positifs non parfaits: Jaccard {_jaccard(left_non_perfect, right_non_perfect):.3f}.
        Positifs à 0: Jaccard {_jaccard(left_zero, right_zero):.3f}.
        Négatifs: Jaccard {_jaccard(set(negative_left.failures), set(negative_right.failures)):.3f}.
      </div>
    </div>

    <div class="footer">
      Basé sur {_escape(positive_left.label)} / {_escape(negative_left.label)} et {_escape(positive_right.label)} / {_escape(negative_right.label)}.
    </div>
  </div>
</body>
</html>
"""
    return html_doc


def main():
    parser = argparse.ArgumentParser(description="Generate overlap report for two checkpoints.")
    parser.add_argument("--dataset-name", default="fenyo/FAQ-MES-WEB")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--positive-left-log", type=pathlib.Path, default=pathlib.Path("results2.log"))
    parser.add_argument("--negative-left-log", type=pathlib.Path, default=pathlib.Path("results3.log"))
    parser.add_argument("--positive-right-log", type=pathlib.Path, default=pathlib.Path("results5.log"))
    parser.add_argument("--negative-right-log", type=pathlib.Path, default=pathlib.Path("results4.log"))
    parser.add_argument("--positive-left-checkpoint", default="checkpoint-750")
    parser.add_argument("--negative-left-checkpoint", default="checkpoint-750")
    parser.add_argument("--positive-right-checkpoint", default="checkpoint-625")
    parser.add_argument("--negative-right-checkpoint", default="checkpoint-625")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("checkpoint-overlap-report.html"))
    args = parser.parse_args()

    dataset_split = load_dataset(args.dataset_name)[args.split]
    positive_split = dataset_split.filter(lambda row: row["type"] == "positive")
    negative_split = dataset_split.filter(lambda row: row["type"] == "negative")

    positive_left = _positive_questions(positive_split, args.positive_left_log, args.positive_left_checkpoint)
    positive_right = _positive_questions(positive_split, args.positive_right_log, args.positive_right_checkpoint)
    negative_left = _negative_questions(negative_split, args.negative_left_log, args.negative_left_checkpoint)
    negative_right = _negative_questions(negative_split, args.negative_right_log, args.negative_right_checkpoint)

    html_doc = _build_html(positive_left, positive_right, negative_left, negative_right)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
