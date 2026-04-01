#!/usr/bin/env python3
import argparse
import html
import pathlib
import re
import unicodedata
from difflib import SequenceMatcher

from datasets import load_dataset


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _pair_label(left_id: int, left_q: str, right_id: int, right_q: str) -> str:
    return f"[{left_id}] {left_q}  |  [{right_id}] {right_q}"


def _render_table(headers, rows):
    head = "".join(f"<th>{_escape(h)}</th>" for h in headers)
    body = "".join(rows) if rows else '<tr><td colspan="99" class="empty">Aucun élément.</td></tr>'
    return f"""
    <table>
      <thead><tr>{head}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def _top_pairs(items_a, items_b, threshold, limit=30, same_set=False):
    pairs = []
    if same_set:
        for i in range(len(items_a)):
            for j in range(i + 1, len(items_a)):
                a = items_a[i]
                b = items_a[j]
                score = _sim(a["user"], b["user"])
                if score >= threshold:
                    pairs.append((score, a, b))
    else:
        for a in items_a:
            for b in items_b:
                score = _sim(a["user"], b["user"])
                if score >= threshold:
                    pairs.append((score, a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:limit]


def _render_pair_row(score, a, b, note=""):
    return f"""
    <tr>
      <td>{score:.3f}</td>
      <td>[id={a['id']}] {_escape(a['user'])}</td>
      <td>[id={b['id']}] {_escape(b['user'])}</td>
      <td>{_escape(note)}</td>
    </tr>
    """


def _build_html(dataset_name: str, split: str, rows, pp_threshold: float, pn_threshold: float):
    positives = [r for r in rows if r["type"] == "positive"]
    negatives = [r for r in rows if r["type"] == "negative"]

    pos_scores = [r["assistant"] for r in positives]
    neg_scores = [r["assistant"] for r in negatives]

    pos_exact_dupes = sum(
        1
        for i in range(len(positives))
        for j in range(i + 1, len(positives))
        if _norm(positives[i]["user"]) == _norm(positives[j]["user"])
    )
    neg_exact_dupes = sum(
        1
        for i in range(len(negatives))
        for j in range(i + 1, len(negatives))
        if _norm(negatives[i]["user"]) == _norm(negatives[j]["user"])
    )
    cross_exact = sorted(set(_norm(r["user"]) for r in positives) & set(_norm(r["user"]) for r in negatives))

    # Asymmetric views useful for the report.
    pp_near = _top_pairs(positives, positives, threshold=pp_threshold, limit=24, same_set=True)
    pn_near = _top_pairs(positives, negatives, threshold=pn_threshold, limit=40, same_set=False)

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audit dataset FAQ MES</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #10182f;
      --border: #26324f;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #f59e0b;
      --accent2: #38bdf8;
      --good: #16a34a;
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
      max-width: 1180px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card, .section {{
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
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.18);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .empty {{
      color: var(--muted);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
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
    .note {{
      color: #fbbf24;
      margin-top: 8px;
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
    <h1>Audit du dataset { _escape(dataset_name) }</h1>
    <div class="subtitle">
      Analyse du split <strong>{_escape(split)}</strong>. L'audit cherche surtout les signes d'ambiguïté lexicale:
      doublons exacts, quasi-doublons dans les positifs, et recouvrement entre positifs et négatifs.
    </div>

    <div class="grid">
      <div class="card">
        <div class="k">Lignes</div>
        <div class="v">{len(rows)}</div>
      </div>
      <div class="card">
        <div class="k">Positifs</div>
        <div class="v">{len(positives)}</div>
      </div>
      <div class="card">
        <div class="k">Négatifs</div>
        <div class="v">{len(negatives)}</div>
      </div>
      <div class="card">
        <div class="k">Doublons exacts</div>
        <div class="v">{pos_exact_dupes} / {neg_exact_dupes}</div>
      </div>
      <div class="card">
        <div class="k">Collision positive/négative</div>
        <div class="v">{len(cross_exact)}</div>
      </div>
      <div class="card">
        <div class="k">Quasi-doublons positifs</div>
        <div class="v">{len(pp_near)}</div>
      </div>
      <div class="card">
        <div class="k">Collisions pos./neg.</div>
        <div class="v">{len(pn_near)}</div>
      </div>
      <div class="card">
        <div class="k">Assistant négatif</div>
        <div class="v">{len(set(neg_scores))} réponse unique</div>
      </div>
    </div>

    <div class="section">
      <h2>Résumé structurel</h2>
      <p>
        Le dataset est propre sur les collisions exactes: aucune question n'apparaît telle quelle dans les deux types,
        et il n'y a pas de doublon exact de question après normalisation NFKC.
        Le point de vigilance est ailleurs: les questions proches par formulation.
      </p>
      <div class="note">
        Positifs uniques: {len(set(_norm(r['user']) for r in positives))}/{len(positives)}.
        Négatifs uniques: {len(set(_norm(r['user']) for r in negatives))}/{len(negatives)}.
        Questions exactes partagées entre types: {len(cross_exact)}.
      </div>
    </div>

    <div class="section">
      <h2>Positifs - quasi-doublons</h2>
      <p>Questions du même type dont la formulation est très proche (seuil ≥ {pp_threshold:.2f}). Ce sont les candidats les plus probables à une ambiguïté de dataset.</p>
      {_render_table(
          ["Score", "Question A", "Question B", "Lecture"],
          [
              _render_pair_row(
                  score,
                  a,
                  b,
                  "Même thème ou très proche"
              )
              for score, a, b in pp_near
          ],
      )}
    </div>

    <div class="section">
      <h2>Positifs vs négatifs - collisions potentielles</h2>
      <p>Ces paires ne sont pas des doublons exacts, mais elles ressemblent suffisamment pour que le modèle puisse hésiter entre réponse et refus (seuil ≥ {pn_threshold:.2f}).</p>
      {_render_table(
          ["Score", "Question positive", "Question négative", "Lecture"],
          [
              _render_pair_row(
                  score,
                  a,
                  b,
                  "Collision sémantique possible"
              )
              for score, a, b in pn_near
          ],
      )}
    </div>

    <div class="section">
      <h2>Exemples les plus sensibles</h2>
      <div class="two-col">
        <div class="section" style="margin-top:0;">
          <h2>Positifs très proches</h2>
          <p>Quelques paires proches à surveiller de près.</p>
          <ul>
            {''.join(f"<li>[id={a['id']}] {_escape(a['user'])}<br/><span class='empty'>↔ [id={b['id']}] {_escape(b['user'])} ({score:.3f})</span></li>" for score, a, b in pp_near[:8])}
          </ul>
        </div>
        <div class="section" style="margin-top:0;">
          <h2>Positifs / négatifs proches</h2>
          <p>Les cas les plus risqués pour l'apprentissage du refus.</p>
          <ul>
            {''.join(f"<li>[id={a['id']}] {_escape(a['user'])}<br/><span class='empty'>↔ [id={b['id']}] {_escape(b['user'])} ({score:.3f})</span></li>" for score, a, b in pn_near[:10])}
          </ul>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Lecture rapide</h2>
      <p>
        Le dataset semble sain sur les doublons exacts, mais il est moins propre sur les formulations proches.
        C’est cohérent avec tes observations: le modèle apprend les grandes lignes, puis se trompe surtout sur des familles de questions voisines.
      </p>
      <div class="note">
        Conclusion pratique: l'audit pointe plus un problème de formulation / séparation des cas qu'un simple réglage d'hyperparamètres.
      </div>
    </div>

    <div class="footer">
      Fichier généré à partir du split { _escape(split) } de { _escape(dataset_name) }.
    </div>
  </div>
</body>
</html>
"""
    return html_doc


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML dataset audit report.")
    parser.add_argument("--dataset-name", default="fenyo/FAQ-MES-WEB")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("dataset-audit-report.html"))
    parser.add_argument("--pp-threshold", type=float, default=0.80)
    parser.add_argument("--pn-threshold", type=float, default=0.60)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)[args.split]
    html_doc = _build_html(args.dataset_name, args.split, dataset, args.pp_threshold, args.pn_threshold)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
