#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard comparing pipeline output vs ground truth.

Usage:
    python tools/segment_dashboard.py <ground_truth.json> <pipeline_output.srt> [--out dashboard.html]

Shows per-segment: lang, speaker, confidence signals, IPA (3 backends), NAH markers,
error type, Whisper text, and GT comparison — all sortable and filterable.
"""

import json
import re
import sys
import html as html_mod
from pathlib import Path
from collections import defaultdict

NAH_IPA_MARKERS = {"ts", "tɬ", "kʷ", "tʃʼ", "kʼ", "tɕ"}
EJECTIVE_MARKERS = {"kʼ", "tʼ", "tsʼ", "tʃʼ", "pʼ"}

ERROR_COSTS = {
    ("nah", "spa"): 2.0, ("nah", "oth"): 1.5, ("nah", "unk"): 0.5,
    ("may", "spa"): 2.0, ("may", "oth"): 1.5, ("may", "unk"): 0.5,
    ("spa", "nah"): 1.5, ("spa", "may"): 1.5,
    "_default": 1.0, "_to_unk": 0.5, "_from_unk": 0.0,
}


def get_error_cost(actual, predicted):
    if actual == predicted:
        return 0.0
    if actual == "unk":
        return ERROR_COSTS["_from_unk"]
    if predicted == "unk":
        return ERROR_COSTS.get((actual, "unk"), ERROR_COSTS["_to_unk"])
    return ERROR_COSTS.get((actual, predicted), ERROR_COSTS["_default"])


def parse_time(tc):
    """Parse SRT timecode to seconds."""
    tc = tc.replace(",", ".")
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path):
    """Parse SRT into segments with all metadata."""
    text = Path(path).read_text(encoding="utf-8")
    segments = {}
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            cue = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1],
        )
        if not time_match:
            continue

        start_tc = time_match.group(1)
        end_tc = time_match.group(2)
        start_s = parse_time(start_tc)
        end_s = parse_time(end_tc)
        duration = end_s - start_s

        # Parse content line (line 3): [LANG|SPEAKER] text...
        content_line = lines[2] if len(lines) > 2 else ""
        tag_match = re.match(r"\[(\w+)\|([^\]]+)\]\s*(.*)", content_line)
        if tag_match:
            pred_lang = tag_match.group(1).lower()
            speaker = tag_match.group(2)
            text_content = tag_match.group(3)
        else:
            pred_lang = "oth"
            speaker = "?"
            text_content = content_line

        lang_map = {
            "nahuatl": "nah", "español": "spa", "spanish": "spa",
            "english": "eng", "deutsch": "deu", "???": "oth",
            "other": "oth", "silence": "sil",
        }
        pred_lang = lang_map.get(pred_lang, pred_lang)

        # Detect source: [FT], [LLM], [IPA], [REC], [UNC]
        source = "whisper"
        for marker in ["[FT]", "[LLM]", "[IPA]", "[REC]", "[UNC]"]:
            if marker in text_content:
                source = marker.strip("[]").lower()
                text_content = text_content.replace(marker, "").strip()
                break

        # Parse metadata lines
        ipa_allo = ipa_w2v2 = ipa_fused = trim = conf = ""
        for l in lines[3:]:
            if l.startswith("♫allo:"):
                ipa_allo = l[6:].strip()
            elif l.startswith("♫w2v2:"):
                ipa_w2v2 = l[6:].strip()
            elif l.startswith("♫fused:"):
                ipa_fused = l[7:].strip()
            elif l.startswith("♫trim:"):
                trim = l[6:].strip()
            elif l.startswith("♫conf:"):
                conf = l[6:].strip()

        # Analyze IPA for NAH markers
        fused_tokens = set(ipa_fused.split()) if ipa_fused else set()
        nah_markers = fused_tokens & NAH_IPA_MARKERS
        ejectives = fused_tokens & EJECTIVE_MARKERS

        segments[cue] = {
            "cue": cue,
            "start_tc": start_tc.replace(",", "."),
            "end_tc": end_tc.replace(",", "."),
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration": round(duration, 2),
            "speaker": speaker,
            "pred_lang": pred_lang,
            "text": text_content,
            "source": source,
            "ipa_allo": ipa_allo,
            "ipa_w2v2": ipa_w2v2,
            "ipa_fused": ipa_fused,
            "trim": trim,
            "conf": conf,
            "nah_markers": sorted(nah_markers),
            "ejectives": sorted(ejectives),
        }
    return segments


def load_gt(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        segs = data.get("segments", [])
        return {i + 1: s["lang"].lower() for i, s in enumerate(segs) if s.get("done", False)}
    gt = {}
    for item in data:
        cue = item.get("cue_index")
        lang = (item.get("correct_lang") or "").lower()
        if cue is not None and lang:
            gt[cue] = lang
    return gt


def generate_html(segments, gt, out_path):
    # Build rows
    rows = []
    all_cues = sorted(set(list(segments.keys()) + list(gt.keys())))
    speakers = sorted(set(s["speaker"] for s in segments.values()))
    langs = sorted(set(list(gt.values()) + [s["pred_lang"] for s in segments.values()]))

    for cue in all_cues:
        seg = segments.get(cue)
        gt_lang = gt.get(cue, "")

        if not seg:
            rows.append({
                "cue": cue, "start_tc": "", "end_tc": "", "duration": 0,
                "speaker": "?", "gt_lang": gt_lang, "pred_lang": "(none)",
                "match": False, "error_type": f"{gt_lang}→(none)" if gt_lang else "",
                "cost": get_error_cost(gt_lang, "oth") if gt_lang else 0,
                "text": "", "source": "", "ipa_allo": "", "ipa_w2v2": "",
                "ipa_fused": "", "trim": "", "conf": "",
                "nah_markers": [], "ejectives": [],
            })
            continue

        match = (gt_lang == seg["pred_lang"]) if gt_lang else None
        error_type = ""
        cost = 0.0
        if gt_lang and not match:
            error_type = f"{gt_lang}→{seg['pred_lang']}"
            cost = get_error_cost(gt_lang, seg["pred_lang"])

        rows.append({
            **seg,
            "gt_lang": gt_lang,
            "match": match,
            "error_type": error_type,
            "cost": cost,
        })

    # Stats
    total_annotated = sum(1 for r in rows if r["gt_lang"])
    total_correct = sum(1 for r in rows if r["match"] is True)
    total_errors = sum(1 for r in rows if r["match"] is False)
    total_cost = sum(r["cost"] for r in rows)

    # Error type counts
    error_counts = defaultdict(int)
    for r in rows:
        if r["error_type"]:
            error_counts[r["error_type"]] += 1

    # Speaker profiles (from GT)
    speaker_gt = defaultdict(lambda: defaultdict(int))
    speaker_pred = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["gt_lang"]:
            speaker_gt[r["speaker"]][r["gt_lang"]] += 1
        speaker_pred[r["speaker"]][r["pred_lang"]] += 1

    rows_json = json.dumps(rows, ensure_ascii=False)
    speakers_json = json.dumps(speakers, ensure_ascii=False)
    langs_json = json.dumps(langs, ensure_ascii=False)
    error_types_json = json.dumps(sorted(error_counts.keys(), key=lambda x: -error_counts[x]),
                                  ensure_ascii=False)

    # Speaker profiles for tooltip
    sp_profiles = {}
    for spk in speakers:
        gt_profile = dict(speaker_gt[spk])
        pred_profile = dict(speaker_pred[spk])
        sp_profiles[spk] = {"gt": gt_profile, "pred": pred_profile}
    sp_json = json.dumps(sp_profiles, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tenepal Segment Dashboard</title>
<style>
:root {{
  --bg: #1a1a2e;
  --bg2: #16213e;
  --bg3: #0f3460;
  --fg: #e0e0e0;
  --fg2: #a0a0b0;
  --accent: #e94560;
  --ok: #4ecca3;
  --warn: #ffc857;
  --err: #e94560;
  --nah: #ff6b6b;
  --spa: #4ecdc4;
  --oth: #888;
  --may: #f7b731;
  --unk: #666;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; background: var(--bg); color: var(--fg); }}
.header {{ background: var(--bg2); padding: 12px 20px; border-bottom: 2px solid var(--bg3);
           display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }}
.header h1 {{ font-size: 16px; color: var(--accent); white-space: nowrap; }}
.stat {{ background: var(--bg3); padding: 4px 10px; border-radius: 4px; font-size: 11px; }}
.stat b {{ color: var(--ok); }}
.stat.err b {{ color: var(--err); }}

.filters {{ background: var(--bg2); padding: 8px 20px; border-bottom: 1px solid var(--bg3);
            display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
.filters label {{ font-size: 11px; color: var(--fg2); }}
.filters select, .filters input {{ background: var(--bg); color: var(--fg); border: 1px solid var(--bg3);
                                    padding: 3px 6px; border-radius: 3px; font-size: 11px; font-family: inherit; }}
.filters select {{ min-width: 90px; }}
.filters input[type=text] {{ width: 160px; }}
.filters .btn {{ background: var(--bg3); color: var(--fg); border: none; padding: 4px 10px;
                 border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; }}
.filters .btn:hover {{ background: var(--accent); }}
.filters .btn.active {{ background: var(--accent); }}

.table-wrap {{ overflow-x: auto; padding: 0 8px; }}
table {{ width: 100%; border-collapse: collapse; table-layout: auto; }}
thead {{ position: sticky; top: 0; z-index: 10; }}
th {{ background: var(--bg3); color: var(--fg); padding: 6px 8px; text-align: left;
     font-size: 11px; font-weight: 600; cursor: pointer; white-space: nowrap;
     border-bottom: 2px solid var(--accent); user-select: none; }}
th:hover {{ color: var(--accent); }}
th .arrow {{ font-size: 9px; margin-left: 3px; }}
td {{ padding: 4px 8px; border-bottom: 1px solid #222; vertical-align: top; max-width: 300px;
     overflow: hidden; text-overflow: ellipsis; }}
tr:hover {{ background: rgba(233, 69, 96, 0.08); }}
tr.match-true {{ }}
tr.match-false {{ background: rgba(233, 69, 96, 0.12); }}
tr.match-null {{ opacity: 0.5; }}

.lang {{ display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px;
         font-weight: 700; text-transform: uppercase; min-width: 30px; text-align: center; }}
.lang-nah {{ background: var(--nah); color: #000; }}
.lang-spa {{ background: var(--spa); color: #000; }}
.lang-oth {{ background: var(--oth); color: #fff; }}
.lang-may {{ background: var(--may); color: #000; }}
.lang-unk {{ background: var(--unk); color: #fff; }}
.lang-eng {{ background: #6c5ce7; color: #fff; }}
.lang-lat {{ background: #a29bfe; color: #000; }}
.lang-sil {{ background: #333; color: #666; }}

.source {{ font-size: 10px; padding: 1px 4px; border-radius: 2px; }}
.source-ft {{ background: #00b894; color: #000; }}
.source-llm {{ background: #fdcb6e; color: #000; }}
.source-whisper {{ background: #636e72; color: #fff; }}
.source-ipa {{ background: #a29bfe; color: #000; }}

.marker {{ background: var(--accent); color: #fff; padding: 0 3px; border-radius: 2px;
           font-size: 10px; margin-right: 2px; }}
.ejective {{ background: var(--may); color: #000; }}

.ipa {{ font-size: 11px; color: var(--fg2); word-break: break-all; }}
.ipa-highlight {{ color: var(--accent); font-weight: 700; }}

.error-type {{ font-size: 10px; font-weight: 700; }}
.cost {{ font-size: 10px; }}
.cost-high {{ color: var(--err); font-weight: 700; }}
.cost-med {{ color: var(--warn); }}
.cost-low {{ color: var(--fg2); }}
.cost-zero {{ color: var(--ok); }}

.text-cell {{ font-size: 11px; max-width: 250px; word-break: break-word; }}
.duration {{ font-size: 10px; color: var(--fg2); }}

.match-icon {{ font-size: 14px; }}
.spk-tip {{ cursor: help; border-bottom: 1px dotted var(--fg2); }}

#count {{ font-size: 11px; color: var(--fg2); padding: 4px 20px; }}

.ipa-cell {{ max-width: 200px; word-break: break-all; font-size: 11px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Tenepal Segment Dashboard</h1>
  <div class="stat">Segments: <b>{len(rows)}</b></div>
  <div class="stat">Annotated: <b>{total_annotated}</b></div>
  <div class="stat">Correct: <b>{total_correct}/{total_annotated} ({total_correct/max(total_annotated,1):.1%})</b></div>
  <div class="stat err">Errors: <b>{total_errors}</b> (cost {total_cost:.0f})</div>
</div>

<div class="filters">
  <label>Show:</label>
  <select id="fShow">
    <option value="all">All segments</option>
    <option value="annotated">Annotated only</option>
    <option value="errors">Errors only</option>
    <option value="correct">Correct only</option>
  </select>
  <label>Speaker:</label>
  <select id="fSpeaker"><option value="">all</option></select>
  <label>GT:</label>
  <select id="fGT"><option value="">all</option></select>
  <label>Pred:</label>
  <select id="fPred"><option value="">all</option></select>
  <label>Error:</label>
  <select id="fError"><option value="">all</option></select>
  <label>Source:</label>
  <select id="fSource">
    <option value="">all</option>
    <option value="ft">FT</option>
    <option value="llm">LLM</option>
    <option value="whisper">Whisper</option>
  </select>
  <label>IPA markers:</label>
  <select id="fMarker">
    <option value="">any</option>
    <option value="has">has NAH marker</option>
    <option value="none">no NAH marker</option>
    <option value="ejective">has ejective</option>
  </select>
  <label>Search:</label>
  <input type="text" id="fSearch" placeholder="text or IPA...">
  <button class="btn" onclick="resetFilters()">Reset</button>
</div>

<div id="count"></div>
<div class="table-wrap">
<table id="mainTable">
<thead>
<tr>
  <th data-col="cue" data-type="num">#</th>
  <th data-col="start_tc">Time</th>
  <th data-col="duration" data-type="num">Dur</th>
  <th data-col="speaker">Speaker</th>
  <th data-col="gt_lang">GT</th>
  <th data-col="pred_lang">Pred</th>
  <th data-col="match">OK</th>
  <th data-col="error_type">Error</th>
  <th data-col="cost" data-type="num">Cost</th>
  <th data-col="source">Src</th>
  <th data-col="text">Whisper Text</th>
  <th data-col="ipa_fused">IPA Fused</th>
  <th data-col="ipa_allo">IPA Allo</th>
  <th data-col="ipa_w2v2">IPA W2V2</th>
  <th data-col="nah_markers">Markers</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<script>
const ROWS = {rows_json};
const SPEAKERS = {speakers_json};
const LANGS = {langs_json};
const ERROR_TYPES = {error_types_json};
const SP_PROFILES = {sp_json};

const NAH_MARKERS = new Set(["ts", "tɬ", "kʷ", "tʃʼ", "kʼ", "tɕ"]);

// Populate filter dropdowns
function populateSelect(id, items) {{
  const sel = document.getElementById(id);
  items.forEach(v => {{
    const o = document.createElement('option');
    o.value = v; o.textContent = v.toUpperCase();
    sel.appendChild(o);
  }});
}}
populateSelect('fSpeaker', SPEAKERS);
populateSelect('fGT', LANGS);
populateSelect('fPred', LANGS);
populateSelect('fError', ERROR_TYPES);

function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

function highlightIPA(ipa) {{
  if (!ipa) return '';
  return ipa.split(' ').map(tok => {{
    if (NAH_MARKERS.has(tok)) return '<span class="ipa-highlight">' + esc(tok) + '</span>';
    return esc(tok);
  }}).join(' ');
}}

function langBadge(lang) {{
  if (!lang) return '';
  return '<span class="lang lang-' + lang + '">' + lang.toUpperCase() + '</span>';
}}

function sourceBadge(src) {{
  if (!src) return '';
  return '<span class="source source-' + src + '">' + src.toUpperCase() + '</span>';
}}

function costClass(c) {{
  if (c >= 2.0) return 'cost-high';
  if (c >= 1.0) return 'cost-med';
  if (c > 0) return 'cost-low';
  return 'cost-zero';
}}

function speakerTip(spk) {{
  const p = SP_PROFILES[spk];
  if (!p) return esc(spk);
  let tip = 'GT: ';
  tip += Object.entries(p.gt).map(([k,v]) => k.toUpperCase() + '=' + v).join(', ');
  tip += '\\nPred: ';
  tip += Object.entries(p.pred).map(([k,v]) => k.toUpperCase() + '=' + v).join(', ');
  return '<span class="spk-tip" title="' + esc(tip) + '">' + esc(spk) + '</span>';
}}

function matchIcon(m) {{
  if (m === true) return '<span class="match-icon" style="color:var(--ok)">&#10003;</span>';
  if (m === false) return '<span class="match-icon" style="color:var(--err)">&#10007;</span>';
  return '<span style="color:var(--unk)">-</span>';
}}

let sortCol = 'cue';
let sortAsc = true;

function getFilters() {{
  return {{
    show: document.getElementById('fShow').value,
    speaker: document.getElementById('fSpeaker').value,
    gt: document.getElementById('fGT').value,
    pred: document.getElementById('fPred').value,
    error: document.getElementById('fError').value,
    source: document.getElementById('fSource').value,
    marker: document.getElementById('fMarker').value,
    search: document.getElementById('fSearch').value.toLowerCase(),
  }};
}}

function filterRow(r, f) {{
  if (f.show === 'annotated' && !r.gt_lang) return false;
  if (f.show === 'errors' && r.match !== false) return false;
  if (f.show === 'correct' && r.match !== true) return false;
  if (f.speaker && r.speaker !== f.speaker) return false;
  if (f.gt && r.gt_lang !== f.gt) return false;
  if (f.pred && r.pred_lang !== f.pred) return false;
  if (f.error && r.error_type !== f.error) return false;
  if (f.source && r.source !== f.source) return false;
  if (f.marker === 'has' && r.nah_markers.length === 0) return false;
  if (f.marker === 'none' && r.nah_markers.length > 0) return false;
  if (f.marker === 'ejective' && r.ejectives.length === 0) return false;
  if (f.search) {{
    const hay = (r.text + ' ' + r.ipa_fused + ' ' + r.ipa_allo + ' ' + r.ipa_w2v2 + ' ' + r.speaker).toLowerCase();
    if (!hay.includes(f.search)) return false;
  }}
  return true;
}}

function sortRows(rows) {{
  const col = sortCol;
  const asc = sortAsc;
  return rows.sort((a, b) => {{
    let va = a[col], vb = b[col];
    if (va == null) va = '';
    if (vb == null) vb = '';
    if (typeof va === 'number' && typeof vb === 'number') return asc ? va - vb : vb - va;
    if (typeof va === 'boolean') {{ va = va ? 1 : 0; vb = vb ? 1 : 0; return asc ? va - vb : vb - va; }}
    va = String(va); vb = String(vb);
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
}}

function render() {{
  const f = getFilters();
  let filtered = ROWS.filter(r => filterRow(r, f));
  filtered = sortRows(filtered);

  document.getElementById('count').textContent =
    'Showing ' + filtered.length + ' of ' + ROWS.length + ' segments' +
    (filtered.length !== ROWS.length ? ' (filtered)' : '');

  const tbody = document.getElementById('tbody');
  // Build HTML in chunks for performance
  const chunks = [];
  for (const r of filtered) {{
    const mc = r.match === true ? 'match-true' : (r.match === false ? 'match-false' : 'match-null');
    const markers = r.nah_markers.map(m => '<span class="marker">' + esc(m) + '</span>').join('') +
                    r.ejectives.map(m => '<span class="marker ejective">' + esc(m) + '</span>').join('');

    chunks.push('<tr class="' + mc + '">' +
      '<td>' + r.cue + '</td>' +
      '<td style="white-space:nowrap;font-size:10px">' + esc(r.start_tc || '') + '</td>' +
      '<td class="duration">' + (r.duration ? r.duration.toFixed(1) + 's' : '') + '</td>' +
      '<td>' + speakerTip(r.speaker) + '</td>' +
      '<td>' + langBadge(r.gt_lang) + '</td>' +
      '<td>' + langBadge(r.pred_lang) + '</td>' +
      '<td style="text-align:center">' + matchIcon(r.match) + '</td>' +
      '<td class="error-type">' + esc(r.error_type) + '</td>' +
      '<td class="cost ' + costClass(r.cost) + '">' + (r.cost ? r.cost.toFixed(1) : '') + '</td>' +
      '<td>' + sourceBadge(r.source) + '</td>' +
      '<td class="text-cell">' + esc(r.text) + '</td>' +
      '<td class="ipa-cell">' + highlightIPA(r.ipa_fused) + '</td>' +
      '<td class="ipa-cell">' + highlightIPA(r.ipa_allo) + '</td>' +
      '<td class="ipa-cell">' + highlightIPA(r.ipa_w2v2) + '</td>' +
      '<td>' + markers + '</td>' +
    '</tr>');
  }}
  tbody.innerHTML = chunks.join('');
}}

// Sort on header click
document.querySelectorAll('th[data-col]').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else {{ sortCol = col; sortAsc = true; }}
    // Update arrows
    document.querySelectorAll('th .arrow').forEach(a => a.remove());
    const arrow = document.createElement('span');
    arrow.className = 'arrow';
    arrow.textContent = sortAsc ? ' \\u25B2' : ' \\u25BC';
    th.appendChild(arrow);
    render();
  }});
}});

// Filter events
['fShow','fSpeaker','fGT','fPred','fError','fSource','fMarker'].forEach(id => {{
  document.getElementById(id).addEventListener('change', render);
}});
document.getElementById('fSearch').addEventListener('input', render);

function resetFilters() {{
  ['fShow','fSpeaker','fGT','fPred','fError','fSource','fMarker'].forEach(id => {{
    document.getElementById(id).value = '';
  }});
  document.getElementById('fShow').value = 'all';
  document.getElementById('fSearch').value = '';
  render();
}}

// Initial render
render();
</script>
</body>
</html>"""

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Dashboard: {out_path} ({len(rows)} segments, {total_errors} errors)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Segment dashboard generator")
    parser.add_argument("gt", help="Ground truth JSON")
    parser.add_argument("srt", help="Pipeline SRT output")
    parser.add_argument("--out", default="tools/segment_dashboard.html", help="Output HTML")
    args = parser.parse_args()

    gt = load_gt(args.gt)
    segments = parse_srt(args.srt)
    generate_html(segments, gt, args.out)


if __name__ == "__main__":
    main()
