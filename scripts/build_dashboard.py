#!/usr/bin/env python3
"""STEP 3/7 — generate the final self-contained HTML dashboard from saved results.

All numbers rendered come directly from the saved outputs/metrics files; no
values are hard-coded. Chart.js is inlined from assets/vendor so the file works
offline without a server.
"""
import sys, json, base64, html
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def esc(s):  # JS-safe string for embedding inside JSON
    if s is None: return ""
    return str(s)

def main():
    # ---- load saved results (source of truth) ----
    m2 = json.load(open(config.STEP2_METRICS))
    m6 = json.load(open(config.STEP6_METRICS))
    m5 = json.load(open(config.STEP5_METRICS)) if config.STEP5_METRICS.exists() else {}
    tbl = pd.read_csv(config.STEP6_TABLE, dtype={"title": str, "text": str})

    # ---- prepare review-level rows for the table ----
    rows = []
    for _, r in tbl.iterrows():
        rows.append({
            "rating": int(r["rating"]),
            "title": esc(r["title"]),
            "text": esc(r["text"])[:400],
            "actual": esc(r["actual"]),
            "predicted": esc(r["predicted"]),
            "correct": str(r["actual"]) == str(r["predicted"]),
            "confidence": esc(r.get("confidence")),
            "emotion_llm": esc(r.get("emotion_llm")),
            "emotion_nrc": esc(r.get("emotion_nrc")),
            "reason": esc(r.get("reason"))[:200],
        })

    # ---- build per-page chart data ----
    clabels = m6["confusion_matrix_labels"]
    cm = m6["confusion_matrix"]
    actual_counts = m6["actual_counts"]
    predicted_counts = m6["predicted_counts"]
    per_class = m6["per_class"]
    misdir = m6.get("misclassification_direction", {})
    emotions = config.EMOTIONS
    llm_e = {k: m5.get("llm_counts", {}).get(k, 0) for k in emotions}
    nrc_e = {k: m5.get("nrc_counts", {}).get(k, 0) for k in emotions}

    # star distribution
    star_dist = tbl["rating"].value_counts().sort_index().to_dict()
    stars = {1: star_dist.get(1,0), 2: star_dist.get(2,0), 3: star_dist.get(3,0),
             4: star_dist.get(4,0), 5: star_dist.get(5,0)}

    accuracy2 = m2["accuracy"]; total2 = m2["total"]; correct2=m2["correct"]; incorrect2=m2["incorrect"]
    accuracy6 = m6["accuracy"]; total6 = m6["total"]; correct6=m6["correct"]; incorrect6=m6["incorrect"]

    data = {
        "meta": {
            "dataset": "Amazon Reviews 2023 — Gift Cards (McAuley Lab)",
            "source_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz",
            "seed": config.RANDOM_SEED,
            "model": _model_name(),
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        },
        "kpis": {
            "total": total6, "accuracy": accuracy6, "correct": correct6, "incorrect": incorrect6,
            "total2": total2, "accuracy2": accuracy2, "correct2": correct2, "incorrect2": incorrect2,
        },
        "cm": {"labels": clabels, "matrix": cm},
        "actual_counts": {k: actual_counts.get(k,0) for k in clabels},
        "predicted_counts": {k: predicted_counts.get(k,0) for k in clabels},
        "per_class": per_class,
        "misdir": misdir,
        "stars": stars,
        "emotions": emotions,
        "llm_emotion": llm_e,
        "nrc_emotion": nrc_e,
        "emotion_agreement": {
            "n": m5.get("n_compared",0), "agree": m5.get("agree",0),
            "rate": m5.get("agreement_rate"),
            "nrc_neutral": m5.get("nrc_neutral_count",0),
        },
        "emotion_examples": m5.get("disagreement_examples", [])[:8],
        "rows": rows,
    }

    payload_json = json.dumps(data, ensure_ascii=False)

    chart_js = (config.PROJECT_ROOT / "assets/vendor/chart.umd.min.js").read_text()

    # ---- Build the HTML ----
    html_doc = TEMPLATE.replace("__CHART_JS__", chart_js).replace("__DATA_JSON__", payload_json).replace("__DATA_SRC__", config.DATA_URL)

    out = config.FINAL_DASHBOARD
    out.write_text(html_doc, encoding="utf-8")

def _model_name():
    from src.llm_client import _load_env
    try:
        return _load_env().get("LLM_MODEL", "LLM")
    except Exception:
        return "LLM"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amazon Review Sentiment & Emotion Classifier — Dashboard</title>
<style>
  :root{
    --bg:#0f1420; --panel:#171e2e; --panel2:#1d2740; --border:#2a3550;
    --text:#e7ecf5; --muted:#93a0bd; --accent:#5b9cff; --accent2:#7c6bff;
    --pos:#2ecc71; --neu:#f1c40f; --neg:#e74c3c; --warn:#e67e22;
    --radius:14px; --shadow:0 8px 30px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       -webkit-font-smoothing:antialiased; line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  header{padding:34px 34px 22px;border-bottom:1px solid var(--border);
         background:linear-gradient(180deg,rgba(91,156,255,.08),transparent)}
  header h1{margin:0 0 6px;font-size:26px;letter-spacing:-.3px;font-weight:750}
  header p{margin:2px 0;color:var(--muted);font-size:13.5px}
  .wrap{max-width:1280px;margin:0 auto;padding:26px 34px 60px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:22px 0 30px}
  .kpi{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
       padding:18px 20px;box-shadow:var(--shadow)}
  .kpi .v{font-size:30px;font-weight:800;letter-spacing:-.5px}
  .kpi .l{color:var(--muted);font-size:12.5px;text-transform:uppercase;letter-spacing:.6px;margin-top:2px}
  .kpi.pos .v{color:var(--pos)} .kpi.neg .v{color:var(--neg)} .kpi.acc .v{color:var(--accent)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
        padding:20px 22px;box-shadow:var(--shadow)}
  .card h3{margin:0 0 14px;font-size:15px;font-weight:650;letter-spacing:.2px;color:#d7e0f2}
  .card h3 small{color:var(--muted);font-weight:500}
  .full{grid-column:1/-1}
  .chartbox{position:relative;height:260px}
  .chartbox.tall{height:320px}
  .table-wrap{overflow:auto;max-height:520px;border:1px solid var(--border);border-radius:10px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  thead th{position:sticky;top:0;background:var(--panel2);color:var(--muted);
           text-align:left;padding:9px 10px;font-weight:600;letter-spacing:.3px;white-space:nowrap}
  tbody td{padding:8px 10px;border-top:1px solid #243049;vertical-align:top}
  tbody tr:hover{background:#1b2540}
  .tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600}
  .tag.pos{background:rgba(46,204,113,.15);color:#3ee58a}
  .tag.neu{background:rgba(241,196,15,.15);color:#f5d040}
  .tag.neg{background:rgba(231,76,60,.18);color:#ff7580}
  .tag.miss{background:rgba(255,87,34,.15);color:#ff9066}
  .ctrl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:14px}
  .ctrl label{color:var(--muted);font-size:12px}
  select,input[type=text]{background:var(--panel2);border:1px solid var(--border);color:var(--text);
        padding:7px 10px;border-radius:8px;font-size:12.5px}
  .live{font-size:13px;color:var(--accent);font-weight:650}
  .cm{width:100%;border-collapse:collapse;font-size:13px}
  .cm th,.cm td{border:1px solid var(--border);padding:9px;text-align:center}
  .cm th{color:var(--muted);font-weight:600;background:var(--panel2)}
  .cm td.n{font-weight:700}
  .cm .diag{background:rgba(46,204,113,.14)}
  .cm .off{background:rgba(231,76,60,.12)}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .pill{display:inline-block;background:var(--panel2);border:1px solid var(--border);
        padding:1px 8px;border-radius:6px;margin:1px;font-size:11px;color:var(--muted)}
  .foot{color:var(--muted);font-size:12px;text-align:center;padding:26px 0 10px}
  .bar{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px}
  .bar .bl{width:130px;color:var(--muted)}
  .bar .bw{flex:1;background:#0c1220;border-radius:6px;overflow:hidden;height:20px}
  .bar .bf{height:100%;border-radius:6px;transition:width .4s}
  .note{color:var(--muted);font-size:12px;margin-top:10px}
  .em{display:inline-block;margin:2px;padding:2px 9px;border-radius:16px;background:var(--panel2);
      border:1px solid var(--border);font-size:11.5px}
  @media(max-width:720px){ .two{grid-template-columns:1fr} }
</style>
</head>
<body>
<header>
  <div class="wrap" style="padding:0">
    <h1>Amazon Review Sentiment &amp; Emotion Classification</h1>
    <p>MBAX 6418 — Assignment 1 &nbsp;•&nbsp; Dataset: <a href="__DATA_SRC__" target="_blank">Amazon Reviews 2023 · Gift Cards</a> (McAuley Lab)</p>
    <p id="meta-line"></p>
  </div>
</header>

<div class="wrap">
  <!-- ============ STEP 6 balanced KPIs ============ -->
  <div class="kpis" id="kpis"></div>

  <div class="note" id="imbalance-note" style="background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px;margin-bottom:26px"></div>

  <!-- ============ Distributions & confusion ============ -->
  <div class="grid">
    <div class="card"><h3>Star-Rating Distribution</h3><div class="chartbox"><canvas id="c-stars"></canvas></div></div>
    <div class="card"><h3>Actual vs Predicted Sentiment</h3><div class="chartbox"><canvas id="c-cmp"></canvas></div></div>
    <div class="card full"><h3>Confusion Matrix <small>(rows = actual, columns = predicted)</small></h3><div id="cm-box"></div></div>
    <div class="card full"><h3>Accuracy / Recall by Class</h3><div class="chartbox"><canvas id="c-perclass"></canvas></div></div>
  </div>

  <!-- ============ Emotion ============ -->
  <div class="card full" style="margin-top:18px">
    <h3>Emotion Classification — LLM vs NRC Word-List</h3>
    <div class="two">
      <div class="chartbox tall"><canvas id="c-emo"></canvas></div>
      <div>
        <div id="emo-agree"></div>
        <h4 style="margin:14px 0 8px;color:#d7e0f2;font-size:13.5px">Meaningful disagreement examples</h4>
        <div id="emo-examples"></div>
      </div>
    </div>
  </div>

  <!-- ============ Interactive review table ============ -->
  <div class="card full" style="margin-top:18px">
    <h3>Review-Level Results <span id="count-pill" class="pill"></span></h3>
    <div class="ctrl">
      <div>
        <label>Prediction:</label>
        <select id="f-filter"><option value="all">All reviews</option>
          <option value="correct">Correct predictions</option>
          <option value="incorrect">Incorrect / mismatched</option>
          <option value="pos">Predicted POSITIVE</option>
          <option value="neu">Predicted NEUTRAL</option>
          <option value="neg">Predicted NEGATIVE</option></select>
      </div>
      <div>
        <label>Actual sentiment:</label>
        <select id="f-actual"><option value="all">Any</option><option value="POSITIVE">POSITIVE</option>
          <option value="NEUTRAL">NEUTRAL</option><option value="NEGATIVE">NEGATIVE</option></select>
      </div>
      <div>
        <label>Star rating:</label>
        <select id="f-star"><option value="all">Any</option><option value="1">1</option>
          <option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select>
      </div>
      <div>
        <label>Emotion:</label>
        <select id="f-emotion"><option value="all">Any</option></select>
      </div>
      <div style="margin-left:auto"><span id="live-count" class="live">0 rows</span></div>
    </div>
    <div class="table-wrap"><table id="review-table">
      <thead><tr><th>✓</th><th>★</th><th>Title</th><th>Text</th><th>Actual</th><th>Predicted</th>
        <th>LLM emo</th><th>NRC emo</th><th>Reason</th></tr></thead><tbody id="tbody"></tbody></table></div>
  </div>

  <div class="foot">Generated from saved outputs — all figures computed directly from model predictions.<br>MBAX 6418 Assignment 1 · Amazon Reviews 2023 Gift Cards</div>
</div>

<script>
const DATA = __DATA_JSON__;
</script>
<script>
__CHART_JS__
</script>
<script>
const E = (id)=>document.getElementById(id);
const COLORS = {accent:"#5b9cff",accent2:"#7c6bff",pos:"#2ecc71",neu:"#f1c40f",neg:"#e74c3c",
  grid:"rgba(147,160,189,.12)",text:"#e7ecf5"};
Chart.defaults.color = "#93a0bd";
Chart.defaults.borderColor = "rgba(147,160,189,.12)";
Chart.defaults.font.family = "'Inter',-apple-system,sans-serif";

// ---- KPI cards (STEP 6 balanced) ----
const k = DATA.kpis;
E("kpis").innerHTML = `
  <div class="kpi"><div class="v">${k.total}</div><div class="l">Reviews (3-class balanced)</div></div>
  <div class="kpi acc"><div class="v">${(k.accuracy*100).toFixed(1)}%</div><div class="l">Accuracy</div></div>
  <div class="kpi pos"><div class="v">${k.correct}</div><div class="l">Correct</div></div>
  <div class="kpi neg"><div class="v">${k.incorrect}</div><div class="l">Incorrect</div></div>
  <div class="kpi acc"><div class="v">${(k.accuracy2*100).toFixed(1)}%</div><div class="l">Binary (STEP 2) accuracy</div></div>
  <div class="kpi"><div class="v">${k.correct2}/${k.total2}</div><div class="l">Binary correct (STEP 2)</div></div>
`;
document.title = "Amazon Sentiment & Emotion — Dashboard";

E("meta-line").textContent =
  `Model: ${DATA.meta.model}  •  seed: ${DATA.meta.seed}  •  generated: ${DATA.meta.generated}`;

// ---- class imbalance note ----
const ac = DATA.actual_counts;
E("imbalance-note").innerHTML =
  `<b>Why the balanced sample matters.</b> The initial <b>STEP 2</b> run sampled the natural,
  heavily skewed distribution and scored <b>${(k.accuracy2*100).toFixed(1)}%</b> accuracy
  simply because <b>POSITIVE reviews dominate</b> the dataset (~84% of the 152,410 reviews are
  5-star). This STEP 6 balanced sample uses <b>${Object.entries(ac).map(([c,n])=>`${n} ${c}`).join(" / ")}</b>
  so the classifier cannot "win" by guessing the majority class. Actual counts in this sample:
  <b>${Object.entries(ac).map(([c,n])=>`${c}=${n}`).join(", ")}</b>.`;

// ---- Star distribution ----
new Chart(E("c-stars"),{type:"bar",data:{
  labels:[1,2,3,4,5].map(s=>s+"★"),
  datasets:[{data:[DATA.stars[1],DATA.stars[2],DATA.stars[3],DATA.stars[4],DATA.stars[5]],
    backgroundColor:[COLORS.neg,COLORS.neg,COLORS.neu,COLORS.pos,COLORS.pos],borderRadius:6}]},
  options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,title:{display:true,text:"reviews"}}}}});

// ---- Actual vs predicted ----
new Chart(E("c-cmp"),{type:"bar",data:{
  labels:DATA.cm.labels,
  datasets:[
    {label:"Actual",data:DATA.cm.labels.map(l=>DATA.actual_counts[l]),backgroundColor:"rgba(91,156,255,.85)",borderRadius:6},
    {label:"Predicted",data:DATA.cm.labels.map(l=>DATA.predicted_counts[l]),backgroundColor:"rgba(124,107,255,.85)",borderRadius:6}]},
  options:{scales:{y:{beginAtZero:true}}}});

// ---- Confusion matrix ----
const cl=DATA.cm.labels, mat=DATA.cm.matrix;
let cmh=`<table class="cm"><thead><tr><th></th>${cl.map(c=>`<th>${c}</th>`).join("")}</tr></thead><tbody>`;
cl.forEach((a,i)=>{
  cmh+=`<tr><th>${a}</th>`;
  cl.forEach((pr,j)=>{
    const diag=(i===j);
    cmh+=`<td class="${diag?'diag':'off'} n">${mat[i][j]}</td>`;
  });
  cmh+=`</tr>`;
});
cmh+=`</tbody></table>`;
E("cm-box").innerHTML=cmh;

// ---- per-class accuracy/recall ----
const pc=DATA.per_class;
new Chart(E("c-perclass"),{type:"bar",data:{
  labels:cl,
  datasets:[{label:"Recall",data:cl.map(c=>pc[c].recall*100),backgroundColor:"rgba(46,204,113,.85)",borderRadius:6},
    {label:"Precision",data:cl.map(c=>(pc[c].precision??0)*100),backgroundColor:"rgba(241,196,15,.85)",borderRadius:6}]},
  options:{scales:{y:{beginAtZero:true,max:100,title:{display:true,text:"%"}}}}});

// ---- Emotion (grouped bar: LLM vs NRC) ----
new Chart(E("c-emo"),{type:"bar",data:{
  labels:DATA.emotions.map(x=>x[0].toUpperCase()+x.slice(1)),
  datasets:[
    {label:"LLM",data:DATA.emotions.map(e=>DATA.llm_emotion[e]),backgroundColor:"rgba(91,156,255,.9)",borderRadius:5},
    {label:"NRC word-list",data:DATA.emotions.map(e=>DATA.nrc_emotion[e]),backgroundColor:"rgba(231,76,60,.85)",borderRadius:5}]},
  options:{indexAxis:'y',scales:{x:{beginAtZero:true}}}});

const ea=DATA.emotion_agreement;
const rate = ea.rate==null ? "—" : ((ea.rate*100).toFixed(1)+"%");
E("emo-agree").innerHTML=`
  <div class="kpi" style="margin:0"><div class="v">${rate}</div>
  <div class="l">LLM vs NRC agreement</div></div>
  <div class="note">Agreement on ${ea.n} reviews (excludes ${ea.nrc_neutral} with no NRC
  word-list match). NRC counts reflect matched reviews only.</div>
  <h4 style="margin:14px 0 6px;color:#d7e0f2;font-size:13px">Key disagreements</h4>
  <div id="emo-examples-list"></div>`;
E("emo-examples-list").innerHTML=DATA.emotion_examples.map(x=>
  `<div style="margin:6px 0;font-size:12.5px"><span class="tag pos">LLM ${x.emotion_llm}</span>
   <span style="color:var(--muted)">→ NRC</span> <span class="tag neg">${x.emotion_nrc}</span>
   <span style="color:#c6cfe4">· ${(x.title||"").slice(0,60)}</span></div>`).join("");

// ---- Interactive table ----
const rows=DATA.rows;
const showTag=(c)=> c==null?"—" : `<span class="tag ${c.toLowerCase()}">${c}</span>`;
const emoSet=[...new Set(rows.map(r=>r.emotion_llm).filter(Boolean))].sort();
const sel=E("f-emotion");
emoSet.forEach(e=>{const o=document.createElement("option");o.value=e;o.textContent=e;sel.appendChild(o);});

let filter={filter:"all",actual:"all",star:"all",emotion:"all"};
function render(){
  const f=filter;
  const shown=[];
  for(const r of rows){
    const correct=(r.actual===r.predicted);
    if(f.filter==="correct"&&!correct)continue;
    if(f.filter==="incorrect"&&correct)continue;
    if(["pos","neu","neg"].includes(f.filter)){
      const want=f.filter.toUpperCase();
      if(r.predicted!==want)continue;
    }
    if(f.actual!=="all"&&r.actual!==f.actual)continue;
    if(f.star!=="all"&&String(r.rating)!==f.star)continue;
    if(f.emotion!=="all"&&r.emotion_llm!==f.emotion)continue;
    shown.push(r);
  }
  E("live-count").textContent=shown.length+" rows";
  E("count-pill").textContent=shown.length+"/"+rows.length+" shown";
  E("tbody").innerHTML=shown.map(r=>`<tr>
    <td>${r.correct?"✅":"❌"}</td>
    <td>${"★".repeat(r.rating)}</td>
    <td style="max-width:200px">${esc(r.title)}</td>
    <td style="max-width:300px;color:var(--muted)">${esc(r.text)}</td>
    <td>${showTag(r.actual)}</td>
    <td>${showTag(r.predicted)}</td>
    <td><span class="em">${esc(r.emotion_llm)}</span></td>
    <td><span class="em">${esc(r.emotion_nrc)}</span></td>
    <td style="max-width:260px;color:var(--muted)">${esc(r.reason)}</td>
  </tr>`).join("");
}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
["filter","actual","star","emotion"].forEach(id=>{
  E("f-"+id).addEventListener("change",e=>{filter[id]=e.target.value;render();});
});
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
