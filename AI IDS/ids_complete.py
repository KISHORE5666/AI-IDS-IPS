import os, csv, json, time, random, threading, warnings
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from flask import (Flask, render_template_string, request,
                   redirect, url_for, session, Response, jsonify)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SECRET_KEY      = 'ids-secret-2025'
USERS           = {'admin': 'admin123', 'analyst': 'analyst123'}
LOG_FILE        = 'logs/attack_history.csv'
ALERT_LOG       = 'logs/alerts.log'
ALERT_THRESHOLD = 61          # risk score >= this fires an alert

RISK_BASE = {
    'Normal':          10,
    'DoS':             75,
    'BruteForce':      70,
    'PortScan':        55,
    'SuspiciousLogin': 60,
}

LOG_HEADERS = ['timestamp','protocol','flag','src_bytes','dst_bytes',
               'num_connections','login_attempts','duration',
               'label','risk_score','risk_level','alert_reasons']

# ─────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────
def generate_dataset():
    np.random.seed(42)
    data = []

    # Normal traffic
    for _ in range(700):
        data.append({
            'duration':        np.random.randint(1, 60),
            'src_bytes':       np.random.randint(100, 5000),
            'dst_bytes':       np.random.randint(100, 5000),
            'num_connections': np.random.randint(1, 20),
            'protocol':        np.random.choice(['TCP','UDP','ICMP'], p=[0.7,0.2,0.1]),
            'flag':            np.random.choice(['SF','S0','REJ'], p=[0.92,0.05,0.03]),
            'login_attempts':  np.random.randint(0, 2),
            'label':           'Normal'
        })

    # DoS
    for _ in range(250):
        data.append({
            'duration':        np.random.randint(0, 5),
            'src_bytes':       np.random.randint(8000, 60000),
            'dst_bytes':       np.random.randint(0, 100),
            'num_connections': np.random.randint(150, 600),
            'protocol':        'TCP',
            'flag':            np.random.choice(['S0','REJ'], p=[0.7,0.3]),
            'login_attempts':  0,
            'label':           'DoS'
        })

    # BruteForce
    for _ in range(200):
        data.append({
            'duration':        np.random.randint(10, 120),
            'src_bytes':       np.random.randint(200, 3000),
            'dst_bytes':       np.random.randint(100, 2000),
            'num_connections': np.random.randint(5, 30),
            'protocol':        'TCP',
            'flag':            np.random.choice(['SF','REJ'], p=[0.5,0.5]),
            'login_attempts':  np.random.randint(5, 15),
            'label':           'BruteForce'
        })

    # PortScan
    for _ in range(200):
        data.append({
            'duration':        np.random.randint(0, 3),
            'src_bytes':       np.random.randint(10, 400),
            'dst_bytes':       np.random.randint(0, 50),
            'num_connections': np.random.randint(60, 400),
            'protocol':        np.random.choice(['TCP','ICMP'], p=[0.6,0.4]),
            'flag':            'S0',
            'login_attempts':  0,
            'label':           'PortScan'
        })

    # SuspiciousLogin
    for _ in range(200):
        data.append({
            'duration':        np.random.randint(5, 60),
            'src_bytes':       np.random.randint(500, 4000),
            'dst_bytes':       np.random.randint(200, 3000),
            'num_connections': np.random.randint(1, 10),
            'protocol':        'TCP',
            'flag':            np.random.choice(['SF','REJ'], p=[0.6,0.4]),
            'login_attempts':  np.random.randint(3, 8),
            'label':           'SuspiciousLogin'
        })

    return pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────
def train_model():
    print("[IDS] Generating dataset (1,550 records)...")
    df = generate_dataset()

    le_p = LabelEncoder().fit(df['protocol'])
    le_f = LabelEncoder().fit(df['flag'])
    le_l = LabelEncoder().fit(df['label'])

    df['protocol_enc'] = le_p.transform(df['protocol'])
    df['flag_enc']     = le_f.transform(df['flag'])
    df['label_enc']    = le_l.transform(df['label'])

    FEATS = ['duration','src_bytes','dst_bytes','num_connections',
             'protocol_enc','flag_enc','login_attempts']
    X, y = df[FEATS], df['label_enc']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("[IDS] Training Random Forest (150 estimators)...")
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred).tolist()
    report = classification_report(y_te, y_pred,
                                   target_names=le_l.classes_, output_dict=True)

    fi = sorted(zip(FEATS, clf.feature_importances_), key=lambda x: -x[1])

    print(f"[IDS] Accuracy: {acc*100:.2f}%")
    print("[IDS] Model ready.\n")

    return {
        'clf': clf, 'le_l': le_l, 'le_p': le_p, 'le_f': le_f,
        'accuracy': round(acc * 100, 2),
        'cm': cm, 'cm_labels': list(le_l.classes_),
        'report': report,
        'feature_importance': [[n, round(float(v),4)] for n,v in fi],
    }

# ─────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────
def get_risk_score(label, pkt):
    score = RISK_BASE.get(label, 10)
    if pkt['num_connections'] > 300: score = min(score + 15, 100)
    if pkt['src_bytes']       > 30000: score = min(score + 10, 100)
    if pkt['login_attempts']  > 8:     score = min(score + 12, 100)
    if pkt['flag'] in ('S0','REJ') and pkt['num_connections'] > 100:
        score = min(score + 8, 100)
    return round(score)

def risk_level(score):
    if score > 85: return 'Critical'
    if score > 60: return 'High'
    if score > 30: return 'Medium'
    return 'Low'

def alert_reasons(pkt):
    r = []
    if pkt['num_connections'] > 100:  r.append('High Connections')
    if pkt['src_bytes'] > 20000:      r.append('Large Src Bytes')
    if pkt['login_attempts'] > 3:     r.append('Brute Force Login')
    if pkt['flag'] in ('S0','REJ') and pkt['num_connections'] > 50:
        r.append('SYN Flood Pattern')
    return r

# ─────────────────────────────────────────────────────────
# PACKET GENERATOR
# ─────────────────────────────────────────────────────────
def make_packet():
    if random.random() < 0.62:
        return {
            'duration':        random.randint(1,60),
            'src_bytes':       random.randint(100,5000),
            'dst_bytes':       random.randint(100,5000),
            'num_connections': random.randint(1,20),
            'protocol':        random.choice(['TCP','UDP','ICMP']),
            'flag':            random.choice(['SF','SF','SF','S0']),
            'login_attempts':  random.randint(0,1),
        }
    t = random.choice(['DoS','BruteForce','PortScan','SuspiciousLogin'])
    if t == 'DoS':
        return {'duration':random.randint(0,4),'src_bytes':random.randint(8000,60000),
                'dst_bytes':random.randint(0,80),'num_connections':random.randint(150,600),
                'protocol':'TCP','flag':random.choice(['S0','REJ']),'login_attempts':0}
    elif t == 'BruteForce':
        return {'duration':random.randint(10,120),'src_bytes':random.randint(200,3000),
                'dst_bytes':random.randint(100,2000),'num_connections':random.randint(5,30),
                'protocol':'TCP','flag':random.choice(['SF','REJ']),'login_attempts':random.randint(5,15)}
    elif t == 'PortScan':
        return {'duration':random.randint(0,2),'src_bytes':random.randint(10,300),
                'dst_bytes':random.randint(0,40),'num_connections':random.randint(80,400),
                'protocol':random.choice(['TCP','ICMP']),'flag':'S0','login_attempts':0}
    else:
        return {'duration':random.randint(5,60),'src_bytes':random.randint(500,4000),
                'dst_bytes':random.randint(200,3000),'num_connections':random.randint(1,10),
                'protocol':'TCP','flag':random.choice(['SF','REJ']),'login_attempts':random.randint(3,8)}

# ─────────────────────────────────────────────────────────
# PREDICT PACKET
# ─────────────────────────────────────────────────────────
def predict_packet(pkt, mdl):
    try:
        pe = mdl['le_p'].transform([pkt['protocol']])[0]
        fe = mdl['le_f'].transform([pkt['flag']])[0]
    except Exception:
        pe, fe = 0, 0

    X     = [[pkt['duration'], pkt['src_bytes'], pkt['dst_bytes'],
              pkt['num_connections'], pe, fe, pkt['login_attempts']]]
    pred  = mdl['clf'].predict(X)[0]
    label = mdl['le_l'].inverse_transform([pred])[0]
    score = get_risk_score(label, pkt)
    level = risk_level(score)
    reasons = alert_reasons(pkt)

    return {
        'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'protocol':        pkt['protocol'],
        'flag':            pkt['flag'],
        'src_bytes':       pkt['src_bytes'],
        'dst_bytes':       pkt['dst_bytes'],
        'num_connections': pkt['num_connections'],
        'login_attempts':  pkt['login_attempts'],
        'duration':        pkt['duration'],
        'label':           label,
        'risk_score':      score,
        'risk_level':      level,
        'alert_reasons':   ', '.join(reasons) if reasons else '—',
    }

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────
def init_log():
    os.makedirs('logs', exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=LOG_HEADERS).writeheader()

def write_log(entry):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=LOG_HEADERS).writerow(
            {k: entry.get(k,'') for k in LOG_HEADERS})

def write_alert(entry):
    with open(ALERT_LOG, 'a') as f:
        f.write(f"[{entry['timestamp']}] {entry['risk_level'].upper()} | "
                f"{entry['label']} | Score:{entry['risk_score']} | "
                f"{entry['alert_reasons']}\n")

def read_log(limit=300):
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    with open(LOG_FILE, 'r') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows[-limit:]

# ─────────────────────────────────────────────────────────
# HTML TEMPLATES
# ─────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IDS Login</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#050d1a;font-family:'Segoe UI',sans-serif;
       display:flex;align-items:center;justify-content:center;min-height:100vh;}
  .card{background:#0a1628;border:1px solid #0d2847;border-radius:12px;
        padding:40px 48px;width:380px;box-shadow:0 20px 60px rgba(0,0,0,.5);}
  .logo{text-align:center;margin-bottom:28px;}
  .logo-icon{width:52px;height:52px;border:2px solid #00d4ff;border-radius:50%;
             display:flex;align-items:center;justify-content:center;
             margin:0 auto 12px;background:rgba(0,212,255,.08);}
  .logo-icon span{color:#00d4ff;font-size:22px;}
  h1{color:#fff;font-size:1.3rem;letter-spacing:2px;text-transform:uppercase;}
  p.sub{color:#3a5a7a;font-size:.75rem;letter-spacing:2px;margin-top:4px;}
  .field{margin-bottom:16px;}
  label{display:block;color:#3a5a7a;font-size:.75rem;letter-spacing:1px;
        text-transform:uppercase;margin-bottom:6px;}
  input{width:100%;background:#060e1c;border:1px solid #0d2847;border-radius:6px;
        padding:10px 14px;color:#c8e0f4;font-size:.9rem;outline:none;transition:.2s;}
  input:focus{border-color:#00d4ff;}
  button{width:100%;background:linear-gradient(135deg,#1a56a0,#0d2f60);
         border:none;border-radius:6px;padding:12px;color:#fff;
         font-size:.95rem;font-weight:600;letter-spacing:1px;cursor:pointer;
         text-transform:uppercase;margin-top:8px;transition:.2s;}
  button:hover{background:linear-gradient(135deg,#2166c0,#1a3f7a);}
  .error{background:rgba(255,61,107,.12);border:1px solid rgba(255,61,107,.3);
         border-radius:6px;padding:10px 14px;color:#ff6b8a;font-size:.85rem;margin-bottom:14px;}
  .hint{color:#3a5a7a;font-size:.72rem;text-align:center;margin-top:18px;}
  .hint span{color:#00d4ff;}
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <div class="logo-icon"><span>🛡</span></div>
    <h1>NetGuard IDS</h1>
    <p class="sub">AI-Powered Intrusion Detection</p>
  </div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <div class="field">
      <label>Username</label>
      <input type="text" name="username" placeholder="Enter username" required autofocus>
    </div>
    <div class="field">
      <label>Password</label>
      <input type="password" name="password" placeholder="Enter password" required>
    </div>
    <button type="submit">Sign In</button>
  </form>
  <p class="hint">admin / <span>admin123</span> &nbsp;·&nbsp; analyst / <span>analyst123</span></p>
</div>
</body>
</html>"""


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NetGuard IDS Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root{
  --bg:#050d1a; --panel:#080f20; --border:#0d2847;
  --accent:#00d4ff; --accent2:#ff3d6b; --green:#00ff88;
  --yellow:#ffd600; --text:#c8e0f4; --muted:#3a5a7a;
  --mono:'Courier New',monospace; --sans:'Segoe UI',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;}
.app{position:relative;z-index:1;padding:14px;max-width:1500px;margin:0 auto;}

/* Header */
header{display:flex;align-items:center;justify-content:space-between;
       padding:14px 20px;margin-bottom:14px;
       background:#0a1628;border:1px solid var(--border);
       border-radius:8px;border-top:2px solid var(--accent);}
.logo{display:flex;align-items:center;gap:10px;}
.logo-ring{width:34px;height:34px;border:2px solid var(--accent);border-radius:50%;
           display:flex;align-items:center;justify-content:center;
           animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,.4)}
                 50%{box-shadow:0 0 0 7px rgba(0,212,255,0)}}
.logo-ring span{color:var(--accent);font-size:14px;}
.logo-text h1{font-size:1.1rem;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;color:#fff;}
.logo-text p{font-size:.65rem;letter-spacing:2px;color:var(--muted);text-transform:uppercase;}
.header-right{display:flex;align-items:center;gap:14px;}
.live-dot{display:flex;align-items:center;gap:5px;
          font-size:.7rem;font-family:var(--mono);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);
     box-shadow:0 0 7px var(--green);animation:blink 1.4s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.acc-badge{background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);
           border-radius:20px;padding:3px 12px;font-size:.75rem;
           font-family:var(--mono);color:var(--accent);}
.user-info{font-size:.75rem;color:var(--muted);}
.user-info span{color:var(--accent);}
.logout{font-size:.75rem;color:var(--muted);text-decoration:none;
        border:1px solid var(--border);border-radius:4px;padding:3px 10px;transition:.2s;}
.logout:hover{border-color:var(--accent2);color:var(--accent2);}

/* Stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px;}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;
      padding:14px 16px;position:relative;overflow:hidden;}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat.blue::before{background:var(--accent)}
.stat.red::before{background:var(--accent2)}
.stat.green::before{background:var(--green)}
.stat.yellow::before{background:var(--yellow)}
.stat-lbl{font-size:.6rem;letter-spacing:2px;text-transform:uppercase;
           color:var(--muted);font-family:var(--mono);margin-bottom:6px;}
.stat-val{font-size:1.9rem;font-weight:800;line-height:1;}
.stat-val.blue{color:var(--accent)}
.stat-val.red{color:var(--accent2)}
.stat-val.green{color:var(--green)}
.stat-val.yellow{color:var(--yellow)}
.stat-sub{font-size:.65rem;color:var(--muted);margin-top:3px;font-family:var(--mono);}

/* Main grid */
.main-grid{display:grid;grid-template-columns:1fr 320px;gap:12px;margin-bottom:12px;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.ph{display:flex;align-items:center;justify-content:space-between;
    padding:10px 14px;border-bottom:1px solid var(--border);
    background:rgba(0,212,255,.04);}
.ph-title{font-size:.68rem;letter-spacing:2px;text-transform:uppercase;
          font-family:var(--mono);color:var(--accent);}
.ph-badge{font-size:.62rem;font-family:var(--mono);padding:2px 8px;
          border-radius:10px;background:rgba(0,212,255,.15);color:var(--accent);}

/* Traffic table */
.t-wrap{overflow-y:auto;max-height:340px;}
.t-wrap::-webkit-scrollbar{width:3px;}
.t-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.7rem;}
thead th{padding:7px 10px;text-align:left;color:var(--muted);font-weight:400;
         letter-spacing:1px;text-transform:uppercase;position:sticky;top:0;
         background:var(--panel);border-bottom:1px solid var(--border);}
tbody tr{border-bottom:1px solid rgba(13,40,71,.5);transition:.15s;
         animation:row-in .3s ease;}
@keyframes row-in{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:none}}
tbody tr:hover{background:rgba(0,212,255,.04);}
tbody td{padding:6px 10px;}
.badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:.62rem;font-weight:600;}
.badge.Normal{background:rgba(0,255,136,.15);color:var(--green);border:1px solid rgba(0,255,136,.3);}
.badge.DoS,.badge.BruteForce{background:rgba(255,61,107,.15);color:var(--accent2);border:1px solid rgba(255,61,107,.3);}
.badge.PortScan{background:rgba(255,214,0,.15);color:var(--yellow);border:1px solid rgba(255,214,0,.3);}
.badge.SuspiciousLogin{background:rgba(255,140,0,.15);color:#ffa040;border:1px solid rgba(255,140,0,.3);}
.risk-Critical{color:#ff3d6b;font-weight:700;}
.risk-High{color:#ffa040;font-weight:700;}
.risk-Medium{color:var(--yellow);}
.risk-Low{color:var(--green);}
.alert-tag{display:inline-block;padding:1px 5px;border-radius:2px;margin:1px;
           font-size:.58rem;background:rgba(255,214,0,.12);color:var(--yellow);
           border:1px solid rgba(255,214,0,.25);}
.proto-tcp{color:var(--accent);}
.proto-udp{color:#7dd3fc;}
.proto-icmp{color:#a78bfa;}
.flag-sf{color:var(--green);}
.flag-bad{color:var(--accent2);}

/* Feature importance */
.fi-list{padding:14px;display:flex;flex-direction:column;gap:9px;}
.fi-top{display:flex;justify-content:space-between;margin-bottom:3px;
        font-family:var(--mono);font-size:.68rem;}
.fi-name{color:var(--text);}
.fi-pct{color:var(--accent);}
.fi-bg{height:5px;background:rgba(0,212,255,.08);border-radius:3px;overflow:hidden;}
.fi-bar{height:100%;border-radius:3px;
        background:linear-gradient(90deg,var(--accent),#0099cc);transition:width .8s;}

/* Bottom grid */
.bottom-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}

/* Charts */
.chart-wrap{padding:14px;height:200px;position:relative;}

/* Alert log */
.alert-log{padding:10px;max-height:180px;overflow-y:auto;
           display:flex;flex-direction:column;gap:5px;}
.alert-log::-webkit-scrollbar{width:3px;}
.alert-log::-webkit-scrollbar-thumb{background:var(--border);}
.alert-entry{display:flex;gap:7px;padding:7px 9px;
             background:rgba(255,61,107,.07);
             border:1px solid rgba(255,61,107,.2);
             border-radius:4px;animation:row-in .3s ease;}
.alert-time{font-family:var(--mono);font-size:.6rem;color:var(--muted);
            white-space:nowrap;margin-top:1px;}
.alert-body{flex:1;}
.alert-title{font-size:.72rem;color:var(--accent2);font-weight:600;margin-bottom:1px;}
.alert-detail{font-family:var(--mono);font-size:.62rem;color:var(--muted);}
.empty-state{text-align:center;color:var(--muted);
             font-family:var(--mono);font-size:.72rem;padding:18px;}

/* Confusion matrix */
.cm-wrap{padding:14px;display:flex;flex-direction:column;align-items:center;gap:10px;}
.cm-title-row{font-family:var(--mono);font-size:.62rem;color:var(--muted);text-align:center;}
.cm-grid{display:grid;gap:3px;}
.cm-cell{padding:5px 8px;border-radius:3px;text-align:center;
         font-family:var(--mono);font-size:.8rem;font-weight:700;min-width:36px;}
.cm-header{padding:4px 8px;color:var(--muted);font-family:var(--mono);
           font-size:.6rem;text-align:center;}
.cm-row-label{padding:4px 6px;color:var(--muted);font-family:var(--mono);
              font-size:.6rem;text-align:right;display:flex;align-items:center;justify-content:flex-end;}
.cm-diag{background:rgba(0,255,136,.2);color:var(--green);}
.cm-off{background:rgba(255,61,107,.15);color:var(--accent2);}

@media(max-width:900px){
  .stats{grid-template-columns:repeat(2,1fr);}
  .main-grid{grid-template-columns:1fr;}
  .bottom-grid{grid-template-columns:1fr;}
}
</style>
</head>
<body>
<div class="app">

<header>
  <div class="logo">
    <div class="logo-ring"><span>🛡</span></div>
    <div class="logo-text">
      <h1>NetGuard IDS</h1>
      <p>AI-Powered Intrusion Detection System</p>
    </div>
  </div>
  <div class="header-right">
    <div class="live-dot"><div class="dot"></div><span>LIVE</span></div>
    <div class="acc-badge">MODEL: <span id="acc-val">--</span>%</div>
    <div class="user-info">Logged in as <span>{{ user }}</span></div>
    <a href="/logout" class="logout">Logout</a>
  </div>
</header>

<!-- Stats -->
<div class="stats">
  <div class="stat blue">
    <div class="stat-lbl">Total Packets</div>
    <div class="stat-val blue" id="s-total">0</div>
    <div class="stat-sub">analyzed</div>
  </div>
  <div class="stat green">
    <div class="stat-lbl">Normal</div>
    <div class="stat-val green" id="s-normal">0</div>
    <div class="stat-sub">clean traffic</div>
  </div>
  <div class="stat red">
    <div class="stat-lbl">Attacks</div>
    <div class="stat-val red" id="s-attack">0</div>
    <div class="stat-sub">threats detected</div>
  </div>
  <div class="stat yellow">
    <div class="stat-lbl">Alerts Fired</div>
    <div class="stat-val yellow" id="s-alerts">0</div>
    <div class="stat-sub">high-risk events</div>
  </div>
</div>

<!-- Main -->
<div class="main-grid">
  <!-- Live traffic -->
  <div class="panel">
    <div class="ph">
      <span class="ph-title">⚡ Live Traffic Feed</span>
      <span class="ph-badge" id="feed-count">0 packets</span>
    </div>
    <div class="t-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>Time</th><th>Proto</th><th>SrcBytes</th>
          <th>Conns</th><th>Flag</th><th>Login</th>
          <th>Label</th><th>Score</th><th>Level</th><th>Alerts</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Feature importance -->
  <div class="panel">
    <div class="ph"><span class="ph-title">📊 Feature Importance</span></div>
    <div class="fi-list" id="fi-list"></div>
  </div>
</div>

<!-- Bottom -->
<div class="bottom-grid">

  <!-- Attack distribution chart -->
  <div class="panel">
    <div class="ph"><span class="ph-title">🎯 Attack Distribution</span></div>
    <div class="chart-wrap">
      <canvas id="donut-chart"></canvas>
    </div>
  </div>

  <!-- Risk over time -->
  <div class="panel">
    <div class="ph"><span class="ph-title">📈 Risk Score Timeline</span></div>
    <div class="chart-wrap">
      <canvas id="line-chart"></canvas>
    </div>
  </div>

  <!-- Alert log -->
  <div class="panel">
    <div class="ph">
      <span class="ph-title">🚨 Alert Log</span>
      <span class="ph-badge" id="alert-count">0 alerts</span>
    </div>
    <div class="alert-log" id="alert-log">
      <div class="empty-state">Waiting for alerts...</div>
    </div>
  </div>

</div>
</div>

<script>
// ── State
const stats = {total:0, normal:0, attack:0, alerts:0};
const attackCounts = {Normal:0, DoS:0, BruteForce:0, PortScan:0, SuspiciousLogin:0};
const riskHistory  = [];
let rowCount = 0, alertCount = 0, firstAlert = true;

function $(id){ return document.getElementById(id); }

// ── Donut chart
const donutCtx = $('donut-chart').getContext('2d');
const donutChart = new Chart(donutCtx, {
  type: 'doughnut',
  data: {
    labels: ['Normal','DoS','BruteForce','PortScan','SuspiciousLogin'],
    datasets:[{ data:[1,0,0,0,0],
      backgroundColor:['#00ff8844','#ff3d6b44','#ffd60044','#00d4ff44','#ffa04044'],
      borderColor:    ['#00ff88','#ff3d6b','#ffd600','#00d4ff','#ffa040'],
      borderWidth:2
    }]
  },
  options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{ labels:{ color:'#c8e0f4', font:{size:10}, boxWidth:12 } } },
    cutout:'60%'
  }
});

// ── Line chart
const lineCtx = $('line-chart').getContext('2d');
const lineChart = new Chart(lineCtx, {
  type:'line',
  data:{
    labels:[],
    datasets:[{
      label:'Risk Score', data:[], borderColor:'#00d4ff',
      backgroundColor:'rgba(0,212,255,.08)', borderWidth:2,
      pointRadius:2, tension:0.3, fill:true
    }]
  },
  options:{
    responsive:true, maintainAspectRatio:false,
    scales:{
      x:{ ticks:{color:'#3a5a7a',font:{size:8},maxTicksLimit:8}, grid:{color:'rgba(13,40,71,.5)'} },
      y:{ min:0, max:100, ticks:{color:'#3a5a7a',font:{size:9}}, grid:{color:'rgba(13,40,71,.5)'} }
    },
    plugins:{ legend:{ labels:{ color:'#c8e0f4', font:{size:10} } } }
  }
});

function updateCharts(d){
  // Donut
  attackCounts[d.label] = (attackCounts[d.label]||0)+1;
  const labels = Object.keys(attackCounts);
  donutChart.data.datasets[0].data = labels.map(k=>attackCounts[k]);
  donutChart.update('none');

  // Line
  riskHistory.push({t: d.timestamp.slice(11,19), s: d.risk_score});
  if(riskHistory.length > 30) riskHistory.shift();
  lineChart.data.labels = riskHistory.map(r=>r.t);
  lineChart.data.datasets[0].data = riskHistory.map(r=>r.s);
  lineChart.update('none');
}

function updateStats(){
  $('s-total').textContent  = stats.total;
  $('s-normal').textContent = stats.normal;
  $('s-attack').textContent = stats.attack;
  $('s-alerts').textContent = stats.alerts;
  $('feed-count').textContent = stats.total + ' packets';
}

function addRow(d){
  rowCount++;
  const tb = $('tbody');
  const tr = document.createElement('tr');
  const pc = d.protocol==='TCP'?'proto-tcp':d.protocol==='UDP'?'proto-udp':'proto-icmp';
  const fc = d.flag==='SF'?'flag-sf':'flag-bad';
  const alerts = d.alert_reasons !== '—'
    ? d.alert_reasons.split(', ').map(a=>`<span class="alert-tag">${a}</span>`).join('')
    : '<span style="color:var(--muted)">—</span>';
  tr.innerHTML = `
    <td style="color:var(--muted)">${rowCount}</td>
    <td style="color:var(--muted);font-size:.62rem">${d.timestamp.slice(11)}</td>
    <td class="${pc}">${d.protocol}</td>
    <td style="color:#7ab3d4">${d.src_bytes.toLocaleString()}</td>
    <td style="color:#7ab3d4">${d.num_connections}</td>
    <td class="${fc}">${d.flag}</td>
    <td style="color:#7ab3d4">${d.login_attempts}</td>
    <td><span class="badge ${d.label}">${d.label}</span></td>
    <td class="risk-${d.risk_level}">${d.risk_score}</td>
    <td class="risk-${d.risk_level}">${d.risk_level}</td>
    <td>${alerts}</td>
  `;
  tb.insertBefore(tr, tb.firstChild);
  while(tb.children.length > 60) tb.removeChild(tb.lastChild);
}

function addAlert(d){
  const log = $('alert-log');
  if(firstAlert){ log.innerHTML=''; firstAlert=false; }
  alertCount++;
  $('alert-count').textContent = alertCount + ' alerts';
  const div = document.createElement('div');
  div.className = 'alert-entry';
  div.innerHTML = `
    <div style="color:var(--accent2);margin-top:1px">⚠</div>
    <div class="alert-body">
      <div class="alert-title">${d.label} — ${d.risk_level}</div>
      <div class="alert-detail">Score:${d.risk_score} · ${d.protocol} · ${d.src_bytes.toLocaleString()}B · ${d.alert_reasons}</div>
    </div>
    <div class="alert-time">${d.timestamp.slice(11)}</div>
  `;
  log.insertBefore(div, log.firstChild);
  while(log.children.length > 25) log.removeChild(log.lastChild);
}

function renderFI(fi){
  $('fi-list').innerHTML = fi.map(([name,val])=>`
    <div>
      <div class="fi-top">
        <span class="fi-name">${name}</span>
        <span class="fi-pct">${(val*100).toFixed(1)}%</span>
      </div>
      <div class="fi-bg"><div class="fi-bar" style="width:${val*100}%"></div></div>
    </div>
  `).join('');
}

// ── SSE
const es = new EventSource('/stream');
es.addEventListener('init', e => {
  const d = JSON.parse(e.data);
  $('acc-val').textContent = d.accuracy;
  renderFI(d.feature_importance);
  Object.assign(stats, d.stats);
  updateStats();
});
es.addEventListener('packet', e => {
  const d = JSON.parse(e.data);
  stats.total++;
  d.label === 'Normal' ? stats.normal++ : stats.attack++;
  if(d.risk_score >= {{ threshold }}) { stats.alerts++; addAlert(d); }
  addRow(d);
  updateStats();
  updateCharts(d);
});
es.onerror = () => console.log('SSE reconnecting...');
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Global model state
MODEL  = {}
live_stats  = {'total':0,'normal':0,'attack':0,'alerts':0}
lock   = threading.Lock()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        u = request.form.get('username','')
        p = request.form.get('password','')
        if USERS.get(u) == p:
            session['user'] = u
            return redirect(url_for('dashboard'))
        error = 'Invalid username or password.'
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    return render_template_string(DASHBOARD_HTML,
                                  user=session['user'],
                                  threshold=ALERT_THRESHOLD)

@app.route('/stream')
@login_required
def stream():
    def generate():
        # Send model info first
        init_payload = {
            'accuracy':          MODEL['accuracy'],
            'feature_importance':MODEL['feature_importance'],
            'stats':             live_stats,
        }
        yield f"event:init\ndata:{json.dumps(init_payload)}\n\n"

        while True:
            pkt    = make_packet()
            result = predict_packet(pkt, MODEL)
            write_log(result)

            with lock:
                live_stats['total'] += 1
                if result['label'] == 'Normal':
                    live_stats['normal'] += 1
                else:
                    live_stats['attack'] += 1
                if result['risk_score'] >= ALERT_THRESHOLD:
                    live_stats['alerts'] += 1
                    write_alert(result)

            yield f"event:packet\ndata:{json.dumps(result)}\n\n"
            time.sleep(random.uniform(0.6, 1.2))

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/api/history')
@login_required
def history():
    return jsonify(read_log(200))

@app.route('/api/stats')
@login_required
def get_stats():
    with lock:
        return jsonify(live_stats)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    MODEL.update(train_model())
    init_log()
    print("=" * 55)
    print("  NetGuard IDS — Running on http://127.0.0.1:5000")
    print("  Login: admin / admin123  |  analyst / analyst123")
    print("=" * 55 + "\n")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
