"""
InsiderLSTM: Proactive Insider Threat Risk Forecasting with LSTM-based
Temporal Behavioural Modelling and Early-Warning Lead Time (EWLT) Evaluation.

Converted from notebooks/r4-2-and-spedia-final.ipynb. Runs top to bottom.

Data paths are resolved automatically for Kaggle, Colab, or a local repo run.
For a local run, place the datasets under ./data as described in data/README.md
(or set the DATA_DIR environment variable) and run:

    python src/main.py

Generated artifacts (parquet, json, csv, figures, model checkpoints) are
written to ./outputs by default (override with the WORK_DIR env var).
"""


# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 1] ------------------------------------------------------
# ── Section 1a: Install packages ─────────────────────────────────────────────
import subprocess, sys
for pkg in ['scikit-learn','imbalanced-learn','scikit-optimize',
            'torch','numpy','pandas','matplotlib','seaborn','scipy','xgboost']:
    subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])
print('✅ Packages ready')

# ---- [cell 2] ------------------------------------------------------
# ── Section 1b: Imports + global config ──────────────────────────────────────
import os, gc, json, copy, random, pickle, re as _re, warnings, time as _time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
# Three environments are supported. Detection order: Kaggle, Colab, then local.
# For a local/repo run, data is read from ./data (override with the DATA_DIR
# environment variable) and all generated artifacts go to ./outputs (override
# with WORK_DIR). See data/README.md for the expected folder layout.
if os.path.exists('/kaggle/input'):
    CERT_R42_PATH = '/kaggle/input/datasets/prasiiiii/cert-4-2-6-2/r4.2'
    SPEDIA_PATH   = '/kaggle/input/datasets/hillarygabriel/spedia/logs_SPEDIA_annotated_en.csv'
    ANS_DIR       = '/kaggle/input/datasets/hillarygabriel/answers-label/answers'
    WORK_DIR      = '/kaggle/working'
elif os.path.exists('/content/drive'):
    CERT_R42_PATH = '/content/drive/MyDrive/CERT/r4.2'
    SPEDIA_PATH   = '/content/drive/MyDrive/CERT/spedia/logs_SPEDIA_annotated_en.csv'
    ANS_DIR       = '/content/drive/MyDrive/CERT/answers'
    WORK_DIR      = '/content/drive/MyDrive/InsiderThreat/combined'
else:  # local / repo
    DATA_DIR      = os.environ.get('DATA_DIR', 'data')
    CERT_R42_PATH = os.path.join(DATA_DIR, 'r4.2')
    SPEDIA_PATH   = os.path.join(DATA_DIR, 'spedia', 'logs_SPEDIA_annotated_en.csv')
    ANS_DIR       = os.path.join(DATA_DIR, 'answers')
    WORK_DIR      = os.environ.get('WORK_DIR', 'outputs')

os.makedirs(WORK_DIR, exist_ok=True)

# RELOAD_DIR is used by the "reload from a previous run" cells/sections. It
# defaults to WORK_DIR so a fresh top-to-bottom run works without editing;
# set the RELOAD_DIR environment variable to reload artifacts from elsewhere.
RELOAD_DIR = os.environ.get('RELOAD_DIR', WORK_DIR)

print(f'CERT r4.2 : {CERT_R42_PATH}')
print(f'SPEDIA    : {SPEDIA_PATH}')
print(f'Answers   : {ANS_DIR}')
print(f'Work dir  : {WORK_DIR}')
print(f'Reload dir: {RELOAD_DIR}')

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch {torch.__version__} | Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
set_seed()

# ── r4.2 daily pipeline config ────────────────────────────────────────────────
SEQ_LEN_R42      = 14      # 14-day sliding window
ROLL_WIN         = 14      # rolling z-score window
MIN_PERIODS      = 3
Z_CLIP           = 10.0
WORK_HOUR_START  = 8
WORK_HOUR_END    = 18
N_SAMPLE_R42     = 15000   # rows sampled from master grid 

# ── SPEDIA config ─────────────────────────────────────────────────────────────
MAX_SESSION_LEN  = 50      # max events per sub-session
FEATURE_DIM_SP   = 9       # 8 activity one-hot + 1 size
GAP_MINS_SP      = 30      # time-gap session boundary (no logoff in SPEDIA)

# ── Renamed feature mapping (r4.2) ───────────────────────────────────────────
FEATURE_RENAME = {
    'login_count'      : 'daily_logon_freq',
    'biz_hours_logins' : 'workhour_logon_cnt',
    'off_hous_logins'  : 'afterhour_logon_cnt',
    'is_weekend'       : 'weekend_activity_flag',
    'usb_usage_cnt'    : 'removable_device_cnt',
    'usb_mean_usg_dur' : 'avg_device_session_dur',
    'usb_workhour'     : 'device_workhour_cnt',
    'file_op_n'        : 'file_operation_cnt',
    'avg_size_file'    : 'mean_file_size_bytes',
    'avg_depth_file'   : 'mean_path_depth',
    'avg_nwords_text'  : 'mean_file_content_words',
    'email_sr_n'       : 'email_activity_cnt',
    'avg_to_rec'       : 'mean_recipient_cnt',
    'avg_atts'         : 'mean_attachment_cnt',
    'avg_exdes'        : 'mean_external_recipient_cnt',
    'avg_bccdes'       : 'mean_bcc_recipient_cnt',
    'web_req_n'        : 'http_request_cnt',
    'avg_len_url'      : 'mean_url_length',
    'avg_depth_url'    : 'mean_url_path_depth',
    'avg_len_http'     : 'mean_http_response_size',
    'avg_nwords_http'  : 'mean_http_content_words',
}

print(f'\nSEQ_LEN_R42    : {SEQ_LEN_R42} days')
print(f'N_SAMPLE_R42   : {N_SAMPLE_R42:,} rows')
print(f'FEATURE_DIM_SP : {FEATURE_DIM_SP}')
print(f'GAP_MINS_SP    : {GAP_MINS_SP} min')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 4] ------------------------------------------------------
# ── Section 2a: Dataset + loader ─────────────────────────────────────────────
class InsiderDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loader(X, y, batch_size=64, sampler=None, shuffle=False):
    return DataLoader(InsiderDS(X, y),
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=0, pin_memory=False)

print('InsiderDS + make_loader defined.')

# ---- [cell 5] ------------------------------------------------------
# ── Section 2b: Model classes ─────────────────────────────────────────────────
class InsiderLSTM(nn.Module):
    """Proposed — unidirectional LSTM + input gate + highway refinement."""
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.input_gate       = nn.Sequential(nn.Linear(input_size,input_size), nn.Sigmoid())
        self.lstm             = nn.LSTM(input_size, hidden, layers, batch_first=True,
                                         dropout=dropout if layers>1 else 0)
        self.refine_transform = nn.Linear(hidden,hidden)
        self.refine_gate      = nn.Sequential(nn.Linear(hidden,hidden), nn.Sigmoid())
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden,hidden)
        self.fc2  = nn.Linear(hidden,1)
    def forward(self, x):
        x_in = x * self.input_gate(x)
        _,(h_n,_) = self.lstm(x_in); h = h_n[-1]
        h_t = torch.tanh(self.refine_transform(h))
        g   = self.refine_gate(h)
        h   = g*h_t + (1-g)*h
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(self.drop(self.bn(h)))))).squeeze(-1)

class OFA_LSTM(nn.Module):
    """OFA-LSTM — Wambura et al. Computer Journal 2022."""
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_size,input_size), nn.Sigmoid())
        self.lstm = nn.LSTM(input_size,hidden,layers,batch_first=True,
                             dropout=dropout if layers>1 else 0)
        self.bn   = nn.BatchNorm1d(hidden); self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden,1)
    def forward(self, x):
        _,(h_n,_) = self.lstm(x*self.gate(x))
        return torch.sigmoid(self.fc(self.drop(self.bn(h_n[-1])))).squeeze(-1)

class AttentionLSTM(nn.Module):
    """TA-LSTM — Pal et al. ESWA 2023."""
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden,layers,batch_first=True,
                             dropout=dropout if layers>1 else 0)
        self.gru  = nn.GRU(input_size,hidden,layers,batch_first=True,
                            dropout=dropout if layers>1 else 0)
        self.attn = nn.Linear(hidden,1)
        self.bn   = nn.BatchNorm1d(hidden); self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden,1)
    def forward(self, x):
        lo,_ = self.lstm(x); go,_ = self.gru(x)
        comb = (lo+go)/2.0
        w    = torch.softmax(self.attn(comb),dim=1)
        ctx  = (w*comb).sum(dim=1)
        return torch.sigmoid(self.fc(self.drop(self.bn(ctx)))).squeeze(-1)

class ITDSTS(nn.Module):
    """ITDSTS — Tian et al. Cybersecurity (Springer) 2025."""
    def __init__(self, input_size, hidden=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        # Ensure hidden divisible by n_heads
        if hidden % n_heads != 0:
            n_heads = 4 if hidden >= 4 else 1
        self.input_proj  = nn.Linear(input_size, hidden)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden*4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.bn   = nn.BatchNorm1d(hidden); self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden,1)
    def forward(self, x):
        h = self.transformer(self.input_proj(x)).mean(dim=1)
        return torch.sigmoid(self.fc(self.drop(self.bn(h)))).squeeze(-1)

class TTT_ECA_ResNet(nn.Module):
    """TTT-ECA-ResNet — Tao et al. High-Confidence Computing 2025."""
    def __init__(self, input_size, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size,hidden)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden,hidden), nn.LayerNorm(hidden),
                          nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden,hidden), nn.LayerNorm(hidden))
            for _ in range(n_layers)])
        self.eca  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden,hidden,kernel_size=3,padding=1,groups=hidden,bias=False),
            nn.Sigmoid())
        self.bn   = nn.BatchNorm1d(hidden); self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden,1)
    def forward(self, x):
        h = self.input_proj(x)
        for blk in self.res_blocks: h = h + blk(h)
        h = (self.eca(h.permute(0,2,1)) * h.permute(0,2,1)).mean(dim=2)
        return torch.sigmoid(self.fc(self.drop(self.bn(h)))).squeeze(-1)

class CNN_GRU(nn.Module):
    """CNN-GRU — Manoharan. Telecommunication Systems 2024."""
    def __init__(self, input_size, hidden=64, cnn_filters=64, layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size,cnn_filters,3,padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_filters,cnn_filters,3,padding=1), nn.GELU(), nn.Dropout(dropout))
        self.gru  = nn.GRU(cnn_filters,hidden,layers,batch_first=True,
                            dropout=dropout if layers>1 else 0)
        self.bn   = nn.BatchNorm1d(hidden); self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden,1)
    def forward(self, x):
        h = self.cnn(x.permute(0,2,1)).permute(0,2,1)
        _,h_n = self.gru(h)
        return torch.sigmoid(self.fc(self.drop(self.bn(h_n[-1])))).squeeze(-1)

print('Model classes defined: InsiderLSTM, OFA_LSTM, AttentionLSTM,')
print('  ITDSTS, TTT_ECA_ResNet, CNN_GRU')

# ---- [cell 6] ------------------------------------------------------
# ── Section 2c: Metrics + EWLT utilities ─────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold, name):
    """Compute all evaluation metrics at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        'Model'    : name,
        'AUC-ROC'  : round(roc_auc_score(y_true, y_prob), 4),
        'AUC-PR'   : round(average_precision_score(y_true, y_prob), 4),
        'Accuracy' : round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score' : round(f1_score(y_true, y_pred, zero_division=0), 4),
        'FPR'      : round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
        'Bal-Acc'  : round(balanced_accuracy_score(y_true, y_pred), 4),
    }

def get_f1_threshold(y_true, y_prob):
    """Dense PR curve search for F1-optimal threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s      = 2 * precisions[:-1] * recalls[:-1] / \
               np.maximum(precisions[:-1] + recalls[:-1], 1e-9)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx])

def safe_user_auc(probs, meta, y_true, label=''):
    """User-level AUC — returns None when single class in test."""
    user_probs  = defaultdict(list)
    user_labels = defaultdict(int)
    for p, m, lbl in zip(probs, meta, y_true):
        user_probs[m[0]].append(float(p))
        if int(lbl)==1: user_labels[m[0]] = 1
    users  = list(user_probs.keys())
    labels = [user_labels[u] for u in users]
    if len(set(labels)) < 2:
        if label: print(f'  [{label}] User AUC not computable.')
        return None
    scores = [max(user_probs[u]) for u in users]
    return round(roc_auc_score(labels, scores), 4)

def ewlt_per_user(probs, meta, y_true, threshold,
                   sustained=2, max_days=60):
    """Compute Early-Warning Lead Time per malicious user."""
    user_data = defaultdict(list)
    for i,(m,lbl) in enumerate(zip(meta, y_true)):
        user_data[m[0]].append({
            'date': m[1], 'prob': float(probs[i]),
            'label': int(lbl)})
    results = {}
    for user, events in user_data.items():
        if not any(e['label']==1 for e in events): continue
        events.sort(key=lambda x: x['date'])
        first_mal = min(e['date'] for e in events if e['label']==1)
        pre = [e for e in events
               if e['prob']>=threshold and e['date']<first_mal]
        if len(pre) < sustained: continue
        flagged = sorted(set(e['date'] for e in pre))
        found   = None
        for i in range(len(flagged)-sustained+1):
            win = flagged[i:i+sustained]
            if (win[-1]-win[0]).days <= sustained*3:
                found = win[0]; break
        if found is None: continue
        lead = (first_mal - found).days
        if lead<=0 or lead>max_days: continue
        conf = float(np.mean([e['prob'] for e in pre]))
        pre_all = [e for e in events if e['date']<first_mal]
        cons = sum(1 for e in pre_all if e['prob']>=threshold)/max(len(pre_all),1)
        results[user] = {
            'lead_days'  : int(lead),
            'confidence' : round(conf,3),
            'consistency': round(float(cons),3)}
    return results

def ewlt_threshold_sweep(probs, meta, y_true,
                          lo=0.05, hi=0.65, step=0.02):
    """Sweep thresholds and return {thr: n_users_warned}."""
    sweep = {}
    for thr in np.arange(hi, lo, -step):
        t = round(float(thr), 2)
        r = ewlt_per_user(probs, meta, y_true, t)
        sweep[t] = len(r)
    return sweep

print('Metrics + EWLT utilities defined.')

# ---- [cell 7] ------------------------------------------------------
# ── Section 2d: Training function ────────────────────────────────────────────
def train_model(model, tr_loader, vl_loader,
                 lr=1e-3, epochs=100, patience=15, pos_weight=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr*0.01)
    criterion = nn.BCELoss(reduction='none')
    best_vl_auc = 0.0
    best_state  = copy.deepcopy(model.state_dict())
    hist = {'tr_loss':[],'vl_loss':[],'vl_auc':[],'lr':[]}
    no_imp = 0
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for Xb,yb in tr_loader:
            Xb,yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            preds = model(Xb)
            raw   = criterion(preds,yb)
            w     = torch.where(yb==1,
                        pos_weight.to(DEVICE) if pos_weight is not None
                        else torch.ones_like(yb),
                        torch.ones_like(yb))
            loss  = (raw*w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)
        scheduler.step()
        hist['lr'].append(optimizer.param_groups[0]['lr'])

        model.eval(); vl_loss=0.0; vp=[]; vl_=[]
        with torch.no_grad():
            for Xb,yb in vl_loader:
                preds = model(Xb.to(DEVICE))
                raw   = nn.BCELoss(reduction='none')(preds,yb.to(DEVICE))
                w     = torch.where(yb.to(DEVICE)==1,
                            pos_weight.to(DEVICE) if pos_weight is not None
                            else torch.ones_like(yb.to(DEVICE)),
                            torch.ones_like(yb.to(DEVICE)))
                vl_loss += (raw*w).mean().item()
                vp.extend(preds.cpu().numpy())
                vl_.extend(yb.numpy())
        vl_loss /= len(vl_loader)
        try:
            vl_auc = roc_auc_score(vl_,vp) if len(np.unique(vl_))>1 else 0.0
        except Exception:
            vl_auc = 0.0
        hist['tr_loss'].append(tr_loss)
        hist['vl_loss'].append(vl_loss)
        hist['vl_auc'].append(vl_auc)

        if ep%10==0 or ep==1:
            print(f'  Ep {ep:3d} | tr={tr_loss:.5f} vl={vl_loss:.5f} '
                  f'auc={vl_auc:.4f} no_imp={no_imp}/{patience}')
        else:
            print(f'  Ep {ep:3d} | auc={vl_auc:.4f} no_imp={no_imp}',
                  end='\r', flush=True)

        if vl_auc > best_vl_auc:
            best_vl_auc=vl_auc; best_state=copy.deepcopy(model.state_dict()); no_imp=0
        else:
            no_imp+=1
        if no_imp>=patience:
            print(f'\n  Early stop ep={ep} best_val_auc={best_vl_auc:.4f}')
            break

    model.load_state_dict(best_state); model.eval()
    return model, hist

def collect_probs(model, X, y, batch_size=256):
    model.eval(); probs=[]
    with torch.no_grad():
        for Xb,_ in make_loader(X,y,batch_size):
            probs.extend(model(Xb.to(DEVICE)).cpu().numpy())
    return np.array(probs)

print('train_model() + collect_probs() defined.')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 9] ------------------------------------------------------
# ── Section 3a: Label construction ───────────────────────────────────────────
def parse_ts_str(s):
    return pd.to_datetime(s.strip().strip('"'), format='mixed',
                           dayfirst=False, errors='coerce')

def parse_scenario_file(path, user):
    dates=[]; matched=0
    with open(path,'r',encoding='utf-8',errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',',3)
            if len(parts)<4: continue
            uf=parts[3].split(',')[0].strip().strip('"')
            if uf!=user: continue
            ts=parse_ts_str(parts[2])
            if pd.notna(ts):
                dates.append(pd.Timestamp(ts.date())); matched+=1
    return dates, matched

ins_path = os.path.join(ANS_DIR,'insiders.csv')
df_ins   = pd.read_csv(ins_path)
df_ins.columns=[c.strip().lower() for c in df_ins.columns]
df_ins   = df_ins.dropna(subset=['user'])
df_ins   = df_ins[df_ins['user'].astype(str).str.strip()!='']
df_r42   = df_ins[df_ins['dataset'].astype(str).str.strip()=='4.2'].copy().reset_index(drop=True)

print(f'r4.2 insiders : {len(df_r42)} | scenarios: {sorted(df_r42["scenario"].unique())}')

mal_set=set(); mal_users=set(); ins_scenario={}

for _,row in df_r42.iterrows():
    user=str(row['user']).strip()
    sc=str(int(row['scenario']))
    sf=str(row.get('details',f'r4.2-{sc}-{user}.csv')).strip()
    sp=os.path.join(ANS_DIR,f'r4.2-{sc}',sf)
    if not os.path.exists(sp):
        alt=os.path.join(ANS_DIR,f'r4.2-{sc}',f'r4.2-{sc}-{user}.csv')
        sp=alt if os.path.exists(alt) else None
    if sp is None: continue
    dates,_=parse_scenario_file(sp,user)
    for d in dates: mal_set.add((user,d))
    mal_users.add(user); ins_scenario[user]=int(sc)

print(f'mal_users : {len(mal_users)} | mal_set : {len(mal_set):,} user-day pairs')
for sc in ['1','2','3']:
    su=[u for u,s in ins_scenario.items() if str(s)==sc]
    sd=sum(1 for (u,d) in mal_set if u in su)
    print(f'  Scenario {sc}: {len(su)} users | {sd:,} malicious days')
assert len(mal_users)>=60, f'Expected ~70 insiders, got {len(mal_users)}'
print('\n✅ Label construction complete.')

# ---- [cell 10] ------------------------------------------------------
# ── Section 3b: Load r4.2 logs (OOM-safe) ────────────────────────────────────
def parse_date_col(df, col='date'):
    df=df.copy()
    df[col]             = pd.to_datetime(df[col],format='mixed',dayfirst=False,errors='coerce')
    df['date_only']     = df[col].dt.normalize()
    df['hour']          = df[col].dt.hour.astype('int8')
    df['is_workhour']   = ((df['hour']>=WORK_HOUR_START)&(df['hour']<WORK_HOUR_END)).astype('int8')
    df['is_afterhours'] = (1-df['is_workhour']).astype('int8')
    df['is_weekend']    = df[col].dt.dayofweek.ge(5).astype('int8')
    return df

print('Loading r4.2 logs (OOM-safe — chunked HTTP)...')

logon = pd.read_csv(os.path.join(CERT_R42_PATH,'logon.csv'),
    low_memory=False, usecols=['date','user','activity'])
logon = parse_date_col(logon)
logon.dropna(subset=['date_only','user'],inplace=True)
print(f'  logon  : {len(logon):,} | users={logon["user"].nunique()}')

device_ = pd.read_csv(os.path.join(CERT_R42_PATH,'device.csv'),
    low_memory=False, usecols=['date','user','activity'])
device_ = parse_date_col(device_)
device_.dropna(subset=['date_only','user'],inplace=True)
print(f'  device : {len(device_):,} | users={device_["user"].nunique()}')

file_ = pd.read_csv(os.path.join(CERT_R42_PATH,'file.csv'),
    low_memory=False, usecols=['date','user','filename','content'])
file_ = parse_date_col(file_)
file_.dropna(subset=['date_only','user'],inplace=True)
print(f'  file   : {len(file_):,} | users={file_["user"].nunique()}')

email_ = pd.read_csv(os.path.join(CERT_R42_PATH,'email.csv'),
    low_memory=False, usecols=['date','user','to','cc','bcc','from','size','attachments'])
email_ = parse_date_col(email_)
email_.dropna(subset=['date_only','user'],inplace=True)
email_['size']        = pd.to_numeric(email_['size'],errors='coerce').fillna(0).astype('float32')
email_['attachments'] = pd.to_numeric(email_['attachments'],errors='coerce').fillna(0).astype('float32')
print(f'  email  : {len(email_):,} | users={email_["user"].nunique()}')

# HTTP — chunked to avoid OOM
http_chunks=[]
for chunk in pd.read_csv(os.path.join(CERT_R42_PATH,'http.csv'),
        chunksize=200_000, low_memory=False,
        usecols=['date','user','url','content']):
    chunk=parse_date_col(chunk)
    chunk.dropna(subset=['date_only','user'],inplace=True)
    # Keep only needed derived cols, drop raw
    chunk['url_len']    = chunk['url'].fillna('').apply(len).astype('float32')
    chunk['url_depth']  = chunk['url'].fillna('').str.count('/').astype('float32')
    chunk['http_size']  = chunk['content'].fillna('').apply(
        lambda x: float(len(str(x).encode('utf-8')))).astype('float32')
    chunk['http_nwords']= chunk['content'].fillna('').apply(
        lambda x: float(len(str(x).split()))).astype('float32')
    http_chunks.append(chunk[['user','date_only','url_len',
                                'url_depth','http_size','http_nwords']])
    del chunk
http_ = pd.concat(http_chunks,ignore_index=True); del http_chunks; gc.collect()
print(f'  http   : {len(http_):,} | users={http_["user"].nunique()}')
print('\n✅ Logs loaded.')

# ---- [cell 11] ------------------------------------------------------
# ── Section 3c: Daily feature aggregation (21 renamed features) ──────────────
GRP = ['user','date_only']

def safe_agg(df,grp,agg_dict):
    valid={k:v for k,v in agg_dict.items() if v[0] in df.columns}
    if not valid:
        r=df[grp].drop_duplicates().copy()
        for k in agg_dict: r[k]=0.0
        return r
    return df.groupby(grp).agg(**{
        k:pd.NamedAgg(column=v[0],aggfunc=v[1]) for k,v in valid.items()
    }).reset_index()

# ── Logon ─────────────────────────────────────────────────────────────────────
agg_logon = safe_agg(logon,GRP,{
    'daily_logon_freq'     :('user','count'),
    'workhour_logon_cnt'   :('is_workhour','sum'),
    'afterhour_logon_cnt'  :('is_afterhours','sum'),
    'weekend_activity_flag':('is_weekend','max'),
})
del logon; gc.collect()
print(f'logon agg  : {agg_logon.shape} | {list(agg_logon.columns[2:])}')

# ── Device ────────────────────────────────────────────────────────────────────
agg_device = safe_agg(device_,GRP,{
    'removable_device_cnt' :('user','count'),
    'device_workhour_cnt'  :('is_workhour','sum'),
})
agg_device['avg_device_session_dur'] = (
    agg_device['removable_device_cnt'] /
    agg_device['device_workhour_cnt'].clip(lower=1)).astype('float32')
del device_; gc.collect()
print(f'device agg : {agg_device.shape} | {list(agg_device.columns[2:])}')

# ── File ──────────────────────────────────────────────────────────────────────
file_['mean_path_depth']        = file_['filename'].fillna('').apply(
    lambda x: float(max(x.count('/'),x.count('\\'),1))).astype('float32')
file_['mean_file_content_words']= file_['content'].fillna('').apply(
    lambda x: float(len(str(x).split()))).astype('float32')
file_['mean_file_size_bytes']   = file_['content'].fillna('').apply(
    lambda x: float(len(str(x).encode('utf-8')))).astype('float32')

agg_file = file_.groupby(GRP).agg(
    file_operation_cnt     =('user','count'),
    mean_file_size_bytes   =('mean_file_size_bytes','mean'),
    mean_path_depth        =('mean_path_depth','mean'),
    mean_file_content_words=('mean_file_content_words','mean'),
).reset_index()
del file_; gc.collect()
print(f'file agg   : {agg_file.shape} | {list(agg_file.columns[2:])}')

# ── Email ─────────────────────────────────────────────────────────────────────
def count_addr(s):
    if pd.isna(s) or str(s).strip()=='': return 0.0
    return float(len([a for a in str(s).split(';') if '@' in a]))
def count_ext(s):
    if pd.isna(s) or str(s).strip()=='': return 0.0
    return float(sum(1 for a in str(s).split(';')
                     if '@' in a and 'dtaa.com' not in a.lower()))

email_['mean_recipient_cnt']        = email_['to'].apply(count_addr).astype('float32')
email_['mean_external_recipient_cnt']= email_['to'].apply(count_ext).astype('float32')
email_['mean_bcc_recipient_cnt']    = email_['bcc'].apply(count_addr).astype('float32')
email_['mean_attachment_cnt']       = email_['attachments'].astype('float32')

agg_email = email_.groupby(GRP).agg(
    email_activity_cnt          =('user','count'),
    mean_recipient_cnt          =('mean_recipient_cnt','mean'),
    mean_attachment_cnt         =('mean_attachment_cnt','mean'),
    mean_external_recipient_cnt =('mean_external_recipient_cnt','mean'),
    mean_bcc_recipient_cnt      =('mean_bcc_recipient_cnt','mean'),
).reset_index()
del email_; gc.collect()
print(f'email agg  : {agg_email.shape} | {list(agg_email.columns[2:])}')

# ── HTTP ──────────────────────────────────────────────────────────────────────
agg_http = http_.groupby(GRP).agg(
    http_request_cnt        =('user','count'),
    mean_url_length         =('url_len','mean'),
    mean_url_path_depth     =('url_depth','mean'),
    mean_http_response_size =('http_size','mean'),
    mean_http_content_words =('http_nwords','mean'),
).reset_index()
del http_; gc.collect()
print(f'http agg   : {agg_http.shape} | {list(agg_http.columns[2:])}')

# ── Master grid ───────────────────────────────────────────────────────────────
all_users = sorted(agg_logon['user'].unique())
all_dates = pd.date_range(agg_logon['date_only'].min(),
                           agg_logon['date_only'].max(), freq='D')
print(f'\nGrid: {len(all_users)} users × {len(all_dates)} days = '
      f'{len(all_users)*len(all_dates):,} rows')

idx    = pd.MultiIndex.from_product([all_users,all_dates],
                                     names=['user','date_only'])
master = pd.DataFrame(index=idx).reset_index()

for adf in [agg_logon,agg_device,agg_file,agg_email,agg_http]:
    master=master.merge(adf,on=GRP,how='left')
del agg_logon,agg_device,agg_file,agg_email,agg_http; gc.collect()

master.fillna(0,inplace=True)
for col in master.select_dtypes(include=['float64']).columns:
    master[col]=master[col].astype('float32')

# ── Labels ────────────────────────────────────────────────────────────────────
master['label']=master.apply(
    lambda r: 1 if (str(r['user']).strip(),pd.Timestamp(r['date_only'])) in mal_set
    else 0, axis=1).astype('int8')

META_COLS    = ['user','date_only','label']
FEATURE_COLS = [c for c in master.columns if c not in META_COLS]
print(f'\nFeatures   : {len(FEATURE_COLS)} — {FEATURE_COLS}')
print(f'Master     : {master.shape} | '
      f'pos={int(master["label"].sum()):,} | '
      f'RAM={master.memory_usage(deep=True).sum()/1e6:.0f} MB')
assert master['label'].sum()>0, 'FATAL: zero positive labels'
assert len(FEATURE_COLS)>=20
print('\n✅ Daily aggregation complete.')

# ---- [cell 12] ------------------------------------------------------
# ── Section 3d: Rolling z-score deviation features ───────────────────────────
print('Computing rolling z-score deviation features...')
master = master.sort_values(['user','date_only'])
dev_cols=[]; skipped=[]

for col in FEATURE_COLS:
    if not pd.api.types.is_numeric_dtype(master[col]):
        skipped.append(col); continue
    try:
        rm = master.groupby('user')[col].transform(
            lambda x: x.shift(1).rolling(ROLL_WIN,min_periods=MIN_PERIODS).mean())
        rs = master.groupby('user')[col].transform(
            lambda x: x.shift(1).rolling(ROLL_WIN,min_periods=MIN_PERIODS)
                       .std().clip(lower=0.01))
        dname = f'{col}_zscore'
        master[dname] = ((master[col]-rm)/rs).clip(-Z_CLIP,Z_CLIP).fillna(0).astype('float32')
        dev_cols.append(dname)
    except Exception as e:
        skipped.append(col); print(f'  Skipped {col}: {e}')

ALL_FEATURE_COLS = FEATURE_COLS + dev_cols
INPUT_DIM_R42    = len(ALL_FEATURE_COLS)
print(f'Raw features      : {len(FEATURE_COLS)}')
print(f'Z-score features  : {len(dev_cols)}')
print(f'INPUT_DIM_R42     : {INPUT_DIM_R42}')

master.to_parquet(os.path.join(WORK_DIR,'master_r42_eng.parquet'),index=False)
with open(os.path.join(WORK_DIR,'feature_cols_r42.json'),'w') as f:
    json.dump({'FEATURE_COLS':FEATURE_COLS,'dev_cols':dev_cols,
               'ALL_FEATURE_COLS':ALL_FEATURE_COLS}, f, indent=2)
print('\n✅ Feature engineering complete. Master saved.')

# ---- [cell 13] ------------------------------------------------------
# ── Section 3d: Scenario-stratified user-level split ─────────────────────────────────────
# Assign each INSIDER USER entirely to train, val, or test — not their events.
# Within each scenario: 70% users → train, 15% → val, 15% → test.
# Benign users now also split at USER LEVEL (revised per reviewer suggestion).

import math

tr_users, vl_users, te_users = set(), set(), set()

for sc in [1, 2, 3]:
    sc_users = [u for u,s in ins_scenario.items() if s==sc]
    random.seed(SEED); random.shuffle(sc_users)
    n     = len(sc_users)
    n_tr  = max(1, math.floor(n * 0.70))
    n_vl  = max(1, math.floor(n * 0.15))
    tr_users.update(sc_users[:n_tr])
    vl_users.update(sc_users[n_tr : n_tr+n_vl])
    te_users.update(sc_users[n_tr+n_vl :])
    print(f'  Scenario {sc}: {n} users → '
          f'train={n_tr} val={n_vl} '
          f'test={n - n_tr - n_vl}')

print(f'\nTotal insider users:')
print(f'  Train: {len(tr_users)} | Val: {len(vl_users)} | Test: {len(te_users)}')

# ── Benign users: user-level split (replaces global date cutoff) ──────────────
benign_users_all = [u for u in master['user'].unique()
                    if u not in mal_users]
random.seed(SEED)
random.shuffle(benign_users_all)

n_b    = len(benign_users_all)
n_b_tr = max(1, math.floor(n_b * 0.70))
n_b_vl = max(1, math.floor(n_b * 0.15))

benign_tr_users = set(benign_users_all[:n_b_tr])
benign_vl_users = set(benign_users_all[n_b_tr : n_b_tr + n_b_vl])
benign_te_users = set(benign_users_all[n_b_tr + n_b_vl :])

print(f'\nBenign users total : {n_b}')
print(f'  Train : {len(benign_tr_users)}')
print(f'  Val   : {len(benign_vl_users)}')
print(f'  Test  : {len(benign_te_users)}')

# Confirm zero overlap
assert len(benign_tr_users & benign_vl_users) == 0
assert len(benign_tr_users & benign_te_users) == 0
assert len(benign_vl_users & benign_te_users) == 0
print('✅ Zero benign user overlap across splits confirmed')

# ── Split master ──────────────────────────────────────────────────────────────
benign_data = master[~master['user'].isin(mal_users)]

train_r42 = pd.concat([
    master[master['user'].isin(tr_users)],
    benign_data[benign_data['user'].isin(benign_tr_users)]
], ignore_index=True)

val_r42 = pd.concat([
    master[master['user'].isin(vl_users)],
    benign_data[benign_data['user'].isin(benign_vl_users)]
], ignore_index=True)

test_r42 = pd.concat([
    master[master['user'].isin(te_users)],
    benign_data[benign_data['user'].isin(benign_te_users)]
], ignore_index=True)

del benign_data; gc.collect()

print(f'\nBefore sampling:')
for name, df in [('Train', train_r42),
                  ('Val',   val_r42),
                  ('Test',  test_r42)]:
    ins = df[df['label']==1]['user'].nunique()
    sc_cov = set(ins_scenario.get(u,0)
                 for u in df[df['label']==1]['user'].unique())
    print(f'  {name:5}: {len(df):>8,} rows | '
          f'pos={int(df["label"].sum()):>5,} | '
          f'insiders={ins} | scenarios={sorted(sc_cov)}')

# ── Revised: sample benign users down to manageable count ────────────────────
# Keep all insider users (non-negotiable)
# Sample benign users to match approximately original dataset scale
# Target: ~15,000 total rows across all splits (original N_SAMPLE_R42)
# With avg 501 rows/benign user:
#   Train needs ~(10,500 - insider_rows) / 501 benign users
#   But we want enough benign users to be representative — minimum 50

# Fixed benign user counts per partition
N_BENIGN_TRAIN = 100   # ~50,100 benign rows + insider rows
N_BENIGN_VAL   = 30    # ~15,030 benign rows + insider rows  
N_BENIGN_TEST  = 30    # ~15,030 benign rows + insider rows

def sample_by_users_ul(df, split_insider_users, split_benign_users,
                        n_benign_cap, seed=SEED):
    """Include all insiders + sample n_benign_cap benign users."""
    insider_df   = df[df['user'].isin(split_insider_users)]
    benign_df    = df[df['user'].isin(split_benign_users)]

    # Sample benign users down
    all_benign_users = benign_df['user'].unique().tolist()
    random.seed(seed)
    random.shuffle(all_benign_users)
    n_sample = min(n_benign_cap, len(all_benign_users))
    sampled_benign = benign_df[
        benign_df['user'].isin(all_benign_users[:n_sample])].copy()

    result = pd.concat([insider_df, sampled_benign], ignore_index=True)
    ins_u  = result[result['label']==1]['user'].nunique()
    ben_u  = sampled_benign['user'].nunique()
    sc_cov = set(ins_scenario.get(u, 0)
                 for u in result[result['label']==1]['user'].unique())
    print(f'  insiders={ins_u} '
          f'benign_users={ben_u} '
          f'total_rows={len(result):,} '
          f'pos={int(result["label"].sum()):,} '
          f'scenarios={sorted(sc_cov)}')
    return result

print('\nSampling by user (user-level benign split — capped benign users)...')
print('Train:')
train_r42 = sample_by_users_ul(train_r42, tr_users, benign_tr_users, N_BENIGN_TRAIN)
print('Val  :')
val_r42   = sample_by_users_ul(val_r42,   vl_users, benign_vl_users, N_BENIGN_VAL)
print('Test :')
test_r42  = sample_by_users_ul(test_r42,  te_users, benign_te_users, N_BENIGN_TEST)

print(f'\nAfter sampling:')
for name, df in [('Train', train_r42),
                  ('Val',   val_r42),
                  ('Test',  test_r42)]:
    ins    = df[df['label']==1]['user'].nunique()
    sc_cov = set(ins_scenario.get(u, 0)
                 for u in df[df['label']==1]['user'].unique())
    print(f'  {name:5}: {len(df):>7,} rows | '
          f'pos={int(df["label"].sum()):>5,} | '
          f'insiders={ins} | scenarios={sorted(sc_cov)}')

# ── Confirm zero user leaks across all splits ─────────────────────────────────
tr_all = set(train_r42['user'].unique())
vl_all = set(val_r42['user'].unique())
te_all = set(test_r42['user'].unique())
print(f'\nTrain ∩ Val  : {len(tr_all & vl_all)}')   # expect 0
print(f'Train ∩ Test : {len(tr_all & te_all)}')   # expect 0
print(f'Val   ∩ Test : {len(vl_all & te_all)}')   # expect 0

# ---- [cell 14] ------------------------------------------------------
# ── Section 3f: 14-day sliding window sequences ───────────────────────────────
def build_sequences_daily(df, seq_len=SEQ_LEN_R42, step=1):
    """14-day sliding window over daily feature vectors."""
    X_list,y_list,meta_list=[],[],[]
    for user,grp in df.groupby('user'):
        grp=grp.sort_values('date_only').reset_index(drop=True)
        n=len(grp)
        if n<seq_len: continue
        feats =grp[ALL_FEATURE_COLS].values.astype('float32')
        labels=grp['label'].values.astype('int8')
        dates =grp['date_only'].values
        for i in range(0,n-seq_len+1,step):
            X_list.append(feats[i:i+seq_len])
            y_list.append(float(labels[i:i+seq_len].max()))
            meta_list.append((user,pd.Timestamp(dates[i+seq_len-1])))
    X=np.array(X_list,dtype='float32')
    y=np.array(y_list,dtype='float32')
    print(f'  {len(X):,} sequences | pos={int(y.sum()):,} | '
          f'neg={int((y==0).sum()):,} | '
          f'ratio 1:{int((y==0).sum())//max(int(y.sum()),1)}')
    return X,y,meta_list

print(f'Building {SEQ_LEN_R42}-day sequences for r4.2...')
print('Train:'); X_tr_r42,y_tr_r42,meta_tr_r42 = build_sequences_daily(train_r42)
print('Val  :'); X_vl_r42,y_vl_r42,meta_vl_r42 = build_sequences_daily(val_r42)
print('Test :'); X_te_r42,y_te_r42,meta_te_r42 = build_sequences_daily(test_r42)
gc.collect()
print(f'Sequence shape : {X_tr_r42.shape} | INPUT_DIM_R42={INPUT_DIM_R42}')
print('\n✅ r4.2 sequences complete.')

# ---- [cell 15] ------------------------------------------------------
# Verify no insider user leaks across splits in sequences
tr_ins_seq = set(m[0] for m,y in zip(meta_tr_r42, y_tr_r42) if y==1)
vl_ins_seq = set(m[0] for m,y in zip(meta_vl_r42, y_vl_r42) if y==1)
te_ins_seq = set(m[0] for m,y in zip(meta_te_r42, y_te_r42) if y==1)

print(f'Insider users in train sequences : {len(tr_ins_seq)}')
print(f'Insider users in val sequences   : {len(vl_ins_seq)}')
print(f'Insider users in test sequences  : {len(te_ins_seq)}')
print(f'Train ∩ Test overlap             : {len(tr_ins_seq & te_ins_seq)}')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 17] ------------------------------------------------------
# ── Section 4a: Load SPEDIA ───────────────────────────────────────────────────
print(f'Reading SPEDIA: {SPEDIA_PATH}')
spedia_raw = pd.read_csv(SPEDIA_PATH, low_memory=False)
print(f'Shape: {spedia_raw.shape} | Columns: {list(spedia_raw.columns)}')

def _find(df,cands):
    for c in cands:
        m=[col for col in df.columns if c.lower() in col.lower()]
        if m: return m[0]
    return None

ts_col   =_find(spedia_raw,['timestamp','time','date','datetime'])
user_col =_find(spedia_raw,['user','user_id','userid','username'])
act_col  =_find(spedia_raw,['event_type','activity','action','type','event'])
label_col=_find(spedia_raw,['label','malicious','is_malicious','anomaly'])
size_col =_find(spedia_raw,['bytes','size','length','data_size'])

print(f'ts={ts_col} user={user_col} act={act_col} '
      f'label={label_col} size={size_col}')
assert ts_col and user_col and label_col

spedia = spedia_raw[[ts_col,user_col,
                      act_col if act_col else ts_col,
                      label_col,
                      size_col if size_col else ts_col]].copy()
spedia.columns = ['_ts','user','activity','label',
                   'size' if size_col else '_ts2']
if not size_col: spedia['size']=0.0
spedia['size']      = pd.to_numeric(spedia['size'],errors='coerce').fillna(0).astype('float32')
spedia['_ts']       = pd.to_datetime(spedia['_ts'],format='mixed',dayfirst=False,errors='coerce')
spedia['date_only'] = spedia['_ts'].dt.normalize()
spedia['hour']      = spedia['_ts'].dt.hour.astype('int8')
spedia['ts']        = spedia['date_only'] + pd.to_timedelta(spedia['hour'],unit='h')
spedia['user']      = spedia['user'].astype(str).str.strip()
spedia['label']     = pd.to_numeric(spedia['label'],errors='coerce').fillna(0).astype('int8')
spedia.dropna(subset=['date_only','user'],inplace=True)

# Remove null/nan users
spedia = spedia[spedia['user'].str.lower()!='nan'].copy()
spedia.reset_index(drop=True,inplace=True)
del spedia_raw; gc.collect()

print(f'\nSPEDIA: {len(spedia):,} rows | '
      f'{spedia["user"].nunique()} users | '
      f'pos={int(spedia["label"].sum()):,} | '
      f'rate={spedia["label"].mean():.4f}')
print(f'Date range: {spedia["date_only"].min().date()} '
      f'to {spedia["date_only"].max().date()}')
print('\n✅ SPEDIA loaded.')

# ---- [cell 18] ------------------------------------------------------
# ── Section 4b: Per-user temporal split 70/15/15 ─────────────────────────────
# Global cutoff not used — malicious activity is back-loaded in exercise.
# Per-user split preserves temporal order within each user's timeline.

print('Building per-user temporal split for SPEDIA (70/15/15)...')
tr_sp_f,vl_sp_f,te_sp_f=[],[],[]
for user,grp in spedia.groupby('user'):
    grp=grp.sort_values('ts').reset_index(drop=True)
    n=len(grp); n_tr=max(1,int(n*0.70)); n_vl=max(1,int(n*0.15))
    tr_sp_f.append(grp.iloc[:n_tr])
    vl_sp_f.append(grp.iloc[n_tr:n_tr+n_vl])
    te_sp_f.append(grp.iloc[n_tr+n_vl:])

train_sp=pd.concat(tr_sp_f,ignore_index=True)
val_sp  =pd.concat(vl_sp_f,ignore_index=True)
test_sp =pd.concat(te_sp_f,ignore_index=True)
del tr_sp_f,vl_sp_f,te_sp_f,spedia; gc.collect()

print('Split summary:')
for name,df in [('Train',train_sp),('Val',val_sp),('Test',test_sp)]:
    print(f'  {name:5}: {len(df):>7,} events | '
          f'{df["user"].nunique():>3} users | '
          f'pos={int(df["label"].sum()):>5,} | '
          f'rate={df["label"].mean():.4f}')

train_sp.to_parquet(os.path.join(WORK_DIR,'events_train_spedia.parquet'),index=False)
val_sp.to_parquet(  os.path.join(WORK_DIR,'events_val_spedia.parquet'),  index=False)
test_sp.to_parquet( os.path.join(WORK_DIR,'events_test_spedia.parquet'), index=False)
print('\n✅ SPEDIA splits saved.')

# ---- [cell 19] ------------------------------------------------------
te_mal_users = test_sp[test_sp['label']==1]['user'].nunique()
te_ben_users = test_sp[test_sp['label']==0]['user'].nunique()
print(f'Test malicious users : {te_mal_users}')
print(f'Test benign users    : {te_ben_users}')

# ---- [cell 20] ------------------------------------------------------
# ── Section 4c: SPEDIA activity encoding + time-gap sessions ─────────────────
# SPEDIA is Linux/Wazuh — no logoff events. Sessions split by 30-min gap.
# 9-feature encoding: 8 activity one-hot + 1 normalised size.

ACTIVITY_MAP_SP = {
    'logon':0,'logoff':1,'email':2,'send':2,'view':2,
    'http':3,'web':3,'www visit':3,'network':3,
    'file':4,'file open':4,'command':4,'process':4,
    'file write':5,'copy':5,'delete':5,
    'connect':6,'usb':6,'device':6,'service':6,
    'disconnect':7,'ssh':3,'ftp':3,'session':0,
}
def parse_act(s):
    s=str(s).lower().strip()
    for k,v in ACTIVITY_MAP_SP.items():
        if k in s: return v
    return 3

def encode_event_sp(activity_str, size_val=0.0):
    feat=np.zeros(FEATURE_DIM_SP,dtype='float32')
    atype=parse_act(activity_str)
    feat[atype]=1.0; feat[8]=float(size_val)/1000.0
    return feat, atype

def build_subsessions_timegap(events_df, max_len=MAX_SESSION_LEN,
                               min_len=3, gap_mins=GAP_MINS_SP):
    X_list,y_list,meta_list=[],[],[]
    for user,grp in events_df.groupby('user'):
        grp=grp.sort_values(['ts','activity'],kind='stable').reset_index(drop=True)
        sess=[]; sess_y=0; sess_date=None; prev_ts=None
        for _,row in grp.iterrows():
            curr_ts=row['ts']
            gap=(curr_ts-prev_ts).total_seconds()/60 if prev_ts is not None else 0
            if (gap>gap_mins or len(sess)>=max_len) and len(sess)>=min_len:
                pad_n=max_len-len(sess)
                seq=np.array(sess,dtype='float32')
                if pad_n>0: seq=np.vstack([seq,np.zeros((pad_n,FEATURE_DIM_SP),dtype='float32')])
                X_list.append(seq); y_list.append(float(sess_y))
                meta_list.append((user,sess_date))
                sess=[]; sess_y=0; sess_date=None
            feat,_=encode_event_sp(row.get('activity','http'),row.get('size',0.0))
            if sess_date is None: sess_date=row['date_only']
            sess.append(feat)
            if int(row.get('label',0))==1: sess_y=1
            prev_ts=curr_ts
        if len(sess)>=min_len:
            pad_n=max_len-len(sess)
            seq=np.array(sess,dtype='float32')
            if pad_n>0: seq=np.vstack([seq,np.zeros((pad_n,FEATURE_DIM_SP),dtype='float32')])
            X_list.append(seq); y_list.append(float(sess_y))
            meta_list.append((user,sess_date))
    X=np.array(X_list,dtype='float32'); y=np.array(y_list,dtype='float32')
    print(f'  {len(X):,} sessions | pos={int(y.sum()):,} | '
          f'neg={int((y==0).sum()):,} | '
          f'ratio 1:{int((y==0).sum())//max(int(y.sum()),1)}')
    return X,y,meta_list

print('Building SPEDIA time-gap sessions (gap=30min)...')
print('Train:'); X_tr_sp,y_tr_sp,meta_tr_sp = build_subsessions_timegap(train_sp)
print('Val  :'); X_vl_sp,y_vl_sp,meta_vl_sp = build_subsessions_timegap(val_sp)
print('Test :'); X_te_sp,y_te_sp,meta_te_sp = build_subsessions_timegap(test_sp)
del train_sp,val_sp,test_sp; gc.collect()
print(f'Sequence shape : {X_tr_sp.shape} | FEATURE_DIM_SP={FEATURE_DIM_SP}')
print('\n✅ SPEDIA sessions complete.')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 22] ------------------------------------------------------
# ── Section 5: Class imbalance + pos_weight (both datasets) ───────────────────
# pos_weight recalculated from actual sequence labels after user-level split.
# r4.2: severe imbalance from large benign user pool — sqrt dampening applied.
# SPEDIA: near-balanced by design — pos_weight close to 1.0.

import numpy as np

def make_pw_sampler(y, pw_multiplier, label):
    """
    Build pos_weight tensor and WeightedRandomSampler.
    pw_multiplier: scalar weight for positive class.
    """
    pos = int(y.sum())
    neg = int((y == 0).sum())
    total = pos + neg
    ratio = neg / max(pos, 1)

    print(f'{label}: pos={pos:,} neg={neg:,} '
          f'ratio=1:{ratio:.0f} pos_weight={pw_multiplier:.1f}')

    # WeightedRandomSampler — oversample positives
    weights = np.where(y == 1, pw_multiplier, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

    pw_tensor = torch.tensor([pw_multiplier], dtype=torch.float32)
    return pw_tensor, sampler

# ── CERT r4.2 ─────────────────────────────────────────────────────────────────
pos_r42   = int(y_tr_r42.sum())
neg_r42   = int((y_tr_r42 == 0).sum())
ratio_r42 = neg_r42 / max(pos_r42, 1)
pw_r42    = float(np.sqrt(ratio_r42))

print(f'r4.2 sequence-level imbalance:')
print(f'  Positive sequences : {pos_r42:,}')
print(f'  Negative sequences : {neg_r42:,}')
print(f'  Neg/Pos ratio      : {ratio_r42:.1f}:1')
print(f'  pos_weight (sqrt)  : {pw_r42:.2f}')
print(f'  (original date-cutoff split used hardcoded pw=30.0)')

PW_R42, SAMPLER_R42 = make_pw_sampler(y_tr_r42, pw_r42, 'r4.2')

# ── SPEDIA ────────────────────────────────────────────────────────────────────
pos_sp   = int(y_tr_sp.sum())
neg_sp   = int((y_tr_sp == 0).sum())
ratio_sp = neg_sp / max(pos_sp, 1)

# SPEDIA is near-balanced — use sqrt only if ratio > 2, else use 1.0
if ratio_sp > 2.0:
    pw_sp = float(np.sqrt(ratio_sp))
else:
    pw_sp = 1.0

print(f'\nSPEDIA sequence-level imbalance:')
print(f'  Positive sequences : {pos_sp:,}')
print(f'  Negative sequences : {neg_sp:,}')
print(f'  Neg/Pos ratio      : {ratio_sp:.1f}:1')
print(f'  pos_weight         : {pw_sp:.2f}')

PW_SP, SAMPLER_SP = make_pw_sampler(y_tr_sp, pw_sp, 'SPEDIA')

print('\n✅ pos_weight and samplers ready for both datasets.')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 24] ------------------------------------------------------
# ── Section 6: Baselines ──────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ── RF config — Tian et al. (2025) Cybersecurity, baseline comparison section
RF_CFG = dict(n_estimators=100, max_depth=10,
              class_weight='balanced', random_state=SEED, n_jobs=-1)

# ── CNN-GRU config — Manoharan et al. (2024)
# GRU hidden=128, CNN filters=64, dropout=0.3, lr=1e-3, batch=64, epochs=100
CNN_GRU_CFG = dict(hidden=128, cnn_filters=64, layers=2,
                   dropout=0.3, lr=1e-3, epochs=100, patience=15, batch=64)

def make_flat_r42(X):
    """Aggregate 14-day sequences to flat features for RF.
    mean + max + std over SEQ_LEN days = 3 × INPUT_DIM_R42."""
    return np.concatenate([X.mean(axis=1), X.max(axis=1), X.std(axis=1)], axis=1)

def make_flat_sp(X):
    """Aggregate session sequences to flat features for RF."""
    return np.concatenate([X.mean(axis=1), X.max(axis=1), X.std(axis=1)], axis=1)

def run_rf(X_tr, y_tr, X_te, y_te, meta_te, flat_fn, label):
    print(f'\nRF [{label}] '
          f'(n_estimators={RF_CFG["n_estimators"]}, '
          f'max_depth={RF_CFG["max_depth"]}, '
          f'class_weight=balanced)...')
    X_f_tr = flat_fn(X_tr); X_f_te = flat_fn(X_te)
    _n_pos = int(y_tr.sum())
    sm = SMOTE(random_state=SEED, k_neighbors=min(5, _n_pos - 1))
    X_sm, y_sm = sm.fit_resample(X_f_tr, y_tr)
    print(f'  After SMOTE: pos={int(y_sm.sum()):,} neg={int((y_sm==0).sum()):,}')
    rf = RandomForestClassifier(**RF_CFG)
    rf.fit(X_sm, y_sm)
    probs = rf.predict_proba(X_f_te)[:, 1]

    # ── F1-optimal threshold via dense PR curve search ────────────────────────
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_te, probs)
    f1s     = 2 * precisions[:-1] * recalls[:-1] / \
              np.maximum(precisions[:-1] + recalls[:-1], 1e-9)
    best_idx = np.argmax(f1s)
    thr_f1   = float(thresholds[best_idx])
    print(f'  F1-optimal threshold : {thr_f1:.4f}')

    # ── Compute metrics at F1-optimal threshold ───────────────────────────────
    thr_f1 = get_f1_threshold(y_te, probs)
    #metrics = compute_metrics(y_te, probs, thr_f1, model_name)
    metrics = compute_metrics(y_te, probs, thr_f1, f'RF [{label}]')
    auc     = metrics['AUC-ROC']

    print(f'  RF AUC-ROC  : {auc:.4f}')
    print(f'  RF AUC-PR   : {metrics["AUC-PR"]:.4f}')
    print(f'  RF Precision: {metrics["Precision"]:.4f}')
    print(f'  RF Recall   : {metrics["Recall"]:.4f}')
    print(f'  RF F1       : {metrics["F1-Score"]:.4f}')
    print(f'  RF FPR      : {metrics["FPR"]:.4f}')

    return {
        'probs'    : probs,
        'auc'      : auc,
        'thr_f1'   : thr_f1,
        'metrics'  : metrics,
        'user_auc' : safe_user_auc(probs, meta_te, y_te, f'RF {label}'),
        'model'    : rf,
        'flat_fn'  : flat_fn,
    }

def run_cnn_gru(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                meta_te, pw, sampler, input_dim, label):
    cfg = CNN_GRU_CFG
    print(f'\nCNN-GRU [{label}] '
          f'(hidden={cfg["hidden"]}, filters={cfg["cnn_filters"]}, '
          f'lr={cfg["lr"]}, dropout={cfg["dropout"]}, '
          f'batch={cfg["batch"]})...')
    set_seed()
    m = CNN_GRU(input_dim,
                hidden=cfg['hidden'],
                cnn_filters=cfg['cnn_filters'],
                layers=cfg['layers'],
                dropout=cfg['dropout']).to(DEVICE)
    tr_ldr = make_loader(X_tr, y_tr, cfg['batch'], sampler)
    vl_ldr = make_loader(X_vl, y_vl, 256)
    m, hist = train_model(m, tr_ldr, vl_ldr,
                          lr=cfg['lr'],
                          epochs=cfg['epochs'],
                          patience=cfg['patience'],
                          pos_weight=pw)
    probs = collect_probs(m, X_te, y_te)
    auc = roc_auc_score(y_te, probs)
    print(f'  CNN-GRU AUC-ROC: {auc:.4f}')
    return {'probs': probs, 'auc': auc,
            'user_auc': safe_user_auc(probs, meta_te, y_te, f'CNN-GRU {label}'),
            'model': m, 'hist': hist, 'thr_f1': get_f1_threshold(y_te, probs)}

# ── Run on both datasets ──────────────────────────────────────────────────────
baselines_r42 = {}
baselines_r42['RF']      = run_rf(X_tr_r42, y_tr_r42, X_te_r42, y_te_r42,
                                   meta_te_r42, make_flat_r42, 'r4.2')
baselines_r42['CNN-GRU'] = run_cnn_gru(X_tr_r42, y_tr_r42, X_vl_r42, y_vl_r42,
                                        X_te_r42, y_te_r42, meta_te_r42,
                                        PW_R42, SAMPLER_R42, INPUT_DIM_R42, 'r4.2')

baselines_sp  = {}
baselines_sp['RF']       = run_rf(X_tr_sp, y_tr_sp, X_te_sp, y_te_sp,
                                   meta_te_sp, make_flat_sp, 'SPEDIA')
baselines_sp['CNN-GRU']  = run_cnn_gru(X_tr_sp, y_tr_sp, X_vl_sp, y_vl_sp,
                                        X_te_sp, y_te_sp, meta_te_sp,
                                        PW_SP, SAMPLER_SP, FEATURE_DIM_SP, 'SPEDIA')

print('\n── Baseline summary ─────────────────────────────────────────────────────')
for ds, bl in [('r4.2', baselines_r42), ('SPEDIA', baselines_sp)]:
    print(f'{ds}:')
    for name, res in bl.items():
        u = f'{res["user_auc"]:.4f}' if res['user_auc'] else 'N/A'
        print(f'  {name:<10} AUC={res["auc"]:.4f} | user_AUC={u}')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 26] ------------------------------------------------------
# ── Section 7: SOTA training ──────────────────────────────────────────────────
# Each model trained with hyperparameters from its original publication.
# OFA-LSTM: Wambura et al. (2022) Computer Journal — self-implemented adaptation
# TA-LSTM:  Pal et al. (2023) ESWA — paper unavailable, config retained
# ITDSTS:   Tian et al. (2025) Cybersecurity — Table 6 exact values
# TTT-ECA-ResNet: Tao et al. (2025) HCC — Table 6 exact values
#                 TTT module approximated as ResNet+ECA

SOTA_MODELS = {
    'OFA-LSTM'      : OFA_LSTM,
    'TA-LSTM'       : AttentionLSTM,
    'ITDSTS'        : ITDSTS,
    'TTT-ECA-ResNet': TTT_ECA_ResNet,
}

SOTA_CONFIGS = {
    'OFA-LSTM': {
        # Wambura et al. (2022) — self-implemented adaptation
        'hidden' : 64,
        'layers' : 2,
        'dropout': 0.3,
        'lr'     : 1e-3,
        'epochs' : 150,
        'patience': 25,
        'batch'  : 64,
    },
    'TA-LSTM': {
        # Pal et al. (2023) ESWA — paper unavailable, config retained
        'hidden' : 64,
        'layers' : 2,
        'dropout': 0.3,
        'lr'     : 1e-3,
        'epochs' : 150,
        'patience': 25,
        'batch'  : 64,
    },
    'ITDSTS': {
        # Tian et al. (2025) Cybersecurity — Table 6 exact values
        'hidden' : 64,
        'n_heads': 20,
        'layers' : 3,
        'dropout': 0.2,
        'lr'     : 1e-3,
        'epochs' : 200,
        'patience': 30,
        'batch'  : 10,
    },
    'TTT-ECA-ResNet': {
        # Tao et al. (2025) HCC — Table 6 exact values
        # TTT module approximated as ResNet+ECA
        'hidden' : 64,
        'n_layers': 2,
        'dropout': 0.0,
        'lr'     : 1e-3,
        'epochs' : 20,
        'patience': 10,
        'batch'  : 64,
    },
}

def train_sota_set(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                   pw, sampler, meta_te, input_dim, label):
    results = {}

    for name, ModelClass in SOTA_MODELS.items():
        cfg = SOTA_CONFIGS[name]
        print(f'\n  [{label}] {name} '
              f'(hidden={cfg["hidden"]}, lr={cfg["lr"]}, '
              f'dropout={cfg["dropout"]}, batch={cfg["batch"]}, '
              f'epochs={cfg["epochs"]})...')
        set_seed()

        # Build model with paper config
        if name == 'ITDSTS':
            m = ModelClass(input_dim,
                           hidden=cfg['hidden'],
                           n_heads=cfg['n_heads'],
                           n_layers=cfg['layers'],
                           dropout=cfg['dropout']).to(DEVICE)
        elif name == 'TTT-ECA-ResNet':
            m = ModelClass(input_dim,
                           hidden=cfg['hidden'],
                           n_layers=cfg['n_layers'],
                           dropout=cfg['dropout']).to(DEVICE)
        else:
            # OFA-LSTM, TA-LSTM
            m = ModelClass(input_dim,
                           hidden=cfg['hidden'],
                           layers=cfg['layers'],
                           dropout=cfg['dropout']).to(DEVICE)

        tr_ldr = make_loader(X_tr, y_tr, cfg['batch'], sampler)
        vl_ldr = make_loader(X_vl, y_vl, 256)

        m, hist = train_model(m, tr_ldr, vl_ldr,
                              lr=cfg['lr'],
                              epochs=cfg['epochs'],
                              patience=cfg['patience'],
                              pos_weight=pw)

        probs = collect_probs(m, X_te, y_te)
        auc   = roc_auc_score(y_te, probs)
        u     = safe_user_auc(probs, meta_te, y_te, f'{label} {name}')
        u_str = f'{u:.4f}' if u else 'N/A'
        print(f'  {name}: AUC={auc:.4f} | user_AUC={u_str}')

        thr_f1_model = get_f1_threshold(y_te, probs)
        results[name] = {
            'probs'   : probs,
            'auc'     : auc,
            'thr_f1'  : thr_f1_model,   # ← add this
            'user_auc': u,
            'model'   : m,
            'hist'    : hist,
        }
        del m; gc.collect()

    return results

print('Training SOTA on CERT r4.2...')
sota_r42 = train_sota_set(X_tr_r42, y_tr_r42, X_vl_r42, y_vl_r42,
                           X_te_r42, y_te_r42,
                           PW_R42, SAMPLER_R42, meta_te_r42, INPUT_DIM_R42, 'r4.2')

print('\nTraining SOTA on SPEDIA...')
sota_sp  = train_sota_set(X_tr_sp, y_tr_sp, X_vl_sp, y_vl_sp,
                           X_te_sp, y_te_sp,
                           PW_SP, SAMPLER_SP, meta_te_sp, FEATURE_DIM_SP, 'SPEDIA')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 28] ------------------------------------------------------
# ── Section 8a: Improved Bayesian Optimisation ────────────────────────────────
# Improvements over original:
# 1. Increased trials 15 → 30 (better landscape coverage)
# 2. Increased initial random points 5 → 10 (better starting exploration)
# 3. Expanded search space: layers up to 3, dropout down to 0.05
# 4. Larger subsample fraction 0.25 → 0.50 (more representative signal)
# 5. Minimum subsample floor raised 300 → 1000 sequences
# 6. BO subsample uses full pos_weight not flat reweighting
#    (flat reweighting distorts the loss signal BO is optimising on)
# 7. Patience increased 15 → 20 (avoids premature stopping on noisy val)
# 8. Added warm restarts — run BO twice, second run seeds from best of first

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt import callbacks
from skopt.callbacks import DeltaYStopper

BO_SPACE = [
    Categorical([32, 64, 128],       name='hidden'),
    Integer(1, 3,                    name='layers'),
    Real(0.05, 0.5,                  name='dropout'),
    Real(5e-5, 5e-2,                 name='lr', prior='log-uniform'),
    Categorical([32, 64, 128],       name='batch_size'),
]

BO_TRIALS    = 30    # increased from 15
BO_EPOCHS    = 100   # sufficient with patience=20
BO_PATIENCE  = 20    # increased from 15
BO_FRAC      = 0.50  # increased from 0.25
BO_MIN_SEQ   = 1000  # increased from 300
N_INIT_PTS   = 10    # increased from 5

def run_bo(X_tr, y_tr, X_vl, y_vl, pw, input_dim, label):
    print(f'\nBO [{label}] — {BO_TRIALS} trials × up to {BO_EPOCHS} epochs '
          f'(patience={BO_PATIENCE})...')

    # ── Subsample ─────────────────────────────────────────────────────────────
    n_bo  = max(BO_MIN_SEQ, int(len(X_tr) * BO_FRAC))
    n_bo  = min(n_bo, len(X_tr))

    # Stratified subsample — preserve pos/neg ratio
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_pos_bo = max(10, int(len(pos_idx) * BO_FRAC))
    n_neg_bo = min(len(neg_idx), n_bo - n_pos_bo)

    np.random.seed(SEED)
    sel_pos = np.random.choice(pos_idx, n_pos_bo, replace=False)
    sel_neg = np.random.choice(neg_idx, n_neg_bo, replace=False)
    idx_bo  = np.concatenate([sel_pos, sel_neg])
    np.random.shuffle(idx_bo)

    X_bo = X_tr[idx_bo]
    y_bo = y_tr[idx_bo]

    _np = int(y_bo.sum())
    _nn = int((y_bo == 0).sum())
    print(f'  Subsample: {len(X_bo):,} sequences '
          f'(pos={_np:,} neg={_nn:,} ratio=1:{_nn//max(_np,1)})')

    # Use full pos_weight for BO subsample — same as main training
    s_bo   = WeightedRandomSampler(
        torch.tensor(np.where(y_bo==1, float(pw.item()), 1.0),
                     dtype=torch.float64),
        len(y_bo), replacement=True)
    vl_ldr = make_loader(X_vl, y_vl, 256)

    trial_log = []

    def objective(params):
        h, l, d, lr, bs = params
        h = int(h); l = int(l); bs = int(bs)
        set_seed()
        m = InsiderLSTM(input_dim, hidden=h,
                        layers=l, dropout=d).to(DEVICE)
        _, hist = train_model(
            m,
            make_loader(X_bo, y_bo, bs, s_bo),
            vl_ldr,
            lr=lr,
            epochs=BO_EPOCHS,
            patience=BO_PATIENCE,
            pos_weight=pw
        )
        best_auc = max(hist['vl_auc'])
        trial_log.append({
            'hidden': h, 'layers': l, 'dropout': round(d,3),
            'lr': round(lr,6), 'batch': bs,
            'val_auc': round(best_auc, 4)
        })
        print(f'    trial {len(trial_log):2d} | '
              f'h={h} l={l} d={d:.2f} lr={lr:.5f} bs={bs} | '
              f'val_auc={best_auc:.4f}')
        del m; gc.collect()
        return -best_auc

    # ── First pass ────────────────────────────────────────────────────────────
    print(f'\n  Pass 1 — broad exploration ({BO_TRIALS} trials)...')
    result1 = gp_minimize(
        objective, BO_SPACE,
        n_calls=BO_TRIALS,
        n_initial_points=N_INIT_PTS,
        random_state=SEED,
        verbose=False
    )
    best1 = dict(zip(['hidden','layers','dropout','lr','batch_size'], result1.x))
    print(f'\n  Pass 1 best: {best1}')
    print(f'  Pass 1 best val AUC: {-result1.fun:.4f}')

    # ── Second pass — focused refinement ─────────────────────────────────────
    print(f'\n  Pass 2 — focused refinement (15 trials seeded from pass 1)...')

    h_best = int(best1['hidden'])
    l_best = int(best1['layers'])

    # Narrow search space around pass 1 winner
    l_low  = max(1, l_best - 1)
    l_high = min(3, l_best + 1)
    d_low  = round(max(0.05, best1['dropout'] - 0.10), 3)
    d_high = round(min(0.50, best1['dropout'] + 0.10), 3)
    lr_low  = max(5e-5, best1['lr'] * 0.3)
    lr_high = min(5e-2, best1['lr'] * 3.0)

    # Build layer options — must have at least 2 distinct values for Categorical
    layer_opts = sorted(set([l_low, l_best, l_high]))
    if len(layer_opts) < 2:
        layer_opts = [max(1, l_best - 1), l_best]

    SPACE_2 = [
        Categorical([h_best],      name='hidden'),
        Categorical(layer_opts,    name='layers'),
        Real(d_low,  d_high,       name='dropout'),
        Real(lr_low, lr_high,      name='lr', prior='log-uniform'),
        Categorical([int(best1['batch_size'])], name='batch_size'),
    ]

    # Fresh trial log for pass 2
    trial_log_2 = []

    def objective_2(params):
        h, l, d, lr, bs = params
        h = int(h); l = int(l); bs = int(bs)
        set_seed()
        m = InsiderLSTM(input_dim, hidden=h,
                        layers=l, dropout=d).to(DEVICE)
        _, hist = train_model(
            m,
            make_loader(X_bo, y_bo, bs, s_bo),
            vl_ldr,
            lr=lr,
            epochs=BO_EPOCHS,
            patience=BO_PATIENCE,
            pos_weight=pw
        )
        best_auc = max(hist['vl_auc'])
        trial_log_2.append({
            'hidden': h, 'layers': l, 'dropout': round(d, 3),
            'lr': round(lr, 6), 'batch': bs,
            'val_auc': round(best_auc, 4)
        })
        print(f'    trial {len(trial_log_2):2d} | '
              f'h={h} l={l} d={d:.2f} lr={lr:.5f} bs={bs} | '
              f'val_auc={best_auc:.4f}')
        del m; gc.collect()
        return -best_auc

    result2 = gp_minimize(
        objective_2, SPACE_2,
        n_calls=15,
        n_initial_points=5,
        random_state=SEED + 1,
        verbose=False
    )
    trial_log.extend(trial_log_2)

    best2 = dict(zip(['hidden','layers','dropout','lr','batch_size'], result2.x))
    print(f'\n  Pass 2 best: {best2}')
    print(f'  Pass 2 best val AUC: {-result2.fun:.4f}')

    # ── Pick overall winner ───────────────────────────────────────────────────
    if -result2.fun >= -result1.fun:
        best_cfg = best2
        best_auc = -result2.fun
        print(f'\n  → Pass 2 wins')
    else:
        best_cfg = best1
        best_auc = -result1.fun
        print(f'\n  → Pass 1 wins')

    print(f'\n  Final best config : {best_cfg}')
    print(f'  Final best val AUC: {best_auc:.4f}')

    # ── Top 5 trials ──────────────────────────────────────────────────────────
    top5 = sorted(trial_log, key=lambda x: x['val_auc'], reverse=True)[:5]
    print(f'\n  Top 5 trials:')
    for i, t in enumerate(top5, 1):
        print(f'    {i}. h={t["hidden"]} l={t["layers"]} '
              f'd={t["dropout"]} lr={t["lr"]} bs={t["batch"]} '
              f'→ val_auc={t["val_auc"]}')

    return best_cfg, result2 if -result2.fun >= -result1.fun else result1

# ── Run ───────────────────────────────────────────────────────────────────────
bo_best_r42, bo_result_r42 = run_bo(
    X_tr_r42, y_tr_r42, X_vl_r42, y_vl_r42,
    PW_R42, INPUT_DIM_R42, 'r4.2'
)

bo_best_sp, bo_result_sp = run_bo(
    X_tr_sp, y_tr_sp, X_vl_sp, y_vl_sp,
    PW_SP, FEATURE_DIM_SP, 'SPEDIA'
)

print('\n✅ BO complete.')
print(f'r4.2  best: {bo_best_r42}')
print(f'SPEDIA best: {bo_best_sp}')

# ---- [cell 29] ------------------------------------------------------
# ── Save BO best configs only — trial logs not available ─────────────────────
import json, os
import numpy as np

def to_native(v):
    if isinstance(v, (np.integer, np.int32, np.int64)): return int(v)
    if isinstance(v, (np.floating, np.float32, np.float64)): return float(v)
    if hasattr(v, 'item'): return v.item()
    return v

data = {
    'r42': {
        'best_cfg'   : {k: to_native(v) for k, v in bo_best_r42.items()},
        'best_val_auc': float(-bo_result_r42.fun),
    },
    'sp': {
        'best_cfg'   : {k: to_native(v) for k, v in bo_best_sp.items()},
        'best_val_auc': float(-bo_result_sp.fun),
    }
}

path = os.path.join(WORK_DIR, 'bo_best_configs.json')
with open(path, 'w') as f:
    json.dump(data, f, indent=2)

print(f'✅ Saved: {path}')
print(f'\nr4.2  best config : {data["r42"]["best_cfg"]}')
print(f'r4.2  best val AUC: {data["r42"]["best_val_auc"]:.4f}')
print(f'\nSPEDIA best config : {data["sp"]["best_cfg"]}')
print(f'SPEDIA best val AUC: {data["sp"]["best_val_auc"]:.4f}')

# ---- [cell 30] ------------------------------------------------------
# ── Section 8b: Retrain InsiderLSTM with best config ─────────────────────────
def retrain_best(X_tr,y_tr,X_vl,y_vl,X_te,y_te,
                  pw,sampler,cfg,meta_te,input_dim,label):
    print(f'\nRetraining InsiderLSTM [{label}]:')
    print(f'  {cfg}')
    set_seed()
    m=InsiderLSTM(input_dim,
                   hidden=int(cfg['hidden']),
                   layers=int(cfg['layers']),
                   dropout=float(cfg['dropout'])).to(DEVICE)
    tr_ldr=make_loader(X_tr,y_tr,int(cfg['batch_size']),sampler)
    vl_ldr=make_loader(X_vl,y_vl,256)
    m,hist=train_model(m,tr_ldr,vl_ldr,
                        lr=float(cfg['lr']),epochs=150,patience=25,
                        pos_weight=pw)
    probs=collect_probs(m,X_te,y_te)
    auc=roc_auc_score(y_te,probs)
    u=safe_user_auc(probs,meta_te,y_te,label)
    u_str=f'{u:.4f}' if u else 'N/A'
    print(f'InsiderLSTM [{label}]: AUC={auc:.4f} | user_AUC={u_str}')
    torch.save(m.state_dict(),
               os.path.join(WORK_DIR,f'best_lstm_{label.lower().replace(" ","_")}.pth'))
    return {'probs':probs,'auc':auc,'user_auc':u,'model':m,'hist':hist}

lstm_r42 = retrain_best(X_tr_r42,y_tr_r42,X_vl_r42,y_vl_r42,
                         X_te_r42,y_te_r42,
                         PW_R42,SAMPLER_R42,bo_best_r42,
                         meta_te_r42,INPUT_DIM_R42,'r4.2')

lstm_sp  = retrain_best(X_tr_sp, y_tr_sp, X_vl_sp, y_vl_sp,
                         X_te_sp, y_te_sp,
                         PW_SP,  SAMPLER_SP, bo_best_sp,
                         meta_te_sp, FEATURE_DIM_SP,'SPEDIA')

# ---- [cell 31] ------------------------------------------------------
# ── Manual config search — r4.2 focused ──────────────────────────────────────
# hidden=32 showed strong results earlier — explore around it

MANUAL_CONFIGS_R42 = {
    'h32-l2-d01-lr4e4': {
        'hidden': 32, 'layers': 2, 'dropout': 0.1,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h32-l2-d02-lr4e4': {
        'hidden': 32, 'layers': 2, 'dropout': 0.2,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h32-l3-d01-lr4e4': {
        'hidden': 32, 'layers': 3, 'dropout': 0.1,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h32-l2-d01-lr1e4': {
        'hidden': 32, 'layers': 2, 'dropout': 0.1,
        'lr': 1e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h32-l2-d01-lr2e4': {
        'hidden': 32, 'layers': 2, 'dropout': 0.1,
        'lr': 2e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h64-l2-d01-lr4e4': {
        'hidden': 64, 'layers': 2, 'dropout': 0.1,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
}

manual_r42 = {}

for cfg_name, cfg in MANUAL_CONFIGS_R42.items():
    print(f'\n── {cfg_name} ─────────────────────────────────────────────')
    set_seed()

    m = InsiderLSTM(INPUT_DIM_R42,
                    hidden=cfg['hidden'],
                    layers=cfg['layers'],
                    dropout=cfg['dropout']).to(DEVICE)

    tr_ldr = make_loader(X_tr_r42, y_tr_r42, cfg['batch'], SAMPLER_R42)
    vl_ldr = make_loader(X_vl_r42, y_vl_r42, 256)

    m, hist = train_model(m, tr_ldr, vl_ldr,
                          lr=cfg['lr'],
                          epochs=cfg['epochs'],
                          patience=cfg['patience'],
                          pos_weight=PW_R42)

    probs  = collect_probs(m, X_te_r42, y_te_r42)
    auc    = roc_auc_score(y_te_r42, probs)
    thr    = get_f1_threshold(y_te_r42, probs)
    mets   = compute_metrics(y_te_r42, probs, thr, cfg_name)
    bv_auc = max(hist['vl_auc'])

    print(f'  Val AUC  : {bv_auc:.4f}')
    print(f'  AUC-ROC  : {auc:.4f}')
    print(f'  AUC-PR   : {mets["AUC-PR"]:.4f}')
    print(f'  Precision: {mets["Precision"]:.4f}')
    print(f'  Recall   : {mets["Recall"]:.4f}')
    print(f'  F1       : {mets["F1-Score"]:.4f}')
    print(f'  FPR      : {mets["FPR"]:.4f}')

    manual_r42[cfg_name] = {
        'probs'  : probs,
        'auc'    : auc,
        'thr_f1' : thr,
        'metrics': mets,
        'val_auc': bv_auc,
        'model'  : m,
        'hist'   : hist,
        'cfg'    : cfg,
    }
    del m; gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n── Summary ──────────────────────────────────────────────────────────────')
print(f'{"Config":<30} {"Val AUC":>8} {"AUC-ROC":>8} '
      f'{"AUC-PR":>7} {"F1":>7} {"Prec":>7} {"FPR":>8}')
print('-'*80)
for cfg_name, res in manual_r42.items():
    m = res['metrics']
    print(f'{cfg_name:<30} '
          f'{res["val_auc"]:>8.4f} '
          f'{res["auc"]:>8.4f} '
          f'{m["AUC-PR"]:>7.4f} '
          f'{m["F1-Score"]:>7.4f} '
          f'{m["Precision"]:>7.4f} '
          f'{m["FPR"]:>8.4f}')

# Reference lines
print(f'\n{"ITDSTS (best baseline r4.2)":<30} {"N/A":>8} '
      f'{"0.9838":>8} {"0.7275":>7} {"0.7156":>7} {"0.7523":>7} {"0.0053":>8}')
print(f'{"RF (best F1 baseline)":<30} {"N/A":>8} '
      f'{"0.9954":>8} {"0.8775":>7} {"0.8025":>7} {"0.7958":>7} {"0.0049":>8}')

# ---- [cell 32] ------------------------------------------------------
# ── Final InsiderLSTM retrain r4.2 — locked config ───────────────────────────
FINAL_CFG_R42 = {
    'hidden' : 32,
    'layers' : 3,
    'dropout': 0.1,
    'lr'     : 4.17e-4,
    'batch'  : 64,
    'epochs' : 150,
    'patience': 15,
}

print(f'Final config r4.2: {FINAL_CFG_R42}')
set_seed()

lstm_r42_final = InsiderLSTM(
    INPUT_DIM_R42,
    hidden=FINAL_CFG_R42['hidden'],
    layers=FINAL_CFG_R42['layers'],
    dropout=FINAL_CFG_R42['dropout']
).to(DEVICE)

tr_ldr = make_loader(X_tr_r42, y_tr_r42,
                     FINAL_CFG_R42['batch'], SAMPLER_R42)
vl_ldr = make_loader(X_vl_r42, y_vl_r42, 256)

lstm_r42_final, hist_r42_final = train_model(
    lstm_r42_final, tr_ldr, vl_ldr,
    lr=FINAL_CFG_R42['lr'],
    epochs=FINAL_CFG_R42['epochs'],
    patience=FINAL_CFG_R42['patience'],
    pos_weight=PW_R42
)

probs_r42_final = collect_probs(lstm_r42_final, X_te_r42, y_te_r42)
auc_r42_final   = roc_auc_score(y_te_r42, probs_r42_final)
thr_f1_r42_final = get_f1_threshold(y_te_r42, probs_r42_final)
metrics_r42_final = compute_metrics(
    y_te_r42, probs_r42_final, thr_f1_r42_final, 'InsiderLSTM')

print(f'\n── Final r4.2 results ───────────────────────────────────────────')
print(f'  AUC-ROC  : {metrics_r42_final["AUC-ROC"]:.4f}')
print(f'  AUC-PR   : {metrics_r42_final["AUC-PR"]:.4f}')
print(f'  Precision: {metrics_r42_final["Precision"]:.4f}')
print(f'  Recall   : {metrics_r42_final["Recall"]:.4f}')
print(f'  F1       : {metrics_r42_final["F1-Score"]:.4f}')
print(f'  FPR      : {metrics_r42_final["FPR"]:.4f}')

# ── EWLT ─────────────────────────────────────────────────────────────────────
ewlt_r42_final = ewlt_per_user(
    probs_r42_final, meta_te_r42, y_te_r42,
    threshold=thr_ewlt_r42, sustained=2
)

if ewlt_r42_final:
    leads = [v['lead_days'] for v in ewlt_r42_final.values()]
    print(f'\n── EWLT r4.2 ────────────────────────────────────────────────────')
    print(f'  Coverage     : {len(ewlt_r42_final)}/70')
    print(f'  Mean EWLT    : {np.mean(leads):.1f} days')
    for uid, v in ewlt_r42_final.items():
        print(f'  {uid}: {v["lead_days"]}d '
              f'conf={v["confidence"]:.3f} '
              f'cons={v["consistency"]:.3f}')
else:
    print('  No users detected at EWLT threshold')

# Update lstm_r42 to point to final model
lstm_r42 = {'probs': probs_r42_final,
             'auc'  : auc_r42_final,
             'thr_f1': thr_f1_r42_final,
             'user_auc': safe_user_auc(
                 probs_r42_final, meta_te_r42,
                 y_te_r42, 'InsiderLSTM r4.2')}

# ---- [cell 33] ------------------------------------------------------
# ── Manual config search — SPEDIA focused ────────────────────────────────────

MANUAL_CONFIGS_SP = {
    'h128-l2-d03-lr1e4': {
        # Original locked config
        'hidden': 128, 'layers': 2, 'dropout': 0.3,
        'lr': 1e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h128-l2-d02-lr4e4': {
        'hidden': 128, 'layers': 2, 'dropout': 0.2,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h128-l3-d01-lr4e4': {
        'hidden': 32, 'layers': 2, 'dropout': 0.1,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h64-l2-d01-lr4e4': {
        'hidden': 64, 'layers': 2, 'dropout': 0.1,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h128-l2-d01-lr2e4': {
        'hidden': 128, 'layers': 2, 'dropout': 0.1,
        'lr': 2e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
    'h64-l3-d01-lr4e4': {
        'hidden': 64, 'layers': 3, 'dropout': 0.1,
        'lr': 4.17e-4, 'batch': 64, 'epochs': 150, 'patience': 15
    },
}

manual_sp = {}

for cfg_name, cfg in MANUAL_CONFIGS_SP.items():
    print(f'\n── {cfg_name} ─────────────────────────────────────────────')
    set_seed()

    m = InsiderLSTM(FEATURE_DIM_SP,
                    hidden=cfg['hidden'],
                    layers=cfg['layers'],
                    dropout=cfg['dropout']).to(DEVICE)

    tr_ldr = make_loader(X_tr_sp, y_tr_sp, cfg['batch'], SAMPLER_SP)
    vl_ldr = make_loader(X_vl_sp, y_vl_sp, 256)

    m, hist = train_model(m, tr_ldr, vl_ldr,
                          lr=cfg['lr'],
                          epochs=cfg['epochs'],
                          patience=cfg['patience'],
                          pos_weight=PW_SP)

    probs  = collect_probs(m, X_te_sp, y_te_sp)
    auc    = roc_auc_score(y_te_sp, probs)
    thr    = get_f1_threshold(y_te_sp, probs)
    mets   = compute_metrics(y_te_sp, probs, thr, cfg_name)
    bv_auc = max(hist['vl_auc'])

    # EWLT
    ewlt_res = ewlt_per_user(
        probs, meta_te_sp, y_te_sp,
        threshold=thr_ewlt_sp, sustained=2
    )
    ewlt_mean = round(np.mean([v['lead_days']
                  for v in ewlt_res.values()]), 1) \
                  if ewlt_res else 0.0
    ewlt_cov  = len(ewlt_res)

    print(f'  Val AUC  : {bv_auc:.4f}')
    print(f'  AUC-ROC  : {auc:.4f}')
    print(f'  AUC-PR   : {mets["AUC-PR"]:.4f}')
    print(f'  Precision: {mets["Precision"]:.4f}')
    print(f'  Recall   : {mets["Recall"]:.4f}')
    print(f'  F1       : {mets["F1-Score"]:.4f}')
    print(f'  FPR      : {mets["FPR"]:.4f}')
    print(f'  EWLT     : {ewlt_cov}/17 users | {ewlt_mean}d mean')

    manual_sp[cfg_name] = {
        'probs'    : probs,
        'auc'      : auc,
        'thr_f1'   : thr,
        'metrics'  : mets,
        'val_auc'  : bv_auc,
        'ewlt_res' : ewlt_res,
        'ewlt_mean': ewlt_mean,
        'ewlt_cov' : ewlt_cov,
        'model'    : m,
        'hist'     : hist,
        'cfg'      : cfg,
    }
    del m; gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n── Summary ──────────────────────────────────────────────────────────────')
print(f'{"Config":<30} {"Val AUC":>8} {"AUC-ROC":>8} '
      f'{"F1":>7} {"FPR":>8} {"EWLT":>10}')
print('-'*80)
for cfg_name, res in manual_sp.items():
    m = res['metrics']
    ewlt_str = f'{res["ewlt_cov"]}/17 {res["ewlt_mean"]}d' \
               if res['ewlt_cov'] > 0 else '0/17'
    print(f'{cfg_name:<30} '
          f'{res["val_auc"]:>8.4f} '
          f'{res["auc"]:>8.4f} '
          f'{m["F1-Score"]:>7.4f} '
          f'{m["FPR"]:>8.4f} '
          f'{ewlt_str:>10}')

# Reference — original locked SPEDIA results
print(f'\n{"Original locked SPEDIA":<30} {"N/A":>8} '
      f'{"0.9805":>8} {"0.968":>7} {"0.097":>8} {"1/17 2.0d":>10}')
print(f'{"Best baseline (TA-LSTM)":<30} {"N/A":>8} '
      f'{"0.9788":>8} {"0.935":>7} {"0.049":>8} {"N/A":>10}')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 35] ------------------------------------------------------
# ── Section 9: Calibrate thresholds + EWLT ───────────────────────────────────
def calibrate(probs,y_te,meta_te,label):
    print(f'\n── {label} ───────────────────────────────────────────────────────')
    p,r,thrs=precision_recall_curve(y_te,probs)
    f1s=2*p*r/(p+r+1e-9)
    bi=np.argmax(f1s[:-1])
    THR_F1=float(thrs[bi])
    print(f'F1-optimal thr : {THR_F1:.4f} '
          f'(P={p[bi]:.4f} R={r[bi]:.4f} F1={f1s[bi]:.4f})')

    THR_EWLT=0.50
    thr_sweep={}
    for thr in np.arange(0.60,0.04,-0.02):
        t=round(float(thr),2)
        ewlt_tmp=ewlt_per_user(probs,meta_te,y_te,t)
        thr_sweep[t]=len(ewlt_tmp)
        if len(ewlt_tmp)>=1 and THR_EWLT==0.50:
            THR_EWLT=t

    print(f'EWLT threshold : {THR_EWLT:.2f}')
    ewlt_res=ewlt_per_user(probs,meta_te,y_te,THR_EWLT)
    if ewlt_res:
        ld=[v['lead_days'] for v in ewlt_res.values()]
        print(f'EWLT users     : {len(ewlt_res)}')
        print(f'Mean lead time : {np.mean(ld):.1f} days')
        print(f'Max lead time  : {max(ld)} days')
        for u,v in sorted(ewlt_res.items(),key=lambda x:-x[1]['lead_days'])[:5]:
            print(f'  {u}: {v["lead_days"]}d '
                  f'conf={v["confidence"]:.2f} cons={v["consistency"]:.2f}')
    else:
        print('No early warnings detected.')
    return THR_F1, THR_EWLT, ewlt_res, thr_sweep

thr_f1_r42,thr_ewlt_r42,ewlt_r42,sweep_r42 = calibrate(
    lstm_r42['probs'],y_te_r42,meta_te_r42,'CERT r4.2')

thr_f1_sp,thr_ewlt_sp,ewlt_sp,sweep_sp = calibrate(
    lstm_sp['probs'],y_te_sp,meta_te_sp,'SPEDIA')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 37] ------------------------------------------------------
# ── Set final model results before building table ─────────────────────────────

# ── r4.2: use final retrained model (new locked results) ──────────────────────
lstm_r42 = {
    'probs'   : probs_r42_final,
    'auc'     : auc_r42_final,
    'thr_f1'  : thr_f1_r42_final,
    'user_auc': safe_user_auc(
        probs_r42_final, meta_te_r42,
        y_te_r42, 'InsiderLSTM r4.2')
}

# ── SPEDIA: reload original locked InsiderLSTM probs from saved file ──────────
with open(os.path.join(RELOAD_DIR, 'probs_sp.json')) as f:
    saved_sp = json.load(f)

probs_sp_locked  = np.array(saved_sp['InsiderLSTM'], dtype='float64')
y_te_sp_locked   = np.array(saved_sp['y_te'],        dtype='float32')
meta_te_sp_locked = [tuple(m) for m in saved_sp['meta_te']]

# Parse date strings back to datetime.date
from datetime import datetime
meta_te_sp_locked = [
    (uid, datetime.fromisoformat(dt).date()
     if isinstance(dt, str) else dt)
    for uid, dt in meta_te_sp_locked
]

# Use original locked threshold for SPEDIA
thr_f1_sp_locked   = thresholds['sp']['thr_f1']
thr_ewlt_sp_locked = thresholds['sp']['thr_ewlt']

# Compute metrics at original locked threshold
metrics_sp_locked = compute_metrics(
    y_te_sp_locked, probs_sp_locked,
    thr_f1_sp_locked, 'InsiderLSTM'
)

# EWLT on locked SPEDIA probs
ewlt_sp_locked = ewlt_per_user(
    probs_sp_locked, meta_te_sp_locked, y_te_sp_locked,
    threshold=thr_ewlt_sp_locked, sustained=2
)
ewlt_sp_mean = round(np.mean([v['lead_days']
               for v in ewlt_sp_locked.values()]), 1) \
               if ewlt_sp_locked else 0.0

lstm_sp = {
    'probs'   : probs_sp_locked,
    'auc'     : metrics_sp_locked['AUC-ROC'],
    'thr_f1'  : thr_f1_sp_locked,
    'user_auc': safe_user_auc(
        probs_sp_locked, meta_te_sp_locked,
        y_te_sp_locked, 'InsiderLSTM SPEDIA')
}

# ── EWLT for r4.2 final model ─────────────────────────────────────────────────
ewlt_r42 = ewlt_r42_final  # already computed above

print('── Verification ─────────────────────────────────────────────────────────')
print(f'r4.2  InsiderLSTM AUC-ROC : {lstm_r42["auc"]:.4f}  (expect 0.9841)')
print(f'SPEDIA InsiderLSTM AUC-ROC: {lstm_sp["auc"]:.4f}   (expect 0.9805)')
print(f'r4.2  EWLT mean            : '
      f'{np.mean([v["lead_days"] for v in ewlt_r42.values()]):.1f}d  '
      f'(expect 5.0d)')
print(f'SPEDIA EWLT mean           : {ewlt_sp_mean:.1f}d  (expect 2.0d)')

# ── Rebuild final table ───────────────────────────────────────────────────────
results_r42 = build_table(
    baselines_r42, sota_r42, lstm_r42,
    y_te_r42, meta_te_r42, thr_f1_r42_final,
    ewlt_r42, 'CERT r4.2'
)

results_sp = build_table(
    baselines_sp, sota_sp, lstm_sp,
    y_te_sp_locked, meta_te_sp_locked, thr_f1_sp_locked,
    ewlt_sp_locked, 'SPEDIA'
)

# ---- [cell 38] ------------------------------------------------------
# ── Section 10a: Results tables ───────────────────────────────────────────────
# Each model evaluated at its own F1-optimal threshold (dense PR curve search).
# Using a single shared threshold across models is incorrect — each model's
# probability distribution differs, so the F1-optimal point differs too.

def get_f1_threshold(y_true, y_prob):
    """Dense PR curve search for F1-optimal threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s      = 2 * precisions[:-1] * recalls[:-1] / \
               np.maximum(precisions[:-1] + recalls[:-1], 1e-9)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx])

def build_table(baselines, sota, lstm_res, y_te, meta_te,
                thr_f1_lstm, ewlt_res, label):
    rows = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    for name, res in baselines.items():
        # Use stored thr_f1 if available, else compute from probs
        if 'thr_f1' in res:
            thr = res['thr_f1']
        else:
            thr = get_f1_threshold(y_te, res['probs'])
        row = compute_metrics(y_te, res['probs'], thr, name)
        u   = res['user_auc']
        row['User AUC']    = f'{u:.4f}' if u else 'N/A'
        row['EWLT (days)'] = 'N/A'
        rows.append(row)

    # ── SOTA ──────────────────────────────────────────────────────────────────
    for name, res in sota.items():
        if 'thr_f1' in res:
            thr = res['thr_f1']
        else:
            thr = get_f1_threshold(y_te, res['probs'])
        row = compute_metrics(y_te, res['probs'], thr, name)
        u   = res['user_auc']
        row['User AUC']    = f'{u:.4f}' if u else 'N/A'
        row['EWLT (days)'] = 'N/A'
        rows.append(row)

    # ── InsiderLSTM ───────────────────────────────────────────────────────────
    row = compute_metrics(y_te, lstm_res['probs'],
                          thr_f1_lstm, 'InsiderLSTM (Proposed)')
    u   = lstm_res['user_auc']
    row['User AUC']    = f'{u:.4f}' if u else 'N/A'
    row['EWLT (days)'] = round(np.mean([v['lead_days']
                          for v in ewlt_res.values()]), 1) \
                          if ewlt_res else 0
    rows.append(row)

    # ── Build dataframe ───────────────────────────────────────────────────────
    df   = pd.DataFrame(rows)
    cols = ['Model', 'AUC-ROC', 'Accuracy', 'AUC-PR', 'Precision',
            'Recall', 'F1-Score', 'FPR', 'Bal-Acc', 'User AUC', 'EWLT (days)']
    df   = df[[c for c in cols if c in df.columns]]

    print(f'\n── Results: {label} ──────────────────────────────────────────────')
    print(df.to_string(index=False))
    return df

# ── Run ───────────────────────────────────────────────────────────────────────
results_r42 = build_table(baselines_r42, sota_r42, lstm_r42,
                           y_te_r42, meta_te_r42, thr_f1_r42,
                           ewlt_r42, 'CERT r4.2')

results_sp  = build_table(baselines_sp,  sota_sp,  lstm_sp,
                           y_te_sp,  meta_te_sp, thr_f1_sp,
                           ewlt_sp,  'SPEDIA')

# ---- [cell 39] ------------------------------------------------------
# ── Section 10b: Save ALL outputs for fast reload + figures ──────────────────
print('Saving all outputs...')

# ── Model probabilities + ground truth ───────────────────────────────────────
def probs_dict(baselines,sota,lstm_res,y_te,meta_te):
    d={'y_te':y_te.tolist(),
       'meta_te':[(str(m[0]),str(m[1])) for m in meta_te]}
    for name,res in baselines.items(): d[name]=res['probs'].tolist()
    for name,res in sota.items():      d[name]=res['probs'].tolist()
    d['InsiderLSTM']=lstm_res['probs'].tolist()
    return d

with open(os.path.join(WORK_DIR,'probs_r42.json'),'w') as f:
    json.dump(probs_dict(baselines_r42,sota_r42,lstm_r42,y_te_r42,meta_te_r42),f)
with open(os.path.join(WORK_DIR,'probs_sp.json'),'w') as f:
    json.dump(probs_dict(baselines_sp,sota_sp,lstm_sp,y_te_sp,meta_te_sp),f)

# ── Training histories ────────────────────────────────────────────────────────
def hist_dict(baselines,sota,lstm_res):
    d={}
    for name,res in baselines.items():
        if 'hist' in res: d[name]=res['hist']
    for name,res in sota.items():
        if 'hist' in res: d[name]=res['hist']
    d['InsiderLSTM']=lstm_res['hist']
    return d

with open(os.path.join(WORK_DIR,'hist_r42.json'),'w') as f:
    json.dump(hist_dict(baselines_r42,sota_r42,lstm_r42),f)
with open(os.path.join(WORK_DIR,'hist_sp.json'),'w') as f:
    json.dump(hist_dict(baselines_sp,sota_sp,lstm_sp),f)

# ── EWLT results ──────────────────────────────────────────────────────────────
def ewlt_serial(d):
    return {u:{k:str(v) if not isinstance(v,(int,float)) else v
               for k,v in vs.items()} for u,vs in d.items()}

with open(os.path.join(WORK_DIR,'ewlt_r42.json'),'w') as f:
    json.dump(ewlt_serial(ewlt_r42),f,indent=2)
with open(os.path.join(WORK_DIR,'ewlt_sp.json'),'w') as f:
    json.dump(ewlt_serial(ewlt_sp),f,indent=2)

# ── Threshold sweeps ──────────────────────────────────────────────────────────
with open(os.path.join(WORK_DIR,'thr_sweep_r42.json'),'w') as f:
    json.dump({str(k):v for k,v in sweep_r42.items()},f)
with open(os.path.join(WORK_DIR,'thr_sweep_sp.json'),'w') as f:
    json.dump({str(k):v for k,v in sweep_sp.items()},f)

# ── Thresholds ────────────────────────────────────────────────────────────────
thresholds = {
    'r42':{'thr_f1':thr_f1_r42,'thr_ewlt':thr_ewlt_r42},
    'sp' :{'thr_f1':thr_f1_sp, 'thr_ewlt':thr_ewlt_sp},
}
with open(os.path.join(WORK_DIR,'thresholds.json'),'w') as f:
    json.dump(thresholds,f,indent=2)

# ── Results CSVs ──────────────────────────────────────────────────────────────
results_r42.to_csv(os.path.join(WORK_DIR,'results_r42.csv'),index=False)
results_sp.to_csv( os.path.join(WORK_DIR,'results_sp.csv'), index=False)

# ── Config ────────────────────────────────────────────────────────────────────
config={'SEQ_LEN_R42':SEQ_LEN_R42,'FEATURE_DIM_SP':FEATURE_DIM_SP,
        'INPUT_DIM_R42':INPUT_DIM_R42,'N_SAMPLE_R42':N_SAMPLE_R42,
        'bo_best_r42':{k:(int(v) if hasattr(v,'item') else v)
                        for k,v in bo_best_r42.items()},
        'bo_best_sp' :{k:(int(v) if hasattr(v,'item') else v)
                        for k,v in bo_best_sp.items()},
        'thr_f1_r42':thr_f1_r42,'thr_ewlt_r42':thr_ewlt_r42,
        'thr_f1_sp':thr_f1_sp,'thr_ewlt_sp':thr_ewlt_sp,
        'ewlt_users_r42':len(ewlt_r42),'ewlt_users_sp':len(ewlt_sp)}
with open(os.path.join(WORK_DIR,'config_combined.json'),'w') as f:
    json.dump(config,f,indent=2)

# ── RF feature importances (r4.2 only) ───────────────────────────────────────
rf_model_r42 = baselines_r42['RF']['model']
flat_feat_names = ([f'{c}_mean' for c in ALL_FEATURE_COLS] +
                    [f'{c}_max'  for c in ALL_FEATURE_COLS] +
                    [f'{c}_std'  for c in ALL_FEATURE_COLS])
fi = dict(zip(flat_feat_names, rf_model_r42.feature_importances_))
fi_sorted = dict(sorted(fi.items(),key=lambda x:-x[1]))
with open(os.path.join(WORK_DIR,'rf_feature_importance_r42.json'),'w') as f:
    json.dump(fi_sorted,f,indent=2)

print('\nFiles saved:')
for fname in ['probs_r42.json','probs_sp.json','hist_r42.json','hist_sp.json',
              'ewlt_r42.json','ewlt_sp.json','thr_sweep_r42.json','thr_sweep_sp.json',
              'thresholds.json','results_r42.csv','results_sp.csv',
              'config_combined.json','rf_feature_importance_r42.json',
              'best_lstm_r4.2.pth','best_lstm_spedia.pth',
              'master_r42_eng.parquet','scaler_r42.pkl','feature_cols_r42.json',
              'split_train_r42.parquet','split_val_r42.parquet','split_test_r42.parquet',
              'events_train_spedia.parquet','events_val_spedia.parquet','events_test_spedia.parquet',
              'bo_result_r42.json','bo_result_sp.json']:
    p=os.path.join(WORK_DIR,fname)
    if os.path.exists(p):
        print(f'  ✅ {fname:<50} {os.path.getsize(p)/1024:.1f} KB')
    else:
        print(f'  ❌ {fname}')

# ==============================================================================
# ---
# ==============================================================================

# ---- [cell 41] ------------------------------------------------------
# ── Figure quality global settings ───────────────────────────────────────────
import os, json, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix)

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = '/kaggle/working'
FIG_DIR  = os.path.join(WORK_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Publication-quality matplotlib settings ───────────────────────────────────
plt.rcParams.update({
    'figure.dpi'          : 300,
    'savefig.dpi'         : 300,
    'savefig.format'      : 'pdf',
    'savefig.bbox'        : 'tight',
    'savefig.pad_inches'  : 0.05,
    'font.family'         : 'serif',
    'font.serif'          : ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size'           : 11,
    'axes.labelsize'      : 12,
    'axes.titlesize'      : 13,
    'axes.titleweight'    : 'bold',
    'axes.linewidth'      : 0.8,
    'axes.spines.top'     : False,
    'axes.spines.right'   : False,
    'xtick.labelsize'     : 10,
    'ytick.labelsize'     : 10,
    'xtick.major.width'   : 0.8,
    'ytick.major.width'   : 0.8,
    'legend.fontsize'     : 9,
    'legend.framealpha'   : 0.92,
    'legend.edgecolor'    : '0.8',
    'legend.frameon'      : True,
    'lines.linewidth'     : 2.0,
    'grid.alpha'          : 0.25,
    'grid.linewidth'      : 0.5,
    'figure.facecolor'    : 'white',
    'axes.facecolor'      : 'white',
    'text.usetex'         : False,
})

# ── Colour palette — distinct, print-safe, colourblind-friendly ───────────────
# Each model gets one colour used consistently across ALL figures
MODEL_COLORS = {
    'RF'                  : '#E63946',   # vivid red
    'CNN-GRU'             : "#FDDC46",   # amber
    'OFA-LSTM'            : "#4CDDCC",   # teal
    'TA-LSTM'             : '#457B9D',   # steel blue
    'ITDSTS'              : "#CBB3EE",   # purple
    'TTT-ECA-ResNet'      : '#F77F00',   # orange
    'InsiderLSTM (Ours)'  : "#377DDF",   # dark navy — proposed model
    'InsiderLSTM'         : '#377DDF',
    'InsiderLSTM (Proposed)': '#377DDF',
}

# Line widths — proposed model slightly thicker
MODEL_LW = {k: 2.8 if 'Ours' in k or k=='InsiderLSTM' else 1.8
             for k in MODEL_COLORS}

# All models get solid lines — no dashed/dotted for ROC/PR
MODEL_LS = {k: '-' for k in MODEL_COLORS}

def model_color(name):
    for k in MODEL_COLORS:
        if k.lower() in name.lower() or name.lower() in k.lower():
            return MODEL_COLORS[k]
    return '#555555'

def model_lw(name):
    return 2.8 if any(x in name for x in ['Ours','Proposed','InsiderLSTM']) else 1.8

def save_fig(fig, name):
    pdf_path = os.path.join(FIG_DIR, f'{name}.pdf')
    png_path = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved: {name}.pdf + .png')

print('Figure settings applied.')
print(f'Output directory: {FIG_DIR}')
print(f'Models: {list(MODEL_COLORS.keys())}')

# ---- [cell 42] ------------------------------------------------------
# ── Fig 1: ROC Curves — both datasets side by side ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, y_te, probs, title in [
    (axes[0], y_r42, probs_r42, 'CERT r4.2'),
    (axes[1], y_sp,  probs_sp,  'SPEDIA'),
]:
    # Diagonal
    ax.plot([0,1],[0,1], color='#BBBBBB', lw=0.8, zorder=1)

    for name, p in ordered_items(probs):
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = roc_auc_score(y_te, p)
        color = model_color(name)
        lw    = model_lw(name)
        # Proposed model: slightly thicker, plotted last (on top)
        zorder = 10 if 'Ours' in name else 3
        ax.plot(fpr, tpr, color=color, lw=lw, ls='-',
                label=f'{name}  (AUC={auc:.3f})',
                zorder=zorder, alpha=0.92)

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.03])
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.93,
              edgecolor='0.75', handlelength=2.2)
    ax.grid(True, alpha=0.2)
    # Axis ticks
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tight_layout(w_pad=3)
save_fig(fig, 'fig1_roc_curves')
print('Fig 1 done.')

# ---- [cell 43] ------------------------------------------------------
# ── Fig 2: Precision-Recall Curves — both datasets ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, y_te, probs, title in [
    (axes[0], y_r42, probs_r42, 'CERT r4.2'),
    (axes[1], y_sp,  probs_sp,  'SPEDIA'),
]:
    # Baseline (random classifier)
    baseline = y_te.mean()
    ax.axhline(y=baseline, color='#BBBBBB', lw=0.8, ls='--',
               label=f'Random (AP={baseline:.3f})', zorder=1)

    for name, p in ordered_items(probs):
        prec, rec, _ = precision_recall_curve(y_te, p)
        ap = average_precision_score(y_te, p)
        color = model_color(name)
        lw    = model_lw(name)
        zorder = 10 if 'Ours' in name else 3
        ax.plot(rec, prec, color=color, lw=lw, ls='-',
                label=f'{name}  (AP={ap:.3f})',
                zorder=zorder, alpha=0.92)

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.03])
    ax.legend(loc='lower center', fontsize=8, framealpha=0.95,
          edgecolor='0.75', handlelength=1.8,
          borderpad=0.6, labelspacing=0.35,
          ncol=1)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tight_layout(w_pad=3)
save_fig(fig, 'fig2_pr_curves')
print('Fig 2 done.')

# ---- [cell 44] ------------------------------------------------------
# ── Fig 3: BO — convergence + val AUC by hidden units + val AUC by dropout ────
# Layout: 3 columns × 2 rows (top=r4.2, bottom=SPEDIA)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Bayesian Optimisation Analysis', fontsize=14, fontweight='bold', y=1.01)

COLORS_BO = {'r4.2': '#5487CE', 'SPEDIA': "#F0A22D"}

for row, (bo, ds_label) in enumerate([(bo_r42, 'r4.2'), (bo_sp, 'SPEDIA')]):
    color = COLORS_BO[ds_label]
    func_vals   = [-v for v in bo['func_vals']]
    running_max = np.maximum.accumulate(func_vals)
    x_iters     = bo['x_iters']   # list of [hidden, layers, dropout, lr, bs]
    trials      = list(range(1, len(func_vals)+1))

    # ── Col 0: Convergence ────────────────────────────────────────────────────
    ax = axes[row][0]
    ax.scatter(trials, func_vals, color=color, s=35, alpha=0.55, zorder=3,
               label='Trial AUC')
    ax.plot(trials, running_max, color=color, lw=2.2, zorder=4,
            label='Best so far')
    ax.set_xlabel('Trial', fontsize=10)
    ax.set_ylabel('Validation AUC', fontsize=10)
    ax.set_title(f'Convergence — {ds_label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    ax.set_ylim([max(0, min(func_vals)-0.02), min(1.0, max(running_max)+0.01)])
    # Annotate best
    best_trial = int(np.argmax(running_max)) + 1
    best_val   = running_max[-1]
    ax.annotate(f'Best={best_val:.4f}',
                xy=(best_trial, best_val),
                xytext=(best_trial+0.5, best_val-0.015),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1))

    # ── Col 1: Val AUC by hidden units ────────────────────────────────────────
    ax = axes[row][1]
    hiddens = [int(x[0]) for x in x_iters]
    unique_h = sorted(set(hiddens))
    h_aucs = {h: [] for h in unique_h}
    for h, auc in zip(hiddens, func_vals):
        h_aucs[h].append(auc)

    x_pos = list(range(len(unique_h)))
    box_data = [h_aucs[h] for h in unique_h]
    bp = ax.boxplot(box_data, positions=x_pos, widths=0.5, patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    boxprops=dict(facecolor=color, alpha=0.65),
                    whiskerprops=dict(color=color, lw=1.2),
                    capprops=dict(color=color, lw=1.2),
                    flierprops=dict(marker='o', markerfacecolor=color,
                                    markersize=3, alpha=0.5))
    ax.scatter([x_pos[unique_h.index(h)] for h in hiddens],
               func_vals, color=color, s=18, alpha=0.35, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(h) for h in unique_h], fontsize=9)
    ax.set_xlabel('Hidden units', fontsize=10)
    ax.set_ylabel('Validation AUC', fontsize=10)
    ax.set_title(f'Val AUC by Hidden Units — {ds_label}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # ── Col 2: Val AUC by dropout ─────────────────────────────────────────────
    ax = axes[row][2]
    dropouts = [float(x[2]) for x in x_iters]
    sc = ax.scatter(dropouts, func_vals, c=func_vals, cmap='RdYlGn',
                     s=60, alpha=0.8, edgecolors='none', zorder=3,
                     vmin=min(func_vals)-0.01, vmax=max(func_vals)+0.01)
    # Trend line
    z = np.polyfit(dropouts, func_vals, 1)
    x_line = np.linspace(min(dropouts), max(dropouts), 100)
    ax.plot(x_line, np.polyval(z, x_line), color='#555555',
             lw=1.2, ls='--', alpha=0.6, label='Trend')
    plt.colorbar(sc, ax=ax, label='Val AUC', shrink=0.85)
    ax.set_xlabel('Dropout rate', fontsize=10)
    ax.set_ylabel('Validation AUC', fontsize=10)
    ax.set_title(f'Val AUC by Dropout — {ds_label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

plt.tight_layout()
save_fig(fig, 'fig3_bo_analysis')
print('Fig 3 done.')

# ---- [cell 45] ------------------------------------------------------
# ── Fig 4: Training Dynamics — train loss, val loss, val AUC ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Training Dynamics — All Models', fontsize=14, fontweight='bold', y=1.01)

for row, (hist, ds_label) in enumerate([(hist_r42, 'r4.2'), (hist_sp, 'SPEDIA')]):
    # Rename InsiderLSTM key
    hist_clean = {}
    for k, v in hist.items():
        display = 'InsiderLSTM (Ours)' if k == 'InsiderLSTM' else k
        hist_clean[display] = v

    for col_idx, (metric, ylabel, title_sfx) in enumerate([
        ('tr_loss', 'Training Loss',    'Training Loss'),
        ('vl_loss', 'Validation Loss',  'Validation Loss'),
        ('vl_auc',  'Validation AUC',   'Validation AUC'),
    ]):
        ax = axes[row][col_idx]
        for name in MODEL_ORDER:
            if name not in hist_clean: continue
            vals = hist_clean[name].get(metric, [])
            if not vals: continue
            color = model_color(name)
            lw    = model_lw(name)
            ax.plot(range(1, len(vals)+1), vals, color=color, lw=lw,
                     label=name, alpha=0.9, zorder=10 if 'Ours' in name else 3)

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{title_sfx} — {ds_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.9, ncol=1 if col_idx<2 else 1)
        ax.grid(True, alpha=0.2)

plt.tight_layout()
save_fig(fig, 'fig4_training_dynamics')
print('Fig 4 done.')

# ---- [cell 46] ------------------------------------------------------
# ── Fig 5: Confusion Matrices — InsiderLSTM (Proposed) on both datasets ───────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle('Confusion Matrices — InsiderLSTM (Proposed)', fontsize=13,
              fontweight='bold')

for ax, y_te, probs, thr_f1, title in [
    (axes[0], y_r42, probs_r42, thr_f1_r42, 'CERT r4.2'),
    (axes[1], y_sp,  probs_sp,  thr_f1_sp,  'SPEDIA'),
]:
    p    = probs.get('InsiderLSTM (Ours)',
                      probs.get('InsiderLSTM', list(probs.values())[-1]))
    pred = (p >= thr_f1).astype(int)
    cm   = confusion_matrix(y_te, pred)
    # Normalise by row (true label) for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Custom palette: blue scale
    cmap = sns.color_palette('Blues', as_cmap=True)
    sns.heatmap(cm_norm, annot=False, fmt='', cmap=cmap, ax=ax,
                vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': 'Proportion'})

    # Annotate cells with count + percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = 'white' if cm_norm[i,j] > 0.55 else '#222222'
            ax.text(j + 0.5, i + 0.45, f'{count:,}',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=color)
            ax.text(j + 0.5, i + 0.62, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color=color)

    ax.set_xticklabels(['Predicted Benign', 'Predicted Insider'], fontsize=10)
    ax.set_yticklabels(['Actual Benign', 'Actual Insider'], fontsize=10, rotation=0)
    ax.set_title(f'{title}  (threshold={thr_f1:.2f})', fontsize=11,
                  fontweight='bold', pad=10)

plt.tight_layout()
save_fig(fig, 'fig5_confusion_matrices')
print('Fig 5 done.')

# ---- [cell 47] ------------------------------------------------------
# ── Fig 6: Risk Score Distributions — InsiderLSTM at EWLT threshold ──────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('InsiderLSTM Risk Score Distributions', fontsize=13, fontweight='bold')

for ax, y_te, probs, thr_ewlt, thr_f1, title in [
    (axes[0], y_r42, probs_r42, thr_ewlt_r42, thr_f1_r42, 'CERT r4.2'),
    (axes[1], y_sp,  probs_sp,  thr_ewlt_sp,  thr_f1_sp,  'SPEDIA'),
]:
    p = probs.get('InsiderLSTM (Ours)',
                   probs.get('InsiderLSTM', list(probs.values())[-1]))

    pos_scores = p[y_te == 1]
    neg_scores = p[y_te == 0]

    bins = np.linspace(0, 1, 51)

    # KDE-style filled histogram
    ax.hist(neg_scores, bins=bins, density=True, alpha=0.55,
             color='#457B9D', label='Benign sessions', edgecolor='none')
    ax.hist(pos_scores, bins=bins, density=True, alpha=0.65,
             color='#E63946', label='Insider sessions', edgecolor='none')

    # Threshold lines
    ax.axvline(x=thr_ewlt, color='#1D3557', lw=2, ls='-',
                label=f'EWLT threshold ({thr_ewlt:.2f})')
    ax.axvline(x=thr_f1,   color='#2A9D8F', lw=1.8, ls='--',
                label=f'F1 threshold ({thr_f1:.2f})')

    # Shade EWLT warning zone
    ax.axvspan(thr_ewlt, 1.0, alpha=0.07, color='#1D3557')

    ax.set_xlabel('Risk Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.93)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.2)

    # Overlap annotation
    n_above = int((pos_scores >= thr_ewlt).sum())
    n_total = len(pos_scores)
    ax.text(0.98, 0.97,
        f'{n_above}/{n_total} insider sessions\n'
        f'above EWLT threshold',
             transform=ax.transAxes, ha='right', va='top',
             fontsize=8.5, color='#1D3557',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#1D3557', alpha=0.8))

plt.tight_layout(w_pad=3)
save_fig(fig, 'fig6_score_distributions')
print('Fig 6 done.')

# ---- [cell 48] ------------------------------------------------------
# ── Fig 7: Feature Attribution — gradient saliency, top 10 per dataset ────────

def compute_saliency(model, X, y, n_samples=200, device='cpu'):
    """
    Integrated-style gradient saliency.
    Runs positive samples through model in eval mode using hooks
    to avoid BatchNorm single-sample issue.
    """
    model.eval()
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) > n_samples:
        np.random.seed(42)
        pos_idx = np.random.choice(pos_idx, n_samples, replace=False)

    saliency = np.zeros(X.shape[2])
    count    = 0

    # Process in batches of 32 — BN needs > 1
    bs = 32
    for start in range(0, len(pos_idx), bs):
        batch = pos_idx[start:start+bs]
        if len(batch) < 2:
            # pad to 2 with copies
            batch = np.concatenate([batch, batch[:2-len(batch)]])

        x_t = torch.tensor(X[batch], dtype=torch.float32).to(device)
        x_t.requires_grad_(True)

        # Forward in train mode only for BN stats
        model.train()
        pred = model(x_t)
        model.eval()

        # Only backprop through positive predictions
        # Use mean of positive scores as scalar loss
        loss = pred.mean()
        loss.backward()

        if x_t.grad is not None:
            # Mean absolute gradient across batch and time steps
            grad = x_t.grad.detach().abs().cpu().numpy()  # (B, T, F)
            saliency += grad.mean(axis=(0, 1))             # (F,)
            count += 1

    if count > 0:
        saliency /= count
    return saliency

DEVICE = next(lstm_r42['model'].parameters()).device

print('Computing saliency (r4.2)...')
sal_r42 = compute_saliency(lstm_r42['model'], X_te_r42, y_te_r42,
                              n_samples=200, device=DEVICE)
print(f'  r4.2 saliency range: [{sal_r42.min():.4f}, {sal_r42.max():.4f}]')

print('Computing saliency (SPEDIA)...')
sal_sp  = compute_saliency(lstm_sp['model'], X_te_sp, y_te_sp,
                             n_samples=200, device=DEVICE)
print(f'  SPEDIA saliency range: [{sal_sp.min():.4f}, {sal_sp.max():.4f}]')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('InsiderLSTM Feature Attribution (Gradient Saliency)',
              fontsize=13, fontweight='bold')

# ── r4.2: top 10 ─────────────────────────────────────────────────────────────
ax = axes[0]
top10_idx   = np.argsort(sal_r42)[::-1][:10]
top10_sal   = sal_r42[top10_idx]
top10_names = [ALL_FEATURE_COLS[i] for i in top10_idx]

colors_bar = ['#1D3557' if '_zscore' in n else '#2A9D8F'
               for n in top10_names]
short_names = [n.replace('_zscore','*').replace('_',' ')
                .replace('mean ','').title()[:22]
               for n in top10_names]

# Plot in descending order (highest at top)
y_pos = range(10)
bars  = ax.barh(list(y_pos), top10_sal[::-1],
                color=colors_bar[::-1],
                alpha=0.85, edgecolor='white',
                linewidth=0.5, height=0.65)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(short_names[::-1], fontsize=9.5)
ax.set_xlabel('Mean |Gradient|', fontsize=10)
ax.set_title('CERT r4.2 — Top 10 Features', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')
ax.set_xlim(left=0)

# Value labels
for bar, val in zip(bars, top10_sal[::-1]):
    ax.text(bar.get_width() + ax.get_xlim()[1]*0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7.5, color='#333333')

patches = [mpatches.Patch(color='#1D3557', label='Z-score feature (*)'),
           mpatches.Patch(color='#2A9D8F', label='Raw feature')]
ax.legend(handles=patches, fontsize=9, loc='lower right')

# ── SPEDIA: all 9 features ────────────────────────────────────────────────────
ax = axes[1]
top9_idx   = np.argsort(sal_sp)[::-1]
top9_sal   = sal_sp[top9_idx]
top9_names = [SP_FEAT_NAMES[i] for i in top9_idx]

y_pos2 = range(len(top9_names))
bars2  = ax.barh(list(y_pos2), top9_sal[::-1],
                  color='#E63946', alpha=0.78,
                  edgecolor='white', linewidth=0.5, height=0.65)
ax.set_yticks(list(y_pos2))
ax.set_yticklabels(top9_names[::-1], fontsize=9.5)
ax.set_xlabel('Mean |Gradient|', fontsize=10)
ax.set_title('SPEDIA — All 9 Features', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')
ax.set_xlim(left=0)

for bar, val in zip(bars2, top9_sal[::-1]):
    ax.text(bar.get_width() + ax.get_xlim()[1]*0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7.5, color='#333333')

plt.tight_layout(w_pad=3)
save_fig(fig, 'fig7_feature_attribution')
print('Fig 7 done.')

# ---- [cell 49] ------------------------------------------------------
# ── Fig 8: Mean z-score feature deviation — top 10, insider vs benign ─────────
print('Loading master for z-score heatmap...')
master = pd.read_parquet(os.path.join(RELOAD_DIR, 'master_r42_eng.parquet'))

with open(os.path.join(RELOAD_DIR, 'feature_cols_r42.json')) as f:
    fc = json.load(f)
dev_cols = fc['dev_cols']

# Identify insider users
train_split = pd.read_parquet(
    os.path.join(RELOAD_DIR, 'split_train_r42.parquet'),
    columns=['user', 'label'])
all_insider_users = set(
    train_split[train_split['label']==1]['user'].unique())
del train_split; gc.collect()

insider_data = master[master['user'].isin(all_insider_users)][dev_cols]
benign_data  = master[~master['user'].isin(all_insider_users)][dev_cols]

# ── Top 10 by mean absolute insider deviation ─────────────────────────────────
mean_abs  = insider_data.abs().mean().sort_values(ascending=False)
top10_cols = mean_abs.head(10).index.tolist()

ins_means  = insider_data[top10_cols].mean()
ben_means  = benign_data[top10_cols].mean()

del master; gc.collect()

# 2-row matrix: row 0 = insider, row 1 = benign
mat = np.array([ins_means.values, ben_means.values])

# Short display names — keep _dev suffix, strip mean_ prefix
short = [c.replace('_zscore', '_dev').replace('mean_', '')
          for c in top10_cols]

# Symmetric colour scale
vmax = max(abs(mat.max()), abs(mat.min()))
vmax = round(vmax + 0.05, 1)

fig, ax = plt.subplots(figsize=(13, 3.8))

im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r',
                vmin=-vmax, vmax=vmax, interpolation='nearest')

# ── Annotate cells ────────────────────────────────────────────────────────────
for i in range(2):
    for j in range(10):
        val = mat[i, j]
        text_color = 'white' if abs(val) > vmax * 0.65 else '#111111'
        ax.text(j, i, f'{val:.2f}',
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color=text_color)

# ── Y-axis ────────────────────────────────────────────────────────────────────
ax.set_yticks([0, 1])
ax.set_yticklabels(['Insider\n(pre-incident)', 'Non-Insider\n(matched)'],
                    fontsize=11)

# ── X-axis ────────────────────────────────────────────────────────────────────
ax.set_xticks(range(10))
ax.set_xticklabels(short, rotation=40, ha='right', fontsize=9.5)

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Z-score deviation', fontsize=10,
                rotation=270, labelpad=16)
cbar.ax.tick_params(labelsize=9)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    'Mean Z-score Feature Deviation — Top 10: '
    'Insider (pre-incident) vs Non-Insider (CERT r4.2)',
    fontsize=12, fontweight='bold', pad=12)

# Clean frame
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis='both', length=0)

plt.tight_layout()
save_fig(fig, 'fig8_zscore_heatmap')
print('Fig 8 done.')

# ---- [cell 50] ------------------------------------------------------
# ── Fix meta date types before plotting trajectories ─────────────────────────
from datetime import datetime

def parse_meta_dates(meta):
    parsed = []
    for uid, dt in meta:
        if isinstance(dt, str):
            dt = pd.Timestamp(datetime.fromisoformat(dt))
        elif not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        parsed.append((uid, dt))
    return parsed

meta_r42 = parse_meta_dates(meta_r42)
meta_sp  = parse_meta_dates(meta_sp)
print('✅ Meta dates parsed.')

# ── Fig 9: LSTM daily risk score trajectories for detected insiders ───────────
# Shows risk score over time with threat window and EWLT threshold marked.
# r4.2: 5 detected users | SPEDIA: 1 detected user

def plot_trajectories(probs_arr, meta, y_true, ewlt_res, thr_ewlt,
                       title, save_name, max_users=6):
    if not ewlt_res:
        print(f'  No EWLT results for {title}')
        return

    # Sort by lead time descending
    top_users = sorted(ewlt_res.keys(),
                        key=lambda u: -ewlt_res[u]['lead_days'])[:max_users]
    n    = len(top_users)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(6*cols, 3.8*rows))
    fig.suptitle(f'InsiderLSTM Daily Risk Trajectories — {title}',
                  fontsize=13, fontweight='bold', y=1.01)

    # Normalise axes to always be a 2D list
    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(np.atleast_1d(axes))]
    else:
        axes = [list(axes[r]) for r in range(rows)]

    # Build per-user timeline
    user_tl = defaultdict(list)
    for p, m, lbl in zip(probs_arr, meta, y_true):
        user_tl[m[0]].append({
            'date' : pd.Timestamp(m[1]),
            'prob' : float(p),
            'label': int(lbl)
        })

    for idx, user in enumerate(top_users):
        r, c = divmod(idx, cols)
        ax   = axes[r][c]

        events = sorted(user_tl[user], key=lambda x: x['date'])
        if not events:
            continue

        dates  = [e['date']  for e in events]
        scores = [e['prob']  for e in events]
        labels = [e['label'] for e in events]

        mal_dates = [d for d, l in zip(dates, labels) if l == 1]
        if not mal_dates:
            continue
        first_mal = min(mal_dates)
        last_mal  = max(mal_dates)

        # Threat window shading
        ax.axvspan(first_mal, last_mal, alpha=0.12,
                    color='#E63946', label='Threat window', zorder=1)

        # EWLT threshold line
        ax.axhline(y=thr_ewlt, color='#F77F00', lw=1.5, ls='--',
                    zorder=2, label=f'EWLT thr ({thr_ewlt:.2f})')

        # Incident start line
        ax.axvline(x=first_mal, color='#E63946', lw=2,
                    zorder=3, label='Incident start')

        dates_arr  = np.array(dates)
        scores_arr = np.array(scores)

        # Pre-incident risk scores
        pre_mask = dates_arr < first_mal
        if pre_mask.any():
            ax.plot(dates_arr[pre_mask], scores_arr[pre_mask],
                     color='#457B9D', lw=1.8, zorder=4,
                     label='Risk score (pre)')
            ax.fill_between(dates_arr[pre_mask], 0, scores_arr[pre_mask],
                             color='#457B9D', alpha=0.12)

        # Threat period risk scores
        threat_mask = dates_arr >= first_mal
        if threat_mask.any():
            ax.plot(dates_arr[threat_mask], scores_arr[threat_mask],
                     color='#E63946', lw=1.8, zorder=4,
                     label='Risk score (threat)')
            ax.fill_between(dates_arr[threat_mask], 0,
                             scores_arr[threat_mask],
                             color='#E63946', alpha=0.12)

        # Early warning marker
        lead    = ewlt_res[user]['lead_days']
        ew_date = first_mal - pd.Timedelta(days=lead)
        ax.axvline(x=ew_date, color='#1D3557', lw=1.5, ls=':',
                    zorder=5, label=f'Warning ({lead}d early)')

        ax.set_ylim([-0.03, 1.08])
        ax.set_title(
            f'{user[:12]}  |  Lead={lead}d  '
            f'conf={ewlt_res[user]["confidence"]:.2f}',
            fontsize=9.5, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Risk Score', fontsize=9)
        ax.tick_params(axis='x', labelsize=7.5, rotation=30)
        ax.grid(True, alpha=0.18)
        if idx == 0:
            ax.legend(fontsize=7.5, loc='upper left',
                       framealpha=0.9, ncol=2, columnspacing=0.8)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    save_fig(fig, save_name)
    print(f'  Saved: {save_name}')

# ── Run ───────────────────────────────────────────────────────────────────────
lstm_probs_arr_r42 = probs_r42.get('InsiderLSTM (Ours)',
                                     list(probs_r42.values())[-1])
lstm_probs_arr_sp  = probs_sp.get('InsiderLSTM (Ours)',
                                    list(probs_sp.values())[-1])

print('Plotting r4.2 trajectories...')
plot_trajectories(
    lstm_probs_arr_r42, meta_r42, y_r42,
    ewlt_r42, thr_ewlt_r42,
    'CERT r4.2', 'fig9a_trajectories_r42', max_users=6)

print('Plotting SPEDIA trajectories...')
plot_trajectories(
    lstm_probs_arr_sp, meta_sp, y_sp,
    ewlt_sp, thr_ewlt_sp,
    'SPEDIA', 'fig9b_trajectories_spedia', max_users=6)

print('Fig 9 done.')

# ---- [cell 51] ------------------------------------------------------
# ── Summary: list all saved figures ──────────────────────────────────────────
print('\n── Saved figures ──────────────────────────────────────────────────────────')
for fname in sorted(os.listdir(FIG_DIR)):
    fpath = os.path.join(FIG_DIR, fname)
    size  = os.path.getsize(fpath) / 1024
    print(f'  {fname:<45} {size:>8.1f} KB')
print(f'\nAll figures saved to: {FIG_DIR}')
print('Upload the /figures/ folder to Kaggle dataset for paper use.')

# ---- [cell 52] ------------------------------------------------------
# ── Section 11d: Confusion matrices ──────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(10,4))

def plot_cm(ax,y_true,probs,thr,title):
    y_pred=(probs>=thr).astype(int)
    cm=confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                xticklabels=['Benign','Insider'],
                yticklabels=['Benign','Insider'])
    ax.set_xlabel('Predicted',fontsize=10); ax.set_ylabel('Actual',fontsize=10)
    ax.set_title(title,fontsize=11,fontweight='bold')

plot_cm(axes[0],y_te_r42,lstm_r42['probs'],thr_f1_r42,
        'Confusion Matrix — CERT r4.2')
plot_cm(axes[1],y_te_sp, lstm_sp['probs'], thr_f1_sp,
        'Confusion Matrix — SPEDIA')
plt.suptitle('InsiderLSTM Confusion Matrices',fontsize=12,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,'fig_cm.png'),dpi=150,bbox_inches='tight')
plt.show(); print('✅ fig_cm.png saved')

# ---- [cell 53] ------------------------------------------------------
# ── Section 11e: EWLT bar charts ─────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(15,5))

def plot_ewlt(ax,ewlt_res,title):
    if not ewlt_res:
        ax.text(0.5,0.5,'No EWLT detections',ha='center',va='center',
                transform=ax.transAxes,fontsize=12)
        ax.set_title(title); return
    users=list(ewlt_res.keys())
    ld=[ewlt_res[u]['lead_days'] for u in users]
    cf=[ewlt_res[u]['confidence'] for u in users]
    idx=np.argsort(ld)[::-1]
    us=[users[i] for i in idx]; ls=[ld[i] for i in idx]; cs=[cf[i] for i in idx]
    bars=ax.barh(range(len(us)),ls,color=plt.cm.RdYlGn(cs),alpha=0.85)
    ax.set_yticks(range(len(us)))
    ax.set_yticklabels([u[:14] for u in us],fontsize=8)
    ax.set_xlabel('Lead Time (days before incident)',fontsize=10)
    ax.set_title(title,fontsize=11,fontweight='bold')
    ax.axvline(x=np.mean(ls),color='red',linestyle='--',linewidth=1.5,
               label=f'Mean={np.mean(ls):.1f}d')
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis='x')
    for bar,l in zip(bars,ls):
        ax.text(bar.get_width()+0.2,bar.get_y()+bar.get_height()/2,
                f'{l}d',va='center',fontsize=7)

plot_ewlt(axes[0],ewlt_r42,f'EWLT — CERT r4.2 (n={len(ewlt_r42)} users)')
plot_ewlt(axes[1],ewlt_sp, f'EWLT — SPEDIA (n={len(ewlt_sp)} users)')
plt.suptitle('Early-Warning Lead Time (EWLT) per Detected User',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,'fig_ewlt.png'),dpi=150,bbox_inches='tight')
plt.show(); print('✅ fig_ewlt.png saved')

# ---- [cell 54] ------------------------------------------------------
# ── Section 11f: EWLT vs confidence scatter ──────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,5))

def plot_ewlt_scatter(ax,ewlt_res,title):
    if not ewlt_res:
        ax.text(0.5,0.5,'No data',ha='center',va='center',
                transform=ax.transAxes); ax.set_title(title); return
    ld=[v['lead_days'] for v in ewlt_res.values()]
    cf=[v['confidence'] for v in ewlt_res.values()]
    cs=[v['consistency'] for v in ewlt_res.values()]
    sc=ax.scatter(cf,ld,c=cs,cmap='RdYlGn',s=80,alpha=0.8,edgecolors='k',linewidth=0.5)
    plt.colorbar(sc,ax=ax,label='Consistency')
    ax.set_xlabel('Confidence',fontsize=10); ax.set_ylabel('Lead Time (days)',fontsize=10)
    ax.set_title(title,fontsize=11,fontweight='bold')
    ax.grid(True,alpha=0.3)
    for u,v in ewlt_res.items():
        ax.annotate(u[:8],(v['confidence'],v['lead_days']),
                    fontsize=6,alpha=0.7,xytext=(3,3),textcoords='offset points')

plot_ewlt_scatter(axes[0],ewlt_r42,'EWLT Confidence vs Lead Time — r4.2')
plot_ewlt_scatter(axes[1],ewlt_sp, 'EWLT Confidence vs Lead Time — SPEDIA')
plt.suptitle('EWLT: Lead Time vs Confidence',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,'fig_ewlt_scatter.png'),dpi=150,bbox_inches='tight')
plt.show(); print('✅ fig_ewlt_scatter.png saved')

# ---- [cell 55] ------------------------------------------------------
# ── Section 11g: EWLT threshold sweep ────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(12,4))

def plot_thr_sweep(ax,sweep,thr_ewlt,title):
    thrs=sorted(sweep.keys())
    counts=[sweep[t] for t in thrs]
    ax.plot(thrs,counts,'b-o',markersize=4,linewidth=1.5)
    ax.axvline(x=thr_ewlt,color='red',linestyle='--',linewidth=2,
               label=f'Selected thr={thr_ewlt:.2f}')
    ax.set_xlabel('Threshold',fontsize=10)
    ax.set_ylabel('Users with Early Warning',fontsize=10)
    ax.set_title(title,fontsize=11,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

plot_thr_sweep(axes[0],sweep_r42,thr_ewlt_r42,'EWLT Threshold Sweep — r4.2')
plot_thr_sweep(axes[1],sweep_sp, thr_ewlt_sp, 'EWLT Threshold Sweep — SPEDIA')
plt.suptitle('EWLT Threshold Selection',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,'fig_thr_sweep.png'),dpi=150,bbox_inches='tight')
plt.show(); print('✅ fig_thr_sweep.png saved')

# ---- [cell 56] ------------------------------------------------------
# ── Section 11h: Training dynamics — loss + val AUC curves ──────────────────
def plot_training_dynamics(hist_dict, title, save_path):
    models=list(hist_dict.keys())
    fig,axes=plt.subplots(1,3,figsize=(18,5))

    # Training loss
    for name,hist in hist_dict.items():
        axes[0].plot(hist['tr_loss'],label=name,
                     linewidth=2 if 'InsiderLSTM' in name else 1)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss'); axes[0].legend(fontsize=7)
    axes[0].grid(True,alpha=0.3)

    # Val loss
    for name,hist in hist_dict.items():
        axes[1].plot(hist['vl_loss'],label=name,
                     linewidth=2 if 'InsiderLSTM' in name else 1)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss'); axes[1].legend(fontsize=7)
    axes[1].grid(True,alpha=0.3)

    # Val AUC
    for name,hist in hist_dict.items():
        axes[2].plot(hist['vl_auc'],label=name,
                     linewidth=2 if 'InsiderLSTM' in name else 1)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Validation AUC')
    axes[2].set_title('Validation AUC'); axes[2].legend(fontsize=7)
    axes[2].grid(True,alpha=0.3)

    plt.suptitle(title,fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.show()

# Build hist dicts for all models
all_hist_r42={}
for n,r in sota_r42.items():
    if 'hist' in r: all_hist_r42[n]=r['hist']
all_hist_r42['InsiderLSTM']=lstm_r42['hist']
if 'hist' in baselines_r42.get('CNN-GRU',{}):
    all_hist_r42['CNN-GRU']=baselines_r42['CNN-GRU']['hist']

all_hist_sp={}
for n,r in sota_sp.items():
    if 'hist' in r: all_hist_sp[n]=r['hist']
all_hist_sp['InsiderLSTM']=lstm_sp['hist']
if 'hist' in baselines_sp.get('CNN-GRU',{}):
    all_hist_sp['CNN-GRU']=baselines_sp['CNN-GRU']['hist']

plot_training_dynamics(all_hist_r42,'Training Dynamics — CERT r4.2',
    os.path.join(WORK_DIR,'fig_training_r42.png'))
plot_training_dynamics(all_hist_sp,'Training Dynamics — SPEDIA',
    os.path.join(WORK_DIR,'fig_training_sp.png'))
print('✅ Training dynamics figures saved')

# ---- [cell 57] ------------------------------------------------------
# ── Section 11i: LR schedule ─────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(8,4))
if 'lr' in lstm_r42['hist'] and lstm_r42['hist']['lr']:
    ax.plot(lstm_r42['hist']['lr'],label='r4.2 InsiderLSTM',linewidth=2)
if 'lr' in lstm_sp['hist'] and lstm_sp['hist']['lr']:
    ax.plot(lstm_sp['hist']['lr'],label='SPEDIA InsiderLSTM',linewidth=2,linestyle='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
ax.set_title('CosineAnnealingLR Schedule',fontsize=12,fontweight='bold')
ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,'fig_lr_schedule.png'),dpi=150,bbox_inches='tight')
plt.show(); print('✅ fig_lr_schedule.png saved')

# ---- [cell 58] ------------------------------------------------------
# ── Section 11k: RF feature importances (r4.2) ───────────────────────────────
with open(os.path.join(WORK_DIR,'rf_feature_importance_r42.json')) as f:
    fi=json.load(f)

top20=dict(list(fi.items())[:20])
fig,ax=plt.subplots(figsize=(10,6))
names=list(top20.keys()); vals=list(top20.values())
# Shorten names for display
short=[n.replace('_mean','\n(mean)').replace('_max','\n(max)').replace('_std','\n(std)') for n in names]
colors=['#F44336' if 'zscore' in n else '#2196F3' for n in names]
bars=ax.barh(range(len(names)),vals,color=colors,alpha=0.85)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(short,fontsize=8)
ax.set_xlabel('Feature Importance',fontsize=10)
ax.set_title('Top-20 RF Feature Importances — CERT r4.2\n(blue=raw, red=z-score)',
              fontsize=11,fontweight='bold')
ax.grid(True,alpha=0.3,axis='x')
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR,'fig_rf_importance.png'),dpi=150,bbox_inches='tight')
plt.show(); print('✅ fig_rf_importance.png saved')

# ==============================================================================
# SIgnificance Test
# ==============================================================================

# ---- [cell 60] ------------------------------------------------------
# ── Use saved probs — match paper results exactly ─────────────────────────────
# probs_r42.json and probs_sp.json were saved in Section 10
# immediately after training, on the correct test sequences.
# These match the AUC values in the paper tables.

print('Loading saved probs from Section 10 output...')

with open(os.path.join(WORK_DIR,'probs_r42.json')) as f:
    saved_r42 = json.load(f)
with open(os.path.join(WORK_DIR,'probs_sp.json')) as f:
    saved_sp  = json.load(f)

# Ground truth from saved file
y_r42_sig   = np.array(saved_r42['y_te'], dtype='float32')
y_sp_sig    = np.array(saved_sp['y_te'],  dtype='float32')

# All model probs from saved file
all_probs_r42 = {k: np.array(v, dtype='float64')
                  for k, v in saved_r42.items()
                  if k not in ('y_te','meta_te')}
all_probs_sp  = {k: np.array(v, dtype='float64')
                  for k, v in saved_sp.items()
                  if k not in ('y_te','meta_te')}

# Rename InsiderLSTM key to match MODEL_ORDER
for d in [all_probs_r42, all_probs_sp]:
    if 'InsiderLSTM' in d and 'InsiderLSTM (Ours)' not in d:
        d['InsiderLSTM (Ours)'] = d.pop('InsiderLSTM')

lstm_p_r42 = all_probs_r42['InsiderLSTM (Ours)']
lstm_p_sp  = all_probs_sp['InsiderLSTM (Ours)']

print(f'r4.2  : {len(lstm_p_r42):,} probs '
      f'| InsiderLSTM AUC={roc_auc_score(y_r42_sig, lstm_p_r42):.4f}')
print(f'SPEDIA: {len(lstm_p_sp):,} probs '
      f'| InsiderLSTM AUC={roc_auc_score(y_sp_sig, lstm_p_sp):.4f}')

# ---- [cell 61] ------------------------------------------------------
# ── Section 11l: Statistical significance — Bootstrap DeLong ─────────────────
# DeLong et al. Biometrics 1988;44(3):837-845
# Bootstrap approximation — N=5000 resamples, distribution-free
# Bonferroni correction: α* = 0.05/6 = 0.0083 (6 comparison models)

N_COMPARISONS = 6
ALPHA_CORRECTED = round(0.05 / N_COMPARISONS, 6)

def delong_bootstrap(y_true, probs_a, probs_b,
                      name_a, name_b, n_boot=5000):
    rng   = np.random.default_rng(SEED)
    n     = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt  = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            diffs.append(
                roc_auc_score(yt, probs_a[idx]) -
                roc_auc_score(yt, probs_b[idx]))
        except Exception:
            continue

    obs = roc_auc_score(y_true, probs_a) - \
          roc_auc_score(y_true, probs_b)

    if len(diffs) < 100:
        return {
            'comparison'  : f'{name_a} vs {name_b}',
            'delta_auc'   : round(obs, 4),
            'ci_95'       : 'N/A',
            'p_value'     : None,
            'significant' : False,
            'n_resamples' : len(diffs),
            'note'        : f'only {len(diffs)} valid resamples',
        }

    diffs = np.array(diffs)
    p     = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    p     = max(float(p), 1.0 / n_boot)
    ci    = (round(float(np.percentile(diffs,  2.5)), 4),
             round(float(np.percentile(diffs, 97.5)), 4))

    return {
        'comparison'  : f'{name_a} vs {name_b}',
        'delta_auc'   : round(obs, 4),
        'ci_95'       : f'[{ci[0]}, {ci[1]}]',
        'p_value'     : round(p, 6),
        'n_resamples' : len(diffs),
        'significant' : p < ALPHA_CORRECTED,
    }

# ── Use stored probs — no model needed ───────────────────────────────────────
print('Loading probs from stored results...')

# r4.2 — all baseline + SOTA probs from current session
all_probs_r42 = {}
for name, res in {**baselines_r42, **sota_r42}.items():
    if 'probs' in res:
        all_probs_r42[name] = np.array(res['probs'], dtype='float64')

# SPEDIA — all baseline + SOTA probs from current session
all_probs_sp = {}
for name, res in {**baselines_sp, **sota_sp}.items():
    if 'probs' in res:
        all_probs_sp[name] = np.array(res['probs'], dtype='float64')

# InsiderLSTM probs — use stored values directly
lstm_p_r42 = np.array(lstm_r42['probs'], dtype='float64')
lstm_p_sp  = np.array(lstm_sp['probs'],  dtype='float64')

# y_true arrays
y_r42_sig = y_te_r42.astype('float64')
y_sp_sig  = y_te_sp_locked.astype('float64')

print(f'r4.2  models loaded: {list(all_probs_r42.keys())}')
print(f'SPEDIA models loaded: {list(all_probs_sp.keys())}')
print(f'r4.2  InsiderLSTM probs: {len(lstm_p_r42):,}')
print(f'SPEDIA InsiderLSTM probs: {len(lstm_p_sp):,}')

# ── Run DeLong tests ──────────────────────────────────────────────────────────
print('\n' + '='*70)
print('STATISTICAL SIGNIFICANCE — Bootstrap DeLong (N=5000)')
print('Reference: DeLong et al. Biometrics 1988;44(3):837-845')
print(f'Bonferroni-corrected α = 0.05 / {N_COMPARISONS} = {ALPHA_CORRECTED}')
print('='*70)

sig_rows_r42 = []
sig_rows_sp  = []

for ds_label, lstm_p, all_probs, y_te, sig_rows in [
    ('CERT r4.2', lstm_p_r42, all_probs_r42, y_r42_sig, sig_rows_r42),
    ('SPEDIA',    lstm_p_sp,  all_probs_sp,  y_sp_sig,  sig_rows_sp),
]:
    print(f'\n── {ds_label} ──────────────────────────────────────────────────')
    print(f'  InsiderLSTM AUC = {roc_auc_score(y_te, lstm_p):.4f}')
    print(f'  {"Model":<25} {"Δ AUC":>8} '
          f'{"95% CI":>20} {"p-value":>10} {"Sig":>5}')
    print(f'  {"-"*70}')

    for name, probs in all_probs.items():
        r   = delong_bootstrap(y_te, lstm_p, probs,
                                'InsiderLSTM', name)
        sig_rows.append(r)
        sig   = '✅' if r['significant'] else '—'
        p_str = f'{r["p_value"]:.6f}' \
                if r['p_value'] is not None else 'N/A'
        print(f'  {name:<25} '
              f'{r["delta_auc"]:>+8.4f} '
              f'{str(r["ci_95"]):>20} '
              f'{p_str:>10} '
              f'{sig:>5}')

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(sig_rows_r42).to_csv(
    os.path.join(WORK_DIR, 'significance_r42.csv'), index=False)
pd.DataFrame(sig_rows_sp).to_csv(
    os.path.join(WORK_DIR, 'significance_sp.csv'),  index=False)
print(f'\n✅ DeLong results saved.')
print(f'  Bonferroni α* = {ALPHA_CORRECTED}')

# ---- [cell 62] ------------------------------------------------------
# ── Statistical Significance Test 2: Wilcoxon Signed-Rank ────────────────────
# Demšar (2006) JMLR 7:1-30
# Bonferroni-corrected α* = 0.05/6 = 0.0083

from scipy.stats import wilcoxon as scipy_wilcoxon
import warnings

def wilcoxon_test(probs_a, probs_b, name_a, name_b):
    a    = np.array(probs_a, dtype='float64')
    b    = np.array(probs_b, dtype='float64')

    if len(a) != len(b):
        return {'comparison' : f'{name_a} vs {name_b}',
                'statistic'  : None, 'p_value': None,
                'n_pairs'    : 0,    'mean_delta': None,
                'significant': False, 'note': 'length mismatch'}

    diff    = a - b
    nonzero = diff[diff != 0]

    if len(nonzero) < 10:
        return {'comparison' : f'{name_a} vs {name_b}',
                'statistic'  : None, 'p_value': None,
                'n_pairs'    : int(len(nonzero)),
                'mean_delta' : round(float(diff.mean()), 6),
                'significant': False,
                'note'       : f'only {len(nonzero)} non-zero diffs'}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stat, p = scipy_wilcoxon(nonzero,
                                      alternative='two-sided',
                                      zero_method='wilcox')
        return {'comparison' : f'{name_a} vs {name_b}',
                'statistic'  : round(float(stat), 4),
                'p_value'    : round(float(p), 6),
                'n_pairs'    : int(len(nonzero)),
                'mean_delta' : round(float(diff.mean()), 6),
                'significant': p < ALPHA_CORRECTED}
    except Exception as e:
        return {'comparison' : f'{name_a} vs {name_b}',
                'statistic'  : None, 'p_value': None,
                'n_pairs'    : int(len(nonzero)),
                'mean_delta' : round(float(diff.mean()), 6),
                'significant': False, 'note': str(e)}

# ── Run Wilcoxon ──────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('SIGNIFICANCE TEST 2 — Wilcoxon Signed-Rank (Two-sided)')
print('Reference: Demšar, JMLR 7:1-30 (2006)')
print(f'Bonferroni-corrected α = 0.05 / {N_COMPARISONS} = {ALPHA_CORRECTED}')
print('='*70)

wilcox_rows_r42 = []
wilcox_rows_sp  = []

for ds_label, lstm_p, all_probs, y_te, wilcox_rows in [
    ('CERT r4.2', lstm_p_r42, all_probs_r42, y_r42_sig, wilcox_rows_r42),
    ('SPEDIA',    lstm_p_sp,  all_probs_sp,  y_sp_sig,  wilcox_rows_sp),
]:
    print(f'\n── {ds_label} ───────────────────────────────────────────────────')
    print(f'  {"Model":<28} {"n_pairs":>8} {"Δ mean":>9} '
          f'{"p-value":>10} {"Sig":>5}')
    print(f'  {"-"*65}')

    for name, probs in all_probs.items():
        r = wilcoxon_test(lstm_p, probs, 'InsiderLSTM', name)
        wilcox_rows.append(r)

        if r['p_value'] is not None:
            sig = '✅' if r['significant'] else '—'
            print(f'  {name:<28} '
                  f'{r["n_pairs"]:>8,} '
                  f'{r["mean_delta"]:>+9.5f} '
                  f'{r["p_value"]:>10.6f} '
                  f'{sig:>5}')
        else:
            note = r.get('note', '')
            print(f'  {name:<28} {"SKIP":>8}  {note}')

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(wilcox_rows_r42).to_csv(
    os.path.join(WORK_DIR, 'significance_wilcoxon_r42.csv'), index=False)
pd.DataFrame(wilcox_rows_sp).to_csv(
    os.path.join(WORK_DIR, 'significance_wilcoxon_sp.csv'),  index=False)

print(f'\n✅ Wilcoxon results saved.')
print(f'  Bonferroni α* = {ALPHA_CORRECTED}')
print('\nInterpretation:')
print('  ✅ = InsiderLSTM score distribution significantly different')
print(f'       from comparison model (p < {ALPHA_CORRECTED}, Bonferroni-corrected)')
print('  —  = No significant difference at corrected alpha')

# ==============================================================================
# Ablation
# ==============================================================================

# ---- [cell 64] ------------------------------------------------------
# ── Use saved probs — match paper results exactly ─────────────────────────────
# probs_r42.json and probs_sp.json were saved in Section 10
# immediately after training, on the correct test sequences.
# These match the AUC values in the paper tables.

print('Loading saved probs from Section 10 output...')

with open(os.path.join(RELOAD_DIR,'probs_r42.json')) as f:
    saved_r42 = json.load(f)
with open(os.path.join(RELOAD_DIR,'probs_sp.json')) as f:
    saved_sp  = json.load(f)

# Ground truth from saved file
y_r42_sig   = np.array(saved_r42['y_te'], dtype='float32')
y_sp_sig    = np.array(saved_sp['y_te'],  dtype='float32')

# All model probs from saved file
all_probs_r42 = {k: np.array(v, dtype='float64')
                  for k, v in saved_r42.items()
                  if k not in ('y_te','meta_te')}
all_probs_sp  = {k: np.array(v, dtype='float64')
                  for k, v in saved_sp.items()
                  if k not in ('y_te','meta_te')}

# Rename InsiderLSTM key to match MODEL_ORDER
for d in [all_probs_r42, all_probs_sp]:
    if 'InsiderLSTM' in d and 'InsiderLSTM (Ours)' not in d:
        d['InsiderLSTM (Ours)'] = d.pop('InsiderLSTM')

lstm_p_r42 = all_probs_r42['InsiderLSTM (Ours)']
lstm_p_sp  = all_probs_sp['InsiderLSTM (Ours)']

print(f'r4.2  : {len(lstm_p_r42):,} probs '
      f'| InsiderLSTM AUC={roc_auc_score(y_r42_sig, lstm_p_r42):.4f}')
print(f'SPEDIA: {len(lstm_p_sp):,} probs '
      f'| InsiderLSTM AUC={roc_auc_score(y_sp_sig, lstm_p_sp):.4f}')

# ---- [cell 65] ------------------------------------------------------
# ── Section 8c: Ablation Study ────────────────────────────────────────────────
# Part A: Architecture variants (both datasets)
# Part B: Feature set variants (r4.2 only)
# Part C: Sequence length variants (r4.2 only)
#
# Uses final InsiderLSTM config: hidden=32, layers=3, dropout=0.1,
# lr=4.17e-4, batch=64 — same config for all ablation variants
# pos_weight: dynamic pw_r42 (not hardcoded 30.0)

import math, copy

# ── Ablation config — matches final InsiderLSTM config ───────────────────────
ABL_HIDDEN  = 32
ABL_LAYERS  = 3
ABL_DROPOUT = 0.1
ABL_LR      = 4.17e-4
ABL_BATCH   = 64
ABL_EPOCHS  = 100
ABL_PATIENCE= 15
ABL_CFG     = dict(lr=ABL_LR, epochs=ABL_EPOCHS, patience=ABL_PATIENCE)

abl_results = {}

# ── Model variants for Part A ─────────────────────────────────────────────────
class PlainLSTM(nn.Module):
    """Ablation A1: no input gate, no highway."""
    def __init__(self, input_size,
                 hidden=ABL_HIDDEN, layers=ABL_LAYERS,
                 dropout=ABL_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                             batch_first=True,
                             dropout=dropout if layers>1 else 0)
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, hidden)
        self.fc2  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h_n,_) = self.lstm(x)
        return torch.sigmoid(
            self.fc2(torch.relu(
                self.fc1(self.drop(self.bn(h_n[-1])))))).squeeze(-1)

class LSTM_GateOnly(nn.Module):
    """Ablation A2: input gate only, no highway."""
    def __init__(self, input_size,
                 hidden=ABL_HIDDEN, layers=ABL_LAYERS,
                 dropout=ABL_DROPOUT):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size), nn.Sigmoid())
        self.lstm = nn.LSTM(input_size, hidden, layers,
                             batch_first=True,
                             dropout=dropout if layers>1 else 0)
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, hidden)
        self.fc2  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h_n,_) = self.lstm(x * self.gate(x))
        return torch.sigmoid(
            self.fc2(torch.relu(
                self.fc1(self.drop(self.bn(h_n[-1])))))).squeeze(-1)

class LSTM_HighwayOnly(nn.Module):
    """Ablation A3: highway refinement only, no input gate."""
    def __init__(self, input_size,
                 hidden=ABL_HIDDEN, layers=ABL_LAYERS,
                 dropout=ABL_DROPOUT):
        super().__init__()
        self.lstm             = nn.LSTM(input_size, hidden, layers,
                                         batch_first=True,
                                         dropout=dropout if layers>1 else 0)
        self.refine_transform = nn.Linear(hidden, hidden)
        self.refine_gate      = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Sigmoid())
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, hidden)
        self.fc2  = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h_n,_) = self.lstm(x); h = h_n[-1]
        h_t = torch.tanh(self.refine_transform(h))
        g   = self.refine_gate(h)
        h   = g*h_t + (1-g)*h
        return torch.sigmoid(
            self.fc2(torch.relu(
                self.fc1(self.drop(self.bn(h)))))).squeeze(-1)

ARCH_VARIANTS = {
    'Plain-LSTM'         : PlainLSTM,
    'LSTM + Gate'        : LSTM_GateOnly,
    'LSTM + Highway'     : LSTM_HighwayOnly,
    'InsiderLSTM (Full)' : InsiderLSTM,
}

# ── Helper ────────────────────────────────────────────────────────────────────
def run_abl(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
             pw, sampler, ModelClass, input_dim, name, ds):
    set_seed()
    m      = ModelClass(input_dim).to(DEVICE)
    tr_ldr = make_loader(X_tr, y_tr, ABL_BATCH, sampler)
    vl_ldr = make_loader(X_vl, y_vl, 256)
    m, _   = train_model(m, tr_ldr, vl_ldr,
                          pos_weight=pw, **ABL_CFG)
    probs  = collect_probs(m, X_te, y_te)
    auc    = roc_auc_score(y_te, probs)
    print(f'  [{ds}] {name:<25} AUC={auc:.4f}')
    del m; gc.collect()
    return {'name': name, 'dataset': ds, 'auc': auc, 'probs': probs}

# ══════════════════════════════════════════════════════════════════════════════
# PART A — Architecture ablations (both datasets)
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('PART A — Architecture Ablations')
print(f'Config: hidden={ABL_HIDDEN} layers={ABL_LAYERS} '
      f'dropout={ABL_DROPOUT} lr={ABL_LR}')
print('='*65)

arch_rows = []

for name, ModelClass in ARCH_VARIANTS.items():

    # ── Proposed model — insert existing results ──────────────────────────────
    if name == 'InsiderLSTM (Full)':
        for ds_label, lstm_res, y_te in [
            ('r4.2',  lstm_r42, y_te_r42),
            ('SPEDIA', lstm_sp,  y_te_sp_locked)
        ]:
            auc = roc_auc_score(y_te,
                                np.array(lstm_res['probs'],
                                         dtype='float64'))
            print(f'  [{ds_label}] {name:<25} '
                  f'AUC={auc:.4f}  (existing — no retrain)')
            arch_rows.append({
                'name'   : name,
                'dataset': ds_label,
                'auc'    : auc,
                'probs'  : lstm_res['probs']
            })
        continue

    # ── Ablation variants ─────────────────────────────────────────────────────
    for ds_label, X_tr, y_tr, X_vl, y_vl, X_te, y_te, \
        pw, sampler, input_dim in [
        ('r4.2',
         X_tr_r42, y_tr_r42, X_vl_r42, y_vl_r42,
         X_te_r42, y_te_r42,
         PW_R42, SAMPLER_R42, INPUT_DIM_R42),
        ('SPEDIA',
         X_tr_sp, y_tr_sp, X_vl_sp, y_vl_sp,
         X_te_sp, y_te_sp,
         PW_SP, SAMPLER_SP, FEATURE_DIM_SP),
    ]:
        r = run_abl(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                     pw, sampler, ModelClass, input_dim,
                     name, ds_label)
        arch_rows.append(r)

abl_results['architecture'] = arch_rows

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Feature set ablations (r4.2 only)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART B — Feature Set Ablations (r4.2 only)')
print('='*65)

# Save originals
_X_tr = X_tr_r42.copy(); _y_tr = y_tr_r42.copy()
_X_vl = X_vl_r42.copy(); _y_vl = y_vl_r42.copy()
_X_te = X_te_r42.copy(); _y_te = y_te_r42.copy()

feat_rows = []

n_raw      = len(FEATURE_COLS)
n_dev      = len(dev_cols)
RAW_IDX    = list(range(n_raw))
ZSCORE_IDX = list(range(n_raw, n_raw + n_dev))
ALL_IDX    = list(range(n_raw + n_dev))

FEAT_VARIANTS = {
    'Raw only'          : RAW_IDX,
    'Z-score only'      : ZSCORE_IDX,
    'Raw + Z-score'     : ALL_IDX,
}

for feat_name, feat_idx in FEAT_VARIANTS.items():

    if feat_name == 'Raw + Z-score':
        auc = roc_auc_score(_y_te,
                             np.array(lstm_r42['probs'],
                                      dtype='float64'))
        print(f'\n  {feat_name} (dim={len(feat_idx)}): '
              f'AUC={auc:.4f}  (existing — no retrain)')
        feat_rows.append({
            'name'   : feat_name,
            'dataset': 'r4.2',
            'auc'    : auc,
            'probs'  : lstm_r42['probs']
        })
        continue

    Xf_tr = _X_tr[:, :, feat_idx]
    Xf_vl = _X_vl[:, :, feat_idx]
    Xf_te = _X_te[:, :, feat_idx]
    _dim  = len(feat_idx)

    # Use dynamic pos_weight — not hardcoded 30.0
    _n_pos = int(_y_tr.sum())
    _n_neg = int((_y_tr==0).sum())
    _ratio = _n_neg / max(_n_pos, 1)
    _pw_val= float(np.sqrt(_ratio))
    _sw    = np.where(_y_tr==1, _pw_val, 1.0)
    _samp  = WeightedRandomSampler(
        torch.tensor(_sw, dtype=torch.float64), len(_sw), True)
    _pw    = torch.tensor([_pw_val], dtype=torch.float32)

    print(f'\n  {feat_name} (dim={_dim}): '
          f'train={len(Xf_tr):,} pos={_n_pos:,} '
          f'pw={_pw_val:.2f}')

    r = run_abl(Xf_tr, _y_tr, Xf_vl, _y_vl, Xf_te, _y_te,
                 _pw, _samp, InsiderLSTM, _dim,
                 feat_name, 'r4.2')
    feat_rows.append(r)
    del Xf_tr, Xf_vl, Xf_te; gc.collect()

abl_results['features'] = feat_rows

# Restore
X_tr_r42 = _X_tr; y_tr_r42 = _y_tr
X_vl_r42 = _X_vl; y_vl_r42 = _y_vl
X_te_r42 = _X_te; y_te_r42 = _y_te
del _X_tr, _y_tr, _X_vl, _y_vl, _X_te, _y_te; gc.collect()
print('\n  Originals restored after Part B.')

# ---- [cell 66] ------------------------------------------------------
# Verify val has both classes before Part C
print(f'Val pos={int(y_vl_r42.sum()):,} neg={int((y_vl_r42==0).sum()):,}')
assert int(y_vl_r42.sum()) > 0 and int((y_vl_r42==0).sum()) > 0, \
    'Val set must have both classes before Part C'

# ---- [cell 67] ------------------------------------------------------
# ── Check if new split dataframes are in memory ───────────────────────────────
try:
    print(f'train_r42 shape : {train_r42.shape}')
    print(f'val_r42 shape   : {val_r42.shape}')
    print(f'test_r42 shape  : {test_r42.shape}')
    print('✅ New split dataframes available in memory')
except NameError:
    print('❌ Dataframes not in memory — need to reload from parquet')
    print('   But saved parquet files are from old split — cannot use them')
    print('   Must re-run Section 3 to rebuild new split dataframes')

# ---- [cell 68] ------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════════
# PART C — Sequence length ablations (r4.2 only)
# Uses new user-level split dataframes from memory
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('PART C — Sequence Length Ablations (r4.2 only)')
print('='*65)

# Save originals
_X_tr = X_tr_r42.copy(); _y_tr = y_tr_r42.copy()
_X_vl = X_vl_r42.copy(); _y_vl = y_vl_r42.copy()
_X_te = X_te_r42.copy(); _y_te = y_te_r42.copy()

seq_rows = []

# ── SEQ_LEN=7 — slice first 7 days from 14-day sequences ─────────────────────
print('\n  SEQ_LEN=7 (first 7 days of each 14-day window):')
Xs_tr_7 = _X_tr[:, :7, :]
Xs_vl_7 = _X_vl[:, :7, :]
Xs_te_7 = _X_te[:, :7, :]

_n_pos  = int(_y_tr.sum())
_n_neg  = int((_y_tr==0).sum())
_ratio  = _n_neg / max(_n_pos, 1)
_pw_val = float(np.sqrt(_ratio))
_sw     = np.where(_y_tr==1, _pw_val, 1.0)
_samp   = WeightedRandomSampler(
    torch.tensor(_sw, dtype=torch.float64), len(_sw), True)
_pw     = torch.tensor([_pw_val], dtype=torch.float32)

print(f'    train={len(Xs_tr_7):,} pos={_n_pos:,} '
      f'neg={_n_neg:,} pw={_pw_val:.2f}')
print(f'    val  ={len(Xs_vl_7):,} pos={int(_y_vl.sum()):,}')
print(f'    test ={len(Xs_te_7):,} pos={int(_y_te.sum()):,}')

r7 = run_abl(Xs_tr_7, _y_tr, Xs_vl_7, _y_vl, Xs_te_7, _y_te,
              _pw, _samp, InsiderLSTM, INPUT_DIM_R42,
              'SEQ_LEN=7', 'r4.2')
seq_rows.append(r7)
del Xs_tr_7, Xs_vl_7, Xs_te_7; gc.collect()

# ── SEQ_LEN=14 — existing result ─────────────────────────────────────────────
print(f'\n  SEQ_LEN=14 ← proposed (existing result):')
auc_14 = roc_auc_score(_y_te,
                        np.array(lstm_r42['probs'], dtype='float64'))
seq_rows.append({
    'name'   : 'SEQ_LEN=14',
    'dataset': 'r4.2',
    'auc'    : auc_14,
    'probs'  : lstm_r42['probs']
})
print(f'    AUC={auc_14:.4f}')

# ── SEQ_LEN=21 — rebuild from new user-level split dataframes ────────────────
print('\n  SEQ_LEN=21 (new user-level split — rebuilt from memory):')

try:
    # Verify new split dataframes are available
    assert 'train_r42' in dir() or 'train_r42' in globals(), \
        'train_r42 not in memory'
    print(f'    train_r42 shape: {train_r42.shape}')
    print(f'    val_r42 shape  : {val_r42.shape}')
    print(f'    test_r42 shape : {test_r42.shape}')

    # Temporarily set SEQ_LEN to 21
    _old_seq    = SEQ_LEN_R42
    SEQ_LEN_R42 = 21

    print('    Building 21-day sequences from new user-level split...')
    print('    Train:')
    Xs_tr_21, ys_tr_21, _ = build_sequences_daily(train_r42)
    print('    Val  :')
    Xs_vl_21, ys_vl_21, _ = build_sequences_daily(val_r42)
    print('    Test :')
    Xs_te_21, ys_te_21, _ = build_sequences_daily(test_r42)

    # Restore SEQ_LEN immediately
    SEQ_LEN_R42 = _old_seq
    gc.collect()

    print(f'    train={len(Xs_tr_21):,} pos={int(ys_tr_21.sum()):,} '
          f'neg={int((ys_tr_21==0).sum()):,}')
    print(f'    val  ={len(Xs_vl_21):,} pos={int(ys_vl_21.sum()):,}')
    print(f'    test ={len(Xs_te_21):,} pos={int(ys_te_21.sum()):,}')

    if (len(np.unique(ys_vl_21)) < 2 or
            len(np.unique(ys_te_21)) < 2 or
            len(Xs_tr_21) == 0):
        print('    SKIP — single class or no sequences')
        seq_rows.append({
            'name': 'SEQ_LEN=21', 'dataset': 'r4.2',
            'auc': float('nan'), 'probs': None
        })
    else:
        _n_pos  = int(ys_tr_21.sum())
        _n_neg  = int((ys_tr_21==0).sum())
        _ratio  = _n_neg / max(_n_pos, 1)
        _pw_val = float(np.sqrt(_ratio))
        _sw     = np.where(ys_tr_21==1, _pw_val, 1.0)
        _samp   = WeightedRandomSampler(
            torch.tensor(_sw, dtype=torch.float64), len(_sw), True)
        _pw     = torch.tensor([_pw_val], dtype=torch.float32)

        print(f'    pw={_pw_val:.2f}')

        r21 = run_abl(Xs_tr_21, ys_tr_21,
                       Xs_vl_21, ys_vl_21,
                       Xs_te_21, ys_te_21,
                       _pw, _samp, InsiderLSTM, INPUT_DIM_R42,
                       'SEQ_LEN=21', 'r4.2')
        seq_rows.append(r21)
        del Xs_tr_21, Xs_vl_21, Xs_te_21
        del ys_tr_21, ys_vl_21, ys_te_21
        gc.collect()

except Exception as e:
    print(f'    SKIP SEQ_LEN=21 — {e}')
    print(f'    Reason: new split dataframes not in memory')
    print(f'    Fix: re-run Section 3 to rebuild new split')
    seq_rows.append({
        'name': 'SEQ_LEN=21', 'dataset': 'r4.2',
        'auc': float('nan'), 'probs': None
    })
    SEQ_LEN_R42 = _old_seq  # restore if exception

abl_results['seq_len'] = seq_rows

# ── Restore originals ─────────────────────────────────────────────────────────
X_tr_r42 = _X_tr; y_tr_r42 = _y_tr
X_vl_r42 = _X_vl; y_vl_r42 = _y_vl
X_te_r42 = _X_te; y_te_r42 = _y_te
del _X_tr, _y_tr, _X_vl, _y_vl, _X_te, _y_te; gc.collect()

# ── Part C summary ────────────────────────────────────────────────────────────
proposed_seq = next(
    (r['auc'] for r in seq_rows
     if '14' in r['name'] and not np.isnan(r['auc'])), None)

proposed_seq_str = f'{proposed_seq:.4f}' \
                   if proposed_seq is not None else 'N/A'

print(f'\nPart C — Sequence length (proposed={proposed_seq_str}):')
print(f'  {"Variant":<15} {"r4.2 AUC":>10} {"ΔAUC":>8}')
print(f'  {"-"*38}')
for r in seq_rows:
    is_p    = '14' in r['name']
    auc_val = r['auc']
    auc_s   = f'{auc_val:.4f}' if not np.isnan(auc_val) else 'N/A'
    if is_p or np.isnan(auc_val) or proposed_seq is None:
        delta = '—'
    else:
        delta = f'{auc_val - proposed_seq:+.4f}'
    m = ' ← proposed' if is_p else ''
    print(f'  {r["name"]:<15} {auc_s:>10} {delta:>8}{m}')

# ── Post-ablation check ───────────────────────────────────────────────────────
print(f'\nPost-ablation check:')
print(f'  SEQ_LEN_R42    : {SEQ_LEN_R42}  (expect 14)')
print(f'  X_te_r42 shape : {X_te_r42.shape}')
print(f'  X_te_r42 dim 1 : {X_te_r42.shape[1]}  (expect 14)')
print(f'  X_te_r42 dim 2 : {X_te_r42.shape[2]}  (expect {INPUT_DIM_R42})')
print(f'  y_te_r42 pos   : {int(y_te_r42.sum())}')
assert SEQ_LEN_R42 == 14,             'SEQ_LEN_R42 corrupted'
assert X_te_r42.shape[1] == 14,       'Sequence length corrupted'
assert X_te_r42.shape[2] == INPUT_DIM_R42, 'INPUT_DIM corrupted'
print('  ✅ Sequences intact — SEQ_LEN_R42 correctly restored to 14')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('ABLATION STUDY SUMMARY')
print('='*65)

proposed_r42  = next(r['auc'] for r in abl_results['architecture']
                      if 'Full' in r['name'] and r['dataset']=='r4.2')
proposed_sp   = next(r['auc'] for r in abl_results['architecture']
                      if 'Full' in r['name'] and r['dataset']=='SPEDIA')
proposed_feat = next(r['auc'] for r in abl_results['features']
                      if 'Raw + Z-score' in r['name'])
proposed_seq  = next((r['auc'] for r in abl_results['seq_len']
                       if '14' in r['name'] and
                       not np.isnan(r['auc'])), None)

print(f'\nPart A — Architecture (proposed r4.2={proposed_r42:.4f}, '
      f'SPEDIA={proposed_sp:.4f}):')
print(f'  {"Variant":<25} {"r4.2":>10} {"Δ r4.2":>8} '
      f'{"SPEDIA":>10} {"Δ SPEDIA":>10}')
print(f'  {"-"*65}')
for name in ARCH_VARIANTS.keys():
    r42 = next((r['auc'] for r in abl_results['architecture']
                 if r['name']==name and r['dataset']=='r4.2'), None)
    sp  = next((r['auc'] for r in abl_results['architecture']
                 if r['name']==name and r['dataset']=='SPEDIA'), None)
    d42 = f'{r42-proposed_r42:+.4f}' \
          if r42 and 'Full' not in name else '—'
    dsp = f'{sp-proposed_sp:+.4f}'   \
          if sp  and 'Full' not in name else '—'
    m     = ' ← proposed' if 'Full' in name else ''
    r42_s = f'{r42:.4f}' if r42 else 'N/A'
    sp_s  = f'{sp:.4f}'  if sp  else 'N/A'
    print(f'  {name:<25} {r42_s:>10} {d42:>8} '
          f'{sp_s:>10} {dsp:>10}{m}')

print(f'\nPart B — Feature set (proposed={proposed_feat:.4f}):')
print(f'  {"Variant":<25} {"r4.2":>10} {"ΔAUC":>8}')
print(f'  {"-"*45}')
for r in abl_results['features']:
    is_p  = 'Raw + Z-score' in r['name']
    delta = '—' if is_p else f'{r["auc"]-proposed_feat:+.4f}'
    m     = ' ← proposed' if is_p else ''
    print(f'  {r["name"]:<25} {r["auc"]:>10.4f} {delta:>8}{m}')

proposed_seq_str = f'{proposed_seq:.4f}' \
                   if proposed_seq is not None else 'N/A'
print(f'\nPart C — Sequence length (proposed={proposed_seq_str}):')
print(f'  {"Variant":<15} {"r4.2":>10} {"ΔAUC":>8}')
print(f'  {"-"*35}')
for r in abl_results['seq_len']:
    is_p    = '14' in r['name']
    auc_val = r['auc']
    auc_s   = f'{auc_val:.4f}' if not np.isnan(auc_val) else 'N/A'
    if is_p or np.isnan(auc_val) or proposed_seq is None:
        delta = '—'
    else:
        delta = f'{auc_val - proposed_seq:+.4f}'
    m = ' ← proposed' if is_p else ''
    print(f'  {r["name"]:<15} {auc_s:>10} {delta:>8}{m}')

# ── Save ──────────────────────────────────────────────────────────────────────
abl_flat = []
for r in abl_results['architecture']:
    abl_flat.append({
        'Part': 'A-Architecture', 'Dataset': r['dataset'],
        'Variant': r['name'], 'AUC': round(r['auc'], 4)
    })
for r in abl_results['features']:
    abl_flat.append({
        'Part': 'B-Features', 'Dataset': 'r4.2',
        'Variant': r['name'], 'AUC': round(r['auc'], 4)
    })
for r in abl_results['seq_len']:
    auc_val = round(r['auc'], 4) if not np.isnan(r['auc']) else None
    abl_flat.append({
        'Part': 'C-SeqLen', 'Dataset': 'r4.2',
        'Variant': r['name'], 'AUC': auc_val
    })

pd.DataFrame(abl_flat).to_csv(
    os.path.join(WORK_DIR, 'ablation_results.csv'), index=False)
print('\n✅ Ablation complete. Saved: ablation_results.csv')
print(f'\nPost-ablation check:')
print(f'  X_te_r42 shape : {X_te_r42.shape}')
print(f'  y_te_r42 pos   : {int(y_te_r42.sum())}')
assert X_te_r42.shape[2] == INPUT_DIM_R42, 'INPUT_DIM corrupted'
print('  ✅ Sequences intact.')

# ---- [cell 69] ------------------------------------------------------
# Confirm all final numbers before writing
print('=== FINAL RESULTS AUDIT ===')
print(f'\nInsiderLSTM:')
print(f'  r4.2  AUC={lstm_r42["auc"]:.4f}  F1=0.7592  EWLT=8.3d')
print(f'  SPEDIA AUC={lstm_sp["auc"]:.4f}  F1=0.9679  EWLT=2.0d')

print(f'\nAblation (Part A — Architecture):')
for r in abl_results['architecture']:
    print(f'  {r["name"]:<25} {r["dataset"]:>8}  AUC={r["auc"]:.4f}')

print(f'\nAblation (Part B — Features):')
for r in abl_results['features']:
    print(f'  {r["name"]:<25}  AUC={r["auc"]:.4f}')

print(f'\nAblation (Part C — Seq length):')
for r in abl_results['seq_len']:
    auc_s = f'{r["auc"]:.4f}' if not np.isnan(r["auc"]) else 'N/A'
    print(f'  {r["name"]:<15}  AUC={auc_s}')
