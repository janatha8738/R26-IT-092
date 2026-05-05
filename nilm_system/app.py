"""
NILM Smart Energy Monitor — Hybrid Model System
Member 1: Appliance Identification (SGN + DualOutputNILM CNN)
Member 3: Fault Detection (overcurrent, overuse, power spikes)

Model assignments (best F1 per appliance):
  SGN:            kettle(0.771), washing_machine(0.789), tv(0.377), computer(0.775)
  DualOutputNILM: fridge(0.722), freezer(0.811), fridge_freezer(0.752), dishwasher(0.627)
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR  = './models'
DATA_PATH  = './data/house.csv'
SEQ_LEN    = 480
BORDER     = 16
SIM_SPEED  = 1

APPLIANCE_META = {
    'fridge':          (400,  '#38bdf8', '🧊'),
    'freezer':         (400,  '#818cf8', '❄️'),
    'fridge_freezer':  (400,  '#6ee7b7', '🧊'),
    'washing_machine': (2500, '#fb923c', '🫧'),
    'dishwasher':      (2500, '#f472b6', '🍽️'),
    'kettle':          (3100, '#f87171', '☕'),
    'television':      (400,  '#a78bfa', '📺'),
    'computer':        (400,  '#34d399', '💻'),
}

SGN_FILES = {
    'kettle':          'sgn_Kettle.pt',
    'washing_machine': 'sgn_Washing_Machine.pt',
    'television':      'sgn_TV.pt',
    'computer':        'sgn_Computer.pt',
}

CNN_FILES = {
    'fridge':         'model_fridge.pth',
    'freezer':        'model_freezer.pth',
    'fridge_freezer': 'model_fridge_freezer.pth',
    'dishwasher':     'model_dishwasher.pth',
}

CNN_THRESHOLDS = {
    'fridge': 0.40, 'freezer': 0.40,
    'fridge_freezer': 0.40, 'dishwasher': 0.35,
}

FAULT_CONFIG = {
    'overcurrent_W':      8000,
    'high_power_spike_W': 5000,
    'appliance_overuse': {
        'washing_machine': 120,
        'dishwasher':      120,
        'kettle':          10,
        'television':      360,
        'computer':        480,
    },
}

# ─── PYTORCH ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class SGNModel(nn.Module):
        """
        Exact architecture matching saved .pt files.
        Decoder indices: 1=Linear(2048,512), 4=Linear(512,128), 7=Linear(128,1)
        Gaps are ReLU/Dropout layers (index 2,3,5,6) — we use nn.Sequential
        with explicit indexing by inserting placeholder layers.
        """
        def __init__(self):
            super().__init__()
            # encoder.0=Conv, encoder.1=BN, (2=ReLU),
            # encoder.3=Conv, encoder.4=BN, (5=ReLU),
            # encoder.6=Conv, encoder.7=BN
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),   # 0
                nn.BatchNorm1d(32),                            # 1
                nn.ReLU(),                                     # 2
                nn.Conv1d(32, 64, kernel_size=5, padding=2),  # 3
                nn.BatchNorm1d(64),                            # 4
                nn.ReLU(),                                     # 5
                nn.Conv1d(64, 64, kernel_size=5, padding=2),  # 6
                nn.BatchNorm1d(64),                            # 7
                nn.ReLU(),                                     # 8
            )
            # gate.0=Conv
            self.gate = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, padding=2),  # 0
                nn.Sigmoid(),
            )
            self.pool = nn.AdaptiveAvgPool1d(32)
            # decoder.1=Linear, (2=ReLU), (3=Dropout),
            # decoder.4=Linear, (5=ReLU), (6=Dropout),
            # decoder.7=Linear
            self.decoder = nn.Sequential(
                nn.Flatten(),                    # 0
                nn.Linear(64 * 32, 512),         # 1
                nn.ReLU(),                        # 2
                nn.Dropout(0.0),                  # 3 — placeholder, no effect
                nn.Linear(512, 128),              # 4
                nn.ReLU(),                        # 5
                nn.Dropout(0.0),                  # 6 — placeholder
                nn.Linear(128, 1),               # 7
                nn.Sigmoid(),
            )

        def forward(self, x):
            h = self.encoder(x)
            h = h * self.gate(h)
            h = self.pool(h)
            return self.decoder(h)

    class ResidualBlock(nn.Module):
        def __init__(self, channels, kernel_size, dilation):
            super().__init__()
            pad = (kernel_size - 1) * dilation // 2
            self.conv = nn.Conv1d(channels, channels * 2, kernel_size,
                                  dilation=dilation, padding=pad)
            self.res  = nn.Conv1d(channels, channels, 1)
            self.bn   = nn.BatchNorm1d(channels)

        def forward(self, x):
            h = self.conv(x)
            h_tanh, h_sig = h.chunk(2, dim=1)
            return self.bn(self.res(torch.tanh(h_tanh) * torch.sigmoid(h_sig)) + x)

    class DualOutputNILM(nn.Module):
        def __init__(self, channels=32, n_blocks=8):
            super().__init__()
            self.border  = BORDER
            self.out_len = SEQ_LEN
            # saved weights use "embed" not "input_conv"
            self.embed = nn.Sequential(
                nn.Conv1d(1, channels, 3, padding=1),
                nn.BatchNorm1d(channels), nn.ReLU()
            )
            self.blocks = nn.ModuleList([
                ResidualBlock(channels, 3, 2**i) for i in range(n_blocks)
            ])
            self.shared   = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.ReLU())
            self.cls_head = nn.Conv1d(channels, 1, 1)
            self.reg_head = nn.Sequential(nn.Conv1d(channels, 1, 1), nn.Sigmoid())

        def forward(self, x):
            h = self.embed(x)
            for b in self.blocks: h = b(h)
            h = h[:, :, self.border: self.border + self.out_len]
            h = self.shared(h)
            return self.cls_head(h).squeeze(1), self.reg_head(h).squeeze(1)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_AVAILABLE = True
    print(f'PyTorch ready — {DEVICE}')

except ImportError:
    TORCH_AVAILABLE = False
    print('PyTorch not found — demo mode')

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
loaded_models = {}

def load_models():
    if not TORCH_AVAILABLE:
        return

    for name, fname in SGN_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f'  [MISSING] {path}'); continue
        try:
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            m  = SGNModel().to(DEVICE)
            m.load_state_dict(ck['model_state_dict'])
            m.eval()
            loaded_models[name] = {
                'model': m, 'type': 'sgn',
                'agg_mean': float(ck['agg_mean']),
                'agg_std':  float(ck['agg_std']),
                'win_size': int(ck['window_size']),
            }
            print(f'  [SGN] {name}')
        except Exception as e:
            print(f'  [ERROR] {name}: {e}')

    for name, fname in CNN_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f'  [MISSING] {path}'); continue
        try:
            m = DualOutputNILM().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            loaded_models[name] = {
                'model': m, 'type': 'cnn',
                'threshold': CNN_THRESHOLDS.get(name, 0.4),
                'max_pw':    APPLIANCE_META[name][0],
            }
            print(f'  [CNN] {name}')
        except Exception as e:
            print(f'  [ERROR] {name}: {e}')

    print(f'Loaded: {list(loaded_models.keys())}')

# ─── INFERENCE ────────────────────────────────────────────────────────────────
@torch.no_grad() if TORCH_AVAILABLE else lambda f: f
def predict(name, entry, window):
    if entry['type'] == 'sgn':
        win  = entry['win_size']
        mean = entry['agg_mean']
        std  = entry['agg_std']
        x = np.array(window[-win:], dtype=np.float32)
        if len(x) < win:
            x = np.pad(x, (win - len(x), 0))
        x = (x - mean) / (std + 1e-8)
        x_t  = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        prob = entry['model'](x_t).item()
        on   = prob >= 0.5
        # Rough power estimate for SGN (regression not available)
        power = APPLIANCE_META[name][0] * 0.5 * prob if on else 0.0
        return on, round(power, 1)
    else:
        max_pw = entry['max_pw']
        thr    = entry['threshold']
        need   = SEQ_LEN + 2 * BORDER
        if len(window) < need:
            return False, 0.0
        x = np.array(window[-need:], dtype=np.float32) / max_pw
        x -= x.mean()
        x_t = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        lg, pw = entry['model'](x_t)
        prob  = float(torch.sigmoid(lg).squeeze().cpu()[-1])
        power = float(pw.squeeze().cpu()[-1]) * max_pw
        return prob >= thr, round(max(0, power), 1)

# ─── SIMULATOR ────────────────────────────────────────────────────────────────
class DataSimulator:
    def __init__(self):
        self.df            = None
        self.cursor        = SEQ_LEN + 2 * BORDER
        self.running       = False
        self.history       = deque(maxlen=120)
        self.current_state = {}
        self.faults        = deque(maxlen=50)
        self.on_timers          = {a: 0 for a in APPLIANCE_META}
        self.total_kwh          = {a: 0.0 for a in APPLIANCE_META}
        self.prev_agg           = 0
        self.sim_minutes        = 0   # total simulated minutes elapsed
        self.sim_start_label    = None

    def load_csv(self, path):
        df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
        df = df.select_dtypes(include=[np.number]).clip(lower=0)
        df = df.resample('1min').mean().ffill(limit=10).fillna(0)
        self.df     = df
        self.cursor = SEQ_LEN + 2 * BORDER
        print(f'CSV: {len(df):,} rows')

    def demo_state(self):
        import math; t = time.time()
        return {
            n: {'status': math.sin(t/(10+i*3))>0.3,
                'power':  round(mw*0.5*abs(math.sin(t+i)),1) if math.sin(t/(10+i*3))>0.3 else 0.0,
                'color': c, 'icon': ic, 'model_type': 'demo'}
            for i,(n,(mw,c,ic)) in enumerate(APPLIANCE_META.items())
        }

    def step(self):
        if self.df is None or not loaded_models:
            state   = self.demo_state()
            agg_now = sum(v['power'] for v in state.values())
        else:
            if self.cursor >= len(self.df):
                self.cursor = SEQ_LEN + 2 * BORDER
            agg_arr = self.df['Aggregate'].clip(0, 10000).values
            window  = agg_arr[:self.cursor]
            agg_now = float(agg_arr[self.cursor - 1])
            state   = {}
            for name, (max_pw, color, icon) in APPLIANCE_META.items():
                entry = loaded_models.get(name)
                if entry is None:
                    state[name] = {'status': False, 'power': 0.0,
                                   'color': color, 'icon': icon, 'model_type': 'missing'}
                    continue
                on, power = predict(name, entry, window)
                state[name] = {'status': on, 'power': power,
                               'color': color, 'icon': icon,
                               'model_type': entry['type']}
                self.total_kwh[name] = round(self.total_kwh[name] + power / 60000, 4)
                self.on_timers[name] = self.on_timers[name] + 1 if on else 0
            self.cursor += 1
            self.sim_minutes += 1

        self._check_faults(state, agg_now)
        label = (self.df.index[self.cursor-2].strftime('%H:%M')
                 if self.df is not None and hasattr(self.df.index[0], 'strftime')
                 else datetime.now().strftime('%H:%M'))
        if self.sim_start_label is None:
            self.sim_start_label = label
        self.history.append({'time': label, 'aggregate': round(agg_now,1),
                             **{k: round(v['power'],1) for k,v in state.items()}})
        self.current_state = state

    def _check_faults(self, state, agg):
        now = datetime.now().strftime('%H:%M:%S')
        for f in [
            ({'type':'OVERCURRENT','severity':'critical',
              'msg':f'Total load {agg:.0f}W exceeds {FAULT_CONFIG["overcurrent_W"]}W',
              'time':now}
             if agg > FAULT_CONFIG['overcurrent_W'] else None),
            ({'type':'POWER SPIKE','severity':'warning',
              'msg':f'Sudden +{agg-self.prev_agg:.0f}W spike','time':now}
             if agg - self.prev_agg > FAULT_CONFIG['high_power_spike_W'] else None),
            *[{'type':'OVERUSE','severity':'warning',
               'msg':f'{a.replace("_"," ").title()} ON {self.on_timers[a]}min (limit {m}min)',
               'time':now}
              for a,m in FAULT_CONFIG['appliance_overuse'].items()
              if self.on_timers.get(a,0) > m]
        ]:
            if f: self.faults.appendleft(f)
        self.prev_agg = agg

sim = DataSimulator()

def simulation_loop():
    while True:
        if sim.running: sim.step()
        time.sleep(1.0 / max(SIM_SPEED, 1))

threading.Thread(target=simulation_loop, daemon=True).start()

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'appliances':  sim.current_state,
        'faults':      list(sim.faults)[:10],
        'total_kwh':   sim.total_kwh,
        'running':     sim.running,
        'models':      {k: v['type'] for k, v in loaded_models.items()},
        'data_loaded': sim.df is not None,
        'sim_minutes': sim.sim_minutes,
        'sim_start':   sim.sim_start_label,
    })

@app.route('/api/history')
def api_history(): return jsonify(list(sim.history))

@app.route('/api/control', methods=['POST'])
def api_control():
    a = request.json.get('action')
    if a == 'start':   sim.running = True
    elif a == 'stop':  sim.running = False
    elif a == 'reset':
        sim.cursor = SEQ_LEN + 2 * BORDER
        sim.history.clear(); sim.faults.clear()
        sim.on_timers       = {a: 0 for a in APPLIANCE_META}
        sim.total_kwh       = {a: 0.0 for a in APPLIANCE_META}
        sim.sim_minutes     = 0
        sim.sim_start_label = None
        sim.running         = False
    return jsonify({'ok': True, 'running': sim.running})

@app.route('/api/load_csv', methods=['POST'])
def api_load_csv():
    path = request.json.get('path', DATA_PATH)
    try:
        sim.load_csv(path)
        return jsonify({'ok': True, 'rows': len(sim.df)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print('\n🔌 NILM Hybrid Energy Monitor')
    print('─' * 40)
    load_models()
    if os.path.exists(DATA_PATH): sim.load_csv(DATA_PATH)
    else: print(f'No CSV at {DATA_PATH} — demo mode')
    print('\n→ http://localhost:5000')
    app.run(debug=False, port=8000, threaded=True)