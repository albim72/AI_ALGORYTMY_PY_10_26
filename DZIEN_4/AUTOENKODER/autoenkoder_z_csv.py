"""
AUTOENKODER na REALNYM CSV — wykrywanie anomalii medycznych
============================================================
Wczytuje dane_medyczne_5000.csv (5000 pacjentów, 12 markerów, braki ~1.5%).

Pełny, uczciwy pipeline:
  1. Wczytanie CSV + imputacja braków (mediana z treningu — bez wycieku)
  2. Podział: trening = TYLKO zdrowi | test = zdrowi + anomalie
  3. Trening autoenkodera na rekonstrukcji zdrowych
  4. Scoring błędem rekonstrukcji + próg z walidacji
  5. Metryki: ROC-AUC, PR-AUC, macierz pomyłek

Uruchom NAJPIERW generuj_csv.py, żeby powstał plik CSV.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MARKERY = ["glukoza", "hba1c", "cholesterol", "ldl", "hdl", "triglicerydy",
           "kreatynina", "egfr", "alt", "ast", "crp", "leukocyty"]


# ---- 1. Wczytanie realnego CSV ----------------------------------------
df = pd.read_csv("dane_medyczne_5000.csv")
print(f"Wczytano {len(df)} rekordów z CSV")
print(f"Braki w markerach: {df[MARKERY].isna().sum().sum()} komórek\n")

zdrowi   = df[df.label == 0].reset_index(drop=True)
anomalie = df[df.label == 1].reset_index(drop=True)

# ---- 2. Podział: trening/walidacja/test (zdrowi) + test (anomalie) -----
n_z = len(zdrowi)
i_train = int(n_z * 0.7)
i_val   = int(n_z * 0.8)

zdr_train = zdrowi.iloc[:i_train]
zdr_val   = zdrowi.iloc[i_train:i_val]
zdr_test  = zdrowi.iloc[i_val:]

# ---- 3. Imputacja + skalowanie DOPASOWANE TYLKO na treningu -----------
imputer = SimpleImputer(strategy="median").fit(zdr_train[MARKERY])
scaler  = StandardScaler().fit(imputer.transform(zdr_train[MARKERY]))

def przygotuj(frame):
    x = scaler.transform(imputer.transform(frame[MARKERY]))
    return torch.tensor(x, dtype=torch.float32)

Xtr = przygotuj(zdr_train)
Xvl = przygotuj(zdr_val)
X_test_frame = pd.concat([zdr_test, anomalie], ignore_index=True)
Xte = przygotuj(X_test_frame)
yte = X_test_frame.label.values

print(f"Trening (zdrowi):  {len(Xtr)}")
print(f"Walidacja (zdrowi): {len(Xvl)}")
print(f"Test: {len(zdr_test)} zdrowych + {len(anomalie)} anomalii\n")


# ---- 4. Model ----------------------------------------------------------
class Autoenkoder(nn.Module):
    def __init__(self, wej=12, ukr=8, kod=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(wej, ukr), nn.ReLU(), nn.Linear(ukr, kod), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(kod, ukr), nn.ReLU(), nn.Linear(ukr, wej))
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoenkoder().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
kryt = nn.MSELoss()


# ---- 5. Trening z early stopping --------------------------------------
Xtr_d, Xvl_d = Xtr.to(DEVICE), Xvl.to(DEVICE)
najlepsza, licznik, cierpl = np.inf, 0, 15
for ep in range(200):
    model.train()
    perm = torch.randperm(len(Xtr_d))
    for i in range(0, len(Xtr_d), 128):
        xb = Xtr_d[perm[i:i+128]]
        optim.zero_grad()
        kryt(model(xb), xb).backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        val = kryt(model(Xvl_d), Xvl_d).item()
    if val < najlepsza - 1e-5:
        najlepsza, licznik = val, 0
        stan = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        licznik += 1
        if licznik >= cierpl:
            print(f"early stopping @ epoka {ep}"); break
    if ep % 25 == 0:
        print(f"  epoka {ep:3d} | walidacja MSE {val:.4f}")
model.load_state_dict(stan)


# ---- 6. Scoring + metryki ---------------------------------------------
def bledy(X):
    model.eval()
    with torch.no_grad():
        rek = model(X.to(DEVICE)).cpu()
    return ((X - rek) ** 2).mean(dim=1).numpy()

prog = np.percentile(bledy(Xvl), 95)
e_test = bledy(Xte)
auc = roc_auc_score(yte, e_test)
ap  = average_precision_score(yte, e_test)
cm  = confusion_matrix(yte, (e_test > prog).astype(int))
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*52}")
print(f"  ROC-AUC:  {auc:.3f}")
print(f"  PR-AUC:   {ap:.3f}")
print(f"  Czułość:   {tp/(tp+fn):6.1%}   (wykryte anomalie)")
print(f"  Swoistość: {tn/(tn+fp):6.1%}   (poprawni zdrowi)")
print(f"  Precyzja:  {tp/(tp+fp):6.1%}   (trafność alarmu)")
print(f"{'='*52}")

# rozbicie skuteczności per wzorzec patologii — mocny slajd
print("\nCzułość per typ patologii:")
maska_anom = X_test_frame.label == 1
for wz in ["metaboliczny", "nerkowy", "watrobowy", "zapalny"]:
    m = (X_test_frame.pattern == wz).values
    if m.sum():
        wykryte = (e_test[m] > prog).mean()
        print(f"  {wz:14s}  {wykryte:5.1%}  (n={m.sum()})")
