import pickle
import os
import csv
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

RDLogger.DisableLog("rdApp.*")

app = FastAPI(title="BioPolymer-AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 1. Charger la base de polymères biosourcés ─────────────
def charger_base_polymeres():
    base = {}
    chemin = 'base_polymeres_biosources.csv'
    if os.path.exists(chemin):
        with open(chemin, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('smiles'):
                    base[row['nom_recherche'].strip().lower()] = row['smiles']
                    base[row['nom_affiche'].strip().lower()]   = row['smiles']
        print(f"✅ Base polymères chargée : {len(base)} entrées")
    else:
        print("⚠️ base_polymeres_biosources.csv non trouvée")
    return base

BASE_POLYMERES = charger_base_polymeres()

# ── 2. Télécharger modele_module2 depuis Google Drive ──────
MODELES_DIR  = "modeles"
FILE_ID_M2   = "1mnm6gLpnhWMJo5-LceaL66CizIVDO5Z1"
os.makedirs(MODELES_DIR, exist_ok=True)

def telecharger_drive(file_id, dest):
    if os.path.exists(dest):
        print(f"✅ {dest} déjà présent")
        return
    print(f"📥 Téléchargement {dest}...")
    session  = requests.Session()
    url      = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url      = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={value}"
            response = session.get(url, stream=True)
            break
    with open(dest, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f.write(chunk)
    print(f"✅ {dest} téléchargé !")

telecharger_drive(FILE_ID_M2, f"{MODELES_DIR}/modele_module2.pkl")

# ── 3. Charger les modèles ─────────────────────────────────
M1 = M2 = M3 = None
for var, path in [("M1", f"{MODELES_DIR}/modele_module1.pkl"),
                  ("M2", f"{MODELES_DIR}/modele_module2.pkl"),
                  ("M3", f"{MODELES_DIR}/modele_module3.pkl")]:
    try:
        with open(path, "rb") as f:
            globals()[var] = pickle.load(f)
        print(f"✅ {var} chargé")
    except Exception as e:
        print(f"⚠️ {var} non chargé : {e}")

FEATURES = ["MolWt","LogP","NumHDonors","NumHAcceptors","TPSA",
            "NumRotatableBonds","NumAromaticRings","NumHeavyAtoms",
            "FractionCSP3","NumAliphaticRings","RingCount","MolMR"]

# ── 4. Fonctions utilitaires ───────────────────────────────
def get_smiles_pubchem(nom):
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(nom)}/property/ConnectivitySMILES,MolecularFormula,MolecularWeight/JSON",
            timeout=10)
        if r.status_code == 200:
            p = r.json()["PropertyTable"]["Properties"][0]
            return {
                "smiles" : p.get("ConnectivitySMILES", p.get("CanonicalSMILES", p.get("IsomericSMILES",""))),
                "iupac"  : nom,
                "mw"     : p.get("MolecularWeight", 0),
                "formula": p.get("MolecularFormula", ""),
            }
    except: pass
    return None

def calc_desc(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return pd.DataFrame([{
            "MolWt": Descriptors.MolWt(mol), "LogP": Crippen.MolLogP(mol),
            "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "MolMR": Crippen.MolMR(mol),
        }])
    except: return None

def predire(smiles):
    X = calc_desc(smiles)
    if X is None: raise HTTPException(400, "SMILES invalide")
    sol_log    = float(M1["model"].predict(X)[0]) if M1 else -2.5
    absorption = round(min(abs(10**sol_log)*100, 100), 2)
    energie    = round(sol_log * -2.303 * 0.592, 2)
    logD       = round(float(X["LogP"].values[0]), 3)
    tox_proba  = float(M2["model"].predict_proba(X)[0][1]) if M2 else 0.05
    tox_pct    = round(tox_proba * 100, 1)
    biodeg_p   = float(M3["model_biodeg"].predict_proba(X)[0][1]) if M3 else 0.85
    logbcf     = float(M3["model_bcf"].predict(X)[0]) if M3 else 1.2
    biodeg_pct = round(biodeg_p * 100, 1)
    jours      = 90 if biodeg_pct>80 else 180 if biodeg_pct>50 else 365 if biodeg_pct>30 else 1000
    score      = round((1-tox_proba)*40 + biodeg_p*40 + (1-min(logbcf/6,1))*20, 1)
    return {
        "module1_eau": {"solubility_log": round(sol_log,3), "absorption_eau_pct": absorption,
                        "energie_solvatation": energie, "logD": logD},
        "module2_toxicite": {"toxicite_pct": tox_pct,
                             "verdict": "Non toxique" if tox_pct<30 else "Toxique",
                             "est_toxique": tox_pct>=30},
        "module3_biodeg": {"biodegradabilite_pct": biodeg_pct,
                           "verdict": "Biodégradable" if biodeg_pct>=50 else "Non biodégradable",
                           "est_biodegradable": biodeg_pct>=50,
                           "logBCF": round(logbcf,3),
                           "verdict_bcf": "Faible" if logbcf<3 else "Élevé",
                           "degradation_jours": jours},
        "score_securite_global": score,
        "descripteurs": {"poids_moleculaire": round(float(X["MolWt"].values[0]),2),
                         "LogP": round(float(X["LogP"].values[0]),3),
                         "TPSA": round(float(X["TPSA"].values[0]),2)}
    }

# ── 5. Modèle de requête ───────────────────────────────────
class PolymerRequest(BaseModel):
    nom: str
    smiles: str = None

# ── 6. Endpoints ───────────────────────────────────────────
@app.get("/")
def accueil():
    return {"message": "BioPolymer-AI API v1.0", "status": "active",
            "base_polymeres": f"{len(BASE_POLYMERES)} entrées",
            "modeles": {"M1": "✅" if M1 else "❌", "M2": "✅" if M2 else "❌", "M3": "✅" if M3 else "❌"}}

@app.get("/health")
def health():
    return {"status": "ok", "modeles": {
        "module1": f"R2={M1['r2']:.3f}" if M1 else "non chargé",
        "module2": f"AUC={M2['auc']:.3f}" if M2 else "non chargé",
        "module3": f"AUC={M3['auc_biodeg']:.3f}" if M3 else "non chargé"}}

@app.get("/smiles")
def get_smiles(nom: str):
    data = get_smiles_pubchem(nom)
    if not data: raise HTTPException(404, f"'{nom}' introuvable")
    return data

@app.post("/predict")
def predict(request: PolymerRequest):
    # Chercher d'abord dans la base locale
    nom_lower    = request.nom.strip().lower()
    smiles_local = BASE_POLYMERES.get(nom_lower)

    if request.smiles:
        smiles = request.smiles
        info   = {"iupac": request.nom, "formula": "", "mw": 0}
    elif smiles_local:
        smiles = smiles_local
        info   = {"iupac": request.nom, "formula": "", "mw": 0}
        print(f"✅ '{request.nom}' trouvé dans la base locale")
    else:
        info = get_smiles_pubchem(request.nom)
        if not info: raise HTTPException(404, f"'{request.nom}' introuvable.")
        smiles = info["smiles"]

    return {
        "polymere": {"nom": request.nom, "smiles": smiles,
                     "iupac": info.get("iupac", request.nom),
                     "formule": info.get("formula",""), "poids_mol": info.get("mw",0)},
        "predictions": predire(smiles)
    }
