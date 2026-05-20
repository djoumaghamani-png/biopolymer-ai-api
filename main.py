import pickle
import os
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

RDLogger.DisableLog("rdApp.*")

app = FastAPI(
    title="BioPolymer-AI API",
    description="Prédiction IA des propriétés des polymères biosourcés",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Téléchargement modele_module2.pkl depuis Google Drive ──
MODELES_DIR  = "modeles"
FILE_ID_M2   = "1mnm6gLpnhWMJo5-LceaL66CizIVDO5Z1"
PATH_M2      = f"{MODELES_DIR}/modele_module2.pkl"

def telecharger_depuis_drive(file_id: str, destination: str):
    """Télécharge un fichier depuis Google Drive."""
    print(f"📥 Téléchargement de {destination} depuis Google Drive...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    # Première requête pour obtenir le token de confirmation
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # Si token trouvé, relancer avec confirmation
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    # Sauvegarder le fichier
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    taille = os.path.getsize(destination) / (1024*1024)
    print(f"✅ Téléchargé : {destination} ({taille:.1f} MB)")

# ── Chargement des modèles ─────────────────────────────────
M1 = M2 = M3 = None

def charger_modeles():
    global M1, M2, M3
    try:
        # Module 1
        with open(f"{MODELES_DIR}/modele_module1.pkl", "rb") as f:
            M1 = pickle.load(f)
        print(f"✅ Module 1 chargé — R²={M1['r2']:.3f}")

        # Module 2 — télécharger depuis Drive si absent
        if not os.path.exists(PATH_M2):
            telecharger_depuis_drive(FILE_ID_M2, PATH_M2)
        with open(PATH_M2, "rb") as f:
            M2 = pickle.load(f)
        print(f"✅ Module 2 chargé — AUC={M2['auc']:.3f}")

        # Module 3
        with open(f"{MODELES_DIR}/modele_module3.pkl", "rb") as f:
            M3 = pickle.load(f)
        print(f"✅ Module 3 chargé — AUC={M3['auc_biodeg']:.3f}")

        print("🎉 Tous les modèles sont prêts !")

    except Exception as e:
        print(f"⚠️ Erreur chargement modèles : {e}")

charger_modeles()

FEATURES = [
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA",
    "NumRotatableBonds", "NumAromaticRings", "NumHeavyAtoms",
    "FractionCSP3", "NumAliphaticRings", "RingCount", "MolMR"
]

# ── Fonctions utilitaires ──────────────────────────────────
def get_smiles_pubchem(nom: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{nom}/property/IsomericSMILES,IUPACName,MolecularWeight,MolecularFormula/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            return {
                "smiles"  : props.get("IsomericSMILES", ""),
                "iupac"   : props.get("IUPACName", nom),
                "mw"      : props.get("MolecularWeight", 0),
                "formula" : props.get("MolecularFormula", ""),
            }
    except:
        pass
    return None

def calc_descripteurs(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return pd.DataFrame([{
            "MolWt"             : Descriptors.MolWt(mol),
            "LogP"              : Crippen.MolLogP(mol),
            "NumHDonors"        : rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors"     : rdMolDescriptors.CalcNumHBA(mol),
            "TPSA"              : Descriptors.TPSA(mol),
            "NumRotatableBonds" : rdMolDescriptors.CalcNumRotatableBonds(mol),
            "NumAromaticRings"  : rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumHeavyAtoms"     : mol.GetNumHeavyAtoms(),
            "FractionCSP3"      : rdMolDescriptors.CalcFractionCSP3(mol),
            "NumAliphaticRings" : rdMolDescriptors.CalcNumAliphaticRings(mol),
            "RingCount"         : rdMolDescriptors.CalcNumRings(mol),
            "MolMR"             : Crippen.MolMR(mol),
        }])
    except:
        return None

def predire(smiles: str):
    X = calc_descripteurs(smiles)
    if X is None:
        raise HTTPException(status_code=400, detail="SMILES invalide")

    # Module 1 — Solubilité
    if M1:
        sol_log    = float(M1["model"].predict(X)[0])
        absorption = round(min(abs(10**sol_log) * 100, 100), 2)
        energie    = round(sol_log * -2.303 * 0.592, 2)
        logD       = round(float(X["LogP"].values[0]), 3)
    else:
        sol_log, absorption, energie, logD = -2.5, 0.5, -3.4, -0.8

    # Module 2 — Toxicité
    if M2:
        tox_proba = float(M2["model"].predict_proba(X)[0][1])
    else:
        tox_proba = 0.05
    tox_pct = round(tox_proba * 100, 1)

    # Module 3 — Biodégradabilité + BCF
    if M3:
        biodeg_proba = float(M3["model_biodeg"].predict_proba(X)[0][1])
        logbcf       = float(M3["model_bcf"].predict(X)[0])
    else:
        biodeg_proba, logbcf = 0.85, 1.2
    biodeg_pct = round(biodeg_proba * 100, 1)

    if biodeg_pct > 80:   jours = 90
    elif biodeg_pct > 50: jours = 180
    elif biodeg_pct > 30: jours = 365
    else:                  jours = 1000

    score = round((1 - tox_proba) * 40 + biodeg_proba * 40 + (1 - min(logbcf/6, 1)) * 20, 1)

    return {
        "module1_eau": {
            "solubility_log"     : round(sol_log, 3),
            "absorption_eau_pct" : absorption,
            "energie_solvatation": energie,
            "logD"               : logD,
        },
        "module2_toxicite": {
            "toxicite_pct" : tox_pct,
            "verdict"      : "Non toxique" if tox_pct < 30 else "Toxique",
            "est_toxique"  : tox_pct >= 30,
        },
        "module3_biodeg": {
            "biodegradabilite_pct" : biodeg_pct,
            "verdict"              : "Biodégradable" if biodeg_pct >= 50 else "Non biodégradable",
            "est_biodegradable"    : biodeg_pct >= 50,
            "logBCF"               : round(logbcf, 3),
            "verdict_bcf"          : "Faible" if logbcf < 3 else "Élevé",
            "degradation_jours"    : jours,
        },
        "score_securite_global": score,
        "descripteurs": {
            "poids_moleculaire" : round(float(X["MolWt"].values[0]), 2),
            "LogP"              : round(float(X["LogP"].values[0]), 3),
            "TPSA"              : round(float(X["TPSA"].values[0]), 2),
        }
    }

# ── Modèle de requête ──────────────────────────────────────
class PolymerRequest(BaseModel):
    nom: str
    smiles: str = None

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def accueil():
    return {
        "message"  : "BioPolymer-AI API v1.0",
        "status"   : "active",
        "modeles"  : {
            "module1": "chargé" if M1 else "non chargé",
            "module2": "chargé" if M2 else "non chargé",
            "module3": "chargé" if M3 else "non chargé",
        },
        "endpoints": ["/predict", "/smiles", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "modeles" : {
            "module1": f"R2={M1['r2']:.3f}" if M1 else "non chargé",
            "module2": f"AUC={M2['auc']:.3f}" if M2 else "non chargé",
            "module3": f"AUC={M3['auc_biodeg']:.3f}" if M3 else "non chargé",
        }
    }

@app.get("/smiles")
def get_smiles(nom: str):
    data = get_smiles_pubchem(nom)
    if data is None:
        raise HTTPException(status_code=404, detail=f"'{nom}' introuvable dans PubChem")
    return data

@app.post("/predict")
def predict(request: PolymerRequest):
    if request.smiles:
        smiles = request.smiles
        info   = {"nom": request.nom, "smiles": smiles, "iupac": request.nom, "formula": "", "mw": 0}
    else:
        info = get_smiles_pubchem(request.nom)
        if info is None:
            raise HTTPException(status_code=404, detail=f"'{request.nom}' introuvable. Essayez le nom du monomère.")
        smiles = info["smiles"]

    resultats = predire(smiles)
    return {
        "polymere"    : {
            "nom"      : request.nom,
            "smiles"   : smiles,
            "iupac"    : info.get("iupac", request.nom),
            "formule"  : info.get("formula", ""),
            "poids_mol": info.get("mw", 0)
        },
        "predictions" : resultats,
    }
import csv, os

def charger_base_polymeres():
    base = {}
    chemin = 'base_polymeres_biosources officiel.csv'
    if os.path.exists(chemin):
        with open(chemin, 'r') as f:
            for row in csv.DictReader(f):
                base[row['nom_recherche'].lower()] = row['smiles']
                base[row['nom_affiche'].lower()] = row['smiles']
    return base

BASE_POLYMERES = charger_base_polymeres()   
