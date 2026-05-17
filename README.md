# BioPolymer-AI API

API FastAPI pour la prédiction des propriétés des polymères biosourcés.

## Structure
```
BioPolymer_API_Render/
├── main.py           ← Code de l'API
├── requirements.txt  ← Dépendances Python
├── render.yaml       ← Configuration Render
└── modeles/          ← Dossier pour vos fichiers .pkl
    ├── modele_module1.pkl
    ├── modele_module2.pkl
    └── modele_module3.pkl
```

## Déploiement sur Render.com

1. Uploadez ce dossier sur GitHub
2. Connectez Render à votre repo GitHub
3. Render déploie automatiquement !

## Endpoints

- `GET /` → Vérification API
- `GET /health` → État des modèles
- `GET /smiles?nom=lactic+acid` → Récupérer SMILES
- `POST /predict` → Prédiction complète
