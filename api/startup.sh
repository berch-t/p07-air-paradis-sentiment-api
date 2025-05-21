#!/bin/bash
cd `dirname $0`

# Définir la variable PORT pour Azure
export PORT=8000

# Activer le mode simulation si nécessaire
export SIMULATION_MODE=true

# Installer les dépendances
pip install -r requirements.txt

# Démarrer l'application
gunicorn main:app --bind=0.0.0.0:$PORT --timeout 600 --workers 2 --worker-class uvicorn.workers.UvicornWorker