import sys
import os
import pytest
from fastapi.testclient import TestClient
import tensorflow as tf
from pathlib import Path

# Forcer l'utilisation du CPU pour les tests
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Configuration pour les tests
os.environ["FORCE_CPU"] = "true"
os.environ["MAX_SEQUENCE_LENGTH"] = "64"
os.environ["SIMULATION_MODE"] = "false"  # Utiliser le vrai modèle

# S'assurer que le chemin du modèle est correct
MODEL_DIR = Path(__file__).parent.parent / "model"
os.environ["MODEL_DIR"] = str(MODEL_DIR)

# Ajout du répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

@pytest.fixture
def client():
    """Fixture fournissant un client de test pour l'application FastAPI."""
    with TestClient(app) as c:
        yield c