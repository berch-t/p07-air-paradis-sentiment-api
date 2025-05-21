# Importation des bibliothèques nécessaires
import os
import json
import tempfile
import shutil
import uuid
import transformers
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
import mlflow
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import requests
from typing import Dict, List, Any
import traceback

# Importation des bibliothèques spécifiques à BERT
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Forcer TensorFlow à utiliser uniquement le CPU si nécessaire
FORCE_CPU = os.getenv("FORCE_CPU", "true").lower() == "true"
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Désactiver l'allocation de mémoire GPU
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, False)
            # Masquer complètement les GPUs
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU désactivé avec succès pour TensorFlow.")
        except Exception as e:
            logger.warning(f"Erreur lors de la désactivation du GPU: {str(e)}")

# Chargement des variables d'environnement
load_dotenv()

# Configuration de MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
run_id = os.getenv("RUN_ID")

# Paramètres pour BERT
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "64"))

# Répertoire local pour sauvegarder les artefacts du modèle
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Variable globale pour stocker le modèle chargé
model_pack = None

# Configuration d'Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler
from applicationinsights import TelemetryClient
from datetime import datetime

# Récupérer la clé d'instrumentation depuis les variables d'environnement
appinsights_key = os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY")
telemetry_client = None

if appinsights_key:
    # Configurer le client Application Insights
    telemetry_client = TelemetryClient(appinsights_key)
    logger.info("Azure Application Insights configuré avec succès.")
    
    try:
        # Ajouter un gestionnaire Azure Log Handler
        logger.addHandler(AzureLogHandler(
            connection_string=f'InstrumentationKey={appinsights_key}'
        ))
        logger.info("Azure Application Insights configuré avec succès.")
    except Exception as e:
        logger.warning(f"Erreur lors de la configuration d'Azure Log Handler: {str(e)}")
else:
    logger.warning("Clé d'instrumentation Application Insights non trouvée. La télémétrie ne sera pas envoyée.")

# Définition du gestionnaire de contexte pour le cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    global model_pack
    
    # Télécharger les artefacts si nécessaire
    try:
        success = download_artifacts_from_mlflow(run_id, MODEL_DIR)
        if not success:
            logger.warning("Impossible de télécharger les artefacts du modèle depuis MLflow.")
            # Tentative de chargement direct depuis le dossier local
            if (MODEL_DIR / "model").exists() and (MODEL_DIR / "tokenizer").exists():
                logger.info("Tentative de chargement du modèle depuis le dossier local...")
                model_pack = load_model_from_local()
                if model_pack:
                    logger.info("Modèle chargé avec succès depuis le dossier local.")
                else:
                    logger.error("Échec du chargement du modèle depuis le dossier local.")
        else:
            # Charger le modèle
            model_pack = load_model_from_mlflow()
            logger.info("Modèle BERT chargé avec succès depuis MLflow et prêt pour les prédictions.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        logger.error(traceback.format_exc())
        # Tentative de chargement en mode simulation
        if os.getenv("SIMULATION_MODE", "false").lower() == "true":
            logger.warning("Mode simulation activé. Un modèle fictif sera utilisé.")
            model_pack = create_dummy_model()
    
    yield  # L'application s'exécute ici
    
    # Code exécuté à l'arrêt
    # Libérer les ressources
    if model_pack is not None and "model" in model_pack:
        logger.info("Libération des ressources du modèle...")
        del model_pack["model"]
        model_pack = None

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de prédiction de sentiment avec BERT",
    description="API pour prédire le sentiment (positif/négatif) d'un tweet en utilisant le modèle BERT fine-tuné",
    version="1.0.0",
    lifespan=lifespan
)

# Support CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines en développement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de données pour les requêtes
class TweetRequest(BaseModel):
    text: str
    model_config = ConfigDict(extra="forbid")

# Modèle de données pour une requête de lot
class BatchTweetRequest(BaseModel):
    texts: List[str]
    model_config = ConfigDict(extra="forbid")

# Modèle de données pour une prédiction individuelle
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    raw_score: float
    model_config = ConfigDict(extra="forbid")

# Modèle de données pour la réponse par lot
class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    model_config = ConfigDict(extra="forbid")

# Modèle de données pour le feedback utilisateur
class FeedbackRequest(BaseModel):
    tweet_text: str
    prediction: str
    confidence: float
    is_correct: bool
    corrected_sentiment: str = ""
    comments: str = ""
    model_config = ConfigDict(extra="forbid")

# Fonction pour télécharger les artefacts depuis MLflow
def download_artifacts_from_mlflow(run_id: str, model_dir: Path) -> bool:
    """
    Télécharge les artefacts BERT nécessaires depuis MLflow.
    """
    if not run_id or not mlflow_tracking_uri:
        logger.warning("Configuration MLflow incomplète. Impossible de télécharger les artefacts.")
        return False
        
    try:
        logger.info(f"Configuration de MLflow avec l'URI: {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Noms des artefacts pour le modèle BERT
        model_path = model_dir / "model"
        tokenizer_path = model_dir / "tokenizer"
        
        # Vérifier si les artefacts existent déjà
        if model_path.exists() and tokenizer_path.exists():
            logger.info("Artefacts BERT déjà présents localement.")
            return True
            
        # Créer des sous-répertoires pour le modèle et le tokenizer
        model_path.mkdir(exist_ok=True)
        tokenizer_path.mkdir(exist_ok=True)
        
        # Télécharger les artefacts
        logger.info(f"Téléchargement des artefacts BERT depuis MLflow pour l'exécution {run_id}")
        
        try:
            # Utiliser l'API MLflow pour télécharger les artefacts
            client = mlflow.tracking.MlflowClient()
            artifact_uri = client.get_run(run_id).info.artifact_uri
            
            # Télécharger le modèle et le tokenizer
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/model", 
                dst_path=str(model_dir)
            )
            
            logger.info("Artefacts BERT téléchargés avec succès.")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des artefacts via MLflow: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des artefacts: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Fonction pour créer un modèle fictif en mode simulation
def create_dummy_model():
    """Crée un modèle fictif pour le mode simulation."""
    logger.info("Création d'un modèle fictif pour le mode simulation.")
    
    class DummyModel:
        def predict(self, inputs):
            # Générer des prédictions aléatoires mais biaisées vers le positif (60/40)
            batch_size = 1 if isinstance(inputs, dict) else len(inputs["input_ids"])
            return {"logits": tf.random.normal([batch_size, 2], mean=[0.4, 0.6], stddev=0.1)}
    
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            # Simuler le comportement d'un tokenizer
            if isinstance(texts, str):
                texts = [texts]
            
            batch_size = len(texts)
            return {
                "input_ids": tf.ones([batch_size, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
                "attention_mask": tf.ones([batch_size, MAX_SEQUENCE_LENGTH], dtype=tf.int32)
            }
    
    return {
        "model": DummyModel(),
        "tokenizer": DummyTokenizer(),
        "max_sequence_length": MAX_SEQUENCE_LENGTH
    }

# Fonction pour charger le modèle BERT depuis MLflow
def load_model_from_mlflow():
    """
    Charge le modèle BERT et le tokenizer depuis les artefacts téléchargés de MLflow.
    """
    try:
        logger.info("Chargement du modèle BERT depuis les artefacts MLflow...")
        
        # Chemins vers les artefacts
        model_path = str(MODEL_DIR / "model")
        tokenizer_path = str(MODEL_DIR / "tokenizer")
        
        # Charger le modèle BERT
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Charger le tokenizer BERT
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info("Modèle BERT et tokenizer chargés avec succès.")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "max_sequence_length": MAX_SEQUENCE_LENGTH
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle BERT: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Fonction pour charger le modèle BERT depuis une source distante
def load_model_from_huggingface():
    """
    Charge un modèle DistilBERT pré-entraîné directement depuis Hugging Face.
    Utilise cette fonction comme fallback si le modèle MLflow n'est pas disponible.
    """
    try:
        logger.info("Tentative de chargement d'un modèle DistilBERT pré-entraîné depuis Hugging Face...")
        
        # Charger un modèle pré-entraîné pour la classification de sentiments
        model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english',
            cache_dir='./hf_cache'
        )
        
        # Charger le tokenizer correspondant
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english',
            cache_dir='./hf_cache'
        )
        
        logger.info("Modèle DistilBERT pré-entraîné chargé avec succès depuis Hugging Face.")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "max_sequence_length": MAX_SEQUENCE_LENGTH
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle depuis Hugging Face: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Fonction pour charger le modèle directement depuis le dossier local (fallback)
def load_model_from_local():
    """
    Charge le modèle BERT directement depuis le dossier local si les artefacts sont déjà présents.
    """
    try:
        logger.info("Chargement du modèle BERT depuis le dossier local...")
        
        # Vérifier si les répertoires model et tokenizer existent
        model_path = MODEL_DIR / "model"
        tokenizer_path = MODEL_DIR / "tokenizer"
        
        if not model_path.exists() or not tokenizer_path.exists():
            logger.error("Les dossiers model et/ou tokenizer n'existent pas localement.")
            return None
        
        # Charger le modèle BERT
        model = TFDistilBertForSequenceClassification.from_pretrained(str(model_path))
        
        # Charger le tokenizer BERT
        tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
        
        logger.info("Modèle BERT et tokenizer chargés avec succès depuis le dossier local.")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "max_sequence_length": MAX_SEQUENCE_LENGTH
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement local du modèle BERT: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Fonction pour prédire le sentiment avec BERT
def predict_with_bert(texts: List[str], model_pack: Dict) -> List[Dict[str, Any]]:
    """
    Prédit le sentiment d'une liste de textes en utilisant le modèle BERT.
    
    Args:
        texts: Liste de textes à analyser
        model_pack: Dictionnaire contenant le modèle et le tokenizer
    
    Returns:
        Liste de dictionnaires contenant les sentiments prédits et les scores
    """
    try:
        model = model_pack["model"]
        tokenizer = model_pack["tokenizer"]
        max_length = model_pack["max_sequence_length"]
        
        # Tokenisation des textes pour BERT
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        
        # Prédiction avec le modèle BERT
        outputs = model(inputs)
        logits = outputs.logits
        
        # Application de softmax pour obtenir les probabilités
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        
        # Interpréter les résultats (en supposant que l'index 1 correspond au sentiment positif)
        results = []
        for prob in probabilities:
            # BERT peut avoir une sortie différente selon l'entraînement
            # Adapté ici pour classer en Positif/Négatif
            sentiment = "Positif" if prob[1] >= 0.5 else "Négatif"
            confidence = float(prob[1]) if sentiment == "Positif" else float(prob[0])
            
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'raw_score': float(prob[1])  # Score brut pour le sentiment positif
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction avec BERT: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Définition des routes de l'API

@app.get("/")
async def root():
    """Endpoint racine."""
    return {"message": "API de prédiction de sentiment pour tweets avec BERT", "status": "opérationnel"}

@app.get("/health")
async def health_check():
    """Endpoint de vérification de santé."""
    if model_pack is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "erreur",
                "message": "Le modèle BERT n'est pas chargé correctement."
            }
        )
    return {"status": "ok", "message": "Le modèle BERT est chargé et prêt pour les prédictions."}

@app.get("/info")
async def get_info():
    """Endpoint pour obtenir des informations sur l'environnement d'exécution."""
    return {
        "tensorflow_version": tf.__version__,
        "transformers_version": transformers.__version__, 
        "devices_available": [device.name for device in tf.config.list_logical_devices()],
        "using_gpu": not FORCE_CPU and len(tf.config.list_physical_devices('GPU')) > 0,
        "environment": {
            "MAX_SEQUENCE_LENGTH": MAX_SEQUENCE_LENGTH,
            "FORCE_CPU": FORCE_CPU
        }
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TweetRequest):
    """
    Endpoint de prédiction du sentiment d'un tweet unique avec BERT.
    """
    global model_pack
    
    if model_pack is None:
        raise HTTPException(status_code=503, detail="Le modèle BERT n'est pas encore chargé. Veuillez réessayer plus tard.")
    
    try:
        # Utiliser la fonction de prédiction avec BERT
        results = predict_with_bert([request.text], model_pack)
        # Renvoyer seulement le premier résultat
        return SentimentResponse(**results[0])
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/predict-batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchTweetRequest):
    """
    Endpoint de prédiction du sentiment pour un lot de tweets avec BERT.
    """
    global model_pack
    
    if model_pack is None:
        raise HTTPException(status_code=503, detail="Le modèle BERT n'est pas encore chargé. Veuillez réessayer plus tard.")
    
    if not request.texts:
        return BatchSentimentResponse(results=[])
    
    try:
        # Utiliser la fonction de prédiction avec BERT
        results = predict_with_bert(request.texts, model_pack)
        return BatchSentimentResponse(results=[SentimentResponse(**result) for result in results])
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction par lot: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction par lot: {str(e)}")
    
@app.post("/feedback")
async def record_feedback(feedback: FeedbackRequest):
    """
    Endpoint pour enregistrer le feedback utilisateur sur les prédictions.
    """
    try:
        # Log details
        logger.info(f"Feedback reçu: {feedback.dict()}")
        
        # Enregistrer dans Application Insights si configuré
        if telemetry_client:
            # Construire les propriétés en incluant seulement les champs non vides
            properties = {
                "tweet": feedback.tweet_text,
                "prediction": feedback.prediction,
                "confidence": str(feedback.confidence),
                "is_correct": str(feedback.is_correct)
            }
            
            # Ajouter les propriétés optionnelles si elles sont présentes
            if feedback.corrected_sentiment:
                properties["corrected_sentiment"] = feedback.corrected_sentiment
            
            if feedback.comments:
                properties["comments"] = feedback.comments
            
            # Enregistrer l'événement de feedback
            telemetry_client.track_event(
                name="model_feedback",
                properties=properties
            )
            
            # Envoyer les données immédiatement
            telemetry_client.flush()
            
        return {"status": "success", "message": "Feedback enregistré avec succès"}
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du feedback: {str(e)}")

@app.get("/test-appinsights")
async def test_appinsights_connection():
    """
    Endpoint pour tester la connexion à Azure Application Insights.
    """
    if not telemetry_client:
        return {
            "status": "error",
            "message": "Aucun client Application Insights n'est configuré. Vérifiez la variable d'environnement APPINSIGHTS_INSTRUMENTATION_KEY."
        }
    
    try:
        # Envoyer un événement de test
        test_id = str(uuid.uuid4())
        telemetry_client.track_event(
            name="appinsights_connection_test",
            properties={
                "timestamp": datetime.now().isoformat(),
                "test_id": test_id
            }
        )
        telemetry_client.flush()
        
        return {
            "status": "success",
            "message": "Événement de test envoyé à Application Insights. Vérifiez le portail Azure pour confirmer la réception.",
            "details": {
                "test_id": test_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors du test de connexion à Application Insights: {str(e)}",
            "details": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

# Point d'entrée pour exécuter l'application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))