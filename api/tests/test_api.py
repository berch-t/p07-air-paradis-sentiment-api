import pytest
from fastapi import HTTPException
import os

def test_health_endpoint(client):
    """Teste l'endpoint de vérification de santé."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data
    assert "Le modèle BERT est chargé" in data["message"]

def test_predict_endpoint_positive(client):
    """Teste l'endpoint de prédiction de sentiment avec un texte positif."""
    test_tweet = "I love flying with Air Paradis! Great service and friendly staff."
    
    response = client.post(
        "/predict", 
        json={"text": test_tweet}
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Vérification de la structure de la réponse
    assert "sentiment" in result
    assert "confidence" in result
    assert "raw_score" in result
    
    # Vérification des types
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["raw_score"], float)
    
    # Vérification des valeurs
    assert result["sentiment"] in ["Positif", "Négatif"]
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["raw_score"] <= 1
    
    # Pour un texte positif, on s'attend à une prédiction positive
    if os.getenv("SIMULATION_MODE", "false").lower() != "true":
        assert result["sentiment"] == "Positif"
        assert result["confidence"] > 0.5

def test_predict_endpoint_negative(client):
    """Teste l'endpoint de prédiction de sentiment avec un texte négatif."""
    test_tweet = "Terrible experience with Air Paradis. Delayed flight and rude staff."
    
    response = client.post(
        "/predict", 
        json={"text": test_tweet}
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Vérification de base
    assert result["sentiment"] in ["Positif", "Négatif"]
    assert 0 <= result["confidence"] <= 1
    
    # Pour un texte négatif, on s'attend à une prédiction négative
    if os.getenv("SIMULATION_MODE", "false").lower() != "true":
        assert result["sentiment"] == "Négatif"
        assert result["confidence"] > 0.5

def test_predict_endpoint_empty_text(client):
    """Teste l'endpoint de prédiction avec un texte vide."""
    response = client.post(
        "/predict", 
        json={"text": ""}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result
    assert "raw_score" in result

def test_predict_batch_endpoint(client):
    """Teste l'endpoint de prédiction par lot."""
    test_tweets = [
        "Great service!",
        "Terrible experience.",
        "It was okay."
    ]
    
    response = client.post(
        "/predict-batch", 
        json={"texts": test_tweets}
    )
    
    assert response.status_code == 200
    results = response.json()["results"]
    
    assert len(results) == len(test_tweets)
    for result in results:
        assert "sentiment" in result
        assert "confidence" in result
        assert "raw_score" in result
        assert result["sentiment"] in ["Positif", "Négatif"]
        assert 0 <= result["confidence"] <= 1

def test_info_endpoint(client):
    """Teste l'endpoint d'informations sur l'environnement."""
    response = client.get("/info")
    assert response.status_code == 200
    
    result = response.json()
    assert "tensorflow_version" in result
    assert "transformers_version" in result
    assert "devices_available" in result
    assert "using_gpu" in result
    assert "environment" in result
    
    # Vérification des valeurs de l'environnement
    env = result["environment"]
    assert "MAX_SEQUENCE_LENGTH" in env
    assert "FORCE_CPU" in env
    assert env["FORCE_CPU"] is True  # On force le CPU dans les tests

def test_feedback_endpoint(client):
    """Teste l'endpoint de feedback."""
    feedback_data = {
        "tweet_text": "Test tweet",
        "prediction": "Positif",
        "confidence": 0.8,
        "is_correct": True,
        "corrected_sentiment": "",
        "comments": "Test feedback"
    }
    
    response = client.post(
        "/feedback",
        json=feedback_data
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert "message" in result