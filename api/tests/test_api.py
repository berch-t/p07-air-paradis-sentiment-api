def test_health_endpoint(client):
    """Teste l'endpoint de vérification de santé."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(client):
    """Teste l'endpoint de prédiction de sentiment."""
    # Requête de test
    test_tweet = "I love flying with Air Paradis! Great service and friendly staff."
    
    # Envoi de la requête POST
    response = client.post(
        "/predict", 
        json={"text": test_tweet}
    )
    
    # Vérification de la réponse
    assert response.status_code == 200
    
    # Vérification du contenu de la réponse
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result
    assert "raw_score" in result
    
    # Vérification des types des valeurs
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["raw_score"], float)
    
    # Le sentiment devrait être soit "Positif" soit "Négatif"
    assert result["sentiment"] in ["Positif", "Négatif"]
    
    # La confiance devrait être entre 0 et 1
    assert 0 <= result["confidence"] <= 1
    
    # Le score brut devrait être entre 0 et 1
    assert 0 <= result["raw_score"] <= 1


def test_info_endpoint(client):
    """Teste l'endpoint d'informations sur l'environnement."""
    response = client.get("/info")
    assert response.status_code == 200
    
    # Vérification des clés attendues
    result = response.json()
    assert "tensorflow_version" in result
    assert "transformers_version" in result
    assert "devices_available" in result
    assert "using_gpu" in result
    assert "environment" in result