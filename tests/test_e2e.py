import pytest
from unittest.mock import Mock

def test_health_endpoint(client):
    """Testa endpoint de health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["merchants_loaded"] >= 0
    assert data["data_insights_loaded"] is True

def test_process_message_endpoint_logistics(client, orchestrator_components):
    """Testa endpoint completo para mensagem de logística"""
    mock_llm = orchestrator_components["mock_llm"]
    
    # Configurar sequência de respostas
    responses = [
        Mock(content='{"intent": "logistics_sharing", "confidence": 0.8}'),
        Mock(content='{"action": "approve", "flags": [], "suggestions": []}'),
        Mock(content='[{"merchant_id": "4", "reason": "mesma cidade", "score": 0.85}]'),
        Mock(content="Encontramos 1 parceiro em Santos para dividir frete!")
    ]
    mock_llm.invoke.side_effect = responses
    
    response = client.post("/process_message", json={
        "message": "Quero dividir frete para entregas em Santos",
        "user_id": "6"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["intent"] == "logistics_sharing"
    assert data["intent_confidence"] == 0.8
    assert len(data["matches"]) == 1
    assert data["matches"][0]["merchant_id"] == "4"
    assert "Encontramos 1 parceiro" in data["response"]
    assert data["source_agent_response"] == "LLM generated response"

def test_process_message_endpoint_off_topic(client, orchestrator_components):
    """Testa endpoint para mensagem off-topic"""
    mock_llm = orchestrator_components["mock_llm"]
    
    router_response = Mock()
    router_response.content = '{"intent": "off_topic", "confidence": 1.0}'
    mock_llm.invoke.return_value = router_response
    
    response = client.post("/process_message", json={
        "message": "Qual foi o último jogo do Palmeiras?",
        "user_id": "456"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["intent"] == "off_topic"
    assert data["chosen_agent"] == "FallbackAgent"
    assert "não trata de parcerias" in data["response"]

def test_get_merchant_profile_endpoint(client):
    """Testa endpoint de perfil do merchant"""
    response = client.get("/merchants/6")
    assert response.status_code == 200
    data = response.json()
    
    assert data["merchant_id"] == "6"
    assert data["city"] == "Santos"
    assert data["business_category"] == "Toy Shops"
    assert "logistics" in data["needs"]

def test_get_merchant_profile_not_found(client):
    """Testa endpoint com merchant inexistente"""
    response = client.get("/merchants/999")
    assert response.status_code == 404

def test_process_message_with_moderation_flags(client, orchestrator_components):
    """Testa mensagem que é flagged pela moderação"""
    mock_llm = orchestrator_components["mock_llm"]
    
    responses = [
        Mock(content='{"intent": "general_inquiry", "confidence": 0.6}'),
        Mock(content='{"action": "improve", "flags": ["too_vague_question"], "suggestions": ["Seja mais específico"]}'),
        Mock(content="Por favor, seja mais específico sobre o que você precisa.")
    ]
    mock_llm.invoke.side_effect = responses
    
    response = client.post("/process_message", json={
        "message": "alguém?",
        "user_id": "5"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert "too_vague_question" in data["moderation_flags"]
    assert len(data["moderation_suggestions"]) > 0
    assert "seja mais específico" in data["response"]