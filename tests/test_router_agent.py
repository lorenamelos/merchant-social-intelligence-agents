# ===== tests/test_router_agent.py =====
import pytest
import json
from unittest.mock import Mock
from scr.merchant_social_agents import RouterAgent, AgentState

def test_router_classification_logistics(mock_llm):
    """Testa classificação de intent de logística"""
    # Configurar resposta específica do mock
    mock_response = Mock()
    mock_response.content = '{"intent": "logistics_sharing", "confidence": 0.8}'
    mock_llm.invoke.return_value = mock_response
    
    router = RouterAgent(mock_llm)
    state: AgentState = {
        "message": "preciso de parceiros de frete",
        "user_profile": {"city": "Santos", "mcc": 5945},
        "agent_workflow": []
    }
    
    result = router.classify_intent(state)
    
    assert result["intent"] == "logistics_sharing"
    assert result["intent_confidence"] == 0.8
    assert len(result["agent_workflow"]) == 1
    assert result["agent_workflow"][0]["agent_name"] == "RouterAgent"

def test_router_classification_off_topic(mock_llm):
    """Testa classificação de mensagem off-topic"""
    mock_response = Mock()
    mock_response.content = '{"intent": "off_topic", "confidence": 1.0}'
    mock_llm.invoke.return_value = mock_response
    
    router = RouterAgent(mock_llm)
    state: AgentState = {
        "message": "Qual foi o último jogo do Palmeiras?",
        "user_profile": {"city": "São Paulo", "mcc": 0},
        "agent_workflow": []
    }
    
    result = router.classify_intent(state)
    
    assert result["intent"] == "off_topic"
    assert result["intent_confidence"] == 1.0

def test_router_classification_low_quality(mock_llm):
    """Testa classificação de mensagem de baixa qualidade"""
    mock_response = Mock()
    mock_response.content = '{"intent": "low_quality", "confidence": 0.9}'
    mock_llm.invoke.return_value = mock_response
    
    router = RouterAgent(mock_llm)
    state: AgentState = {
        "message": "alguém?",
        "user_profile": {"city": "Santos", "mcc": 5945},
        "agent_workflow": []
    }
    
    result = router.classify_intent(state)
    
    assert result["intent"] == "low_quality"
    assert result["intent_confidence"] == 0.9