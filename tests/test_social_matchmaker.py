import pytest
import json
from unittest.mock import Mock
from scr.merchant_social_agents import SocialMatchmaker, AgentState

def test_matchmaker_finds_matches(mock_llm, mock_data_processor):
    """Testa se o matchmaker encontra correspondências"""
    # Configurar resposta do LLM com matches
    mock_response = Mock()
    mock_response.content = '''[
        {"merchant_id": "4", "reason": "mesma cidade e área", "score": 0.85},
        {"merchant_id": "9", "reason": "serviços complementares", "score": 0.7}
    ]'''
    mock_llm.invoke.return_value = mock_response
    
    matcher = SocialMatchmaker(mock_llm, mock_data_processor)
    state: AgentState = {
        "user_id": "6",
        "message": "preciso de parceiros de frete",
        "user_profile": {"city": "Santos", "mcc": 5945},
        "intent": "logistics_sharing",
        "agent_workflow": []
    }
    
    result = matcher.find_matches(state)
    
    assert len(result["matches"]) == 2
    assert result["matches"][0]["merchant_id"] == "4"
    assert result["matches"][0]["score"] == 0.85
    assert result["agent_workflow"][-1]["agent_name"] == "SocialMatchmaker"

def test_matchmaker_no_matches(mock_llm, mock_data_processor):
    """Testa quando não há matches"""
    mock_response = Mock()
    mock_response.content = '[]'
    mock_llm.invoke.return_value = mock_response
    
    matcher = SocialMatchmaker(mock_llm, mock_data_processor)
    state: AgentState = {
        "user_id": "123",
        "message": "procuro algo muito específico",
        "user_profile": {"city": "São Paulo", "mcc": 5814},
        "intent": "supplier_search",
        "agent_workflow": []
    }
    
    result = matcher.find_matches(state)
    
    assert len(result["matches"]) == 0
    assert result["agent_workflow"][-1]["agent_name"] == "SocialMatchmaker"