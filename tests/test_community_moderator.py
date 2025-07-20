import pytest
import json
from unittest.mock import Mock
from scr.merchant_social_agents import CommunityModerator, AgentState

def test_moderator_approves_good_message(mock_llm):
    """Testa aprovação de mensagem de boa qualidade"""
    mock_response = Mock()
    mock_response.content = '{"action": "approve", "flags": [], "suggestions": []}'
    mock_llm.invoke.return_value = mock_response
    
    moderator = CommunityModerator(mock_llm)
    state: AgentState = {
        "message": "Sou dono de uma loja de brinquedos em Santos e estou procurando parceiros para dividir custos de frete.",
        "user_id": "6",
        "agent_workflow": []
    }
    
    result = moderator.moderate_message(state)
    
    assert result["moderation_flags"] == []
    assert result["moderation_suggestions"] == []
    assert result["agent_workflow"][-1]["agent_name"] == "CommunityModerator"

def test_moderator_flags_poor_quality(mock_llm):
    """Testa detecção de mensagem de baixa qualidade"""
    mock_response = Mock()
    mock_response.content = '''{"action": "improve", "flags": ["too_vague_question"], "suggestions": ["Complete sua pergunta com mais detalhes sobre o que você precisa"]}'''
    mock_llm.invoke.return_value = mock_response
    
    moderator = CommunityModerator(mock_llm)
    state: AgentState = {
        "message": "alguém?",
        "user_id": "5",
        "agent_workflow": []
    }
    
    result = moderator.moderate_message(state)
    
    assert "too_vague_question" in result["moderation_flags"]
    assert len(result["moderation_suggestions"]) > 0
    assert "Complete sua pergunta" in result["moderation_suggestions"][0]

def test_moderator_block_request(mock_llm):
    """Testa detecção de pedido de bloqueio"""
    mock_response = Mock()
    mock_response.content = '''{"action": "review", "flags": ["user_block_request"], "suggestions": ["Para bloquear notificações, acesse configurações do seu perfil"]}'''
    mock_llm.invoke.return_value = mock_response
    
    moderator = CommunityModerator(mock_llm)
    state: AgentState = {
        "message": "Eu tô cansado de receber pedido de negócios. Tem como bloquear isso?",
        "user_id": "134",
        "agent_workflow": []
    }
    
    result = moderator.moderate_message(state)
    
    assert "user_block_request" in result["moderation_flags"]
    assert any("bloquear" in suggestion for suggestion in result["moderation_suggestions"])