import pytest
from unittest.mock import Mock
from scr.merchant_social_agents import ResponseGenerator, AgentState

def test_response_generator_with_matches(mock_llm):
    """Testa geração de resposta com matches encontrados"""
    mock_response = Mock()
    mock_response.content = "Encontramos 2 parceiros em Santos que podem ajudar com frete. Quer que eu faça a introdução?"
    mock_llm.invoke.return_value = mock_response
    
    generator = ResponseGenerator(mock_llm)
    state: AgentState = {
        "intent": "logistics_sharing",
        "matches": [
            {"merchant_id": "4", "reason": "mesma cidade", "score": 0.85},
            {"merchant_id": "9", "reason": "área similar", "score": 0.7}
        ],
        "moderation_flags": [],
        "agent_workflow": []
    }
    
    result = generator.generate_response(state)
    
    assert "Encontramos 2 parceiros" in result["final_response"]
    assert result["source_agent_response"] == "LLM generated response"
    assert result["agent_workflow"][-1]["agent_name"] == "ResponseGenerator"

def test_response_generator_with_moderation_flags(mock_llm):
    """Testa geração de resposta com flags de moderação"""
    mock_response = Mock()
    mock_response.content = "Por favor, seja mais específico sobre o que você precisa para que possamos ajudar melhor."
    mock_llm.invoke.return_value = mock_response
    
    generator = ResponseGenerator(mock_llm)
    state: AgentState = {
        "intent": "general_inquiry",
        "matches": [],
        "moderation_flags": ["too_vague_question"],
        "agent_workflow": []
    }
    
    result = generator.generate_response(state)
    
    assert "seja mais específico" in result["final_response"]
    assert result["agent_workflow"][-1]["agent_name"] == "ResponseGenerator"

def test_response_generator_no_matches(mock_llm):
    """Testa geração de resposta sem matches"""
    mock_response = Mock()
    mock_response.content = "Não encontramos parceiros para sua necessidade no momento. Tente reformular sua mensagem ou aguarde novos membros."
    mock_llm.invoke.return_value = mock_response
    
    generator = ResponseGenerator(mock_llm)
    state: AgentState = {
        "intent": "supplier_search",
        "matches": [],
        "moderation_flags": [],
        "agent_workflow": []
    }
    
    result = generator.generate_response(state)
    
    assert "Não encontramos parceiros" in result["final_response"]