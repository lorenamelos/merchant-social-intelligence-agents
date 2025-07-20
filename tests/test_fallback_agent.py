import pytest
from unittest.mock import Mock
from scr.merchant_social_agents import FallbackAgent, AgentState

def test_fallback_agent_response():
    """Testa resposta do agente fallback"""
    fallback = FallbackAgent()
    state: AgentState = {
        "message": "Qual o time de futebol melhor do mundo?",
        "user_id": "456",
        "intent": "off_topic",
        "agent_workflow": []
    }
    
    result = fallback.respond(state)
    
    assert "não trata de parcerias de negócio" in result["final_response"]
    assert result["source_agent_response"] == "FallbackAgent"
    assert result["agent_workflow"][-1]["agent_name"] == "FallbackAgent"