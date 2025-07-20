import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()

# Garante que a raiz do projeto esteja no sys.path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import scr.merchant_social_agents as merchant_social_agents
from scr.merchant_social_agents import (
    RouterAgent,
    SocialMatchmaker,
    CommunityModerator,
    ResponseGenerator,
    MerchantSocialOrchestrator,
    FallbackAgent,
)

@pytest.fixture
def mock_llm():
    """
    Mock do LLM que retorna JSON válido para cada agente
    """
    mock = Mock()
    
    # Mock response object
    mock_response = Mock()
    mock_response.content = '{"intent": "logistics_sharing", "confidence": 0.8}'
    mock.invoke.return_value = mock_response
    
    return mock

@pytest.fixture
def mock_data_processor():
    """
    Mock completo do MerchantDataProcessor
    """
    dp = Mock()
    dp.merchant_profiles = {
        "6": type("Profile", (), {
            "merchant_id": "6",
            "city": "Santos",
            "mcc_code": 5945,
            "mcc_description": "Toy Shops",
            "messages": ["preciso de frete"],
            "needs": ["logistics"],
            "offerings": [],
        })(),
        "4": type("Profile", (), {
            "merchant_id": "4",
            "city": "Santos",
            "mcc_code": 7299,
            "mcc_description": "Services",
            "messages": ["faço marketing"],
            "needs": [],
            "offerings": ["marketing_services"],
        })(),
        "9": type("Profile", (), {
            "merchant_id": "9",
            "city": "Sorocaba",
            "mcc_code": 7299,
            "mcc_description": "Marketing Services",
            "messages": ["faço posts no face"],
            "needs": [],
            "offerings": ["marketing_services"],
        })(),
        "123": type("Profile", (), {
            "merchant_id": "123",
            "city": "São Paulo",
            "mcc_code": 5814,
            "mcc_description": "Fast Food",
            "messages": ["preciso doces"],
            "needs": ["suppliers"],
            "offerings": [],
        })(),
        "134": type("Profile", (), {
            "merchant_id": "134",
            "city": "Santos",
            "mcc_code": 7299,
            "mcc_description": "Personal Services",
            "messages": ["cansado de pedidos"],
            "needs": [],
            "offerings": [],
        })(),
        "345": type("Profile", (), {
            "merchant_id": "345",
            "city": "Campinas",
            "mcc_code": 5814,
            "mcc_description": "Fast Food",
            "messages": ["dividir frete campinas"],
            "needs": ["logistics"],
            "offerings": [],
        })(),
    }
    
    # Mock vectorstore
    dp.vectorstore = Mock()
    dp.vectorstore.similarity_search.return_value = []
    
    return dp

@pytest.fixture
def orchestrator_components(mock_llm, mock_data_processor, tmp_path, monkeypatch):
    """
    Fixture que prepara todos os componentes necessários para o orchestrator
    """
    # Mock ChatOpenAI
    monkeypatch.setattr(
        merchant_social_agents,
        "ChatOpenAI",
        lambda **kwargs: mock_llm
    )

    # Mock OpenAIEmbeddings
    mock_embeddings = Mock()
    monkeypatch.setattr(
        merchant_social_agents,
        "OpenAIEmbeddings",
        lambda **kwargs: mock_embeddings
    )

    # Mock MerchantDataProcessor
    monkeypatch.setattr(
        merchant_social_agents,
        "MerchantDataProcessor",
        lambda csv_path: mock_data_processor
    )

    # Criar CSV fake
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("merchant_id,city,mcc_code,mcc_description,message\n")

    return {
        "mock_llm": mock_llm,
        "mock_data_processor": mock_data_processor,
        "fake_csv_path": str(fake_csv)
    }

@pytest.fixture
def orchestrator(orchestrator_components):
    """
    Fixture do orchestrator configurado com mocks
    """
    components = orchestrator_components
    return MerchantSocialOrchestrator("fake-key", components["fake_csv_path"])

@pytest.fixture
def client(orchestrator):
    """
    Fixture do cliente FastAPI para testes E2E
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    
    # Schemas
    class MessageRequest(BaseModel):
        message: str
        user_id: str

    class MessageResponse(BaseModel):
        response: str
        source_agent_response: str
        agent_workflow: List[Dict[str, Any]]
        intent: Optional[str]
        intent_confidence: Optional[float]
        matches: List[Dict[str, Any]]
        moderation_flags: List[str]
        moderation_suggestions: List[str]
        success_probability: float
        chosen_agent: str

    # Criar app de teste
    test_app = FastAPI(title="Test API")
    
    @test_app.post("/process_message", response_model=MessageResponse)
    async def process_message(request: MessageRequest):
        try:
            result = orchestrator.process_message(request.message, request.user_id)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "merchants_loaded": len(orchestrator.data_processor.merchant_profiles),
            "data_insights_loaded": True,
            "confirmed_clusters": len(orchestrator.insights.CONFIRMED_CLUSTERS)
        }

    @test_app.get("/merchants/{merchant_id}")
    async def get_merchant_profile(merchant_id: str):
        profile = orchestrator.data_processor.merchant_profiles.get(merchant_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Merchant not found")
        return {
            "merchant_id": profile.merchant_id,
            "city": profile.city,
            "business_category": profile.mcc_description,
            "needs": profile.needs,
            "offerings": profile.offerings,
            "cluster_memberships": orchestrator._get_cluster_memberships(int(merchant_id))
        }

    return TestClient(test_app)

@pytest.fixture(autouse=True)
def set_openai_key(monkeypatch):
    """
    Garante que exista OPENAI_API_KEY para todos os testes
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")