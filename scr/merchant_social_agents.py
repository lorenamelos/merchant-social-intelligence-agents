# merchant_social_agents.py
# Implementação atualizada com insights dos dados reais

from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import re
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# ============================================================

class DummyVectorStore:
    def similarity_search(self, *args, **kwargs):
        return []


# ==================== REAL DATA INSIGHTS ====================


class RealDataInsights:
    """
    Insights extraídos da análise real dos dados
    """
    
    # Geographic clusters identificados
    GEOGRAPHIC_HOTSPOTS = {
        'Santos': {
            'merchants': [3, 4, 6, 8, 10, 15, 20],  # 7 merchants
            'total_messages': 34,
            'density': 4.9,
            'status': 'HOTSPOT'
        },
        'Sorocaba': {
            'merchants': [5, 9, 11, 12, 17, 18],    # 6 merchants  
            'total_messages': 30,
            'density': 5.0,
            'status': 'HOTSPOT'
        }
    }
    
    # Business opportunities confirmadas
    CONFIRMED_CLUSTERS = {
        'santos_logistics': {
            'members': [4, 6, 10, 15, 20],          # 5 merchants confirmados
            'keywords': ['frete', 'entrega', 'parceiro'],
            'success_probability': 0.85,
            'priority': 'HIGH'
        },
        'campinas_delivery': {
            'members': [15, 19, 20],                # 3 merchants confirmados
            'keywords': ['campinas', 'dividir custo', 'entrega'],
            'success_probability': 0.75,
            'priority': 'MEDIUM'
        },
        'marketing_services': {
            'provider': [9],                        # 1 provider confirmado
            'needers': [6, 7, 18],                 # 3 needers confirmados
            'match_type': 'service_provider',
            'success_probability': 0.90,
            'priority': 'HIGH'
        }
    }
    
    # Message patterns com frequências reais
    MESSAGE_PATTERNS = {
        'partnerships': {'count': 33, 'percentage': 0.33},
        'suppliers': {'count': 14, 'percentage': 0.14},
        'marketing': {'count': 13, 'percentage': 0.13},
        'logistics': {'count': 11, 'percentage': 0.11},
        'networking': {'count': 13, 'percentage': 0.13}
    }
    
    # Quality issues identificados
    QUALITY_ISSUES = {
        'total_problematic_messages': 32,
        'problematic_merchants': [1, 3, 5, 6, 7],  # 2+ problemas cada
        'issue_types': {
            'too_generic': 22,
            'vague_connection_request': 7,
            'too_short': 6,
            'platform_question': 5,
            'social_greeting': 3
        },
        'specific_patterns': {
            'como funciona?': 'platform_question',
            'alguém?': 'too_vague_question',
            'quero vender mais': 'too_generic',
            'conectar parceiro': 'vague_connection_request',
            'oi td bom?': 'social_greeting'
        }
    }
    
    # MCC insights baseados nos dados
    MCC_INSIGHTS = {
        5814: {  # Fast Food Restaurants
            'message_count': 26,
            'activity_level': 'HIGH',
            'common_needs': ['suppliers', 'packaging', 'delivery'],
            'merchants': [7, 10, 13, 15, 20]
        },
        7299: {  # Misc. Personal Services  
            'message_count': 23,
            'activity_level': 'HIGH',
            'common_needs': ['marketing', 'networking'],
            'risk_level': 'MEDIUM',  # Categoria muito diversa
            'merchants': [3, 4, 9, 18, 19]
        },
        5945: {  # Hobby, Toy, and Game Shops
            'message_count': 15,
            'activity_level': 'MEDIUM',
            'common_needs': ['partnerships', 'cross_selling'],
            'merchants': [6, 8, 16]
        }
    }

# ==================== UPDATED AGENTS ====================


from typing import TypedDict, Dict, Any, List, Optional

class AgentState(TypedDict, total=False):
    message: str
    user_id: str
    user_profile: Dict[str, Any]
    intent: Optional[str]
    intent_confidence: Optional[float]
    matches: List[Dict[str, Any]]
    moderation_flags: List[str]
    moderation_suggestions: List[str]
    moderation_confidence: Optional[float]
    final_response: Optional[str]
    source_agent_response: Optional[str]
    agent_workflow: List[Dict[str, Any]]

class FallbackAgent:
    """
    Responde a intents off_topic ou general_inquiry com uma mensagem genérica.
    """
    def respond(self, state: AgentState) -> AgentState:
        response = (
            "Parece que sua pergunta não trata de parcerias de negócio locais. "
            "Posso ajudar com outra coisa ou você quer saber mais sobre nossa plataforma?"
        )
        # preenche o final_response
        state["final_response"] = response
        # PREENCHE TAMBÉM source_agent_response
        state["source_agent_response"] = "FallbackAgent"
        # registra no workflow
        state["agent_workflow"].append({
            "agent_name": "FallbackAgent",
            "response": response
        })
        return state
    



from langchain.schema import SystemMessage, HumanMessage

class RouterAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.insights = RealDataInsights()

    def classify_intent(self, state: AgentState) -> AgentState:
        message = state["message"]
        profile = state["user_profile"]
        insights = {
            "hotspots": self.insights.GEOGRAPHIC_HOTSPOTS,
            "clusters": self.insights.CONFIRMED_CLUSTERS,
            "quality_patterns": self.insights.QUALITY_ISSUES["specific_patterns"]
        }

        system_prompt = """
You are RouterAgent. Classify into one of:
["partnership_request","supplier_search","marketing_help","logistics_sharing","service_offer","general_inquiry","off_topic","low_quality"].
Return JSON: {"intent": "...", "confidence": 0.0–1.0}.
"""

        user_prompt = f"""
DATA INSIGHTS:
{json.dumps(insights, ensure_ascii=False, indent=2)}

MERCHANT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

MESSAGE:
\"\"\"{message}\"\"\"
"""

        # **CORREÇÃO**: passar **lista** de mensagens, não só uma string
        resp = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        # agora resp.content é algo como '{"intent":"off_topic","confidence":1.0}'
        try:
            parsed = json.loads(resp.content)
        except json.JSONDecodeError:
            raise RuntimeError(f"RouterAgent retornou texto não-JSON: {resp.content!r}")

        state["intent"]            = parsed["intent"]
        state["intent_confidence"] = parsed["confidence"]
        state.setdefault("agent_workflow", []).append({
            "agent_name": "RouterAgent",
            "classification": parsed["intent"],
            "confidence": parsed["confidence"],
        })
        return state


class SocialMatchmaker:
    def __init__(self, llm: ChatOpenAI, data_processor):
        self.llm = llm
        self.data_processor = data_processor
        self.insights = RealDataInsights()

    def find_matches(self, state: AgentState) -> AgentState:
        message = state["message"]
        user_id = state["user_id"]
        profile = state["user_profile"]
        clusters = self.insights.CONFIRMED_CLUSTERS
        hotspots = self.insights.GEOGRAPHIC_HOTSPOTS

        system_prompt = """
You are the SocialMatchmaker.  
Given a merchant’s profile, message, and data-driven cluster/hotspot info, recommend up to 3 other merchants to connect:
Return a JSON array of objects with keys:
- merchant_id (str), reason (str), score (0.0–1.0)
Example:
  [{"merchant_id":"4","reason":"mesma cidade","score":0.85}, ...]
"""
        user_prompt = f"""
HOTSPOTS:
{json.dumps(hotspots, ensure_ascii=False, indent=2)}

CLUSTERS:
{json.dumps(clusters, ensure_ascii=False, indent=2)}

USER PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

MESSAGE:
\"\"\"{message}\"\"\"

Now recommend up to 3 matches.
"""
        resp = self.llm.invoke(system_prompt + "\n" + user_prompt)
        matches = json.loads(resp.content)

        state["matches"] = matches
        state.setdefault("agent_workflow", []).append({
            "agent_name": "SocialMatchmaker",
            "strategy": "llm_based",
            "matches_found": len(matches)
        })
        return state


class CommunityModerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.insights = RealDataInsights()

    def moderate_message(self, state: AgentState) -> AgentState:
        message = state["message"]
        quality = self.insights.QUALITY_ISSUES

        system_prompt = """
You are the CommunityModerator.  
Given a text message and historical quality issues data, decide:
- action: one of ["approve","review","improve"]
- flags: a list of issue keys (e.g. "too_short","too_generic","problematic_user_history")
- suggestions: a list of human-readable suggestions (can be empty)

Return JSON with keys: action (str), flags (list), suggestions (list).
"""
        user_prompt = f"""
QUALITY ISSUES:
{json.dumps(quality, ensure_ascii=False, indent=2)}

MESSAGE:
\"\"\"{message}\"\"\"

Example:
  Input: "alguém?"
  Output: {{"action":"improve","flags":["too_vague_question"],"suggestions":["Complete sua pergunta com mais detalhes..."]}}

Now analyze the MESSAGE.
"""
        resp = self.llm.invoke(system_prompt + "\n" + user_prompt)
        parsed = json.loads(resp.content)

        state["moderation_flags"]       = parsed["flags"]
        state["moderation_suggestions"] = parsed["suggestions"]
        state["moderation_confidence"]  = 1.0  # ou retire se não usar
        state.setdefault("agent_workflow", []).append({
            "agent_name": "CommunityModerator",
            "action": parsed["action"],
            "flags": parsed["flags"],
            "suggestions": parsed["suggestions"]
        })
        return state


class ResponseGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def generate_response(self, state: AgentState) -> AgentState:
        intent  = state["intent"]
        matches = state.get("matches", [])
        flags   = state.get("moderation_flags", [])

        system_prompt = """
You are the final ResponseGenerator.  
Given an intent, optional moderation flags, and a list of matches, produce a user-friendly textual response.
"""
        user_prompt = f"""
INTENT: {intent}
FLAGS: {flags}

MATCHES:
{json.dumps(matches, ensure_ascii=False, indent=2)}

Examples:
- intent logistics_sharing with matches -> "Encontramos 2 parceiros... Quer intro?"
- flags non-empty -> "Por favor melhore sua mensagem: ..."

Now produce the RESPONSE text only.
"""
        resp = self.llm.invoke(system_prompt + "\n" + user_prompt)
        text = resp.content.strip()

        state["final_response"]         = text
        state["source_agent_response"]  = f"LLM generated response"
        state.setdefault("agent_workflow", []).append({
            "agent_name": "ResponseGenerator",
            "generated": text[:50]  # só um snippet
        })
        return state




# ==================== UPDATED ORCHESTRATOR ====================

class MerchantSocialOrchestrator:
    def __init__(self, openai_api_key: str, csv_path: str):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        self.data_processor = MerchantDataProcessor(csv_path)
        self.insights = RealDataInsights()
        
        # Initialize agents
        self.router = RouterAgent(self.llm)
        self.moderator = CommunityModerator(self.llm)
        self.matchmaker = SocialMatchmaker(self.llm, self.data_processor)
        self.response_generator = ResponseGenerator(self.llm)
        self.fallback = FallbackAgent()                 
        
        
        # Create workflow
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self.router.classify_intent)
        workflow.add_node("moderator", self.moderator.moderate_message)
        workflow.add_node("matchmaker", self.matchmaker.find_matches)
        workflow.add_node("response_generator", self.response_generator.generate_response)
        workflow.add_node("fallback", self.fallback.respond)

        
        workflow.set_entry_point("router")
      
        
        def should_continue_to_matchmaker(state: AgentState) -> str:
            # Se tem flags críticos, pular matchmaker
            critical_flags = ['too_vague_question', 'too_short', 'too_generic', 'user_block_request']
            if any(flag in state.get("moderation_flags", []) for flag in critical_flags):
                return "response_generator"
            return "matchmaker"
            
        workflow.add_conditional_edges(
            "moderator",
            should_continue_to_matchmaker,
            {
                "matchmaker": "matchmaker",
                "response_generator": "response_generator"
            }
        )
        
        def next_after_router(state: AgentState) -> str:
        #se off_topic ou general_inquiry, vai para o fallback
            if state["intent"] == "off_topic":
                return "fallback"
            return "moderator"

        workflow.add_conditional_edges(
            "router",
            next_after_router,
            {
                "fallback": "fallback",
                "moderator": "moderator"
            }
        )

        workflow.add_edge("matchmaker", "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("fallback", END)

        return workflow.compile()
    
    def process_message(self, message: str, user_id: str) -> Dict:
        # Enhanced user profile with real data insights
        user_profile = {}
        if user_id in self.data_processor.merchant_profiles:
            profile = self.data_processor.merchant_profiles[user_id]
            user_profile = {
                'city': profile.city,
                'business': profile.mcc_description,
                'mcc': profile.mcc_code,
                'needs': profile.needs,
                'offerings': profile.offerings,
                'is_hotspot_member': profile.city in ['Santos', 'Sorocaba'],
                'cluster_memberships': self._get_cluster_memberships(int(user_id))
            }
        
        initial_state = {
            "message": message,
            "user_id": user_id,
            "user_profile": user_profile,
            "intent": None,
            "intent_confidence": None,
            "matches": [],
            "moderation_flags": [],
            "moderation_suggestions": [],
            "moderation_confidence": None,
            "final_response": None,
            "source_agent_response": None,
            "agent_workflow": []
            
        }
        
        result = self.workflow.invoke(initial_state)
        chosen = result["agent_workflow"][-1]["agent_name"]

        return {
            "response": result["final_response"],
            "source_agent_response": result["source_agent_response"],
            "agent_workflow": result["agent_workflow"],
            "intent": result.get("intent"),
            "intent_confidence": result.get("intent_confidence"),
            "matches": result.get("matches", []),
            "moderation_flags": result.get("moderation_flags", []),
            "moderation_suggestions": result.get("moderation_suggestions", []),
            "success_probability": self._calculate_overall_success_probability(result),
            "chosen_agent": chosen 
        }
    
    def _get_cluster_memberships(self, user_id: int) -> List[str]:
        """
        Identifica quais clusters o merchant pertence baseado nos dados reais
        """
        memberships = []
        
        if user_id in self.insights.CONFIRMED_CLUSTERS['santos_logistics']['members']:
            memberships.append('santos_logistics')
        
        if user_id in self.insights.CONFIRMED_CLUSTERS['campinas_delivery']['members']:
            memberships.append('campinas_delivery')
            
        if user_id in self.insights.CONFIRMED_CLUSTERS['marketing_services']['needers']:
            memberships.append('marketing_needers')
        elif user_id in self.insights.CONFIRMED_CLUSTERS['marketing_services']['provider']:
            memberships.append('marketing_provider')
            
        return memberships
    
    def _calculate_overall_success_probability(self, result: Dict) -> float:
        """
        Calcula probabilidade geral de sucesso baseada nos matches e estratégia
        """
        matches = result.get("matches", [])
        if not matches:
            return 0.0
        
        # Média das probabilidades dos matches
        probabilities = [match.get('success_probability', 0.5) for match in matches]
        return sum(probabilities) / len(probabilities)

# ==================== MISSING DATA PROCESSOR ====================

class MerchantDataProcessor:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.merchant_profiles = self._create_profiles()
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._create_vectorstore()
        
    def _create_profiles(self) -> Dict[str, any]:
        profiles = {}
        
        for merchant_id in self.df['merchant_id'].unique():
            merchant_data = self.df[self.df['merchant_id'] == merchant_id]
            first_row = merchant_data.iloc[0]
            
            messages = merchant_data['message'].tolist()
            needs, offerings = self._extract_needs_offerings(messages)
            
            # Create simple object instead of dataclass for easier access
            profiles[str(merchant_id)] = type('Profile', (), {
                'merchant_id': str(merchant_id),
                'city': first_row['city'],
                'mcc_code': first_row['mcc_code'],
                'mcc_description': first_row['mcc_description'],
                'messages': [msg.lower() for msg in messages],
                'needs': needs,
                'offerings': offerings
            })()
            
        return profiles
    
    def _extract_needs_offerings(self, messages: List[str]) -> tuple[List[str], List[str]]:
        needs = []
        offerings = []
        
        for msg in messages:
            msg_lower = msg.lower()
            
            # Extract needs
            if any(word in msg_lower for word in ['preciso', 'procuro', 'busco', 'quero']):
                if 'fornecedor' in msg_lower:
                    needs.append('suppliers')
                if 'frete' in msg_lower or 'entrega' in msg_lower:
                    needs.append('logistics')
                if 'marketing' in msg_lower or 'divulga' in msg_lower:
                    needs.append('marketing')
                if 'parceiro' in msg_lower or 'parceria' in msg_lower:
                    needs.append('partnerships')
                if 'embalagem' in msg_lower:
                    needs.append('packaging')
                    
            # Extract offerings  
            if any(word in msg_lower for word in ['faço', 'ofereço', 'vendo', 'conheço']):
                if 'marketing' in msg_lower or 'post' in msg_lower:
                    offerings.append('marketing_services')
                if 'fornecedor' in msg_lower:
                    offerings.append('supplier_connections')
                if 'eletronic' in msg_lower:
                    offerings.append('electronics')
                    
        return list(set(needs)), list(set(offerings))
    
    def _create_vectorstore(self) -> FAISS:
        texts = []
        metadatas = []
        
        for profile in self.merchant_profiles.values():
            profile_text = f"{profile.city} {profile.mcc_description} {' '.join(profile.messages)}"
            texts.append(profile_text)
            metadatas.append({
                'merchant_id': profile.merchant_id,
                'city': profile.city,
                'mcc_code': profile.mcc_code
            })

        # Se não temos nenhum texto, devolve um dummy que só retorna []
        if not texts:
            return DummyVectorStore()
            
        return FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

# ==================== FASTAPI APPLICATION ====================

import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ——— 1. Definição dos Schemas ———
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

# ——— 2. Lifespan para inicialização ———
@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    app.state.orchestrator = MerchantSocialOrchestrator(api_key, "fake_merchant_dataset.csv")
    yield
    # Aqui você pode limpar recursos, se necessário

app = FastAPI(title="Merchant Social Intelligence API", lifespan=lifespan)

# --- Rota raiz de sanity check ---
@app.get("/")
async def read_root():
    return {"message": "API está ativa e healthy!"}

# --- Rotas principais ---
@app.post("/process_message", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    try:
        return app.state.orchestrator.process_message(request.message, request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/merchants/{merchant_id}")
async def get_merchant_profile(merchant_id: str):
    orchestrator = app.state.orchestrator
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

@app.get("/insights")
async def get_data_insights():
    orchestrator = app.state.orchestrator
    return {
        "geographic_hotspots": orchestrator.insights.GEOGRAPHIC_HOTSPOTS,
        "confirmed_clusters": orchestrator.insights.CONFIRMED_CLUSTERS,
        "message_patterns": orchestrator.insights.MESSAGE_PATTERNS,
        "quality_issues": orchestrator.insights.QUALITY_ISSUES,
        "mcc_insights": orchestrator.insights.MCC_INSIGHTS
    }

@app.get("/health")
async def health_check():
    orch = app.state.orchestrator
    return {
        "status": "healthy",
        "merchants_loaded": len(orch.data_processor.merchant_profiles),
        "data_insights_loaded": True,
        "confirmed_clusters": len(orch.insights.CONFIRMED_CLUSTERS)
    }

# ==================== COMPREHENSIVE TESTING ====================

import pytest
from unittest.mock import Mock, patch

class TestMerchantSocialAgents:
    
    @pytest.fixture
    def mock_llm(self):
        mock = Mock()
        mock.invoke.return_value.content = "logistics_sharing"
        return mock
    
    @pytest.fixture 
    def sample_data_processor(self):
        processor = Mock()
        processor.merchant_profiles = {
            "6": type('Profile', (), {
                'merchant_id': "6",
                'city': "Santos", 
                'mcc_code': 5945,
                'mcc_description': "Hobby, Toy, and Game Shops",
                'messages': ["preciso de parceiros de frete"],
                'needs': ["logistics"],
                'offerings': []
            })(),
            "4": type('Profile', (), {
                'merchant_id': "4",
                'city': "Santos",
                'mcc_code': 7299, 
                'mcc_description': "Misc. Personal Services",
                'messages': ["preciso encontrar parceiros de frete"],
                'needs': ["logistics"],
                'offerings': []
            })()
        }
        return processor
    
    def test_santos_logistics_cluster_detection(self, mock_llm, sample_data_processor):
        """
        Testa se o Santos logistics cluster é detectado corretamente
        """
        matchmaker = SocialMatchmaker(mock_llm, sample_data_processor)
        
        state = {
            "user_id": "6",
            "message": "preciso de parceiros de frete",
            "intent": "logistics_sharing",
            "user_profile": {"city": "Santos", "mcc": 5945},
            "agent_workflow": []
        }
        
        result = matchmaker.find_matches(state)
        
        # Deve encontrar matches do cluster Santos
        assert len(result["matches"]) > 0
        assert any(match["merchant_id"] == "4" for match in result["matches"])
        
        # Deve usar a estratégia correta
        workflow_step = result["agent_workflow"][-1]
        assert workflow_step["strategy"] == "santos_logistics_cluster"
    
    def test_marketing_service_provider_match(self, mock_llm, sample_data_processor):
        """
        Testa matching entre marketing provider e needer
        """
        # Adicionar mock para merchant 9 (provider)
        sample_data_processor.merchant_profiles["9"] = type('Profile', (), {
            'merchant_id': "9",
            'city': "Sorocaba",
            'mcc_code': 7299,
            'mcc_description': "Misc. Personal Services",
            'messages': ["faço posts no face tiktok insta"],
            'needs': [],
            'offerings': ["marketing_services"]
        })()
        
        matchmaker = SocialMatchmaker(mock_llm, sample_data_processor)
        
        state = {
            "user_id": "6",  # Merchant 6 está na lista de needers
            "message": "preciso de ajuda com marketing digital",
            "intent": "marketing_help",
            "user_profile": {"city": "Santos", "mcc": 5945},
            "agent_workflow": []
        }
        
        result = matchmaker.find_matches(state)
        
        # Deve encontrar o provider específico
        assert len(result["matches"]) > 0
        assert result["matches"][0]["merchant_id"] == "9"
        assert result["matches"][0]["success_probability"] == 0.9
    
    def test_quality_moderation_specific_patterns(self, mock_llm):
        """
        Testa moderação baseada em padrões específicos identificados
        """
        moderator = CommunityModerator(mock_llm)
        
        # Test mensagem específica identificada na análise
        state = {
            "message": "alguém?",
            "user_id": "5",  # Merchant na watch list
            "agent_workflow": []
        }
        
        result = moderator.moderate_message(state)
        
        # Deve detectar problema específico
        assert "too_vague_question" in result["moderation_flags"]
        assert len(result["moderation_suggestions"]) > 0
        assert "Complete sua pergunta" in result["moderation_suggestions"][0]
        
        # Deve ter penalidade por estar na watch list
        assert result["moderation_confidence"] < 0.8
    
    def test_router_intent_classification_with_data_insights(self, mock_llm):
        """
        Testa classificação com boost baseado nos insights dos dados
        """
        router = RouterAgent(mock_llm)
        
        # Santos + frete deve ter boost
        state = {
            "message": "preciso encontrar parceiros de frete",
            "user_profile": {"city": "Santos", "mcc": 5945},
            "agent_workflow": []
        }
        
        result = router.classify_intent(state)
        
        # Deve classificar como logistics_sharing com alta confiança
        assert result["intent"] == "logistics_sharing"
        assert result["intent_confidence"] > 0.8  # Boost por Santos cluster
    
    def test_full_workflow_santos_logistics(self, sample_data_processor):
        """
        Teste de integração completo para Santos logistics cluster
        """
        with patch('merchant_social_agents.ChatOpenAI') as mock_openai:
            mock_openai.return_value.invoke.return_value.content = "logistics_sharing"
            
            orchestrator = MerchantSocialOrchestrator("fake-api-key", "test.csv")
            orchestrator.data_processor = sample_data_processor
            
            result = orchestrator.process_message(
                "olá, preciso encontrar parceiros de frete",
                "6"
            )
            
            # Verificações finais
            assert "response" in result
            assert "Santos" in result["response"]  # Deve mencionar Santos
            assert "cluster" in result["response"].lower()
            assert result.get("success_probability", 0) > 0.8
            assert len(result.get("matches", [])) > 0

# ==================== SCRIPT DE DEMONSTRAÇÃO ====================

def demonstrate_real_data_insights():
    """
    Script para demonstrar como os insights dos dados reais impactam o sistema
    """
    print("=== DEMONSTRAÇÃO: AGENTES BASEADOS EM DADOS REAIS ===")
    
    # Simular alguns cenários
    test_cases = [
        {
            "name": "Santos Logistics Cluster",
            "message": "preciso de parceiros de frete",
            "user_id": "6",
            "expected": "Santos cluster match com alta probabilidade"
        },
        {
            "name": "Marketing Service Match", 
            "message": "preciso de ajuda com marketing digital",
            "user_id": "7",
            "expected": "Match direto com Merchant 9 (provider confirmado)"
        },
        {
            "name": "Quality Improvement",
            "message": "alguém?",
            "user_id": "5",
            "expected": "Moderação com sugestão específica baseada em padrões reais"
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Input: {case['message']} (User {case['user_id']})")
        print(f"Expected: {case['expected']}")
        # Em um sistema real, chamaria o orchestrator aqui
        print("✓ Sistema processaria baseado em dados reais identificados")

if __name__ == "__main__":
    # Para demonstração
    demonstrate_real_data_insights()
    
    # Para executar API
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)