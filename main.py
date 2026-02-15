# main.py

import json
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import ast
from datasets import load_dataset
from dotenv import load_dotenv
from tavily import TavilyClient
import os

# .env 파일 로드
load_dotenv()

# 1. 데이터 로드 (Salesforce xLAM 데이터셋 연동)
# ---------------------------------------------------------
# main.py

def load_data():
    tools = []
    
    # ---------------------------------------------------------
    # 1. 기본 데이터 로드 (JSON 파일 또는 HuggingFace)
    # ---------------------------------------------------------
    try:
        # 먼저 로컬에 샘플링해둔 파일이 있는지 확인
        with open("selected_tools.json", "r", encoding="utf-8") as f:
            tools = json.load(f)
            print(f"📂 로컬 데이터셋(selected_tools.json) 로드 완료: {len(tools)}개")
    except FileNotFoundError:
        print("⚠️ 로컬 파일이 없어 HuggingFace에서 스트리밍으로 로드합니다 (100개 제한).")
        try:
            # 로컬 파일 없으면 HuggingFace에서 실시간 로드 (기존 로직)
            dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
            for i, item in enumerate(dataset):
                if i >= 100: break
                if 'tools' in item:
                    try:
                        tool_list = item['tools'] if isinstance(item['tools'], list) else json.loads(item['tools'])
                        tools.extend(tool_list)
                    except:
                        continue
        except Exception as e:
            print(f"⚠️ 데이터셋 로드 실패: {e}")

    # ---------------------------------------------------------
    # 2. [핵심] 커스텀 도구(Tavily 등) 강제 주입 (Injection)
    # ---------------------------------------------------------
    # 이 부분이 빠져 있어서 검색이 안 되었던 것입니다!
    custom_tools = [
        {
            "name": "search_web",
            "description": (
                "A powerful internet search engine. "
                "Use this for 'latest news', 'current events', 'AI trends'. "
                "한국어 질문: '웹 검색', '최신 뉴스', '트렌드', '정보 검색'이 필요할 때 반드시 이 도구를 사용하세요."  # <--- 한국어 키워드 추가!
            ),
            "parameters": {
                "type": "object", 
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"]
            }
        },
        {
            "name": "get_weather",
            "description": "Get the current weather for a specific location.",
            "parameters": {
                "type": "object", 
                "properties": {"city": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}},
                "required": ["city"]
            }
        },
        {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object", 
                "properties": {"ticker": {"type": "string", "description": "The stock ticker symbol, e.g. AAPL"}},
                "required": ["ticker"]
            }
        }
    ]

    print(f"💉 커스텀 도구 {len(custom_tools)}개를 도구 풀에 주입합니다.")
    tools.extend(custom_tools)
    
    # 중복 제거 (혹시 모를 중복 방지)
    unique_tools = {t['name']: t for t in tools if 'name' in t}
    final_tools = list(unique_tools.values())
    
    print(f"✅ 최종 도구 풀 크기: {len(final_tools)}개")
    
    # [검증] search_web이 진짜 들어갔는지 확인
    if any(t['name'] == 'search_web' for t in final_tools):
        print("🔍 확인: 'search_web' 도구가 성공적으로 포함되었습니다.")
    else:
        print("🚨 경고: 'search_web' 도구가 포함되지 않았습니다!")

    return final_tools

# ---------------------------------------------------------
# 2. Hybrid Retriever & Reranking (검색 고도화)
# ---------------------------------------------------------
class HybridRetriever:
    def __init__(self, tools, embeddings, client):
        self.tools = tools
        self.embeddings = embeddings  # 미리 임베딩된 도구 설명 벡터들
        self.client = client
        
        # BM25 인덱싱 (키워드 검색용)
        tokenized_corpus = [tool['description'].lower().split() for tool in tools]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, query_embedding, top_k=20):
        """
        BM25(키워드)와 Vector(의미) 검색을 결합한 Hybrid Search
        """
        # 1. Vector Search
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # 점수 정규화 (0~1)
        # (간단한 예시: min-max normalization 로직 추가 가능)
        
        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 점수 정규화 필요 (BM25는 점수 범위가 큼, 여기서는 단순 합산 예시)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)

        # 3. Hybrid Score (가중치 조절 가능: Vector 0.7 + BM25 0.3)
        hybrid_scores = 0.7 * similarities + 0.3 * bm25_scores
        
        # Top-K 추출
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        candidates = [self.tools[i] for i in top_indices]
        
        return self.rerank(query, candidates)

    def rerank(self, query, candidates):
        """
        GPT-4o-mini를 리랭커(Reranker)로 사용하여 후보군 압축
        """
        candidate_str = "\n".join([f"{i}. {t['name']}: {t['description']}" for i, t in enumerate(candidates)])
        
        prompt = f"""
        User Query: "{query}"
        
        Below is a list of potential tools. Select the top 3 tools that are most relevant to solving the user's query.
        Return ONLY the indices of the tools (e.g., [0, 2, 5]).
        
        Tools:
        {candidate_str}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            indices = json.loads(response.choices[0].message.content)
            final_tools = [candidates[i] for i in indices if i < len(candidates)]
            return final_tools if final_tools else candidates[:5]
        except:
            return candidates[:3] # 에러 시 상위 3개 반환

# ---------------------------------------------------------
# 3. Agent Class (Planning + Self-Correction)
# ---------------------------------------------------------
class Agent:
    def __init__(self, client, retriever):
        self.client = client
        self.retriever = retriever
        self.max_retries = 2
        self.history = [] # 대화 및 생각의 흐름 저장

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    # [수정 3] 안전한 JSON 파싱 헬퍼 함수
    def parse_json_safely(self, text):
        try:
            # 1. 마크다운 코드 블록 제거 (```json ... ```)
            cleaned_text = text.strip()
            if "```" in cleaned_text:
                cleaned_text = cleaned_text.split("```")[1]
                if cleaned_text.startswith("json"):
                    cleaned_text = cleaned_text[4:]
            
            # 2. JSON 파싱
            return json.loads(cleaned_text.strip())
        except:
            # JSON이 아니면 (일반 텍스트 답변이면) None 반환
            return None
        
    def run(self, user_query, status_container=None): # status_container 추가
        self.history = [{"role": "user", "content": user_query}]
        logs = [] 

        # 1. 초기 도구 검색
        query_embedding = self.get_embedding(user_query)
        relevant_tools = self.retriever.search(user_query, query_embedding)
        logs.append({"step": "Retrieval", "content": relevant_tools})
        
        # 시스템 프롬프트 강화: 도구 사용 시와 최종 답변 시를 명확히 구분
        system_prompt = f"""
        You are an intelligent agent.
        You have access to the following tools:
        {json.dumps(relevant_tools, indent=2)}

        [INSTRUCTIONS]
        1. To use a tool, you MUST output a JSON block like this:
        ```json
        {{
            "tool_name": "tool_name_here",
            "arguments": {{ "arg_name": "value" }}
        }}
        ```
        2. If you have the final answer or if no tool is relevant, just write the answer in plain text.
        3. Do NOT include any explanations outside the JSON when calling a tool.
        """
        
        # history의 첫 번째에 시스템 프롬프트가 오도록 설정 (매번 갱신)
        messages = [{"role": "system", "content": system_prompt}] + self.history

        # 2. Planning & Execution Loop
        max_steps = 5
        for step in range(max_steps):
            
            # UI에 진행상황 업데이트 (사용자가 멈췄다고 느끼지 않게)
            if status_container:
                status_container.markdown(f"🔄 **Step {step+1}/{max_steps}**: 생각하고 추론하는 중...")

            # [수정 1] response_format 제거 -> 텍스트와 JSON 자유롭게 사용
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0 
            )
            content = response.choices[0].message.content
            
            # [수정 2] 마크다운 백틱 제거 및 JSON 파싱 시도
            action = self.parse_json_safely(content)

            # A. 도구 호출인 경우 (JSON 파싱 성공 및 tool_name 존재)
            if action and "tool_name" in action:
                tool_name = action["tool_name"]
                args = action.get("arguments", {})
                
                logs.append({"step": "Plan", "content": f"Decided to call {tool_name} with {args}"})
                
                # Mock Execution
                result, is_error = self.mock_execute(tool_name, args)
                
                logs.append({"step": "Execution", "content": f"Result: {result}"})
                
                # 대화 기록에 추가 (LLM이 결과를 알아야 함)
                self.history.append({"role": "assistant", "content": content})
                self.history.append({"role": "user", "content": f"Tool output: {result}"})
                messages = [{"role": "system", "content": system_prompt}] + self.history

            # B. 최종 답변인 경우 (JSON이 아니거나 tool_name이 없음)
            else:
                logs.append({"step": "Final Answer", "content": content})
                self.history.append({"role": "assistant", "content": content})
                return content, logs
                
        return "죄송합니다. 너무 많은 단계가 소요되어 답변을 중단했습니다.", logs

    # Agent 클래스 내부
    def mock_execute(self, tool_name, args):
        """
        [Hybrid Execution]
        1. 검색 도구 -> Tavily API 실시간 호출 (Real)
        2. 날씨/주식 -> 데모용 가짜 데이터 (Mock)
        3. 그 외 -> 실행 성공 로그만 반환 (Simulation)
        """
        
        # ---------------------------------------------------------
        # Case 1: [Real] 웹 검색 (Tavily 연동)
        # ---------------------------------------------------------
        # xLAM이 'search_web', 'google_search', 'bing_search' 등 뭘 가져오든
        # 이름에 'search', 'web', 'news'가 있으면 Tavily로 처리합니다.
        if any(k in tool_name.lower() for k in ["search", "web", "news"]):
            try:
                # 1. API 키 로드
                tavily_key = os.getenv("TAVILY_API_KEY")
                if not tavily_key:
                    return "Error: TAVILY_API_KEY not found in .env", True

                # 2. 클라이언트 초기화
                tavily = TavilyClient(api_key=tavily_key)
                
                # 3. 검색 쿼리 추출 (xLAM 도구마다 파라미터 이름이 다를 수 있음)
                query = args.get("query") or args.get("q") or args.get("search_term")
                if not query:
                    return "Error: No query provided in arguments", True

                # 4. 실제 검색 실행 (요약본만 가져오기)
                print(f"🌍 Tavily 검색 실행: {query}")
                search_result = tavily.search(query=query, search_depth="basic", max_results=3)
                
                # 5. 결과 반환 (LLM이 읽을 수 있게 JSON 문자열로 변환)
                # context 리스트만 뽑아서 줍니다.
                results = search_result.get("results", [])
                return json.dumps(results, ensure_ascii=False), False

            except Exception as e:
                return f"Error during Tavily search: {str(e)}", True

        # ---------------------------------------------------------
        # Case 2: [Mock] 날씨 (데모용)
        # ---------------------------------------------------------
        elif "weather" in tool_name.lower():
            city = args.get("city", "Unknown City")
            return json.dumps({
                "city": city,
                "temperature": "22°C", 
                "condition": "Partly Cloudy", 
                "humidity": "45%",
                "note": "This is mock data."
            }), False
            
        # ---------------------------------------------------------
        # Case 3: [Mock] 주식 (데모용)
        # ---------------------------------------------------------
        elif "stock" in tool_name.lower():
            ticker = args.get("ticker", "UNKNOWN")
            return json.dumps({
                "ticker": ticker,
                "price": "$150.25", 
                "change": "+1.25%",
                "status": "Market Open",
                "note": "This is mock data."
            }), False

        # ---------------------------------------------------------
        # Case 4: [Generic] 그 외 모든 도구
        # ---------------------------------------------------------
        else:
            return f"✅ [Simulation] Tool '{tool_name}' executed successfully. (No real action performed)", False

# ---------------------------------------------------------
# 초기화 헬퍼 함수
# ---------------------------------------------------------
def initialize_system(api_key, tools_data, tool_embeddings):
    client = OpenAI(api_key=api_key)
    retriever = HybridRetriever(tools_data, tool_embeddings, client)
    agent = Agent(client, retriever)
    return agent