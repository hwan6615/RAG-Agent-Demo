import streamlit as st
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

# 1. 환경 설정
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ .env 파일에 OPENAI_API_KEY가 없습니다.")
    st.stop()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
hf_token = os.getenv("HF_TOKEN")

# ==========================================
# [핵심 1] 도구 검색기 (RAG Engine) 클래스
# ==========================================
class ToolRetriever:
    def __init__(self, client):
        self.client = client
        self.tool_pool = []       # 모든 도구 저장소 (JSON)
        self.tool_descriptions = [] # 검색용 텍스트 (이름 + 설명)
        self.embeddings = None    # 벡터 데이터

    def add_tools(self, tools):
        """도구를 저장소에 추가합니다."""
        for tool in tools:
            # 중복 방지 (이름 기준)
            if any(t['function']['name'] == tool['function']['name'] for t in self.tool_pool):
                continue
            
            self.tool_pool.append(tool)
            # 검색의 정확도를 높이기 위해 '이름: 설명' 형태로 텍스트 생성
            desc = f"{tool['function']['name']}: {tool['function']['description']}"
            self.tool_descriptions.append(desc)

    def build_index(self):
        """저장된 도구들의 임베딩을 생성합니다."""
        if not self.tool_descriptions: return
        
        with st.spinner(f"🔧 {len(self.tool_descriptions)}개의 도구를 학습(Embedding) 중..."):
            response = self.client.embeddings.create(
                input=self.tool_descriptions,
                model="text-embedding-3-small" # 가볍고 성능 좋은 모델
            )
            self.embeddings = np.array([data.embedding for data in response.data])

    def retrieve(self, query, top_k=5):
        """질문과 가장 관련된 도구 Top K를 찾습니다."""
        if self.embeddings is None: return []

        # 1. 질문 임베딩
        q_resp = self.client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        q_vec = np.array([q_resp.data[0].embedding])

        # 2. 코사인 유사도 계산
        similarities = cosine_similarity(q_vec, self.embeddings)[0]

        # 3. 상위 K개 인덱스 추출
        top_indices = similarities.argsort()[-top_k:][::-1]

        # 4. 결과 반환
        results = []
        for idx in top_indices:
            results.append({
                "tool": self.tool_pool[idx],
                "score": similarities[idx]
            })
        return results

# ==========================================
# [유틸리티] 데이터 로드 및 정제
# ==========================================
@st.cache_data
def load_and_prepare_rag_data(num_samples=20):
    """
    데이터셋을 불러오고, 거기 있는 모든 도구를 긁어모아 
    하나의 거대한 'Tool Registry'를 만듭니다.
    """
    dataset_name = "Salesforce/xlam-function-calling-60k"
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
        raw_data = list(dataset.take(num_samples))
        
        all_tools_raw = []
        samples = []

        for item in raw_data:
            samples.append(item)
            # 문자열로 된 도구 리스트 파싱
            tools = format_tools_for_openai(item['tools'])
            all_tools_raw.extend(tools)
            
        return samples, all_tools_raw
    except Exception as e:
        st.error(f"데이터셋 로드 실패: {e}")
        return [], []

def sanitize_parameters(params):
    """(기존과 동일) 파라미터 타입 청소"""
    if not params or str(params).lower() == "none":
        return {"type": "object", "properties": {}}
    if isinstance(params, str):
        try: params = json.loads(params.replace("'", '"'))
        except: return {"type": "object", "properties": {}}

    sanitized = {"type": "object", "properties": {}, "required": params.get("required", [])}
    raw_props = params.get("properties", {})
    if not isinstance(raw_props, dict): raw_props = {}

    for k, v in raw_props.items():
        if not isinstance(v, dict): continue
        clean = {"description": str(v.get("description", ""))}
        rt = str(v.get("type", "string")).lower()
        
        if "int" in rt: clean["type"] = "integer"
        elif "float" in rt or "number" in rt: clean["type"] = "number"
        elif "bool" in rt: clean["type"] = "boolean"
        elif "list" in rt or "array" in rt: clean["type"] = "array"
        else: clean["type"] = "string"
        
        if "enum" in v and isinstance(v["enum"], list):
            clean["enum"] = [str(e) for e in v["enum"]]
            
        sanitized["properties"][k] = clean
    return sanitized

def format_tools_for_openai(tools_input):
    formatted = []
    if isinstance(tools_input, str):
        try: t_list = json.loads(tools_input)
        except: return []
    elif isinstance(tools_input, list): t_list = tools_input
    else: return []

    for func in t_list:
        if not func: continue
        raw_p = func.get("parameters", {})
        clean_p = sanitize_parameters(raw_p)
        formatted.append({
            "type": "function",
            "function": {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "parameters": clean_params
            }
        })
    return formatted

def format_tools_for_openai(tools_input):
    """(기존과 동일) 도구 포맷 변환"""
    formatted_tools = []
    
    if isinstance(tools_input, str):
        try:
            tools_list = json.loads(tools_input)
        except:
            return []
    elif isinstance(tools_input, list):
        tools_list = tools_input
    else:
        return []

    for func in tools_list:
        if not func: continue
        
        raw_params = func.get("parameters", {})
        clean_params = sanitize_parameters(raw_params) # 위에서 정의한 함수 사용

        tool = {
            "type": "function",
            "function": {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "parameters": clean_params
            }
        }
        formatted_tools.append(tool)
    
    return formatted_tools

# ==========================================
# [메인 로직] Streamlit UI
# ==========================================
st.set_page_config(page_title="RAG Agent Demo", page_icon="🧠", layout="wide")

st.title("🧠 RAG 기반 AI Agent (Tool Retrieval)")
st.markdown("""
이 에이전트는 정답 도구를 미리 알지 못합니다. 
**전체 도구 라이브러리**에서 질문과 가장 관련 있는 도구를 **스스로 검색(Retrieval)**하여 사용합니다.
""")

# 1. 사이드바: 데이터 준비
with st.sidebar:
    st.header("📚 Tool Registry")
    
    # 데이터셋 로드 (샘플 30개 -> 도구 약 100~200개 확보)
    samples, all_tools = load_and_prepare_rag_data(num_samples=30)
    
    if not samples:
        st.stop()
        
    # Retriever 초기화 (세션 상태에 저장하여 재학습 방지)
    if "retriever" not in st.session_state:
        retriever = ToolRetriever(client)
        retriever.add_tools(all_tools)
        retriever.build_index() # 여기서 임베딩 비용 발생 (소량)
        st.session_state.retriever = retriever
        st.success(f"✅ {len(retriever.tool_pool)}개의 도구가 벡터 DB에 저장되었습니다!")
    else:
        retriever = st.session_state.retriever
        st.info(f"💾 {len(retriever.tool_pool)}개의 도구 로드됨")

    st.divider()
    
    # 예제 선택
    options = [f"Q{i+1}: {s['query'][:20]}..." for i, s in enumerate(samples)]
    idx = st.selectbox("테스트 예제 선택", range(len(samples)), format_func=lambda x: options[x])
    
    current_sample = samples[idx]
    
    # 정답 파싱
    try:
        ans_raw = current_sample['answers']
        if isinstance(ans_raw, str): ans = json.loads(ans_raw)
        else: ans = ans_raw
        expected_func = ans[0]['name']
    except:
        expected_func = "Unknown"

# 2. 메인 화면
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣ User Query")
    query_text = st.text_area("질문 입력", value=current_sample['query'], height=100)
    
    # 검색할 도구 개수 설정
    top_k = st.slider("검색할 도구 개수 (Top K)", min_value=1, max_value=10, value=3)
    
    search_btn = st.button("🔍 도구 검색 및 실행", type="primary")

with col2:
    st.subheader("2️⃣ Retrieved Tools (RAG 결과)")
    result_container = st.container()

# 3. 실행 로직
if search_btn:
    # [Step 1] 도구 검색 (Retrieval)
    with st.spinner("📚 전체 라이브러리에서 관련 도구를 검색 중..."):
        retrieved_results = retriever.retrieve(query_text, top_k=top_k)
        
    # 검색 결과 UI 표시
    retrieved_tools = []
    with result_container:
        if not retrieved_results:
            st.warning("관련된 도구를 찾지 못했습니다.")
        else:
            for i, res in enumerate(retrieved_results):
                tool = res['tool']
                score = res['score']
                is_correct = (tool['function']['name'] == expected_func)
                
                # 시각적 피드백
                emoji = "🎯" if is_correct else "🔧"
                color = "green" if is_correct else "blue"
                
                with st.expander(f"{emoji} [{score:.3f}] {tool['function']['name']}"):
                    st.json(tool)
                
                retrieved_tools.append(tool)

    # [Step 2] 에이전트 실행 (Generation)
    st.divider()
    st.subheader("3️⃣ Agent Execution")
    
    with st.status("에이전트가 생각 중입니다...", expanded=True):
        st.write("Retrieved Tools를 모델에게 전달하는 중...")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use the supplied tools to answer the user's question."},
                    {"role": "user", "content": query_text}
                ],
                tools=retrieved_tools, # 검색된 도구만 전달!
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            if msg.tool_calls:
                pred_name = msg.tool_calls[0].function.name
                pred_args = msg.tool_calls[0].function.arguments
                st.success(f"**Selected Tool:** `{pred_name}`")
                st.code(pred_args, language="json")
                
                if pred_name == expected_func:
                    st.balloons()
                    st.toast("정답입니다!", icon="🎉")
                else:
                    st.error(f"오답입니다. (Expected: {expected_func})")
            else:
                st.warning("모델이 도구를 사용하지 않았습니다. (일반 답변)")
                st.write(msg.content)
                
        except Exception as e:
            st.error(f"에러 발생: {e}")