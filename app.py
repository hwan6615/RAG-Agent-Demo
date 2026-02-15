# app.py

import streamlit as st
import pandas as pd
import numpy as np
from main import initialize_system, load_data
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="Agentic RAG Demo", layout="wide")

st.title("🧩 Advanced Agentic Workflow Demo")
st.markdown("""
이 데모는 **GPT-4o-mini**를 활용하여 다음 기능들을 시연합니다:
1. **Hybrid Retrieval**: BM25 + Vector Search
2. **Self-Correction**: 오류 발생 시 스스로 수정
3. **Multi-Step Planning**: 복합적인 도구 사용 계획
""")

# 사이드바 설정 (수정된 로직)
with st.sidebar:
    # 1. 환경 변수에서 먼저 찾기
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    if env_api_key:
        api_key = env_api_key
        st.success("✅ API Key loaded from .env")
    else:
        # 2. 없으면 수동 입력 받기
        api_key = st.text_input("OpenAI API Key", type="password")
        st.info("Salesforce xLAM 데이터셋 기반")

if not api_key:
    st.warning("API Key를 입력하거나 .env 파일에 설정해주세요.")
    st.stop()

client = OpenAI(api_key=api_key) if api_key else None

# ---------------------------------------------------------
# 세션 상태 초기화 (데이터 로드 등)
# ---------------------------------------------------------
if "agent" not in st.session_state and api_key:
    with st.spinner("시스템 초기화 중... (도구 로드 및 임베딩 생성)"):
        
        # 1. 도구 로드 (main.py에서 selected_tools.json + 커스텀 도구 로드)
        tools = load_data()
        
        # 2. [핵심 수정] 진짜 임베딩 생성 (Random 제거!)
        # 도구의 'description' 텍스트를 모아서 한 번에 임베딩합니다.
        tool_descriptions = [t['description'] for t in tools]
        
        try:
            # OpenAI API로 임베딩 요청 (한 번에 배치 처리)
            # 비용은 매우 저렴하니 걱정 마세요.
            response = client.embeddings.create(
                input=tool_descriptions,
                model="text-embedding-3-small"
            )
            
            # 결과 벡터 추출 (이게 진짜 의미 벡터입니다)
            tool_embeddings = np.array([data.embedding for data in response.data])
            st.success(f"✅ {len(tools)}개 도구에 대한 임베딩 생성 완료!")
            
            # 3. 에이전트 초기화 (진짜 임베딩 전달)
            st.session_state.agent = initialize_system(api_key, tools, tool_embeddings)
            
        except Exception as e:
            st.error(f"임베딩 생성 실패: {e}")
            st.stop()

# ---------------------------------------------------------
# 메인 인터페이스
# ---------------------------------------------------------
query = st.text_input("질문을 입력하세요", "AI 트렌드 검색해줘 or 서울 날씨 어때?")

if st.button("에이전트 실행"):
    if not query:
        st.error("질문을 입력해주세요.")
    else:
        agent = st.session_state.agent
        
        # 상태 메시지를 보여줄 빈 공간 확보
        status_container = st.empty()
    
        with st.spinner("에이전트가 생각 중입니다..."):
            final_answer, logs = agent.run(query, status_container)
        
        # 1. 최종 결과 출력
        st.subheader("🤖 최종 답변")
        st.success(final_answer)
        
        # 2. 사고 과정 시각화 (XAI)
        st.subheader("🧠 에이전트 사고 과정 (Chain of Thought)")
        
        for log in logs:
            step_type = log["step"]
            content = log["content"]
            
            if step_type == "Retrieval":
                with st.expander(f"🔍 [검색] 관련 도구 탐색 ({step_type})", expanded=False):
                    # [수정] 데이터가 리스트/딕셔너리면 json, 아니면 그냥 출력
                    if isinstance(content, (dict, list)):
                        st.json(content)
                    else:
                        st.info(content)
            elif step_type == "Plan":
                with st.expander(f"🤔 [계획] 도구 사용 결정 ({step_type})", expanded=True):
                    st.info(content)
            elif step_type == "Execution":
                with st.expander(f"⚡ [실행] 도구 실행 결과 ({step_type})", expanded=True):
                    # [추가] 결과가 리스트(검색 결과)인 경우 보기 좋게 출력
                    if isinstance(content, str) and content.startswith("[{"):
                         try:
                             import json
                             results = json.loads(content)
                             # 검색 결과라면 제목과 링크만 깔끔하게 보여주기
                             if isinstance(results, list) and "title" in results[0]:
                                 st.success("✅ 검색 정보를 성공적으로 가져왔습니다.")
                                 for item in results:
                                     st.markdown(f"**🔗 [{item['title']}]({item['url']})**")
                                     st.caption(item['content'][:200] + "...") # 내용 요약
                             else:
                                 st.code(content)
                         except:
                             st.code(content)
                    else:
                        st.code(content)
            elif step_type == "Error":
                with st.expander(f"🚨 [오류] 실행 실패 및 자가 수정 ({step_type})", expanded=True):
                    st.error(content)
                    st.markdown("**👉 에이전트가 오류를 감지하고 재시도를 준비합니다.**")
            elif step_type == "Final Answer":
                 with st.expander(f"✅ [완료] 최종 정리 ({step_type})", expanded=False):
                    st.write(content)

# ---------------------------------------------------------
# 포트폴리오 팁 섹션
# ---------------------------------------------------------
st.divider()
st.markdown("### 💡 Portfolio Point")
st.caption("이 프로젝트는 단순 RAG를 넘어, 에이전트가 도구 실행 결과를 보고 스스로 판단하고 수정(Self-Correction)하는 루프를 구현했습니다.")