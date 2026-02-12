# 🤖 AI Agent: Function Calling & Tool Retrieval (RAG)

이 프로젝트는 OpenAI의 Function Calling 기능과 RAG(검색 증강 생성) 기술을 결합하여, 수많은 도구 라이브러리 중 질문에 가장 적합한 도구를 스스로 찾아 실행하는 지능형 에이전트 데모입니다.

---

## 🌟 Basic vs. RAG Agent 비교

| 구분 | Basic Function Calling | **RAG-based Agent (현재 버전)** |
| :--- | :--- | :--- |
| **도구 제공 방식** | 정답 후보 도구(3~5개)를 LLM에게 직접 전달 | **전체 도구 풀(100개+)**에서 검색 |
| **작동 원리** | 단순 분류 및 파라미터 추출 | Semantic Search + LLM Reasoning |
| **확장성** | 도구가 많아지면 토큰 제한으로 실행 불가 | 수만 개의 도구도 임베딩 기반으로 대응 가능 |

---

## 🖼️ Dashboard Demo

![Dashboard Demo](image_0b3fc0.png)
*RAG 기반 에이전트가 전체 도구 라이브러리에서 의미론적 유사도를 기반으로 도구를 추출하고 추론하는 과정입니다.*

---

## 🚀 주요 기능
* **RAG 기반 도구 검색(Tool Retrieval)**: 텍스트 임베딩을 통해 수백 개의 도구 중 질문과 연관된 상위 K개의 도구를 코사인 유사도 기반으로 동적 추출합니다.
* **지능형 함수 호출(Function Calling)**: gpt-4o-mini 모델이 검색된 도구 명세를 이해하고 정확한 인자(Arguments)를 생성합니다.
* **데이터 자동 정제(Sanitization)**: 비표준 데이터셋 타입(str, optional 등)을 OpenAI 규격인 JSON Schema 표준으로 자동 변환하여 API 400 에러를 100% 방지합니다.
* **인터랙티브 대시보드**: Streamlit을 활용하여 검색 결과, 유사도 점수, 에이전트의 추론 과정을 실시간으로 확인 가능합니다.

---

## 🛠️ 기술 스택
* **Language**: Python 3.12
* **Package Manager**: uv
* **LLM API**: OpenAI (gpt-4o-mini, text-embedding-3-small)
* **Framework**: Streamlit
* **Dataset**: Salesforce/xlam-function-calling-60k (Hugging Face)
* **Library**: scikit-learn (유사도 계산), numpy, datasets

---

## 🧠 시스템 아키텍처
1. **Tool Registry**: Hugging Face 데이터셋에서 도구 명세를 로드하여 전역 도구 저장소를 구축합니다.
2. **Vector Indexing**: 도구 설명(Description)을 벡터화하여 메모리 인덱스에 저장합니다.
3. **User Query**: 사용자가 질문을 입력합니다.
4. **Semantic Retrieval**: 질문과 유사도가 높은 상위 K개의 도구를 코사인 유사도 기반으로 검색합니다.
5. **Agent Reasoning**: 검색된 도구 명세와 질문을 LLM에 전달하여 최종 행동을 결정합니다.
    - 검색 점수가 낮더라도 LLM이 도구의 설명을 논리적으로 판단하여 최적의 도구를 선택하는 Self-Correction 과정을 거칩니다.

---

## 📦 설치 및 실행 방법
### 1. 환경 설정 및 의존성 설치
 - uv를 사용하여 가상환경을 구축하고 필요한 패키지를 설치합니다.
```bash
# uv 설치 (이미 설치된 경우 생략)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치 및 가상환경 동기화
uv sync
```

### 2. 환경 변수 설정
 - 프로젝트 최상단에 .env 파일을 생성하고 API 키를 입력합니다.
```bash
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_access_token
```

### 3. 실행
```bash
# Streamlit 데모 실행
uv run streamlit run app.py

# CLI 평가 스크립트 실행
uv run python main.py
```

---

## 📈 학습 및 성과
 - 의존성 관리 최적화: uv를 활용하여 Python 3.12/3.13 버전 호환성 및 라이브러리 충돌 문제를 해결했습니다.
 - 데이터 엔지니어링: 실무 데이터의 비표준 포맷을 API 규격에 맞게 정제하는 과정에서 강력한 Parameter Sanitization 로직을 구현했습니다.
 - RAG의 필요성 확인: 모든 도구를 LLM에게 주입하는 대신, 검색 단계를 추가하여 비용 절감과 정확도 향상을 동시에 달성했습니다.