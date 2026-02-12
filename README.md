🤖 AI Agent: Function Calling & Tool Retrieval (RAG)
이 프로젝트는 OpenAI의 Function Calling 기능과 RAG(검색 증강 생성) 기술을 결합하여, 수많은 도구 라이브러리 중 질문에 가장 적합한 도구를 스스로 찾아 실행하는 지능형 에이전트 데모입니다.

🚀 주요 기능
 - RAG 기반 도구 검색(Tool Retrieval): 텍스트 임베딩을 통해 수백 개의 도구 중 질문과 연관된 상위 K개의 도구만 동적으로 추출합니다.
 - 지능형 함수 호출(Function Calling): gpt-4o-mini 모델이 선택된 도구 명세를 이해하고 적절한 인자(Arguments)를 생성합니다.
 - 데이터 자동 정제(Sanitization): 비표준 데이터셋 타입을 OpenAI가 요구하는 JSON Schema 표준으로 자동 변환하여 호환성을 확보합니다.
 - 인터랙티브 대시보드: Streamlit을 활용하여 검색 결과, 유사도 점수, 에이전트의 추론 과정을 한눈에 확인 가능합니다.

🛠️ 기술 스택
 - Language: Python 3.12
 - Package Manager: uv
 - LLM API: OpenAI (gpt-4o-mini, text-embedding-3-small)
 - Framework: Streamlit
 - Dataset: Salesforce/xlam-function-calling-60k (Hugging Face)
 - Library: scikit-learn (유사도 계산), numpy, datasets


📦 설치 및 실행 방법
1. 환경 설정 및 의존성 설치
uv를 사용하여 가상환경을 구축하고 필요한 패키지를 설치합니다.

# uv 설치 (이미 설치된 경우 생략)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치
uv sync

2. 환경 변수 설정
프로젝트 최상단에 .env 파일을 생성하고 API 키를 입력합니다.

OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_access_token

3. 대시보드 실행
uv run streamlit run app.py


🧠 시스템 아키텍처
1. Tool Registry: Hugging Face에서 데이터셋을 로드하여 대규모 도구 풀을 생성합니다.
2. Vector Indexing: 도구의 설명을 벡터화하여 메모리에 저장합니다.
3. User Query: 사용자가 질문을 입력합니다.
4. Retrieval Step: 질문과 유사도가 높은 상위 K개의 도구를 코사인 유사도 기반으로 검색합니다.
5. Agent Logic: 검색된 도구 명세와 질문을 LLM에 전달하여 최종 행동을 결정합니다.


📈 학습 및 성과
 - 의존성 관리 최적화: uv를 활용하여 복잡한 라이브러리 충돌을 해결하고 빠른 개발 환경을 구축했습니다.
 - 데이터 엔지니어링: 실무 데이터의 비표준 포맷을 API 규격에 맞게 정제하는 과정에서 데이터 파이프라인의 중요성을 경험했습니다.
 - RAG의 필요성 확인: 모든 도구를 LLM에게 주입하는 대신, 검색 단계를 추가하여 비용 절감과 정확도 향상을 동시에 달성했습니다.