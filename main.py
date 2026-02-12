import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

# 1. 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
hf_token = os.getenv("HF_TOKEN")

# 2. 데이터셋 로드
def load_salesforce_dataset(num_samples=3):
    dataset_name = "Salesforce/xlam-function-calling-60k"
    print(f"Dataset 로딩 시도 중: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
        return list(dataset.take(num_samples))
    except Exception as e:
        print(f"\n[Error] 데이터셋 로드 실패: {e}")
        return []

# ==========================================
# [핵심] 강력한 파라미터 청소 (Strict Mode)
# ==========================================
def sanitize_parameters(params):
    """
    OpenAI API 400 에러를 방지하기 위해, 
    오직 허용된 필드(type, description)만 남기고 
    나머지(default, optional 등)는 제거하며 타입을 표준화합니다.
    """
    # 1. 파라미터가 없거나 None인 경우
    if not params or str(params).lower() == "none":
        return {"type": "object", "properties": {}}

    # 2. 문자열로 들어온 경우 JSON 파싱
    if isinstance(params, str):
        try:
            params = json.loads(params.replace("'", '"'))
        except:
            return {"type": "object", "properties": {}}

    # 3. 기본 구조 생성
    sanitized_params = {
        "type": "object",
        "properties": {},
        "required": params.get("required", [])
    }

    # 4. 속성별 청소 (Deep Cleaning)
    raw_props = params.get("properties", {})
    if not isinstance(raw_props, dict):
        raw_props = {}

    for prop_name, prop_info in raw_props.items():
        if not isinstance(prop_info, dict):
            continue
            
        # 새 딕셔너리 생성 (기존 더러운 필드들을 버리기 위함)
        clean_prop = {}
        
        # (1) Description 복사
        clean_prop["description"] = str(prop_info.get("description", ""))
        
        # (2) Type 매핑 (여기가 핵심)
        # 'str, optional', 'int', 'List[str]' 같은 것들을 표준으로 변환
        raw_type = str(prop_info.get("type", "string")).lower()
        
        if "int" in raw_type:
            clean_type = "integer"
        elif "float" in raw_type or "number" in raw_type:
            clean_type = "number"
        elif "bool" in raw_type:
            clean_type = "boolean"
        elif "list" in raw_type or "array" in raw_type:
            clean_type = "array"
            # array인 경우 items 정의가 필요할 수 있으나, 복잡성 방지를 위해 생략하거나 string 처리
        else:
            clean_type = "string" # 'str, optional' 등은 여기서 string으로 통일됨

        clean_prop["type"] = clean_type
        
        # (3) Enum 처리 (값이 명확한 경우만 유지)
        if "enum" in prop_info and isinstance(prop_info["enum"], list):
            # enum 값들이 모두 문자열인지 확인 (OpenAI 제약)
            clean_prop["enum"] = [str(e) for e in prop_info["enum"]]

        # (4) sanitizing 완료된 속성 추가
        sanitized_params["properties"][prop_name] = clean_prop

    return sanitized_params

# 3. 도구 포맷 변환
def format_tools_for_openai(tools_input):
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
        # 여기서 강력해진 청소 함수 호출
        clean_params = sanitize_parameters(raw_params)

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

# 4. 에이전트 실행
def run_agent(question, tools):
    try:
        if not tools: return "No Tools", None

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the supplied tools to answer the user's question."},
                {"role": "user", "content": question}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            return message.tool_calls[0].function.name, message.tool_calls[0].function.arguments
        else:
            return "No Tool Call", None
            
    except Exception as e:
        # 에러 메시지를 좀 더 명확히 출력
        print(f"API Error: {e}")
        return "Error", None

# 5. 메인 실행
def main():
    data_samples = load_salesforce_dataset(num_samples=3)
    
    if not data_samples:
        print("테스트할 데이터가 없습니다.")
        return

    correct_count = 0
    total_count = len(data_samples)

    print("\n=== 에이전트 평가 시작 (Salesforce Dataset) ===\n")

    for i, item in enumerate(data_samples):
        question = item['query']
        raw_tools = item['tools']
        raw_answers = item['answers']

        # 1. 도구 변환
        tools = format_tools_for_openai(raw_tools)
        
        # 2. 정답 파싱
        try:
            if isinstance(raw_answers, str):
                answers_list = json.loads(raw_answers)
            else:
                answers_list = raw_answers
            expected_name = answers_list[0]['name']
        except:
            expected_name = "Parsing Error"

        # 3. 에이전트 실행
        predicted_name, predicted_args = run_agent(question, tools)
        
        # 4. 결과 출력
        print(f"[Case {i+1}]")
        print(f"Q: {question}")
        print(f"Expected: {expected_name}")
        print(f"Predicted: {predicted_name}")
        
        if predicted_name == expected_name:
            print("Result: ✅ Success")
            correct_count += 1
        else:
            print("Result: ❌ Fail")
            if predicted_args:
                 print(f"   Args: {predicted_args}")

        print("-" * 30)

    print(f"\n최종 결과: {correct_count}/{total_count} 성공 ({correct_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    main()