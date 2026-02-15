# tools_sampler.py
import json
import random
from datasets import load_dataset
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def create_sampled_dataset():
    print("📥 xLAM 데이터셋 다운로드 및 스트리밍 중...")
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
    
    # 1. 우리가 원하는 핵심 카테고리 키워드
    target_keywords = ["weather", "math", "stock", "finance", "news", "search", "email", "calendar"]
    
    selected_tools = []
    seen_names = set()
    
    print("🔍 도구 필터링 중...")
    
    # 전체 데이터를 순회하며 선별 (최대 10,000개만 확인)
    for i, item in enumerate(dataset):
        if i > 10000: break 
        
        try:
            # 도구 파싱
            tools = item['tools'] if isinstance(item['tools'], list) else json.loads(item['tools'])
            
            for tool in tools:
                name = tool['name'].lower()
                desc = tool['description'].lower()
                
                # 중복 제거
                if name in seen_names: continue
                
                # 전략 1: 핵심 카테고리는 무조건 포함
                is_target = any(k in name or k in desc for k in target_keywords)
                
                # 전략 2: 핵심이 아니더라도 랜덤하게 5% 확률로 포함 (RAG 성능 테스트용 노이즈)
                is_random = random.random() < 0.05
                
                if is_target or is_random:
                    selected_tools.append(tool)
                    seen_names.add(name)
                    
        except Exception as e:
            continue

    print(f"✅ 총 {len(selected_tools)}개의 도구가 선별되었습니다.")
    
    # 파일로 저장 (main.py에서 이걸 로드해서 쓰면 됨)
    with open("selected_tools.json", "w", encoding="utf-8") as f:
        json.dump(selected_tools, f, indent=2, ensure_ascii=False)
    
    print("💾 'selected_tools.json' 저장 완료!")

if __name__ == "__main__":
    create_sampled_dataset()