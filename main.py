import json
import os
import google.generativeai as genai
import concurrent.futures
import time
from google.api_core.exceptions import ResourceExhausted
from config import *


class DataManager:
    @staticmethod
    def load_json(file):
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    @staticmethod
    def save_json(file, data):
        folder = os.path.dirname(file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def history_str(buf):
        s = ''
        for it in buf:
            if isinstance(it, list):
                for sub in it:
                    s += f"{sub['role']}: {sub['content']}\n"
            elif isinstance(it, dict):
                s += f"{it['role']}: {it['content']}\n"
        return s


class MemoryManager:    
    def __init__(self):
        self.data_manager = DataManager()
        self.ensure_memory_files()
    
    def ensure_memory_files(self):
        files = [ALL_MEMORY, SEP_MEMORY, BUF_MEMORY, SUM_MEMORY, ENV_MEMORY]
        for file in files:
            if not os.path.exists(file):
                if file == ENV_MEMORY:
                    self.data_manager.save_json(file, {'BUF_COUNTER': 0, 'CON_COUNTER': 0})
                else:
                    self.data_manager.save_json(file, [])
    
    def save_to_all_memory(self, conversation):
        mem = self.data_manager.load_json(ALL_MEMORY)
        mem.append(conversation)
        self.data_manager.save_json(ALL_MEMORY, mem)
    
    def get_env_counters(self):
        env = self.data_manager.load_json(ENV_MEMORY) if os.path.exists(ENV_MEMORY) else {'BUF_COUNTER': 0, 'CON_COUNTER': 0}
        return env['BUF_COUNTER'], env['CON_COUNTER']
    
    def update_env_counters(self, buf_counter, con_counter):
        self.data_manager.save_json(ENV_MEMORY, {'BUF_COUNTER': buf_counter, 'CON_COUNTER': con_counter})


class AIManager:
    
    @staticmethod
    def call_ai(prompt='테스트', system='지침', history=None, fine=None, api_key=None, retries=0):
        if api_key is None:
            api_key = API_KEY['API_1']

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)

        if fine:
            ex = ''.join([f"user: {q}\nassistant: {a}\n" for q, a in fine])
            combined = f"{ex}user: {prompt}"
        else:
            his = DataManager.history_str(history if history is not None else [])
            combined = f"{his}user: {prompt}"

        attempt = 0
        while True:
            try:
                resp = model.start_chat(history=[]).send_message(combined)
                txt = resp._result.candidates[0].content.parts[0].text.strip()
                result = txt[9:].strip() if txt.lower().startswith('assistant:') else txt
                return result
            except ResourceExhausted:
                attempt += 1
                if attempt > retries:
                    return ''
                wait = 2 ** attempt
                time.sleep(wait)


class AuxiliaryAI:
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.ai_manager = AIManager()
        self.load_ai = LoadAI(memory_manager)
    
    def manage_memory(self, conversation):
        
        self.memory_manager.save_to_all_memory(conversation)
        
        buf_counter, con_counter = self.memory_manager.get_env_counters()
        
        current_summary = self.memory_manager.data_manager.load_json(SUM_MEMORY) if os.path.exists(SUM_MEMORY) else None
        
        if not current_summary:
            new_summary = self.create_comprehensive_summary(conversation)
            self.memory_manager.data_manager.save_json(SUM_MEMORY, new_summary)
            
            buf_memory = self.memory_manager.data_manager.load_json(BUF_MEMORY)
            buf_memory.append(conversation)
            self.memory_manager.data_manager.save_json(BUF_MEMORY, buf_memory)
            return
        
        topic_changed = self.detect_topic_change(current_summary, conversation)
        
        if topic_changed:
            self.move_to_separated_memory(current_summary, buf_counter, con_counter)
            
            new_summary = self.create_comprehensive_summary(conversation)
            self.memory_manager.data_manager.save_json(SUM_MEMORY, new_summary)
            self.memory_manager.data_manager.save_json(BUF_MEMORY, [conversation])
            
            self.memory_manager.update_env_counters(buf_counter + 1, con_counter + 1)
        else:
            updated_summary = self.update_summary(current_summary, conversation)
            self.memory_manager.data_manager.save_json(SUM_MEMORY, updated_summary)
            
            buf_memory = self.memory_manager.data_manager.load_json(BUF_MEMORY)
            buf_memory.append(conversation)
            self.memory_manager.data_manager.save_json(BUF_MEMORY, buf_memory)
            
            self.memory_manager.update_env_counters(buf_counter, con_counter + 1)
    
    def create_comprehensive_summary(self, conversation):
        system_prompt = """당신은 대화 내용을 정확하고 포괄적으로 요약하는 전문가다.
다음 원칙에 따라 요약하라:

1. 사용자가 말한 내용과 AI가 응답한 내용을 모두 포함하라
2. 핵심 주제, 중요한 정보, 구체적인 세부사항을 놓치지 마라
3. 대화의 맥락과 흐름을 유지하라
4. 요약문에는 따옴표("), 백슬래시(\), 작은따옴표(')를 사용하지 마라
5. 간결하면서도 완전한 정보를 담아야 한다

형식: "사용자가 [사용자 내용 요약]에 대해 이야기했고, AI는 [AI 응답 요약]로 답변했다."
"""
        
        user_content = conversation[0]['content']
        ai_content = conversation[1]['content']
        
        prompt = f"사용자: {user_content}\nAI: {ai_content}\n\n위 대화를 요약해달라."
        
        return self.ai_manager.call_ai(prompt=prompt, system=system_prompt)
    
    def update_summary(self, current_summary, new_conversation):
        system_prompt = """당신은 기존 대화 요약에 새로운 대화 내용을 통합하는 전문가다.
다음 원칙에 따라 요약을 업데이트하라:

1. 기존 요약의 내용을 유지하면서 새로운 내용을 자연스럽게 통합하라
2. 사용자 발언과 AI 응답을 모두 포함하라
3. 중복되는 내용은 간결하게 정리하라
4. 새로운 정보나 주제 전개를 명확히 반영하라
5. 요약문에는 따옴표("), 백슬래시(\), 작은따옴표(')를 사용하지 마라

최종 요약은 전체 대화 흐름을 이해할 수 있도록 작성하라.
"""
        
        user_content = new_conversation[0]['content']
        ai_content = new_conversation[1]['content']
        
        prompt = f"""기존 요약: {current_summary}

새로운 대화:
사용자: {user_content}
AI: {ai_content}

기존 요약에 새로운 대화를 통합하여 업데이트된 요약을 작성해달라."""
        
        return self.ai_manager.call_ai(prompt=prompt, system=system_prompt)
    
    def detect_topic_change(self, current_summary, new_conversation):
        system_prompt = """당신은 대화의 주제 변화를 감지하는 전문가다.
기존 대화 요약과 새로운 대화를 비교하여 주제가 크게 바뀌었는지 정확히 판단하라.

주제 변경으로 판단하는 경우 (True):
- 완전히 다른 대상이나 물건에 대한 이야기 (예: 사과 → 포도, 영화 → 음식)
- 전혀 다른 분야나 영역으로의 전환 (예: 건강 → 여행, 공부 → 게임)
- 이전 대화와 논리적 연관성이 거의 없는 새로운 주제
- 상황이나 맥락이 완전히 바뀐 경우

주제 연속으로 판단하는 경우 (False):
- 같은 사람에 대한 추가 정보 (예: 이름 → 나이 → 학교)
- 같은 대상에 대한 세부 질문 (예: 사과에 대한 설명 → 사과 씨앗 → 사과 영양소)
- 관련된 하위 주제로의 자연스러운 전개
- 이전 내용과 논리적 연관성이 있는 질문이나 대화

예시:
- "내 이름은 홍길동" → "나는 20살" = False (같은 사람 정보)
- "사과 설명" → "사과 씨앗" = False (같은 대상의 세부사항)
- "사과 이야기" → "포도 이야기" = True (다른 과일)
- "영화 추천" → "음식 추천" = True (다른 분야)

반드시 "True" (주제 변경) 또는 "False" (주제 연속)로만 답하라."""
        
        user_content = new_conversation[0]['content']
        ai_content = new_conversation[1]['content']
        
        prompt = f"""기존 대화 요약: {current_summary}

새로운 대화:
사용자: {user_content}
AI: {ai_content}

위 새로운 대화가 기존 요약의 주제와 크게 다른지 판단하라."""
        
        result = self.ai_manager.call_ai(prompt=prompt, system=system_prompt, fine=SEPFINE)
        return result.strip() == 'True'
    
    def move_to_separated_memory(self, summary, buf_counter, con_counter):
        sep_memory = self.memory_manager.data_manager.load_json(SEP_MEMORY)
        sep_memory.append([{
            'SUMMARIZATION': summary,
            'NUM': buf_counter,
            'LOAD': con_counter
        }])
        self.memory_manager.data_manager.save_json(SEP_MEMORY, sep_memory)
    
    def retrieve_relevant_memories(self, user_query):
        return self.load_ai.search_memories(user_query)


class LoadAI:
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.ai_manager = AIManager()
    
    def search_memories(self, query):
        buf_counter, con_counter = self.memory_manager.get_env_counters()
        
        sep_memory = self.memory_manager.data_manager.load_json(SEP_MEMORY)
        current_summary = self.memory_manager.data_manager.load_json(SUM_MEMORY) if os.path.exists(SUM_MEMORY) else None
        
        all_summaries = []
        for i, entry in enumerate(sep_memory):
            all_summaries.append((i, entry[0]['SUMMARIZATION']))
        
        if current_summary:
            all_summaries.append(('current', current_summary))
        
        if not all_summaries:
            return None
        
        print("start_load")
        related_indices = self.find_relevant_summaries_async(all_summaries, query)
        print("end_load")

        if not related_indices:
            return None
        
        return self.extract_conversation_data(related_indices, sep_memory)
    
    def find_relevant_summaries_async(self, summaries_with_index, query):
        system_prompt = """당신은 사용자의 질문과 요약 데이터의 연관성을 판단하는 전문가다.
반드시 "True" 또는 "False"만 출력하라."""
        
        related_indices = []
        
        available_api_keys = list(API_KEY.values())
        num_summaries = len(summaries_with_index)
        num_api_keys = len(available_api_keys)
        
        api_key_assignments = []
        for i, (idx, summary) in enumerate(summaries_with_index):
            assigned_api_key = available_api_keys[i % num_api_keys]
            api_key_assignments.append((idx, summary, assigned_api_key))
        
        max_workers = min(1, num_summaries, num_api_keys)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.check_single_relevance, query, summary, system_prompt, api_key): idx
                for idx, summary, api_key in api_key_assignments
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    is_related = future.result()
                    
                    if is_related:
                        print(f"index : {idx} : {is_related}")
                        related_indices.append(idx)
                except Exception as e:
                    pass
        
        return related_indices

    def find_relevant_summaries(self, summaries_with_index, query):
        return self.find_relevant_summaries_async(summaries_with_index, query)
    
    def check_single_relevance(self, query, summary, system_prompt, api_key=None):
        prompt = f"사용자 질문: {query}\n요약 데이터: {summary}\n\n위 질문과 요약 데이터가 관련이 있는지 판단하라."
        
        result = self.ai_manager.call_ai(prompt=prompt, system=system_prompt, fine=ISSAMEFINE, api_key=api_key)

        return result.strip() == 'True'
    
    def extract_conversation_data(self, related_indices, sep_memory):
        all_memory = self.memory_manager.data_manager.load_json(ALL_MEMORY)
        conversation_data = []
        
        for idx in related_indices:
            if idx == 'current':
                buf_memory = self.memory_manager.data_manager.load_json(BUF_MEMORY)
                for conv in buf_memory:
                    if isinstance(conv, list):
                        conversation_data.extend(conv)
                    else:
                        conversation_data.append(conv)
            else:
                if idx < len(sep_memory):
                    entry = sep_memory[idx][0]
                    start_idx = 0 if idx == 0 else sep_memory[idx-1][0]['LOAD'] + 1
                    end_idx = entry['LOAD'] + 1
                    
                    for conv in all_memory[start_idx:end_idx]:
                        if isinstance(conv, list):
                            conversation_data.extend(conv)
                        else:
                            conversation_data.extend([conv[0], conv[1]])
        
        return conversation_data if conversation_data else None


class MainAI:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.auxiliary_ai = AuxiliaryAI(self.memory_manager)
        self.ai_manager = AIManager()
    
    def chat(self, user_input):
        
        if not user_input or user_input.strip() == 'False':
            return 'NONE'
        
        needs_memory = self.check_memory_need(user_input)
        
        if needs_memory:
            relevant_memories = self.auxiliary_ai.retrieve_relevant_memories(user_input)
            if relevant_memories:
                response = self._generate_response_with_memory(user_input, relevant_memories)
            else:
                response = self._generate_simple_response(user_input)
        else:
            response = self._generate_simple_response(user_input)
        
        conversation = [
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': response}
        ]
        self.auxiliary_ai.manage_memory(conversation)
        
        return response
    
    def check_memory_need(self, user_input):
        system_prompt = """당신은 사용자의 말을 분석하여 과거의 대화 내용이 필요한지 판단하는 전문가다.
반드시 "True" 또는 "False"만 출력하라."""

        prompt = f"사용자 입력: {user_input}\n위 입력이 과거 대화 내용을 필요로 하는지 판단하라."
        
        result = self.ai_manager.call_ai(prompt=prompt, system=system_prompt, fine=JUDGEFINE)
        return result.strip() == 'True'
    
    def _generate_simple_response(self, user_input):
        system_prompt = """전문적이면서도 친근한 톤으로 대화하라.
이모티콘을 사용하지 말고 간단하고 명확하게 응답하라."""
        
        return self.ai_manager.call_ai(prompt=user_input, system=system_prompt)
    
    def _generate_response_with_memory(self, user_input, memories):
        system_prompt = """
사용자의 질문에 답하기 위해 과거 대화 내용이 제공되었다.
이 정보를 참고하여 정확하고 맥락에 맞는 답변을 제공하라.
과거 대화 내용을 직접 인용하거나 언급할 때는 자연스럽게 표현하라.
사용자가 기억하고 있는 내용과 일치하는 정확한 정보를 제공하는 것이 중요하다.
이모티콘을 사용하지 말고 간단하고 명확하게 응답하라."""
        
        return self.ai_manager.call_ai(prompt=user_input, system=system_prompt, history=memories)


def main_ai(prompt='False'):
    main_ai_instance = MainAI()
    return main_ai_instance.chat(prompt)


if __name__ == '__main__':
    main_ai_instance = MainAI()

    Q = [
        # 질문 예시 작성
    ]

    for q in Q:
        response = main_ai_instance.chat(q)
        print(f"Q: {q}\nA: {response}\n")

    # while True:
    #     user_input = input("사용자: ")
    #     if user_input.lower() in ['exit', 'quit', '종료', '그만']:
    #         print("AI: 대화를 종료합니다.")
    #         break
    #     response = main_ai_instance.chat(user_input)
    #     print(f"AI: {response}")
