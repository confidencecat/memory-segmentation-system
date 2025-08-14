import os
from dotenv import load_dotenv

load_dotenv()

ALL_MEMORY = 'memory/all_memory.json'
SEP_MEMORY = 'memory/sep_memory.json'
BUF_MEMORY = 'memory/buf_memory.json'
SUM_MEMORY = 'memory/sum_memory.json'
ENV_MEMORY = 'memory/env_memory.json'

GEMINI_MODEL = "gemini-2.5-flash"

def prd(PR):
    print('=' * len(PR))
    print(PR)
    print('=' * len(PR))

# LOAD API 키들을 자동으로 로드
LOAD_API_KEYS = []
i = 1
while True:
    key = os.getenv(f'LOAD_{i}')
    
    if not key:
        break
    LOAD_API_KEYS.append(key)
    i += 1

API_KEY = {
    'API_1': os.getenv('API_1'),
    'API_2': os.getenv('API_2'),
}

JUDGEFINE = [
    ["저번에 우리가 사과의 기원에 대해서 이야기 했었지?", "True"],
    ["저번에 이야기한 사과에 대해서 설명해줘", "True"],
    ["사과에 대해서 이야기 했었나?", "True"],
    ["저번에 얘기한 물건에 대해서 알려줘", "True"],
    ["오늘 날씨는 어때?", "False"],
    ["내 이름은 OOO이야", "False"],
    ["내가 다니는 학교는 대건고등학교야", "False"],
    ["물건에 대해서 알려줘", "False"]
]


SEPFINE = [
    # 같은 사람에 대한 정보 연속 - False
    [f"이전 대화의 요약: 사용자가 자신의 이름이 서재민이라고 말했다.\n아래는 현재 대화 내용이다.\nuser: 나는 19살이다.\nassistant : 안녕하세요! 19살이시군요!", "False"],
    [f"이전 대화의 요약: 사용자가 자신의 이름이 홍길동이라고 밝혔다.\n아래는 현재 대화 내용이다.\nuser: 내가 다니는 학교는 대건고등학교야.\nassistant : 대건고등학교에 다니시는군요!", "False"],
    
    # 같은 대상에 대한 세부 질문 - False  
    [f"이전 대화의 요약: 사용자가 사과에 대해 설명해달라고 요청했고, AI가 사과의 기본 정보를 설명했다.\n아래는 현재 대화 내용이다.\nuser: 사과에는 씨앗이 있는가?\nassistant : 네, 사과에는 씨앗이 있습니다.", "False"],
    [f"이전 대화의 요약: 사용자가 영화 추천을 받았고, AI가 여러 영화를 추천했다.\n아래는 현재 대화 내용이다.\nuser: 그 영화의 줄거리를 알려줘.\nassistant : 그 영화의 줄거리는 다음과 같습니다.", "False"],
    
    # 완전히 다른 대상/주제 - True
    [f"이전 대화의 요약: 사용자가 사과에 대해 질문하고 AI가 사과의 영양소에 대해 설명했다.\n아래는 현재 대화 내용이다.\nuser: 포도에는 씨가 있는가?\nassistant : 네, 포도 품종에 따라 씨가 있을 수 있습니다.", "True"],
    [f"이전 대화의 요약: 사용자와 AI가 날씨에 대해 이야기했다.\n아래는 현재 대화 내용이다.\nuser: 오늘 점심 뭐 먹지?\nassistant : 기분에 따라 따뜻한 국물이 있는 음식은 어떠세요?", "True"],
    [f"이전 대화의 요약: 사용자와 AI가 고양이에 대해 이야기했다.\n아래는 현재 대화 내용이다.\nuser: 어제 학교에서 시험 봤어.\nassistant : 시험은 잘 보셨나요?", "True"],
    
    # 관련된 하위 주제 - False
    [f"이전 대화의 요약: 사용자가 인사했고, 자신의 이름이 서재민임을 밝히고 대건고등학교에 재학 중이라는 사실을 알렸다.\n아래는 현재 대화 내용이다.\nuser: 학교에서 어떤 공부를 해?\nassistant : 대건고등학교에서는 문과와 이과로 나뉘어 다양한 과목을 배우고 있을 거예요.", "False"],
    [f"이전 대화의 요약: 사용자가 최근 감기에 걸렸던 경험을 이야기했고, 인공지능이 건강 조언을 했다.\n아래는 현재 대화 내용이다.\nuser: 이번 주말에 친구랑 놀이공원 갈 거야!\nassistant : 오, 재밌겠네요! 아직 몸 상태는 괜찮으신가요?", "True"],
]

ISSAMEFINE = [
    ["사용자의 프롬프트 : 지난번에 추천해준 영화 제목 뭐였지?\n요약 데이터 : 사용자가 최근 본 영화에 대해 이야기하며 인공지능은 그 영화의 줄거리와 인상 깊은 장면을 정리해서 알려주었고, 유사한 장르의 다른 영화를 함께 추천했다.", "True"],
    ["사용자의 프롬프트 : 오늘 저녁 메뉴 추천해줘\n요약 데이터 : 사용자가 친구와의 갈등 상황에 대해 이야기했고, 인공지능은 감정을 솔직하게 표현하는 방법과 갈등을 푸는 대화법을 제안했다.", "False"],
    ["사용자의 프롬프트 : 전에 알려준 공부 루틴 다시 말해줘\n요약 데이터 : 사용자가 집중력이 떨어진다고 이야기하자, 인공지능은 시간대별 공부 계획, 타이머 기법, 방해 요소 차단 방법 등을 조언하였다.", "True"],
    ["사용자의 프롬프트 : 내일 비 오려나?\n요약 데이터 : 사용자가 좋아하는 작가의 신작 소설에 대해 이야기했고, 인공지능은 해당 작가의 문체와 주제를 분석하여 설명했다.", "False"]
]

