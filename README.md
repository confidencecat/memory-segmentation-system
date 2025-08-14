# AI 대화형 기억 시스템

## 1. 프로젝트 개요

본 프로젝트는 장기적인 대화 내용을 효과적으로 기억하고 관리하는 AI 챗봇 시스템을 구현하는 것을 목표로 합니다. 사용자와의 대화를 주제별로 자동 분리하고 요약하여, 필요할 때 관련 정보를 신속하게 찾아내어 맥락에 맞는 답변을 생성합니다.

## 2. 시스템 아키텍처

시스템은 사용자와 직접 상호작용하는 `MainAI`, 기억을 총괄 관리하는 `AuxiliaryAI`, 그리고 분리된 기억 덩어리를 담당하는 `LoadAI`로 구성됩니다. 기억은 단기 기억(Buffer), 현재 대화 요약(Summary), 그리고 주제별로 분리된 장기 기억(Separated Memory)으로 나뉩니다.

```
+------------------+      1. User Input      +----------+
|      User        | ----------------------> |  MainAI  |
+------------------+                         +----------+
       ^      ^                                |  |  ^
       |      | 8. Generate Response           |  |  | 7. Provide Context
       |      +--------------------------------+  |  |
       |                                          |  | 2. Needs Memory?
       |                                          |  |
       +------------------------------------------+  |
                                                     |
+------------------+      3. Retrieve Memory         v
|   AuxiliaryAI    | <--------------------------+----------+
| (Memory Manager) |                            |  LoadAI  |
+------------------+                            +----------+
  | |         ^                                    ^   | 4. Search Memories
  | | 6. Save |                                    |   | (Concurrent)
  | |    Memory                                    |   v
  | v         | 5. Topic Change?                   |
+----------------------+----------------------+----+---------------------+
| Buffer Memory        | Summary Memory       | Separated Memory         |
| (buf_memory.json)    | (sum_memory.json)    | (sep_memory.json)        |
+----------------------+----------------------+--------------------------+
```

## 3. 핵심 로직

1.  **입력 분석**: `MainAI`는 사용자 입력(`U`)이 과거의 기억을 필요로 하는지 판단합니다 (`f_need`).
2.  **기억 검색**: 기억이 필요하다면(`f_need(U) → True`), `LoadAI`가 모든 분리된 기억(`M_sep`)과 현재 대화 요약(`M_sum`)의 연관성을 병렬적으로 검사하여 관련 기억(`M_rel`)을 찾습니다.
3.  **응답 생성**: `MainAI`는 검색된 기억(`M_rel`)을 바탕으로 사용자에게 답변(`R`)을 생성합니다. 기억이 필요 없다면, 단순 응답을 생성합니다.
4.  **기억 관리**: `AuxiliaryAI`는 새로운 대화(`U`, `R`)와 현재 대화 요약(`M_sum`)을 비교하여 주제 변경 여부를 판단합니다 (`f_change`).
5.  **기억 분리 및 저장**:
    *   주제가 변경되었다면(`f_change → True`), 현재까지의 대화 내용(`M_buf`)과 요약(`M_sum`)을 `Separated Memory`에 저장하고 새로운 대화로 `Buffer`와 `Summary`를 초기화합니다.
    *   주제가 연속된다면(`f_change → False`), 현재 `Summary`를 업데이트하고 대화 내용을 `Buffer`에 추가합니다.

## 4. 프로세스 수식 표현

-   `U`: 사용자 입력 (User Input)
-   `R`: AI 응답 (Response)
-   `M_sep`: 분리된 기억의 집합 {`m_1`, `m_2`, ...}
-   `M_buf`: 현재 대화 버퍼
-   `S(m)`: 기억 `m`의 요약본
-   `f_need(U)`: 기억 필요 여부 판단 함수 (→ `True`/`False`)
-   `f_rel(U, S(m))`: 입력과 요약의 연관성 판단 함수 (→ `True`/`False`)
-   `f_change(S(M_buf), U)`: 주제 변경 감지 함수 (→ `True`/`False`)
-   `G(U, M)`: 입력 `U`와 기억 `M`을 기반으로 응답을 생성하는 함수

1.  **기억 검색 (Memory Retrieval)**:
    `M_rel = {m ∈ M_sep ∪ {M_buf} | f_need(U) ∧ f_rel(U, S(m))}`

2.  **응답 생성 (Response Generation)**:
    `R = G(U, M_rel)`

3.  **기억 업데이트 (Memory Update)**:
    -   If `f_change(S(M_buf), U)` is `True`:
        `M_sep_new = M_sep ∪ {M_buf}`
        `M_buf_new = {(U, R)}`
    -   Else:
        `M_buf_new = M_buf ∪ {(U, R)}`

## 5. 주요 컴포넌트

-   **`main.py`**:
    -   `MainAI`: 사용자 인터페이스 및 대화 흐름 제어
    -   `AuxiliaryAI`: 기억 생성, 요약, 분리 등 메모리 관리 총괄
    -   `LoadAI`: 분리된 기억에서 관련 정보 병렬 검색
    -   `MemoryManager`: JSON 파일 기반의 메모리 데이터 입출력 관리
    -   `DataManager`: 파일 시스템 관리 유틸리티
-   **`config.py`**:
    -   API 키, 파일 경로, 모델명 등 설정 변수 관리
    -   AI 판단 모델(기억 필요, 주제 분리, 연관성)을 위한 Few-shot 예제 데이터
-   **`memory/`**:
    -   `all_memory.json`: 전체 대화 기록
    -   `sep_memory.json`: 주제별로 분리된 장기 기억
    -   `buf_memory.json`: 현재 주제의 단기 대화 기록
    -   `sum_memory.json`: 현재 주제의 대화 요약
    -   `env_memory.json`: 메모리 카운터 등 환경 정보
