# MAYPL 모델 변수 사전

## 1. Dataloader (`dataloader.py` 생성)

### 1.1 기본 데이터 저장소 (Raw Data Storage)

| 변수명 | 자료구조 | 역할 |
|---|---|---|
| `ent2id_train` | `dict` | **(학습용) Entity → ID** 변환 사전 |
| `rel2id_train` | `dict` | **(학습용) Relation → ID** 변환 사전 |
| `ent2id_inf` | `dict` | **(추론용) Entity → ID** 변환 사전 |
| `rel2id_inf` | `dict` | **(추론용) Relation → ID** 변환 사전 |
| `train` | `List[List[int]]` | **(학습용)** `train.txt`의 `fact`들을 ID로 변환하여 저장한 리스트 |
| `inference` | `List[List[int]]` | **(추론용)** 메시지 전달의 기반이 될 `fact` 목록 (`msg.txt` + `train.txt`*) |
| `valid` | `List[List[int]]` | `valid.txt`의 `fact`들을 ID로 변환하여 저장한 리스트 |
| `test` | `List[List[int]]` | `test.txt`의 `fact`들을 ID로 변환하여 저장한 리스트 |

_* `msg_add_tr=True`일 경우_

---

### 1.2 평가용 데이터 (Evaluation Data)

| 변수명 | 모양 (Shape) / 자료구조 | 역할 |
|---|---|---|
| `eval_filter_dict` | `dict` | **정답 필터링용 사전.** `key`: `(질문)`, `value`: `[모든 정답 ID]` |
| `valid_query` | `List[Tuple]` | **검증용 '빈칸 문제'** 목록. `tuple`로 구성됨 |
| `valid_answer` | `List[List[int]]` | `valid_query`의 각 문제에 대한 **'정답 ID 리스트'** |
| `test_query` | `List[Tuple]` | **테스트용 '빈칸 문제'** 목록 |
| `test_answer` | `List[List[int]]` | `test_query`의 각 문제에 대한 **'정답 ID 리스트'** |

---

### 1.3 모델 입력용 텐서 (`_train` / 학습용)

| 변수명 | 모양 (Shape) | 역할 |
|---|---|---|
| `pri_train` | `(num_train, 3)` | 훈련 데이터의 `primary triplet`만 모아놓은 텐서 |
| `qual_train` | `(총 qual 수, 2)` | 훈련 데이터의 모든 `qualifier` `[rel, ent]` 쌍을 평탄화한 텐서 |
| `ent_train` | `(총 ent 수,)` | 훈련 데이터의 모든 `entity` ID를 평탄화한 텐서 |
| `qual2fact_train`| `(총 qual 수,)` | 각 `qualifier`가 몇 번 `fact`에 속하는지 알려주는 매핑 텐서 |
| `ent2fact_train` | `(총 ent 수,)` | 각 `entity`가 몇 번 `fact`에 속하는지 알려주는 매핑 텐서 |
| `fact2entstart_train` | `(num_train,)` | 각 `fact`의 `entity`들이 `ent_train` 텐서의 어디서 시작하는지 알려주는 포인터 |
| `rel_mask_train` | `(num_rel_train,)` | 훈련 데이터에 등장한 `relation`만 `True`인 boolean 마스크 |
| `hpair_train` | `(고유 hpair 수, 2)` | 훈련 데이터에 등장한 고유한 `(head, pri_rel)` 조합 목록 |
| `fact2hpair_train` | `(num_train,)` | 각 `fact`가 몇 번 `hpair`를 참조하는지에 대한 매핑 텐서 |

---

### 1.4 모델 입력용 텐서 (`_inf` / 추론용)

| 변수명 | 모양 (Shape) | 역할 |
|---|---|---|
| `pri_inf` | `(num_inf, 3)` | 추론 그래프의 `primary triplet`만 모아놓은 텐서 |
| `qual_inf` | `(총 qual 수, 2)` | 추론 그래프의 모든 `qualifier`를 평탄화한 텐서 |
| `hpair_inf` | `(고유 hpair 수, 2)` | 추론 그래프에 등장한 고유한 `(head, pri_rel)` 조합 목록 |
| `hpair_freq_inf`| `(고유 hpair 수,)` | 각 `hpair`가 추론 그래프에서 몇 번 등장했는지에 대한 빈도수 |
| ... | ... | _... 위 `_train` 변수들과 동일한 역할을 하는 추론용 버전들_ |