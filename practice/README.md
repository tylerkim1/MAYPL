# MAYPL 간단한 버전 구현 (처음부터)

이 폴더는 **원본 MAYPL 코드를 완전히 이해하기 위해 간단한 버전부터 직접 구현**하는 프로젝트입니다.

## 프로젝트 목표

1. **학습 방식**: 밑바닥부터 직접 코딩하면서 배우기
2. **구현 범위**: Primary triplet만 사용 (Qualifier는 나중에 추가 예정)
3. **최종 목표**: 실제 데이터셋(WD20K100v1)으로 학습 & 평가

---

## 📁 파일 구조

```
practice/
├── my_maypl.py           # SimpleKG 클래스 (Knowledge Graph 데이터 구조)
├── init_layer.py         # SimpleInitLayer 클래스 (구조 기반 초기화)
├── simple_model.py       # SimpleModel 클래스 (전체 모델 통합)
├── train_eval.py         # 학습 & 평가 파이프라인 (CrossEntropyLoss 버전)
└── README.md            # 이 파일
```

---

## 🔧 구현된 코드 상세

### 1. **my_maypl.py** - Knowledge Graph 데이터 구조

**목적**: Entity와 Relation을 ID로 관리하고, Facts를 저장

**클래스**: `SimpleKG`

#### 주요 속성
```python
self.ent2id = {}        # entity 이름 → ID 매핑
self.id2ent = []        # ID → entity 이름 매핑
self.num_ent = 0        # entity 개수

self.rel2id = {}        # relation 이름 → ID 매핑
self.id2rel = []        # ID → relation 이름 매핑
self.num_rel = 0        # relation 개수

self.facts = []         # [(head_id, rel_id, tail_id), ...]
```

#### 주요 메소드
```python
def add_fact(head: str, relation: str, tail: str):
    """Fact 추가 (자동으로 entity/relation 등록)"""

def to_tensors():
    """모든 facts를 torch.tensor로 변환"""
    # 반환: (num_fact, 3) shape의 tensor

def print_info():
    """데이터셋 정보 출력"""
```

#### 구현 상태
✅ **완료** - 테스트 완료

---

### 2. **init_layer.py** - 구조 기반 초기화 레이어

**목적**: Entity와 Relation이 그래프 구조를 통해 메시지를 주고받으며 임베딩 업데이트

**클래스**: `SimpleInitLayer`

#### 아키텍처
```
Entity → Relation 메시지:
- proj_ent_to_rel: nn.Linear(dim, dim)
- 각 relation이 연결된 head/tail entity로부터 메시지 수집
- 평균 후 LayerNorm 적용

Relation → Entity 메시지:
- proj_rel_to_ent: nn.Linear(dim, dim)
- 각 entity가 연결된 relation으로부터 메시지 수집
- 평균 후 LayerNorm 적용
```

#### 주요 파라미터
```python
self.proj_ent_to_rel = nn.Linear(dim, dim)
self.proj_rel_to_ent = nn.Linear(dim, dim)
self.ent_ln = nn.LayerNorm(dim)
self.rel_ln = nn.LayerNorm(dim)
```

#### Forward 과정
```python
def forward(emb_ent, emb_rel, pri):
    """
    Args:
        emb_ent: (num_ent, dim)
        emb_rel: (num_rel, dim)
        pri: (num_fact, 3) - [head_id, rel_id, tail_id]

    Returns:
        new_emb_ent: (num_ent, dim)
        new_emb_rel: (num_rel, dim)
    """
    # 1. Entity → Relation
    # 2. Relation 업데이트
    # 3. Relation → Entity
    # 4. Entity 업데이트
```

#### 구현 상태
✅ **완료** - 테스트 완료

#### 주요 연산
- `torch.index_select()`: 특정 인덱스의 임베딩 선택
- `torch.index_add_()`: 인덱스별로 메시지 집계
- `torch.bincount()`: 각 인덱스에 메시지가 몇 개 왔는지 카운트

---

### 3. **simple_model.py** - 전체 모델 통합

**목적**: SimpleKG + SimpleInitLayer를 통합하여 완전한 모델 구성

**클래스**: `SimpleModel`

#### 아키텍처
```
초기 임베딩 (학습 가능)
    ↓
Init_Layer 1 (구조 정보 학습)
    ↓
Init_Layer 2
    ↓
...
    ↓
Init_Layer N
    ↓
최종 임베딩 → Prediction
```

#### 주요 파라미터
```python
self.init_emb_ent = nn.Parameter(torch.randn(1, dim))
self.init_emb_rel = nn.Parameter(torch.randn(1, dim))
self.layers = nn.ModuleList([
    SimpleInitLayer(dim, num_ent, num_rel)
    for _ in range(num_layers)
])
```

#### Forward 과정
```python
def forward(pri):
    """
    Args:
        pri: (num_fact, 3) - 전체 facts

    Returns:
        emb_ents: (num_ent, dim) - 최종 entity 임베딩
        emb_rels: (num_rel, dim) - 최종 relation 임베딩
    """
    # 1. 초기 임베딩 복제
    emb_ents = self.init_emb_ent.repeat(num_ent, 1)
    emb_rels = self.init_emb_rel.repeat(num_rel, 1)

    # 2. 모든 Init_Layer 통과
    for layer in self.layers:
        emb_ents, emb_rels = layer(emb_ents, emb_rels, pri)

    # 3. 최종 임베딩 반환
    return emb_ents, emb_rels
```

#### Prediction
```python
def predict(query_pri, pri):
    """
    Args:
        query_pri: (1, 3) - [head_id, rel_id, -1] 형태 (예측할 위치는 -1)
        pri: (num_fact, 3) - 전체 facts

    Returns:
        scores: (num_candidates,) - 각 후보의 점수
    """
    # 1. Forward로 임베딩 얻기
    emb_ents, emb_rels = self.forward(pri)

    # 2. -1 위치 찾기 (0=head, 1=rel, 2=tail)
    pred_pos = (query_pri == -1).nonzero(as_tuple=True)[1].item()

    # 3. Query 임베딩 생성 (예: head + rel)
    query_emb = emb_ents[head_id] + emb_rels[rel_id]

    # 4. 모든 후보와 내적
    scores = torch.inner(query_emb, emb_ents)  # or emb_rels

    return scores
```

#### 구현 상태
✅ **완료** - 테스트 완료

#### 버그 수정 이력
1. ~~`emb_rels[head_id]` → `emb_ents[head_id]`~~ (수정 완료)
2. ~~Loop 변수 이름 불일치~~ (수정 완료)

---

### 4. **train_eval.py** - 학습 & 평가 파이프라인

**목적**: 실제 데이터셋으로 모델 학습 및 평가

**중요**: CrossEntropyLoss 사용 (원본 MAYPL 방식)

---

#### 4.1 **DataLoader 클래스**

**역할**: 데이터셋 파일 읽어서 SimpleKG로 변환

```python
class DataLoader:
    def __init__(self, data_dir, dataset_name):
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.kg = SimpleKG()  # SimpleKG 활용!
        self.valid_facts = []
        self.test_facts = []
```

**주요 메소드**:
```python
def _parse_fact_line(line):
    """
    입력: "Q1000	P530	Q148	P805	Q16957740\n"
    출력: ("Q1000", "P530", "Q148")

    Qualifier는 무시하고 primary triplet만 추출
    """

def load_train():
    """train.txt → self.kg.add_fact()로 추가"""

def load_valid():
    """valid.txt → self.valid_facts에 저장"""

def load_test():
    """test.txt → self.test_facts에 저장"""
```

**구현 상태**: ⚠️ TODO 구현 필요

---

#### 4.2 **SimpleTrainer 클래스** (CrossEntropyLoss)

**핵심 변경**: Margin Ranking Loss → CrossEntropyLoss

**이유**:
- ✅ 복수 정답 문제 자동 해결
- ✅ Negative sampling 불필요
- ✅ 모든 entity를 후보로 고려

```python
class SimpleTrainer:
    def __init__(self, model, kg, lr=0.001, label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='mean'
        )
```

**학습 과정**:
```python
def train_epoch(pri_tensor, batch_size=128):
    """
    1. pri_tensor를 batch_size로 분할 (랜덤 셔플)
    2. 각 배치에 대해:
       - compute_loss() 계산
       - backward & optimizer step
    3. 평균 loss 반환
    """
```

**손실 함수 (핵심!)**:
```python
def compute_loss(pri_tensor):
    """
    CrossEntropyLoss 계산

    과정:
    1. 전체 그래프로 forward
       all_facts = self.kg.to_tensors()
       emb_ents, emb_rels = self.model.forward(all_facts)

    2. 배치에서 heads, rels, tails 추출
       heads = pri_tensor[:, 0]
       rels = pri_tensor[:, 1]
       tails = pri_tensor[:, 2]  # 정답 ID

    3. Query 임베딩 계산
       query_emb = emb_ents[heads] + emb_rels[rels]

    4. 모든 entity와 내적
       scores = torch.matmul(query_emb, emb_ents.T)
       # scores: (batch_size, num_ent)

    5. CrossEntropy 계산
       loss = self.criterion(scores, tails)
       return loss
    """
```

**시각적 예시**:
```
Fact: (Alice, knows, Bob)

Step 1: Query 임베딩
query = emb[Alice] + emb[knows]

Step 2: 모든 entity와 내적
scores = [2.1, 4.5, 1.8, 0.9, 1.2]
        Alice Bob  Charlie David Eve

Step 3: Softmax (자동)
probs = [0.10, 0.74, 0.08, 0.03, 0.04]

Step 4: CrossEntropy
loss = -log(probs[Bob]) = -log(0.74) = 0.30
```

**복수 정답 처리**:
```
Facts: (Alice, knows, Bob), (Alice, knows, Charlie)

→ 각각 따로 학습됨
→ 여러 에폭 거치면 둘 다 학습
→ Negative sampling 필요 없음!
```

**구현 상태**: ⚠️ TODO 구현 필요

---

#### 4.3 **SimpleEvaluator 클래스** (Filtered Evaluation)

**핵심**: 복수 정답을 고려한 공정한 평가

```python
class SimpleEvaluator:
    def __init__(self, model, kg):
        self.answer_dict = {}  # {(h, r): {t1, t2, ...}}
        self._build_answer_dict()
```

**Answer Dictionary 구축**:
```python
def _build_answer_dict():
    """
    예시:
    Facts: (Alice, knows, Bob), (Alice, knows, Charlie)

    answer_dict = {
        (Alice_id, knows_id): {Bob_id, Charlie_id}
    }
    """
    for (h, r, t) in self.kg.facts:
        key = (h, r)
        if key not in self.answer_dict:
            self.answer_dict[key] = set()
        self.answer_dict[key].add(t)
```

**평가 과정**:
```python
def evaluate(test_facts, pri_tensor):
    """
    1. 전체 그래프로 한번만 forward
       emb_ents, emb_rels = model.forward(pri_tensor)

    2. 각 test fact에 대해:
       - Query 임베딩 계산
       - 모든 entity와 점수 계산
       - Filter 준비 (다른 정답들)
       - Rank 계산

    3. MRR, Hits@K 계산
    """
```

**Rank 계산 (Filtered)**:
```python
def calculate_rank(scores, target_id, filter_ids):
    """
    원본 MAYPL 방식

    시각적 예시:
    ==========================================
    Query: (Alice, knows, ?)
    Test 정답: Bob
    Train 정답: Bob, Charlie, David

    원래 점수:
    Alice   ██████ 3.2
    Bob     ████████████████ 8.5  ← target
    Charlie ██████████████ 7.8    ← filter!
    David   █████████████ 7.2     ← filter!
    Eve     ████ 2.1

    Filter 적용 후:
    Alice   ██████ 3.2
    Bob     ████████████████ 8.5  ← target
    Charlie ███████ 7.5            ← 강제로 낮춤!
    David   ██████ 7.5             ← 강제로 낮춤!
    Eve     ████ 2.1

    Bob보다 높은 점수: 0개
    Bob의 rank: 1 ✓
    ==========================================

    구현:
    1. scores를 numpy로 변환
    2. target 점수 저장
    3. filter_ids 점수를 target-1로 낮춤
    4. target보다 높은 점수 개수 세기
    """
```

**구현 상태**: ⚠️ TODO 구현 필요

---

## 🚀 사용 방법

### 1. 데이터 로드
```python
from train_eval import DataLoader

loader = DataLoader("../data", "WD20K100v1")
loader.load_train()
loader.load_valid()
loader.load_test()

pri_tensor = loader.kg.to_tensors()
```

### 2. 모델 생성
```python
from simple_model import SimpleModel

model = SimpleModel(
    num_ent=loader.kg.num_ent,
    num_rel=loader.kg.num_rel,
    dim=64,           # 임베딩 차원
    num_layers=2      # Init_Layer 개수
)
```

### 3. 학습
```python
from train_eval import SimpleTrainer

trainer = SimpleTrainer(
    model=model,
    kg=loader.kg,
    lr=0.001,
    label_smoothing=0.1
)

for epoch in range(100):
    loss = trainer.train_epoch(pri_tensor, batch_size=128)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

### 4. 평가
```python
from train_eval import SimpleEvaluator

evaluator = SimpleEvaluator(model, loader.kg)
metrics = evaluator.evaluate(loader.test_facts, pri_tensor)

print("Test Results:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

---

## 📊 구현 진행 상황

| 파일 | 클래스/함수 | 상태 |
|------|------------|------|
| **my_maypl.py** | SimpleKG | ✅ 완료 |
| **init_layer.py** | SimpleInitLayer | ✅ 완료 |
| **simple_model.py** | SimpleModel | ✅ 완료 |
| **train_eval.py** | DataLoader._parse_fact_line | ✅ 완료 |
| **train_eval.py** | DataLoader.load_train/valid/test | ⚠️ TODO |
| **train_eval.py** | SimpleTrainer.train_epoch | ⚠️ TODO |
| **train_eval.py** | SimpleTrainer.compute_loss | ⚠️ TODO |
| **train_eval.py** | SimpleEvaluator._build_answer_dict | ⚠️ TODO |
| **train_eval.py** | SimpleEvaluator.evaluate | ⚠️ TODO |
| **train_eval.py** | SimpleEvaluator.calculate_rank | ⚠️ TODO |

---

## 🔑 핵심 개념 정리

### 1. **구조 기반 초기화 (Structure-driven Initialization)**
- Entity와 Relation이 그래프 구조를 통해 메시지 교환
- 초기 임베딩이 무작위가 아닌 구조 정보를 반영

### 2. **CrossEntropyLoss vs Margin Ranking Loss**

| 항목 | Margin Ranking Loss | CrossEntropyLoss |
|------|---------------------|------------------|
| Negative 샘플 | 1개 랜덤 선택 | 모든 entity 고려 |
| 복수 정답 | 문제 발생 가능 | 자동 해결 |
| 계산량 | 적음 | 많음 (O(num_ent)) |
| 성능 | 좋음 | 더 좋음 |

### 3. **Filtered Evaluation**
- **문제**: 같은 (h, r)에 여러 정답이 있을 때 불공평
- **해결**: 다른 정답들을 filter로 제외하고 rank 계산
- **결과**: 모든 정답이 공정하게 평가됨

### 4. **Inner Product (내적) 유사도**
```python
# 두 벡터가 비슷한 방향 → 내적 큼 → 유사함
query_emb = head_emb + rel_emb
scores = torch.inner(query_emb, entity_embs)
```

---

## 🐛 알려진 이슈 & 해결

### 1. ~~버그: Entity embedding 잘못 접근~~
- **문제**: `emb_rels[head_id]` 대신 `emb_ents[head_id]` 사용해야 함
- **위치**: simple_model.py의 predict 함수
- **상태**: ✅ 수정 완료

### 2. ~~버그: Loop 변수 업데이트 안됨~~
- **문제**: `new_emb_ents` 생성 후 loop에서 `emb_ents` 사용
- **위치**: simple_model.py의 forward 함수
- **상태**: ✅ 수정 완료

---

## 📚 참고 자료

### 원본 MAYPL 코드 위치
```
../code/
├── model.py        # MAYPL, ANMP_Layer, Attn 클래스
├── dataloader.py   # HKG 클래스
├── train.py        # 학습 스크립트
├── test.py         # 평가 스크립트
└── utils.py        # calculate_rank, metrics 함수
```

### 주요 차이점

| 항목 | 원본 MAYPL | 간단한 버전 (practice/) |
|------|-----------|------------------------|
| Qualifier | 지원 | 미지원 (나중에 추가 예정) |
| ANMP Layer | 있음 | 없음 (Init_Layer만) |
| Attention | 있음 | 없음 |
| Pair embedding | 있음 | 없음 |
| 데이터 구조 | 복잡 | 간단 (SimpleKG) |

---

## 🎯 다음 단계

1. **현재 단계**: train_eval.py의 TODO 구현
   - DataLoader 완성
   - SimpleTrainer.compute_loss() 구현
   - SimpleEvaluator 구현

2. **다음 단계**: 학습 & 평가 실행
   - WD20K100v1 데이터셋으로 학습
   - 성능 측정 (MRR, Hits@K)

3. **향후 계획**:
   - Qualifier 지원 추가
   - ANMP_Layer 추가
   - 성능 최적화

---

## 💡 학습 노트

### 자주 묻는 질문들

**Q1: `torch.sum(query_emb * tail_emb, dim=1)`이 뭐하는 건가요?**
```python
# 요소별 곱셈 후 합 = 내적
query_emb = [0.5, 0.3, 0.8]
tail_emb = [0.4, 0.6, 0.7]
result = 0.5*0.4 + 0.3*0.6 + 0.8*0.7 = 0.94
```

**Q2: 왜 `nn.ModuleList`를 사용하나요?**
- 일반 list: 파라미터 등록 안됨, GPU 이동 안됨
- nn.ModuleList: 파라미터 자동 등록, GPU 이동 자동

**Q3: Negative sampling에서 정답이 뽑히면?**
- Margin Ranking Loss: 문제 발생 (정답을 negative로 학습)
- CrossEntropyLoss: 문제 없음 (모든 entity 고려)

**Q4: 복수 정답이 있으면?**
- 학습: 각 fact마다 따로 학습 (여러 에폭으로 커버)
- 평가: Filtered evaluation으로 공정하게 처리

---

## 📝 Git 정보

- **Branch**: `practice`
- **Gitignore**: `.claude/` 폴더 제외됨
- **최근 커밋**: "Complete SimpleModel implementation with bug fixes"

---

**마지막 업데이트**: 2025년 (대화 종료 시점)
**작성자**: 사용자 직접 구현 (Claude 가이드)
**상태**: train_eval.py TODO 구현 필요
