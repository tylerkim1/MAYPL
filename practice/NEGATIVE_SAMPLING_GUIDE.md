# Negative Sampling 가이드

## 문제: 랜덤 Negative Sampling의 위험성

### 현재 코드 (173-175번 줄)
```python
neg_tails = torch.randint(0, self.kg.num_ent, (len(pri_tensor), ))
neg_tail_emb = emb_ents[neg_tails]
neg_scores = torch.sum(query_emb * neg_tail_emb, dim=1)
```

### 문제점
```
Knowledge Graph:
- (Alice, knows, Bob)
- (Alice, knows, Charlie)
- (Alice, knows, David)

학습 중:
Positive: (Alice, knows, Bob)
Negative: 랜덤 선택 → Charlie 선택됨!

문제: Charlie도 정답인데 negative로 학습 ✗
```

## 해결책 1: Answer Dictionary 사용 (추천)

### Step 1: __init__에 answer_dict 추가

```python
class SimpleTrainer:
    def __init__(self, model, kg, lr=0.001):
        self.model = model
        self.kg = kg
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # 복수 정답 처리를 위한 dictionary
        self.answer_dict = {}  # {(head_id, rel_id): set(tail_ids)}
        self._build_answer_dict()

    def _build_answer_dict(self):
        """각 (head, relation) 쌍에 대한 모든 정답 tail 저장"""
        for (h, r, t) in self.kg.facts:
            key = (h, r)
            if key not in self.answer_dict:
                self.answer_dict[key] = set()
            self.answer_dict[key].add(t)
```

### Step 2: compute_loss 수정

```python
def compute_loss(self, emb_ents, emb_rels, pri_tensor):
    # ... (positive scores 계산은 그대로)

    # Negative sampling (정답 제외!)
    neg_tails = []
    for i in range(len(pri_tensor)):
        h = heads[i].item()
        r = rels[i].item()

        # 이 (h, r)에 대한 모든 정답 가져오기
        answers = self.answer_dict.get((h, r), set())

        # 정답이 아닌 entity 선택
        while True:
            neg = torch.randint(0, self.kg.num_ent, (1,)).item()
            if neg not in answers:
                neg_tails.append(neg)
                break

    neg_tails = torch.tensor(neg_tails, dtype=torch.long)
    neg_tail_emb = emb_ents[neg_tails]
    neg_scores = torch.sum(query_emb * neg_tail_emb, dim=1)

    # ... (loss 계산은 그대로)
```

## 해결책 2: 간단한 버전 (확률적 접근)

```python
# 여러 개 뽑고 중복 제거 (간단하지만 완벽하지 않음)
num_negatives = 10
neg_candidates = torch.randint(0, self.kg.num_ent, (len(pri_tensor), num_negatives))

# 첫 번째 negative만 사용 (정답 포함 확률 낮춤)
# 하지만 완벽하지 않음!
```

## 시각적 비교

### Before (문제 있음)
```
Facts: (Alice, knows, Bob), (Alice, knows, Charlie), (Alice, knows, David)

Positive: Bob
Negative 후보: Alice, Bob, Charlie, David, Eve, Frank
                     ↑    ↑       ↑      ← 모두 정답인데 선택될 수 있음!
```

### After (정답 제외)
```
Facts: (Alice, knows, Bob), (Alice, knows, Charlie), (Alice, knows, David)

Positive: Bob
정답 집합: {Bob, Charlie, David}
Negative 후보: Alice, Eve, Frank
                ↑    ↑    ↑     ← 정답이 아닌 것만 선택!
```

## 성능 고려사항

### 방법 1의 단점
- 루프가 있어서 조금 느림
- Entity가 적고 정답이 많으면 while 루프가 오래 걸릴 수 있음

### 개선 방법
```python
# 한번에 여러 negative 샘플링
num_retries = 100
for i in range(len(pri_tensor)):
    h = heads[i].item()
    r = rels[i].item()
    answers = self.answer_dict.get((h, r), set())

    # 한번에 여러 개 뽑아서 첫 번째 valid한 것 사용
    candidates = torch.randint(0, self.kg.num_ent, (num_retries,))
    for neg in candidates:
        if neg.item() not in answers:
            neg_tails.append(neg.item())
            break
```

## 추천 순서

1. **일단 간단하게**: 현재 코드 그대로 사용 (작은 데이터셋에서는 문제 적음)
2. **답변 dictionary 추가**: `_build_answer_dict()` 구현
3. **compute_loss 수정**: while 루프로 정답 제외
4. **나중에 최적화**: 필요하면 더 빠른 방법 사용

## 적용 방법

`train_eval.py`의 `SimpleTrainer` 클래스를:
1. `__init__`에 `_build_answer_dict()` 호출 추가
2. `_build_answer_dict()` 메소드 구현
3. `compute_loss`의 negative sampling 부분 수정
