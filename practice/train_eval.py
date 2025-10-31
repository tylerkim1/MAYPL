import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

from my_maypl import SimpleKG
from simple_model import SimpleModel


class DataLoader:
    """실제 데이터셋 파일을 읽어서 SimpleKG로 변환하는 클래스"""

    def __init__(self, data_dir, dataset_name, setting='Transductive'):
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.setting = setting  # 'Transductive' or 'Inductive'

        # Train용 KG (학습 데이터)
        self.kg_train = SimpleKG()

        # Inference용 KG (평가 시 사용)
        if setting == 'Transductive':
            # Transductive: train과 inference가 같음
            self.kg = self.kg_train  # kg는 kg_train의 별칭
        elif setting == 'Inductive':
            # Inductive: 별도의 inference graph (msg.txt)
            self.kg_inf = SimpleKG()
        else:
            raise ValueError(f"Unknown setting: {setting}")

        # valid/test 데이터는 평가용으로 별도 저장
        self.valid_facts = []
        self.test_facts = []

    def _parse_fact_line(self, line):
        """
        한 줄의 fact를 파싱
        형식: head relation tail [qualifier_rel qualifier_ent]*

        예시 입력: "Q1000	P530	Q148	P805	Q16957740\n"
        예시 출력: ("Q1000", "P530", "Q148")

        설명:
        - 데이터 파일의 한 줄을 읽어서 primary triplet만 추출
        - Qualifier 부분(P805 Q16957740)은 무시
        - 반환값: (head_name, relation_name, tail_name) 튜플
        """
        # TODO:
        # 1. line.strip()으로 앞뒤 공백/개행 제거
        line_clean = line.strip()
        #    line_clean = line.strip()
        #
        # 2. split()으로 탭/공백 기준 분리 (자동으로 처리됨)
        tokens = line_clean.split()
        #    tokens = line_clean.split()
        #    결과 예시: ["Q1000", "P530", "Q148", "P805", "Q16957740"]
        #
        # 3. 앞 3개만 추출 (head, relation, tail)
        head = tokens[0]
        relation = tokens[1]
        tail = tokens[2]
        #    head = tokens[0]
        #    relation = tokens[1]
        #    tail = tokens[2]
        #
        # 4. 튜플로 반환
        return (head, relation, tail)
        #    return (head, relation, tail)
        #
        # 힌트: 한 줄로 줄이면 이렇게도 가능
        #    tokens = line.strip().split()
        #    return (tokens[0], tokens[1], tokens[2])

    def load_train(self):
        """train.txt 파일 로드하여 SimpleKG에 추가"""
        # TODO:
        train_path = os.path.join(self.data_dir, "train.txt")
        with open(train_path, 'r') as f:
            for line in f:
                head, relation, tail = self._parse_fact_line(line)
                self.kg_train.add_fact(head, relation, tail)  # Train KG에 추가
        # 1. train_path = os.path.join(self.data_dir, "train.txt")
        # 2. with open(train_path, 'r') as f:
        # 3.     for line in f:
        # 4.         head, relation, tail = self._parse_fact_line(line)
        # 5.         self.kg.add_fact(head, relation, tail)
        #
        # 힌트: SimpleKG.add_fact()가 자동으로 entity/relation 매핑 처리함!

    def load_valid(self):
        """valid.txt 파일 로드"""
        valid_path = os.path.join(self.data_dir, "valid.txt")
        with open(valid_path, 'r') as f:
            for line in f:
                head, relation, tail = self._parse_fact_line(line)

                if self.setting == 'Transductive':
                    # Transductive: train KG에 entity/relation 등록 (facts는 추가 안 함)
                    kg = self.kg_train
                    if head not in kg.ent2id:
                        kg.ent2id[head] = kg.num_ent
                        kg.id2ent.append(head)
                        kg.num_ent += 1
                    if tail not in kg.ent2id:
                        kg.ent2id[tail] = kg.num_ent
                        kg.id2ent.append(tail)
                        kg.num_ent += 1
                    if relation not in kg.rel2id:
                        kg.rel2id[relation] = kg.num_rel
                        kg.id2rel.append(relation)
                        kg.num_rel += 1

                elif self.setting == 'Inductive':
                    # Inductive: inference KG에만 등록
                    if head not in self.kg_inf.ent2id:
                        self.kg_inf.ent2id[head] = self.kg_inf.num_ent
                        self.kg_inf.id2ent.append(head)
                        self.kg_inf.num_ent += 1
                    if tail not in self.kg_inf.ent2id:
                        self.kg_inf.ent2id[tail] = self.kg_inf.num_ent
                        self.kg_inf.id2ent.append(tail)
                        self.kg_inf.num_ent += 1
                    if relation not in self.kg_inf.rel2id:
                        self.kg_inf.rel2id[relation] = self.kg_inf.num_rel
                        self.kg_inf.id2rel.append(relation)
                        self.kg_inf.num_rel += 1

                self.valid_facts.append((head, relation, tail))
        # 1. valid_path = os.path.join(self.data_dir, "valid.txt")
        # 2. with open(valid_path, 'r') as f:
        # 3.     for line in f:
        # 4.         fact = self._parse_fact_line(line)
        # 5.         self.valid_facts.append(fact)
        #
        # 힌트: valid는 평가용이라 self.valid_facts 리스트에 저장
        
    def load_test(self):
        """test.txt 파일 로드"""
        test_path = os.path.join(self.data_dir, "test.txt")
        with open(test_path, 'r') as f:
            for line in f:
                head, relation, tail = self._parse_fact_line(line)

                if self.setting == 'Transductive':
                    # Transductive: train KG에 entity/relation 등록 (facts는 추가 안 함)
                    kg = self.kg_train
                    if head not in kg.ent2id:
                        kg.ent2id[head] = kg.num_ent
                        kg.id2ent.append(head)
                        kg.num_ent += 1
                    if tail not in kg.ent2id:
                        kg.ent2id[tail] = kg.num_ent
                        kg.id2ent.append(tail)
                        kg.num_ent += 1
                    if relation not in kg.rel2id:
                        kg.rel2id[relation] = kg.num_rel
                        kg.id2rel.append(relation)
                        kg.num_rel += 1

                elif self.setting == 'Inductive':
                    # Inductive: inference KG에만 등록
                    if head not in self.kg_inf.ent2id:
                        self.kg_inf.ent2id[head] = self.kg_inf.num_ent
                        self.kg_inf.id2ent.append(head)
                        self.kg_inf.num_ent += 1
                    if tail not in self.kg_inf.ent2id:
                        self.kg_inf.ent2id[tail] = self.kg_inf.num_ent
                        self.kg_inf.id2ent.append(tail)
                        self.kg_inf.num_ent += 1
                    if relation not in self.kg_inf.rel2id:
                        self.kg_inf.rel2id[relation] = self.kg_inf.num_rel
                        self.kg_inf.id2rel.append(relation)
                        self.kg_inf.num_rel += 1

                self.test_facts.append((head, relation, tail))

    def load_msg(self):
        """msg.txt 파일 로드 (Inductive setting 전용)"""
        if self.setting != 'Inductive':
            print("Warning: load_msg() is only for Inductive setting")
            return

        msg_path = os.path.join(self.data_dir, "msg.txt")
        if not os.path.exists(msg_path):
            print(f"Warning: {msg_path} not found. Skipping msg.txt loading.")
            return

        with open(msg_path, 'r') as f:
            for line in f:
                head, relation, tail = self._parse_fact_line(line)
                # msg.txt는 inference KG의 facts에 추가
                self.kg_inf.add_fact(head, relation, tail)


class SimpleTrainer:
    """
    학습을 위한 클래스 (CrossEntropyLoss 사용)

    원본 MAYPL 방식:
    - 각 fact (h, r, t)에 대해 모든 entity와의 점수를 계산
    - CrossEntropyLoss로 정답 entity 학습
    - 복수 정답 문제 자동 해결!
    """

    def __init__(self, model, kg_train, lr=0.001, label_smoothing=0.0):
        self.model = model
        self.kg_train = kg_train  # Train용 KG
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # CrossEntropyLoss 설정 (원본 MAYPL 방식)
        # label_smoothing: 정답을 1보다 작게, 오답을 0보다 크게 (과적합 방지)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='mean'  # batch 평균
        )

    def train_epoch(self, pri_tensor, batch_size=128):
        """
        한 에폭 학습

        Args:
            pri_tensor: 전체 training facts (num_fact, 3)
            batch_size: 배치 크기

        Returns:
            평균 loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 전체 그래프 facts (모든 배치에서 재사용)
        all_facts = self.kg_train.to_tensors()

        # TODO: 구현해야 할 것들
        # 1. pri_tensor를 랜덤하게 섞기
        pri_index = torch.randperm(len(pri_tensor))
        #    힌트: torch.randperm()을 사용하면 랜덤 인덱스를 생성할 수 있음
        #    힌트: len(pri_tensor)로 전체 fact 개수를 얻을 수 있음
        #
        # 2. 섞인 인덱스를 batch_size씩 나눠서 반복
        pbar = tqdm(range(0, len(pri_index), batch_size), desc="Training", leave=False)
        for i in pbar:
            batch_pri = pri_tensor[pri_index[i:i+batch_size]]  # 버그 수정!
        #    힌트: range(시작, 끝, 간격)을 사용
        #    힌트: 각 배치는 pri_tensor에서 해당 인덱스들로 선택
        #
        # 3. 각 배치마다 수행할 것들 (일반적인 PyTorch 학습 루프):
            self.optimizer.zero_grad()

            # 매 배치마다 forward (gradient graph 재생성)
            emb_ents, emb_rels = self.model.forward(all_facts)
            loss = self.compute_loss(batch_pri, emb_ents, emb_rels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            # 실시간 loss 표시
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        #    - optimizer의 gradient를 0으로 초기화
        #    - compute_loss()로 loss 계산
        #    - backward()로 gradient 계산
        #    - optimizer로 파라미터 업데이트
        #    - loss 값 누적 (total_loss에 더하기, loss.item() 사용)
        #    - 배치 개수 증가
        #
        # 4. 평균 loss 반환
        avg_loss = total_loss / num_batches
        #    힌트: 전체 loss를 배치 개수로 나누기
        return avg_loss

    def compute_loss(self, pri_tensor, emb_ents, emb_rels):
        """
        CrossEntropyLoss 계산 (원본 MAYPL 방식)

        Args:
            pri_tensor: 현재 배치의 facts (batch_size, 3)
            emb_ents: 전체 entity 임베딩 (num_ent, dim)
            emb_rels: 전체 relation 임베딩 (num_rel, dim)

        과정:
        1. Query 임베딩 = head + relation
        2. 모든 entity와의 내적 = 점수
        3. CrossEntropyLoss(점수, 정답 ID)

        시각적 예시:
        ==========================================
        Fact: (Alice, knows, Bob)

        Step 1: Query 임베딩
        query = emb[Alice] + emb[knows]

        Step 2: 모든 entity와 내적
        scores = query · emb[모든 entity]
        scores = [2.1, 4.5, 1.8, 0.9, 1.2]
                Alice Bob  Charlie David Eve

        Step 3: Softmax (자동)
        probs = [0.10, 0.74, 0.08, 0.03, 0.04]

        Step 4: CrossEntropy
        loss = -log(probs[Bob])
             = -log(0.74)
             = 0.30

        복수 정답 처리:
        - (Alice, knows, Bob)와 (Alice, knows, Charlie) 모두 있으면
        - 각각 따로 학습됨 (여러 에폭 거치면 둘 다 학습)
        - Negative sampling 필요 없음!
        ==========================================
        """
        # 1. pri_tensor에서 head, relation, tail ID 추출
        heads = pri_tensor[:, 0]
        rels = pri_tensor[:, 1]
        tails = pri_tensor[:, 2]
        #    힌트: pri_tensor는 (batch_size, 3) shape
        #    힌트: 각 컬럼은 [:, 0], [:, 1], [:, 2]로 접근 가능
        #    힌트: tail은 정답 entity ID가 됨 (나중에 loss 계산 시 사용)
        #
        # 3. Query 임베딩 만들기
        head_embs = emb_ents[heads]
        rel_embs = emb_rels[rels]
        query_embs = head_embs + rel_embs
        #    힌트: README에서 설명한 대로 query = head_emb + rel_emb
        #    힌트: 임베딩에서 특정 ID들을 선택하려면 indexing 사용 (예: emb_ents[heads])

        #
        # 4. 모든 entity와의 유사도(점수) 계산
        scores = torch.matmul(query_embs, emb_ents.T)
        #    힌트: torch.matmul()을 사용해서 query와 모든 entity의 내적 계산
        #    힌트: emb_ents.T는 entity 임베딩을 transpose한 것
        #    힌트: 결과 shape은 (batch_size, num_ent)가 되어야 함
        #
        # 5. CrossEntropyLoss 계산 후 반환
        CE = self.criterion(scores, tails.long())  # tails를 long 타입으로 변환
        #    힌트: self.criterion(점수, 정답_ID) 형태로 호출
        #    힌트: CrossEntropyLoss는 자동으로 softmax 적용 후 -log(확률) 계산
        return CE


class SimpleEvaluator:
    """
    평가를 위한 클래스 (Filtered Evaluation)

    Filtered setting:
    - 복수 정답을 고려한 공정한 평가
    - Train/valid/test의 모든 정답을 filter로 제외
    """

    def __init__(self, model, kg_train, kg_inf=None):
        self.model = model
        self.kg_train = kg_train  # Train용 KG (answer_dict 생성용)
        self.kg_inf = kg_inf if kg_inf is not None else kg_train  # Inference용 KG (임베딩 생성용)

        # 복수 정답 처리: 모든 (h, r)에 대한 정답 tail 수집
        self.answer_dict = {}  # {(head_id, rel_id): {tail_id1, tail_id2, ...}}
        self._build_answer_dict()

    def _build_answer_dict(self):
        """
        각 (head, relation) 쌍에 대한 모든 정답 tail 저장

        예시:
        Facts: (Alice, knows, Bob), (Alice, knows, Charlie), (Bob, knows, Eve)

        answer_dict = {
            (Alice_id, knows_id): {Bob_id, Charlie_id},
            (Bob_id, knows_id): {Eve_id}
        }
        """
        # TODO: 구현해야 할 것들
        # self.kg_train.facts를 순회하면서:
        for head_id, rel_id, tail_id in self.kg_train.facts:
            if (head_id, rel_id) not in self.answer_dict:
                self.answer_dict[(head_id, rel_id)] = {tail_id}
            else:
                self.answer_dict[(head_id, rel_id)].add(tail_id)
        #   - 각 fact는 (head_id, rel_id, tail_id) 튜플
        #   - (head_id, rel_id)를 key로 사용
        #   - 해당 key에 대한 set이 없으면 새로 만들기
        #   - set에 tail_id 추가
        # 힌트: self.answer_dict는 딕셔너리, 값은 set 타입
        # 힌트: set은 중복을 자동으로 제거해줌

    def evaluate(self, test_facts, pri_tensor):
        """
        Test facts에 대해 평가 (Filtered Evaluation)

        각 fact (h, r, t)에 대해:
        1. 모든 entity와의 점수 계산
        2. 다른 정답들은 filter로 제외
        3. Rank 계산
        4. MRR, Hits@K 계산
        """
        self.model.eval()
        ranks = []

        with torch.no_grad():
            # TODO: 구현해야 할 것들
            # 1. 전체 그래프로 한 번만 forward 해서 모든 임베딩 얻기
            #    힌트: compute_loss에서 했던 것과 동일
            #    힌트: model.forward()를 호출하면 entity와 relation 임베딩 반환
            #    주의: Inference KG (kg_inf)를 사용!
            emb_ents, emb_rels = self.model.forward(pri_tensor)
            #
            # 2. test_facts를 하나씩 순회
            #    힌트: 각 fact는 (head_name, relation_name, tail_name) 튜플
            for head_name, rel_name, tail_name in tqdm(test_facts, desc="Evaluating", leave=False):
            #
            #    2-1. Entity/relation 이름을 ID로 변환
            #         힌트: kg_inf의 ent2id, rel2id 사용 (Transductive에서는 kg_train과 같음)
                head_id = self.kg_inf.ent2id[head_name]
                rel_id = self.kg_inf.rel2id[rel_name]
                tail_id = self.kg_inf.ent2id[tail_name]
            #
            #    2-2. Query 임베딩 계산 (학습 때와 동일)
            #         힌트: head + relation 방식
                query_emb = emb_ents[head_id] + emb_rels[rel_id]
            #
            #    2-3. 모든 entity와의 점수 계산
            #         힌트: torch.inner()를 사용해서 query와 모든 entity의 내적
            #         힌트: 결과 shape은 (num_ent,)
                scores = torch.inner(query_emb, emb_ents)
            #
            #    2-4. Filter 준비: 이 (h, r)에 대한 모든 정답 tail들 가져오기
            #         힌트: self.answer_dict에서 (head_id, rel_id)를 key로 조회
            #         힌트: .get(key, 기본값)을 사용하면 key가 없을 때 안전
            #         힌트: set을 list로 변환해야 함
            #   
                answer_ids = list(self.answer_dict.get((head_id, rel_id), set()))
                ranks.append(self.calculate_rank(scores, tail_id, answer_ids))
            #    2-5. Rank 계산 후 ranks 리스트에 추가
            #         힌트: self.calculate_rank() 메소드 사용

        # Metrics 계산
        ranks = np.array(ranks)
        mr = np.mean(ranks)
        mrr = np.mean(1.0 / ranks)
        hits1 = np.mean(ranks <= 1)
        hits3 = np.mean(ranks <= 3)
        hits10 = np.mean(ranks <= 10)

        return {
            'MR': mr,
            'MRR': mrr,
            'Hits@1': hits1,
            'Hits@3': hits3,
            'Hits@10': hits10
        }

    def calculate_rank(self, scores, target_id, filter_ids):
        """
        원본 MAYPL의 calculate_rank 함수 사용

        Filtered setting: 이미 정답으로 알려진 entity들은 제외하고 rank 계산

        예시:
        Query: (Alice, knows, ?)
        정답: Bob, Charlie, David (모두 train에 있음)

        Bob 테스트 시:
        - filter_ids = [Bob_id, Charlie_id, David_id]  ← 모든 정답들
        - Charlie, David의 점수를 Bob 점수보다 낮게 설정 (Bob 제외)
        - 이렇게 하면 Bob의 rank가 공정하게 계산됨
        """
        # TODO: 구현해야 할 것들
        # 1. scores를 numpy로 변환하고 복사
        cpy_scores = scores.cpu().numpy().copy()
        #    힌트: .cpu().numpy()로 tensor를 numpy로 변환
        #    힌트: .copy()로 복사해야 원본이 수정되지 않음
        #
        # 2. 현재 테스트 중인 정답(target)의 점수 저장
        #    힌트: scores_np[target_id]로 접근
        answer_score = cpy_scores[target_id]
        #
        # 3. filter_ids(모든 정답들)를 순회하면서 처리
        #    힌트: 현재 테스트 중인 target은 제외해야 함 (if 조건 사용)
        #    힌트: 다른 정답들의 점수를 target 점수보다 낮게 설정
        #    힌트: 예를 들어 target_score - 1로 설정
        cpy_scores[filter_ids] = answer_score - 1
        #
        # 4. Target보다 높은 점수를 가진 entity 개수 세기
        #    힌트: numpy 비교 연산 (scores_np > target_score)는 boolean 배열 반환
        #    힌트: np.sum()으로 True 개수 세기
        #    힌트: rank는 1부터 시작하므로 +1 필요
        rank = np.sum(cpy_scores > answer_score) + 1
        #
        # 시각적 이해:
        # 원래 점수:    [3.2, 8.5, 7.8, 7.2, 2.1]  (Bob=8.5가 target)
        # Filter 후:    [3.2, 8.5, 7.5, 7.5, 2.1]  (Charlie, David 낮춤)
        # Bob보다 높은 점수: 0개 → rank = 1
        return rank

if __name__ == "__main__":
    # ========================================
    # 전체 학습 & 평가 파이프라인
    # ========================================

    print("=== 1. 데이터 로드 ===")
    data_dir = "../data"
    dataset_name = "WD20K100v1"
    setting = 'Transductive'  # 'Transductive' or 'Inductive'

    # 데이터 로드
    dataloader = DataLoader(data_dir, dataset_name, setting=setting)
    dataloader.load_train()
    dataloader.load_valid()
    dataloader.load_test()

    if setting == 'Inductive':
        dataloader.load_msg()  # msg.txt 로드 (Inductive 전용)

    # Train facts tensor 변환
    pri_tensor_train = dataloader.kg_train.to_tensors()

    # Inference facts tensor 변환
    if setting == 'Transductive':
        pri_tensor_inf = pri_tensor_train  # Train과 동일
        kg_for_eval = dataloader.kg_train
    else:
        pri_tensor_inf = dataloader.kg_inf.to_tensors()
        kg_for_eval = dataloader.kg_inf

    # 통계 출력
    print(f"Setting: {setting}")
    print(f"[Train] Entity: {dataloader.kg_train.num_ent}, Relation: {dataloader.kg_train.num_rel}, Facts: {len(dataloader.kg_train.facts)}")
    if setting == 'Inductive':
        print(f"[Inference] Entity: {dataloader.kg_inf.num_ent}, Relation: {dataloader.kg_inf.num_rel}, Facts: {len(dataloader.kg_inf.facts)}")
    print(f"Valid facts: {len(dataloader.valid_facts)}")
    print(f"Test facts: {len(dataloader.test_facts)}")

    print("\n=== 2. 모델 생성 ===")
    # 모델은 train KG 기준으로 생성
    model = SimpleModel(
        num_ent=dataloader.kg_train.num_ent,
        num_rel=dataloader.kg_train.num_rel,
        dim=32,
        num_layers=2
    )
    print(f"모델 생성 완료: dim=32, layers=2")

    print("\n=== 3. 학습 시작 ===")
    trainer = SimpleTrainer(
        model=model,
        kg_train=dataloader.kg_train,
        lr=1e-3,
        label_smoothing=0.1
    )
    num_epochs = 100
    batch_size = 128
    for i in tqdm(range(num_epochs), desc="Overall Progress"):
        loss = trainer.train_epoch(pri_tensor_train, batch_size=batch_size)
        if (i+1) % 10 == 0:
            tqdm.write(f"Epoch {i+1}/{num_epochs}, Loss: {loss:.4f}")

    print("\n=== 4. 평가 시작 ===")
    evaluator = SimpleEvaluator(model, dataloader.kg_train, kg_for_eval)
    print("Evaluating on test set...")
    metrics = evaluator.evaluate(dataloader.test_facts, pri_tensor_inf)
    # 힌트: SimpleEvaluator 인스턴스 생성
    # 힌트: evaluate() 호출해서 metrics 얻기
    # 힌트: metrics는 딕셔너리 (MR, MRR, Hits@1, Hits@3, Hits@10)
    # 힌트: 결과를 보기 좋게 출력
    print("\n" + "=" * 60)
    print("Test Results (Filtered Evaluation)")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name:10s}: {value:.4f}")
    print("=" * 60)