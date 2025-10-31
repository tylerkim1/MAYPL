"""
Simple MAYPL Model (간단한 버전)

이 파일은 당신이 직접 구현할 공간입니다.
SimpleKG + SimpleInitLayer를 통합하여 완전한 모델을 만들어봅시다!
"""

import torch
import torch.nn as nn
from typing import Tuple
from my_maypl import SimpleKG
from init_layer import SimpleInitLayer


# ============================================================
# SimpleModel을 만들어보세요
# ============================================================
# 요구사항:
# - __init__: Init_Layer를 여러 개 쌓기, 초기 임베딩 생성
# - forward: 모든 Init_Layer를 통과시켜서 최종 임베딩 생성
# - predict: 주어진 query의 빈칸을 예측

class SimpleModel(nn.Module):
    """간단한 MAYPL 모델

    구조:
    1. 초기 임베딩 (모든 entity/relation이 같은 값으로 시작)
    2. Init_Layer를 여러 번 통과 (구조 정보 학습)
    3. 최종 임베딩으로 예측
    """

    def __init__(self, num_ent: int, num_rel: int, dim: int = 16, num_layers: int = 2):
        """
        Args:
            num_ent: Entity 개수
            num_rel: Relation 개수
            dim: Embedding 차원
            num_layers: Init_Layer 개수
        """
        # TODO: 다음을 구현하세요
        # 1. super().__init__() 호출
        super().__init__()

        # 2. 파라미터 저장
        # self.num_ent = num_ent
        self.num_ent = num_ent
        # self.num_rel = num_rel
        self.num_rel = num_rel
        # self.dim = dim
        self.dim = dim
        # self.num_layers = num_layers
        self.num_layers = num_layers

        # 3. 초기 임베딩 생성 (학습 가능한 파라미터)
        # 각 entity/relation을 다르게 초기화 (차별화 보장)
        self.emb_ent = nn.Parameter(torch.empty(num_ent, dim))
        self.emb_rel = nn.Parameter(torch.empty(num_rel, dim))
        nn.init.xavier_normal_(self.emb_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.emb_rel, gain=nn.init.calculate_gain('relu'))
        # 힌트: nn.Parameter는 학습 가능한 텐서를 만듭니다

        # 4. Init_Layer들을 리스트로 저장
        # layers = []
        layers = []
        for _ in range(self.num_layers):
            layers.append(SimpleInitLayer(self.dim, self.num_ent, self.num_rel))
        self.layers = nn.ModuleList(layers)
        # for _ in range(num_layers):
        #     layers.append(SimpleInitLayer(dim, num_ent, num_rel))
        # self.layers = nn.ModuleList(layers)
        # 힌트: nn.ModuleList는 여러 레이어를 리스트로 관리합니다

        pass

    def forward(self, pri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pri: Primary triplets (num_fact, 3)

        Returns:
            emb_ent: 최종 Entity 임베딩 (num_ent, dim)
            emb_rel: 최종 Relation 임베딩 (num_rel, dim)
        """
        # TODO: 다음 단계를 구현하세요

        # Step 1: 초기 임베딩 사용 (각 entity/relation이 독립적)
        emb_ents = self.emb_ent.clone()
        emb_rels = self.emb_rel.clone()
        # - init_emb_ent/rel을 num_ent/num_rel개만큼 복제
        # 힌트: repeat(n, 1)으로 n번 복제하면서 gradient 계산 가능

        # Step 2: 모든 Init_Layer를 순차적으로 통과
        for layer in self.layers:
            emb_ents, emb_rels = layer(emb_ents, emb_rels, pri)
        # for layer in self.layers:
        #     emb_ent, emb_rel = layer(emb_ent, emb_rel, pri)

        # Step 3: 최종 임베딩 반환
        return emb_ents, emb_rels

    def predict(self, query_pri: torch.Tensor, pri: torch.Tensor) -> torch.Tensor:
        """주어진 query의 빈칸을 예측

        Args:
            query_pri: Query triplet (1, 3) - 예측할 위치는 -1로 표시
                      예: [0, -1, 2] → relation 예측
                          [-1, 0, 1] → head entity 예측
            pri: 전체 facts (num_fact, 3)

        Returns:
            scores: 각 후보에 대한 점수 (num_candidates,)
        """
        # TODO: 다음 단계를 구현하세요

        # Step 1: Forward pass로 최종 임베딩 얻기
        emb_ents, emb_rels = self.forward(pri)
        # emb_ent, emb_rel = self.forward(pri)

        # Step 2: Query에서 어느 위치를 예측할지 찾기
        pred_pos = (query_pri == -1).nonzero(as_tuple=True)[1].item()
        # 힌트: -1의 위치를 찾습니다 (0=head, 1=rel, 2=tail)

        # Step 3: 예측 대상에 따라 다르게 처리
        if pred_pos == 0:  # head entity 예측
            rel_id = query_pri[0, 1].item()
            tail_id = query_pri[0, 2].item()

            rel_emb = emb_rels[rel_id]
            tail_emb = emb_ents[tail_id]

            query_emb = rel_emb + tail_emb
            scores = torch.inner(query_emb, emb_ents)
        elif pred_pos == 1:  # relation 예측
            head_id = query_pri[0, 0].item()
            tail_id = query_pri[0, 2].item()

            head_emb = emb_ents[head_id]
            tail_emb = emb_ents[tail_id]

            query_emb = head_emb + tail_emb
            scores = torch.inner(query_emb, emb_rels)
        else:  # tail entity 예측
            head_id = query_pri[0, 0].item()
            rel_id = query_pri[0, 1].item()

            head_emb = emb_ents[head_id]
            rel_emb = emb_rels[rel_id]

            query_emb = head_emb + rel_emb
            scores = torch.inner(query_emb, emb_ents)

        return scores

# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=== SimpleModel 테스트 ===\n")

    # 1. Knowledge Graph 생성
    kg = SimpleKG()
    facts = [
        ("Alice", "knows", "Bob"),
        ("Bob", "knows", "Charlie"),
        ("Alice", "works_at", "Google"),
        ("Bob", "works_at", "Facebook"),
        ("Charlie", "lives_in", "Seoul"),
    ]

    for head, rel, tail in facts:
        kg.add_fact(head, rel, tail)

    kg.print_info()
    pri = kg.to_tensors()

    # 2. 모델 생성
    model = SimpleModel(
        num_ent=kg.num_ent,
        num_rel=kg.num_rel,
        dim=16,
        num_layers=2
    )

    print(f"\n=== 모델 구조 ===")
    print(f"Entity 개수: {kg.num_ent}")
    print(f"Relation 개수: {kg.num_rel}")
    print(f"Embedding 차원: 16")
    print(f"Init_Layer 개수: 2")

    # 3. Forward pass 테스트
    print(f"\n=== Forward Pass ===")
    emb_ent, emb_rel = model.forward(pri)
    print(f"Entity Embedding shape: {emb_ent.shape}")
    print(f"Relation Embedding shape: {emb_rel.shape}")

    # 4. Prediction 테스트
    print(f"\n=== Prediction Test ===")
    # Query: (Alice, ?, Bob) → relation 예측
    query = torch.tensor([[kg.ent2id["Alice"], -1, kg.ent2id["Bob"]]])
    scores = model.predict(query, pri)
    print(f"Query: (Alice, ?, Bob)")
    print(f"Scores shape: {scores.shape}")
    print(f"Predicted relation ID: {torch.argmax(scores).item()}")
    print(f"Predicted relation: {kg.id2rel[torch.argmax(scores).item()]}")
