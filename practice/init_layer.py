"""
Init_Layer 구현 (Structure-driven Initialization)

이 파일은 당신이 직접 구현할 공간입니다.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ============================================================
# SimpleInitLayer를 만들어보세요
# ============================================================
# 요구사항:
# - __init__: 선형 변환 레이어와 LayerNorm 정의
# - forward: Entity와 Relation이 메시지를 주고받는 로직

class SimpleInitLayer(nn.Module):
    """Structure-driven Initialization Layer

    Entity와 Relation이 메시지를 주고받으면서 구조 정보를 학습합니다.

    메시지 흐름:
    1. Entity → Relation: 각 entity가 relation에게 메시지 보냄
    2. Relation → Entity: relation이 entity에게 메시지 보냄
    """

    def __init__(self, dim: int, num_ent: int, num_rel: int):
        """
        Args:
            dim: Embedding 차원 (예: 8, 16, 32, ...)
            num_ent: Entity 개수
            num_rel: Relation 개수
        """
        # TODO: 다음을 구현하세요
        # 1. super().__init__() 호출 (nn.Module 초기화)
        super().__init__()
        # 2. self.dim, self.num_ent, self.num_rel 저장
        self.dim = dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        # 3. self.proj_ent_to_rel: Entity → Relation 메시지용 선형 변환
        #    (dim → dim 크기의 Linear layer)
        self.proj_ent_to_rel = nn.Linear(dim, dim)
        # 4. self.proj_rel_to_ent: Relation → Entity 메시지용 선형 변환
        self.proj_rel_to_ent = nn.Linear(dim, dim)
        # 5. self.ent_ln: Entity 임베딩용 LayerNorm
        self.ent_ln = nn.LayerNorm(dim)
        # 6. self.rel_ln: Relation 임베딩용 LayerNorm
        self.rel_ln = nn.LayerNorm(dim)

    def forward(self, emb_ent: torch.Tensor, emb_rel: torch.Tensor,
                pri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            emb_ent: Entity 임베딩 (num_ent, dim)
            emb_rel: Relation 임베딩 (num_rel, dim)
            pri: Primary triplets (num_fact, 3) - [head_id, rel_id, tail_id]

        Returns:
            new_emb_ent: 업데이트된 Entity 임베딩 (num_ent, dim)
            new_emb_rel: 업데이트된 Relation 임베딩 (num_rel, dim)
        """
        # TODO: 다음 단계를 구현하세요

        # Step 1: pri에서 heads, rels, tails 추출
        # heads = pri[:, 0]
        heads = pri[:, 0]
        # rels = pri[:, 1]
        rels = pri[:, 1]
        # tails = pri[:, 2]
        tails = pri[:, 2]

        # Step 2: Entity → Relation 메시지
        # - proj_ent_to_rel(emb_ent)를 계산
        heads_emb = torch.index_select(emb_ent, 0, heads)
        tails_emb = torch.index_select(emb_ent, 0, tails)
        heads_to_rel = self.proj_ent_to_rel(heads_emb)
        tails_to_rel = self.proj_ent_to_rel(tails_emb)
        # - heads와 tails에 해당하는 메시지를 더하기
        msg_ent_to_rel = heads_to_rel + tails_to_rel
        # - 결과: msg_ent_to_rel (shape: num_fact, dim)

        # Step 3: Relation 임베딩 업데이트
        # - msg_ent_to_rel를 rels별로 합산 (같은 relation끼리 모으기)
        msg_rels = torch.zeros((self.num_rel, self.dim))
        msg_rels.index_add_(0, rels, msg_ent_to_rel)
        # - 각 relation이 받은 메시지 개수로 나누기 (평균)
        msg_count_rels = torch.bincount(rels, minlength=self.num_rel).unsqueeze(1)
        msg_count_rels = torch.where(msg_count_rels == 0, 1, msg_count_rels)
        msg_rels /= msg_count_rels
        # - LayerNorm 적용
        normalized_msg_rels = self.rel_ln(msg_rels)
        # - 기존 emb_rel과 더하기: emb_rel + normalized_msg
        new_emb_rel = emb_rel + normalized_msg_rels

        # Step 4: Relation → Entity 메시지
        # - proj_rel_to_ent(new_emb_rel)를 계산
        # - rels에 해당하는 메시지 선택
        rels_emb = torch.index_select(new_emb_rel, 0, rels)
        rels_to_ent = self.proj_rel_to_ent(rels_emb)
        # - 결과: msg_rel_to_ent (shape: num_fact, dim)
        msg_rel_to_ent = rels_to_ent

        # Step 5: Entity 임베딩 업데이트
        # - msg_rel_to_ent를 heads와 tails별로 합산 (같은 entity끼리 모으기)
        msg_ents = torch.zeros((self.num_ent, self.dim))
        msg_ents.index_add_(0, heads, msg_rel_to_ent)
        msg_ents.index_add_(0, tails, msg_rel_to_ent)
        # - 각 entity가 받은 메시지 개수로 나누기 (평균)
        msg_count_heads = torch.bincount(heads, minlength=self.num_ent).unsqueeze(1)
        msg_count_heads = torch.where(msg_count_heads == 0, 1, msg_count_heads)
        msg_count_tails = torch.bincount(tails, minlength=self.num_ent).unsqueeze(1)
        msg_count_tails = torch.where(msg_count_tails == 0, 1, msg_count_tails)
        msg_count_ents = msg_count_heads + msg_count_tails
        msg_ents /= msg_count_ents
        # - LayerNorm 적용
        normalized_msg_ents = self.ent_ln(msg_ents)
        # - 기존 emb_ent와 더하기: emb_ent + normalized_msg
        new_emb_ent = emb_ent + normalized_msg_ents

        # Step 6: new_emb_ent, new_emb_rel 반환
        return new_emb_ent, new_emb_rel


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=== SimpleInitLayer 테스트 ===\n")

    # 간단한 데이터 생성
    num_ent = 4
    num_rel = 2
    dim = 8

    # 임베딩 초기화 (랜덤)
    emb_ent = torch.randn((num_ent, dim))
    emb_rel = torch.randn((num_rel, dim))

    # Facts 생성 (예: (0,0,1), (1,0,2), (0,1,3))
    pri = torch.tensor([
        [0, 0, 1],  # Alice -knows-> Bob
        [1, 0, 2],  # Bob -knows-> Charlie
        [0, 1, 3],  # Alice -works_at-> Google
    ], dtype=torch.long)

    print(f"Entity embedding shape: {emb_ent.shape}")
    print(f"Relation embedding shape: {emb_rel.shape}")
    print(f"Facts shape: {pri.shape}\n")

    # Init_Layer 생성
    init_layer = SimpleInitLayer(dim=dim, num_ent=num_ent, num_rel=num_rel)

    # Forward pass
    new_emb_ent, new_emb_rel = init_layer(emb_ent, emb_rel, pri)

    print(f"After SimpleInitLayer:")
    print(f"  new_emb_ent shape: {new_emb_ent.shape}")
    print(f"  new_emb_rel shape: {new_emb_rel.shape}")
    print(f"\nEntity embeddings (first 2):\n{new_emb_ent[:2]}")
    print(f"\nRelation embeddings:\n{new_emb_rel}")
