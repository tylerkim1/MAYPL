"""
나만의 MAYPL 구현 (완전 처음부터)

이 파일은 당신이 직접 구현할 공간입니다.
각 단계별로 필요한 코드를 작성해보세요!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


# ============================================================
# 1단계: 간단한 Knowledge Graph 클래스를 만들어보세요
# ============================================================
# 요구사항:
# - Entity와 Relation을 이름에서 ID로 변환 가능
# - add_fact(head, relation, tail) 함수
# - to_tensors() 함수로 facts를 tensor로 변환
# - print_info() 함수로 정보 출력

class SimpleKG:
    """간단한 Knowledge Graph 데이터 로더"""

    def __init__(self):
        # TODO: Entity와 Relation을 ID로 변환하기 위한 자료구조 정의
        # - ent2id: entity_name -> id (dict)
        self.ent2id = {}
        # - id2ent: id -> entity_name (list)
        self.id2ent = []
        # - rel2id: relation_name -> id (dict)
        self.rel2id = {}
        # - id2rel: id -> relation_name (list)
        self.id2rel = []
        # - facts: [(head_id, rel_id, tail_id), ...] (list)
        self.facts = []
        self.num_ent = 0
        self.num_rel = 0

    def add_fact(self, head: str, relation: str, tail: str):
        """Fact를 데이터셋에 추가"""
        # TODO: 다음 단계를 구현하세요
        # 1. head entity가 ent2id에 없으면 등록
        if head not in self.ent2id:
            self.ent2id[head] = self.num_ent
            self.id2ent.append(head)
            self.num_ent += 1
        # 2. tail entity가 ent2id에 없으면 등록
        if tail not in self.ent2id:
            self.ent2id[tail] = self.num_ent
            self.id2ent.append(tail)
            self.num_ent += 1
        # 3. relation이 rel2id에 없으면 등록
        if relation not in self.rel2id:
            self.rel2id[relation] = self.num_rel
            self.id2rel.append(relation)
            self.num_rel += 1
        # 4. fact를 ID로 변환해서 self.facts에 추가
        self.facts.append((self.ent2id[head], self.rel2id[relation], self.ent2id[tail]))

    def to_tensors(self):
        """모든 facts를 tensor로 변환"""
        # TODO: facts를 torch.tensor로 변환해서 반환
        if len(self.facts) == 0:
            return torch.tensor((0, 3), dtype=torch.long)
        facts_tensor = torch.tensor(self.facts, dtype=torch.long)
        # 반환 형태: (num_fact, 3) 크기의 tensor
        return facts_tensor

    def print_info(self):
        """데이터셋 정보 출력"""
        # TODO: 다음 정보를 출력하세요
        # - Entity 개수
        print(f"Entity 개수: {self.num_ent}")
        # - Relation 개수
        print(f"Relation 개수: {self.num_rel}")
        # - Fact 개수
        print(f"Fact 개수: {len(self.facts)}")
        # - Entity 목록
        print(f"Entity 목록: {self.id2ent}")
        # - Relation 목록
        print(f"Relation 목록: {self.id2rel}")
        # - 모든 Facts (이름 형태로)
        print(f"모든 Facts")
        for head, rel, tail in self.facts:
            print((self.id2ent[head], self.id2rel[rel], self.id2ent[tail]))


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=== Step 1: SimpleKG 구현 테스트 ===\n")

    # KG 생성
    kg = SimpleKG()

    # Facts 추가
    facts = [
        ("Alice", "knows", "Bob"),
        ("Bob", "knows", "Charlie"),
        ("Alice", "works_at", "Google"),
        ("Bob", "works_at", "Facebook"),
    ]

    for head, rel, tail in facts:
        kg.add_fact(head, rel, tail)

    # 정보 출력
    kg.print_info()

    # Tensor 변환
    pri = kg.to_tensors()
    print(f"\n=== Tensor Format ===")
    print(f"pri shape: {pri.shape}")
    print(f"pri:\n{pri}")
