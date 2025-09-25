import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import copy
import time

class HKG(Dataset):
    def __init__(self, datasets_dir, dataset_name, logger, setting = 'Transductive', msg_add_tr = False):
        self.dataset_dir = os.path.join(datasets_dir, dataset_name)+"/"
        self.logger = logger
        self.msg_add_tr = msg_add_tr # Inductive 세팅에서 msg.txt에 train.txt의 fact를 추가할지 여부 (추론 단계에서 관측하는 fact의 풀을 늘림)
        if self.msg_add_tr: # msg_add_tr이 설정되어 있다면 setting은 반드시 Inductive여야 함
            assert setting == 'Inductive'
        
        logger.info("Loading Dataset...")

        self.ent2id_train = {} # entity를 key로, id를 value로 하는 dictionary
        self.id2ent_train = [] # id를 index로, entity를 value로 하는 list
        self.num_ent_train = 0 # 현재까지 등록된 entity의 개수

        self.ent2id_inf = {} # inference 단계에서 entity를 key로, id를 value로 하는 dictionary
        self.id2ent_inf = [] # inference 단계에서 id를 index로, entity를 value로 하는 list
        self.num_ent_inf = 0 # inference 단계에서 현재까지 등록된 entity의 개수

        self.rel2id_train = {} # relation을 key로, id를 value로 하는 dictionary
        self.rel_idxs_train = [] # 현재까지 등록된 relation의 id를 담는 list
        self.id2rel_train = [] # id를 index로, relation을 value로 하는 list
        self.num_rel_train = 0 # 현재까지 등록된 relation의 개수

        self.rel2id_inf = {} # inference 단계에서 relation을 key로, id를 value로 하는 dictionary
        self.rel_idxs_inf = [] # inference 단계에서 현재까지 등록된 relation의 id를 담는 list
        self.id2rel_inf = [] # inference 단계에서 id를 index로, relation을 value로 하는 list
        self.num_rel_inf = 0 # inference 단계에서 현재까지 등록된 relation의 개수

        self.train = [] # train에 있는 모든 fact를 담는 list (fact는 entity와 relation의 id로 이루어진 list)
        self.inference = []  # inference에 사용할 모든 fact를 담는 list
        self.valid = [] # validation에 사용할 모든 fact를 담는 list
        self.test = [] # test에 사용할 모든 fact를 담는 list
        if setting == 'Transductive': # transductive 세팅
            self.parse_facts()  
            # transductive 세팅에서는 inference에 train 데이터를 그대로 사용
            self.inference = self.train
            self.ent2id_inf = self.ent2id_train
            self.id2ent_inf = self.id2ent_train
            self.num_ent_inf = self.num_ent_train

            self.rel2id_inf = self.rel2id_train
            self.rel_idxs_inf = self.rel_idxs_train
            self.id2rel_inf = self.id2rel_train
            self.num_rel_inf = self.num_rel_train

        elif setting == 'Inductive': # inductive 세팅
            self.parse_facts_ind()
        
        else:
            raise NotImplementedError

        logger.info("Dataset Loaded Successfully!")
        self.num_train = len(self.train) # train에 있는 fact의 개수
        self.num_inf = len(self.inference) # inference에 있는 fact의 개수

        logger.info("Generating Dictionaries for Filtering...")
        self.eval_filter_dict = {} # evaluation 시 정답으로 인정될 수 있는 모든 것들의 목록을 담는 dictionary (key: (primary_triplet, qualifiers), value: [정답 id list])
        self.construct_eval_filter_dict() # eval_filter_dict 생성
        logger.info("Successfully Generated Filtering Dictionary!")

        logger.info("Model Input Construction...")
        # pri_train과 train의 길이는 같다.
        self.pri_train = [] # train에 등장하는 primary triplet을 (head, relation, tail)의 튜플 형태로 담는 list (중복 포함), fact와 index를 나란히 함
        self.qual_train = [] # train에 등장하는 qualifier를 (qualifier_relation, qualifier_entity)의 튜플 형태로 담는 list (중복 포함)
        self.qual2fact_train = [] # qualifier에 해당하는 fact의 index를 담는 list (index: qualifier의 index, value: qualifier가 등장하는 fact의 index)
        self.ent2rel_train = [] # entity가 연결된 relation의 id를 담는 list (index: entity의 index, value: entity가 연결된 relation의 id)
        self.ent_train = [] # train에 등장하는 모든 entity의 id를 담는 list (중복 포함)
        self.ent2fact_train = [] # entity가 등장하는 fact의 index를 담는 list (index: entity의 index, value: entity가 등장하는 fact의 index)
        self.entloc_train = [] # train에 등장하는 모든 entity의 위치 정보를 담는 list (entity의 위치는 fact에서의 index)
        self.fact2len_train = [] # fact의 길이를 담는 list (index: fact의 index, value: fact의 길이)
        self.fact2nument_train = [] # fact에 등장하는 entity의 개수를 담는 list (index: fact의 index, value: fact에 등장하는 entity의 개수)
        self.fact2qualstart_train = [] # qual_train에서 fact에 등장하는 qualifier의 시작 index를 담는 list
        self.fact2entstart_train = [] # ent_train에서 fact에 등장하는 entity의 시작 index를 담는 list
        self.fact2entlocstart_train = [] # entloc_train에서 어디를 보면 되는지 index를 담는 list

        # inference 단계에서 사용하는 데이터
        self.pri_inf = []
        self.qual_inf = []
        self.qual2fact_inf = []
        self.ent2rel_inf = []
        self.ent_inf = []
        self.ent2fact_inf = []
        self.entloc_inf = []
        self.fact2len_inf = []
        self.fact2nument_inf = []
        self.fact2qualstart_inf = []
        self.fact2entstart_inf = []
        self.fact2entlocstart_inf = []

        self.construct_inputs() # train 데이터를 GPU에서 효율적으로 연산할 수 있도록 구조화
        if setting == 'Transductive': # transductive 세팅에서는 inference에 train 데이터를 그대로 사용
            self.pri_inf = self.pri_train
            self.qual_inf = self.qual_train
            self.qual2fact_inf = self.qual2fact_train
            self.ent2rel_inf = self.ent2rel_train
            self.ent_inf = self.ent_train
            self.ent2fact_inf = self.ent2fact_train
            self.entloc_inf = self.entloc_train
            self.fact2len_inf = self.fact2len_train
            self.fact2nument_inf = self.fact2nument_train
            self.fact2qualstart_inf = self.fact2qualstart_train
            self.fact2entstart_inf = self.fact2entstart_train
            self.fact2entlocstart_inf = self.fact2entlocstart_train
            self.rel_mask_inf = self.rel_mask_train

            self.hpair_inf = self.hpair_train
            self.fact2hpair_inf = self.fact2hpair_train
           
            self.tpair_inf = self.tpair_train
            self.fact2tpair_inf = self.fact2tpair_train

            self.qpair_inf = self.qpair_train
            self.qual2qpair_inf = self.qual2qpair_train
        
        elif setting == 'Inductive':
            self.construct_inputs_inf()

        else:
            raise NotImplementedError
        
        # bincount: 각 정수가 등장하는 횟수를 세는 함수
        # minlength: 결과 tensor의 최소 길이 지정 (pair 목록 길이로 설정하여, 등장하지 않은 pair의 빈도도 0으로 포함)
        self.hpair_freq_inf = torch.bincount(self.fact2hpair_inf, minlength = len(self.hpair_inf)) # hpair_freq_inf: inference에 등장하는 hpair의 빈도
        self.tpair_freq_inf = torch.bincount(self.fact2tpair_inf, minlength = len(self.tpair_inf)) # tpair_freq_inf: inference에 등장하는 tpair의 빈도
        self.qpair_freq_inf = torch.bincount(self.qual2qpair_inf, minlength = len(self.qpair_inf)) # qpair_freq_inf: inference에 등장하는 qpair의 빈도

        logger.info("Successfully Saved Model Inputs!")

        logger.info("Evaluation Query Generation...")

        self.valid_query = [] # validation에 사용할 query를 담는 list (query는 (primary_triplet, qualifiers)의 튜플 형태)
        self.valid_answer = [] # validation에 사용할 query에 대한 정답을 담는 list (index: query의 index, value: 정답 id list)
        self.test_query = [] # test에 사용할 query를 담는 list
        self.test_answer = [] # test에 사용할 query에 대한 정답을 담는 list
        self.construct_query() # validation, test에 있는 fact를 바탕으로 평가 시 사용할 query와 정답을 생성
        logger.info("Evaluation Query Generated Successfully!")

        self.train_sorted_by_qual_len = {}
        if self.train: # self.train이 비어있지 않을 때만 실행
            # 각 fact의 원본 인덱스와 qualifier 개수를 함께 저장
            facts_with_len = [(idx, (len(fact) - 3) // 2) for idx, fact in enumerate(self.train)]

            # qualifier 개수를 기준으로 오름차순 정렬
            facts_with_len.sort(key=lambda x: x[1])

            # qualifier 개수별로 fact의 원본 인덱스를 그룹화하여 사전에 저장
            for idx, num_quals in facts_with_len:
                if num_quals not in self.train_sorted_by_qual_len:
                    self.train_sorted_by_qual_len[num_quals] = []
                self.train_sorted_by_qual_len[num_quals].append(idx)

        
    def parse_facts(self): # transductive 세팅에서 train, valid, test 파일을 읽어서 entity, relation, fact를 등록하는 함수
        # index가 짝수면 entity, 홀수면 relation
        with open(self.dataset_dir + "train.txt") as f: # train할 fact를 저장하는 블록
            for line in f.readlines():
                fact = [] # fact는 현재 줄에 있는 entity와 relation의 id로 이루어진 list
                elements = line.strip().split("\t") # elements는 fact의 요소
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        if elements[idx] not in self.ent2id_train: # entity가 아직 등록되지 않은 경우
                            self.id2ent_train.append(elements[idx]) # id2ent_train에 entity 추가
                            self.ent2id_train[elements[idx]] = self.num_ent_train # ent2id_train에 entity를 key로 하고, value가 id인 쌍 추가
                            self.num_ent_train += 1 # entity 개수 증가
                        fact.append(self.ent2id_train[elements[idx]]) # fact에 entity id 추가
                    else: # 홀수 index -> relation
                        if elements[idx] not in self.rel2id_train: # relation이 아직 등록되지 않은 경우
                            self.rel_idxs_train.append(self.num_rel_train) # rel_idxs_train에 relation id 추가 (num_rel_train과 같음)
                            self.rel2id_train[elements[idx]] = self.num_rel_train # rel2id_train에 relation을 key로 하고, value가 id인 쌍 추가
                            self.id2rel_train.append(elements[idx]) # id2rel_train에 relation 추가
                            self.num_rel_train += 1 # relation 개수 증가
                        fact.append(self.rel2id_train[elements[idx]]) # fact에 relation id 추가
                self.train.append(fact) # train에 fact 추가

        with open(self.dataset_dir + "valid.txt") as f: # validation에 사용할 fact를 저장하는 블록
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        if elements[idx] not in self.ent2id_train: # entity가 train에서도 아직 등록되지 않은 경우
                            self.id2ent_train.append(elements[idx]) # id2ent_train에 entity 추가
                            self.ent2id_train[elements[idx]] = self.num_ent_train # ent2id_train에 entity를 key로 하고, value가 id인 쌍 추가
                            self.num_ent_train += 1 # entity 개수 증가
                        fact.append(self.ent2id_train[elements[idx]]) # fact에 entity id 추가
                    else: # 홀수 index -> relation
                        if elements[idx] not in self.rel2id_train: # relation이 train에서도 아직 등록되지 않은 경우
                            self.rel_idxs_train.append(self.num_rel_train) # rel_idxs_train에 relation id 추가 (num_rel_train과 같음)
                            self.rel2id_train[elements[idx]] = self.num_rel_train # rel2id_train에 relation을 key로 하고, value가 id인 쌍 추가
                            self.id2rel_train.append(elements[idx]) # id2rel_train에 relation 추가
                            self.num_rel_train += 1 # relation 개수 증가
                        fact.append(self.rel2id_train[elements[idx]]) # fact에 relation id 추가
                self.valid.append(fact) # valid에 fact 추가

        with open(self.dataset_dir + "test.txt") as f: # test에 사용할 fact를 저장하는 블록
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        if elements[idx] not in self.ent2id_train: # entity가 train, valid에서도 아직 등록되지 않은 경우
                            self.id2ent_train.append(elements[idx]) # id2ent_train에 entity 추가
                            self.ent2id_train[elements[idx]] = self.num_ent_train # ent2id_train에 entity를 key로 하고, value가 id인 쌍 추가
                            self.num_ent_train += 1 # entity 개수 증가
                        fact.append(self.ent2id_train[elements[idx]]) # fact에 entity id 추가
                    else: # 홀수 index -> relation
                        if elements[idx] not in self.rel2id_train: # relation이 train, valid에서도 아직 등록되지 않은 경우
                            self.rel_idxs_train.append(self.num_rel_train) # rel_idxs_train에 relation id 추가 (num_rel_train과 같음)
                            self.rel2id_train[elements[idx]] = self.num_rel_train # rel2id_train에 relation을 key로 하고, value가 id인 쌍 추가
                            self.id2rel_train.append(elements[idx]) # id2rel_train에 relation 추가
                            self.num_rel_train += 1 # relation 개수 증가
                        fact.append(self.rel2id_train[elements[idx]]) # fact에 relation id 추가
                self.test.append(fact) # test에 fact 추가

    def parse_facts_ind(self): # inductive 세팅에서 train, msg, valid, test 파일을 읽어서 entity, relation, fact를 등록하는 함수
        with open(self.dataset_dir + "train.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        if elements[idx] not in self.ent2id_train: # entity가 아직 등록되지 않은 경우
                            self.id2ent_train.append(elements[idx]) # id2ent_train에 entity 추가
                            self.ent2id_train[elements[idx]] = self.num_ent_train # ent2id_train에 entity를 key로 하고, value가 id인 쌍 추가
                            self.num_ent_train += 1 # entity 개수 증가
                        fact.append(self.ent2id_train[elements[idx]]) # fact에 entity id 추가
                    else: # 홀수 index -> relation
                        if elements[idx] not in self.rel2id_train: # relation이 아직 등록되지 않은 경우
                            self.rel_idxs_train.append(self.num_rel_train) # rel_idxs_train에 relation id 추가 (num_rel_train과 같음)
                            self.rel2id_train[elements[idx]] = self.num_rel_train # rel2id_train에 relation을 key로 하고, value가 id인 쌍 추가
                            self.id2rel_train.append(elements[idx]) # id2rel_train에 relation 추가
                            self.num_rel_train += 1 # relation 개수 증가
                        fact.append(self.rel2id_train[elements[idx]]) # fact에 relation id 추가
                self.train.append(fact) # train에 fact 추가
        if self.msg_add_tr: # msg_add_tr이 설정되어 있다면 추론 단계에서 사용하는 msg.txt에 train.txt의 fact를 추가
            with open(self.dataset_dir + "train.txt") as f:
                for line in f.readlines():
                    fact = []
                    elements = line.strip().split("\t")
                    for idx in range(len(elements)):
                        if idx % 2 == 0: # 짝수 index -> entity
                            if elements[idx] not in self.ent2id_inf: # entity가 아직 등록되지 않은 경우
                                self.id2ent_inf.append(elements[idx]) # id2ent_inf에 entity 추가
                                self.ent2id_inf[elements[idx]] = self.num_ent_inf # ent2id_inf에 entity를 key로 하고, value가 id인 쌍 추가
                                self.num_ent_inf += 1 # entity 개수 증가
                            fact.append(self.ent2id_inf[elements[idx]]) # fact에 entity id 추가
                        else: # 홀수 index -> relation
                            if elements[idx] not in self.rel2id_inf: # relation이 아직 등록되지 않은 경우
                                self.rel_idxs_inf.append(self.num_rel_inf) # rel_idxs_inf에 relation id 추가 (num_rel_inf와 같음)
                                self.rel2id_inf[elements[idx]] = self.num_rel_inf # rel2id_inf에 relation을 key로 하고, value가 id인 쌍 추가
                                self.id2rel_inf.append(elements[idx]) # id2rel_inf에 relation 추가
                                self.num_rel_inf += 1 # relation 개수 증가
                            fact.append(self.rel2id_inf[elements[idx]]) # fact에 relation id 추가
                    self.inference.append(fact) # inference에 fact 추가

        with open(self.dataset_dir + "msg.txt") as f: # 그래프에서 추론을 수행하기 전에 사용하는 관찰된 사실을 담고 있는 파일
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        if elements[idx] not in self.ent2id_inf: # entity가 아직 등록되지 않은 경우
                            self.id2ent_inf.append(elements[idx]) # id2ent_inf에 entity 추가
                            self.ent2id_inf[elements[idx]] = self.num_ent_inf # ent2id_inf에 entity를 key로 하고, value가 id인 쌍 추가
                            self.num_ent_inf += 1 # entity 개수 증가
                        fact.append(self.ent2id_inf[elements[idx]]) # fact에 entity id 추가
                    else: # 홀수 index -> relation
                        if elements[idx] not in self.rel2id_inf: # relation이 아직 등록되지 않은 경우
                            self.rel_idxs_inf.append(self.num_rel_inf) # rel_idxs_inf에 relation id 추가 (num_rel_inf와 같음)
                            self.rel2id_inf[elements[idx]] = self.num_rel_inf # rel2id_inf에 relation을 key로 하고, value가 id인 쌍 추가
                            self.id2rel_inf.append(elements[idx]) # id2rel_inf에 relation 추가
                            self.num_rel_inf += 1 # relation 개수 증가
                        fact.append(self.rel2id_inf[elements[idx]]) # fact에 relation id 추가
                self.inference.append(fact) # inference에 fact 추가
        with open(self.dataset_dir + "valid.txt") as f: # validation에 사용할 fact를 저장하는 블록
            for line in f.readlines(): 
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        fact.append(self.ent2id_inf[elements[idx]]) # validation 데이터에 있는 entity는 반드시 msg.txt에 존재하므로 바로 id로 변환
                    else: # 홀수 index -> relation 
                        fact.append(self.rel2id_inf[elements[idx]]) # validation 데이터에 있는 relation는 반드시 msg.txt에 존재하므로 바로 id로 변환
                self.valid.append(fact) # valid에 fact 추가
        
        with open(self.dataset_dir + "test.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0: # 짝수 index -> entity
                        fact.append(self.ent2id_inf[elements[idx]]) # test 데이터에 있는 entity는 반드시 msg.txt에 존재하므로 바로 id로 변환
                    else: # 홀수 index -> relation
                        fact.append(self.rel2id_inf[elements[idx]]) # test 데이터에 있는 relation는 반드시 msg.txt에 존재하므로 바로 id로 변환
                self.test.append(fact) # test에 fact 추가

    def construct_eval_filter_dict(self): # evaluation 시 정답으로 인정될 수 있는 모든 것들의 목록을 담는 dictionary를 생성하는 함수 (key: (primary_triplet, qualifiers), value: [정답 id list])
        for split in [self.inference, self.valid, self.test]: # split은 inference, valid, test에 있는 fact를 담고 있음
            for fact in split: # fact는 entity와 relation의 id로 이루어진 list
                for idx in range(len(fact)):
                    corrupted_fact = fact[:idx] + [-1] + fact[idx+1:] # idx 위치에 -1을 넣어서 해당 위치의 entity 또는 relation을 예측해야 함을 표시
                    primary_triplet = tuple(corrupted_fact[:3]) # (head, relation, tail)로 이루어진 튜플
                    qualifiers = [] # primary triplet 뒤에 붙는 (qualifier_relation, qualifier_entity)의 튜플을 담는 list
                    for i in range(len(corrupted_fact[3::2])): # qualifier는 길이가 2씩 증가하므로 2칸씩 건너뛰면서 확인
                        qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i])) # qualifiers list에 (qualifier_relation, qualifier_entity) 튜플 추가
                    filter_key_list = [primary_triplet] # filtering에 사용할 key를 담는 list
                    if len(qualifiers) != 0: # qualifier가 존재하는 경우, key에 qualifier도 포함
                        filter_key_list += sorted(qualifiers) # qualifier의 순서가 바뀌어도 동일한 key가 생성되도록, 정렬하여 추가
                    filter_key = tuple(filter_key_list) # filter_key_list를 튜플로 변환하여 filter_key로 설정
                    if filter_key not in self.eval_filter_dict: # filter_key가 아직 eval_filter_dict에 없는 경우, dict에 해당 key에 대한 자리 생성
                        self.eval_filter_dict[filter_key] = [] 
                    self.eval_filter_dict[filter_key].append(fact[idx]) # -1 위치에 들어가야 하는 entity 또는 relation의 id를 답으로 정하여 value로 추가

    def construct_query(self): # validation, test에 있는 fact를 바탕으로 평가 시 사용할 query와 정답을 생성하는 함수
        # validation에 사용할 query와 정답 생성
        vquery2idx = {} # (primary_triplet, qualifiers)를 key로, query의 id를 value로 하는 dictionary
        for fact_valid in self.valid: # fact_valid는 entity와 relation의 id로 이루어진 list
            for idx in range(len(fact_valid)):
                if idx % 2 == 1 or (idx > 0 and not self.rel_mask_inf[fact_valid[idx-1]]): # relation 예측이거나, inference 단계에서 등장하지 않는 relation과 연결된 entity 에 대한 예측인 경우 건너뜀
                    continue
                corrupted_fact = fact_valid[:idx] + [-1] + fact_valid[idx+1:] # entity 위치에 -1을 넣어서 해당 위치의 entity를 예측해야 함을 표시
                primary_triplet = tuple(corrupted_fact[:3]) # query의 primary triplet 부분 (head, relation, tail)
                qualifiers = []
                for i in range(len(corrupted_fact[3::2])): # qualifiers에 qualifier를 추가
                    qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i])) # (qualifier_relation, qualifier_entity) 튜플을 qualifiers에 추가
                filter_key_list = [primary_triplet] # filtering에 사용할 key를 담는 list, 우선 primary triplet 추가
                if len(qualifiers) != 0: # qualifier가 존재하는 경우, key에 qualifier도 포함
                    filter_key_list += sorted(qualifiers) # qualifier의 순서가 바뀌어도 동일한 key가 생성되도록, 정렬하여 추가
                filter_key = tuple(filter_key_list) # filter_key_list를 튜플로 변환하여 filter_key로 설정
                if filter_key not in vquery2idx: # filter_key가 아직 vquery2idx에 없는 경우, vquery2idx에 해당 key에 대한 자리 생성
                    vquery2idx[filter_key] = len(self.valid_query) # valid_query: query를 담는 list / filter_key를 key로 하고, query의 id를 value로 하는 dictionary
                    self.valid_query.append(filter_key) # valid_query에 query 추가 
                    self.valid_answer.append([fact_valid[idx]]) # valid_answer: query에 대한 정답을 담는 list (index: query의 id, value: 정답 id list)
                else: # filter_key가 이미 vquery2idx에 있는 경우, 해당 query에 대한 정답 리스트에 정답 추가
                    loc = vquery2idx[filter_key] # query의 id
                    self.valid_answer[loc].append(fact_valid[idx]) # valid_answer의 해당 query에 대한 정답 리스트에 정답 추가
        
        self.valid_answer = [a for _, a in sorted(zip(self.valid_query, self.valid_answer))] # query 순서에 맞게 valid_answer 정렬
        self.valid_query = sorted(self.valid_query) # query 정렬
        # test에 사용할 query와 정답 생성
        tquery2idx = {}
        for fact_test in self.test:
            for idx in range(len(fact_test)):
                if idx % 2 == 1 or (idx > 0 and not self.rel_mask_inf[fact_test[idx-1]]):
                    continue
                corrupted_fact = fact_test[:idx] + [-1] + fact_test[idx+1:]
                primary_triplet = tuple(corrupted_fact[:3])
                qualifiers = []
                for i in range(len(corrupted_fact[3::2])):
                    qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i]))
                filter_key_list = [primary_triplet]
                if len(qualifiers) != 0:
                    filter_key_list += sorted(qualifiers)
                filter_key = tuple(filter_key_list)
                if filter_key not in tquery2idx:
                    tquery2idx[filter_key] = len(self.test_query)
                    self.test_query.append(filter_key)
                    self.test_answer.append([fact_test[idx]])
                else:
                    loc = tquery2idx[filter_key]
                    self.test_answer[loc].append(fact_test[idx])
        self.test_answer = [a for _, a in sorted(zip(self.test_query, self.test_answer))]
        self.test_query = sorted(self.test_query)

    def construct_inputs(self): # 모델이 GPU에서 효율적으로 message passing을 수행할 수 있도록, 입력 데이터를 구조화하는 함수(tensor로 변환하고 GPU에 올림)
        self.rel_mask_train = torch.zeros((self.num_rel_train,)).bool().cuda() # relation의 id 중에서 train에 등장하는 relation의 id 위치를 True로 설정하는 mask
        self.rel_mask_train[self.rel_idxs_train] = 1 # train에 등장하는 relation의 id 위치를 True로 설정
        hpair2idx = {} # (head, relation) 쌍을 key로, id를 value로 하는 dictionary
        tpair2idx = {} # (tail, relation) 쌍을 key로, id를 value로 하는 dictionary
        qpair2idx = {} # (qualifier_relation, qualifier_entity) 쌍을 key로, id를 value로 하는 dictionary
        self.hpair_train = [] # train에 등장하는 hpair(head, relation)를 list 형태로 담는 list
        self.fact2hpair_train = [] # fact에 해당하는 hpair를 매핑하는 list (index: fact의 index, value: hpair의 id)
        self.tpair_train = [] # train에 등장하는 tpair(tail, relation)를 list 형태로 담는 list
        self.fact2tpair_train = [] # fact에 해당하는 tpair를 매핑하는 list (index: fact의 index, value: tpair의 id)
        self.qpair_train = [] # train에 등장하는 qpair(qualifier_relation, qualifier_entity)를 list 형태로 담는 list
        self.qual2qpair_train = [] # qualifier에 해당하는 qpair를 매핑하는 list (index: qualifier의 index, value: qpair의 id)
        for idx in range(self.num_train): # num_train은 train에 있는 fact의 개수, train에서 본 모든 fact에 대해 반복
            fact = self.train[idx] # fact는 entity와 relation의 id로 이루어진 list
            self.pri_train.append([fact[0], fact[1], fact[2]]) # pri_train에 (head, relation, tail) 추가 (id 형태로)
            self.fact2len_train.append(len(fact)) # fact의 길이를 저장하는 list (fact2len_train의 index는 fact의 index와 대응)
            self.fact2nument_train.append(1) # fact에 등장하는 entity의 개수를 저장하는 list (fact2nument_train의 index는 fact의 index와 대응)
            # GPU에서 효율적인 연산을 위해, fact에 등장하는 entity, relation, qualifier의 위치 정보를 미리 저장 (포인터 역할)
            self.fact2entlocstart_train.append(len(self.entloc_train)) # fact에 등장하는 entity의 위치 정보의 시작 위치 (entloc_train에서의 시작 위치)
            self.fact2qualstart_train.append(len(self.qual_train)) # fact에 등장하는 qualifier의 시작 위치 (qual_train에서의 시작 위치)
            self.fact2entstart_train.append(len(self.ent_train)) # fact에 등장하는 entity의 시작 위치 (ent_train에서의 시작 위치)
            hpair = (fact[0], fact[1]) # (head, relation)
            tpair = (fact[2], fact[1]) # (tail, relation)
            if hpair not in hpair2idx: # hpair가 아직 등록되지 않은 경우, hpair2idx에 등록하고 hpair_train에 추가
                hpair2idx[hpair] = len(self.hpair_train) # hpair2idx에 hpair를 key로 하고, value가 id인 쌍 추가
                self.hpair_train.append(list(hpair)) # hpair_train에 hpair 추가
            if tpair not in tpair2idx: # tpair가 아직 등록되지 않은 경우, tpair2idx에 등록하고 tpair_train에 추가
                tpair2idx[tpair] = len(self.tpair_train) # tpair2idx에 tpair를 key로 하고, value가 id인 쌍 추가
                self.tpair_train.append(list(tpair)) # tpair_train에 tpair 추가
            self.fact2hpair_train.append(hpair2idx[hpair]) # fact의 index에 해당하는 fact2hpair_train에 hpair의 id 추가
            self.fact2tpair_train.append(tpair2idx[tpair]) # fact의 index에 해당하는 fact2tpair_train에 tpair의 id 추가
            for i, ent in enumerate(fact[::2]): # fact에서 짝수 index에 해당하는 entity에 대해 반복, i는 entity의 순서 (0: head, 1: tail, 2 이상: qualifier entity)
                if i > 1: # i가 2 이상인 경우, 즉 세 번째 entity부터는 qualifier로 간주
                    qpair = (fact[i*2-1], fact[i*2]) # (qualifier_relation, qualifier_entity)
                    self.qual2fact_train.append(idx) # qual2fact_train에 qualifier가 등장하는 fact의 index 추가 (index: qualifier의 index, value: qualifier가 등장하는 fact의 index)
                    self.qual_train.append(list(qpair)) # qual_train에 qpair 추가
                    if qpair not in qpair2idx: # qpair가 아직 등록되지 않은 경우, qpair2idx에 등록하고 qpair_train에 추가
                        qpair2idx[qpair] = len(self.qpair_train) # qpair2idx에 qpair를 key로 하고, value가 id인 쌍 추가
                        self.qpair_train.append(list(qpair)) # qpair_train에 qpair 추가
                    self.qual2qpair_train.append(qpair2idx[qpair]) # qual2qpair_train에 qualifier의 index에 해당하는 qpair의 id 추가
                if i > 0: # i가 1 이상인 경우, 즉 두 번째 entity부터는 tail entity 또는 qualifier entity이므로, 바로 앞의 relation과 연결
                    self.ent2rel_train.append(fact[i*2-1]) # 해당 entity와 연결된 relation을 ent2rel_train에 추가 (fact[i*2-1]은 바로 앞의 relation)
                    if self.rel_mask_train[fact[i*2-1]]: # 해당 relation이 train에 등장하는 relation인 경우, entity 개수 증가 및 위치 정보 추가
                        self.fact2nument_train[-1] += 1 # fact에 등장하는 entity의 개수 증가
                        self.entloc_train.append(i*2) # entity의 위치 정보를 entloc_train에 추가 (i*2는 fact에서의 index)
                else: # i == 0인 경우, 즉 첫 번째 entity: head entity
                    self.ent2rel_train.append(fact[1]) # fact[1]은 primary triplet의 relation
                    self.entloc_train.append(i*2) # i*2 == 0, entloc_train은 entity의 위치 정보를 담는 list
                self.ent_train.append(fact[i*2]) # ent_train에 entity id 추가
                self.ent2fact_train.append(idx) # ent2fact_train에 entity가 등장하는 fact의 index 추가
        # 모든 입력 데이터를 tensor로 변환하고 GPU에 올림
        self.pri_train = torch.tensor(self.pri_train).cuda()
        self.qual2fact_train = torch.tensor(self.qual2fact_train).long().cuda() # index가 매우 커질 수 있으므로 long 타입으로 변환
        self.qual_train = torch.tensor(self.qual_train).cuda()
        self.ent2rel_train = torch.tensor(self.ent2rel_train).cuda()
        self.ent_train = torch.tensor(self.ent_train).cuda()
        self.ent2fact_train = torch.tensor(self.ent2fact_train).cuda()
        self.entloc_train = torch.tensor(self.entloc_train).cuda()
        self.fact2len_train = torch.tensor(self.fact2len_train).cuda()
        self.fact2nument_train = torch.tensor(self.fact2nument_train).cuda()
        self.fact2entstart_train = torch.tensor(self.fact2entstart_train).cuda()
        self.fact2qualstart_train = torch.tensor(self.fact2qualstart_train).cuda()
        self.fact2entlocstart_train = torch.tensor(self.fact2entlocstart_train).cuda()

        self.hpair_train = torch.tensor(self.hpair_train).cuda()
        self.fact2hpair_train = torch.tensor(self.fact2hpair_train).cuda()
        self.tpair_train = torch.tensor(self.tpair_train).cuda()
        self.fact2tpair_train = torch.tensor(self.fact2tpair_train).cuda()
        self.qpair_train = torch.tensor(self.qpair_train).cuda()
        self.qual2qpair_train = torch.tensor(self.qual2qpair_train).long().cuda() # index가 매우 커질 수 있으므로 long 타입으로 변환

    def construct_inputs_inf(self): # construct_inputs와 동일한 역할을 하지만, inductive 세팅에서 inference 데이터를 구조화하는 함수
        self.rel_mask_inf = torch.zeros((self.num_rel_inf,)).bool().cuda()
        self.rel_mask_inf[self.rel_idxs_inf] = 1
        hpair2idx = {}
        tpair2idx = {}
        qpair2idx = {}
        self.hpair_inf = []
        self.fact2hpair_inf = []
        self.tpair_inf = []
        self.fact2tpair_inf = []
        self.qpair_inf = []
        self.qual2qpair_inf = []
        for idx in range(self.num_inf):
            fact = self.inference[idx]
            self.pri_inf.append([fact[0], fact[1], fact[2]])
            self.fact2len_inf.append(len(fact))
            self.fact2nument_inf.append(1)
            self.fact2entlocstart_inf.append(len(self.entloc_inf))
            self.fact2qualstart_inf.append(len(self.qual_inf))
            self.fact2entstart_inf.append(len(self.ent_inf))
            hpair = (fact[0], fact[1])
            tpair = (fact[2], fact[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(self.hpair_inf)
                self.hpair_inf.append(list(hpair))
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(self.tpair_inf)
                self.tpair_inf.append(list(tpair))
            self.fact2hpair_inf.append(hpair2idx[hpair])
            self.fact2tpair_inf.append(tpair2idx[tpair])
            for i, ent in enumerate(fact[::2]):
                if i > 1:
                    qpair = (fact[i*2-1], fact[i*2])
                    self.qual2fact_inf.append(idx)
                    self.qual_inf.append(list(qpair))
                    if qpair not in qpair2idx:
                        qpair2idx[qpair] = len(self.qpair_inf)
                        self.qpair_inf.append(list(qpair))
                    self.qual2qpair_inf.append(qpair2idx[qpair])
                if i > 0:
                    self.ent2rel_inf.append(fact[i*2-1])
                    if self.rel_mask_inf[fact[i*2-1]]:
                        self.fact2nument_inf[-1] += 1
                        self.entloc_inf.append(i*2)
                else:
                    self.ent2rel_inf.append(fact[1])
                    self.entloc_inf.append(i*2)
                self.ent_inf.append(fact[i*2])
                self.ent2fact_inf.append(idx)
        self.pri_inf = torch.tensor(self.pri_inf).cuda()
        self.qual2fact_inf = torch.tensor(self.qual2fact_inf).long().cuda()
        self.qual_inf = torch.tensor(self.qual_inf).cuda()
        self.ent2rel_inf = torch.tensor(self.ent2rel_inf).cuda()
        self.ent_inf = torch.tensor(self.ent_inf).cuda()
        self.ent2fact_inf = torch.tensor(self.ent2fact_inf).cuda()
        self.entloc_inf = torch.tensor(self.entloc_inf).cuda()
        self.fact2len_inf = torch.tensor(self.fact2len_inf).cuda()
        self.fact2nument_inf = torch.tensor(self.fact2nument_inf).cuda()
        self.fact2entstart_inf = torch.tensor(self.fact2entstart_inf).cuda()
        self.fact2qualstart_inf = torch.tensor(self.fact2qualstart_inf).cuda()
        self.fact2entlocstart_inf = torch.tensor(self.fact2entlocstart_inf).cuda()

        self.hpair_inf = torch.tensor(self.hpair_inf).cuda()
        self.fact2hpair_inf = torch.tensor(self.fact2hpair_inf).cuda()
        self.tpair_inf = torch.tensor(self.tpair_inf).cuda()
        self.fact2tpair_inf = torch.tensor(self.fact2tpair_inf).cuda()
        self.qpair_inf = torch.tensor(self.qpair_inf).cuda()
        self.qual2qpair_inf = torch.tensor(self.qual2qpair_inf).long().cuda()

    def train_split(self, base_idxs): # train 데이터에서 일정 비율로 base(본 적이 있는)와 query(본 적이 없는)로 나누는 함수
        # base_idxs: 일정 비율로 무작위로 선택된 train 데이터의 index 리스트
        # fact의 index라고 생각하면 됨
        base_idxs, _ = base_idxs.sort()
        seen_ent = torch.zeros(self.num_ent_train).bool().cuda() # entity의 id 중에서 base_idxs에 등장하는 entity의 id 위치를 True로 설정하는 mask
        seen_rel = torch.zeros(self.num_rel_train).bool().cuda() # relation의 id 중에서 base_idxs에 등장하는 relation의 id 위치를 True로 설정하는 mask
        base_mask = torch.zeros(len(self.pri_train)).bool().cuda() # primary triplet 중에서 base_idxs에 해당하는 위치를 True로 설정하는 mask
        base_mask[base_idxs] = True # base_idxs에 해당하는 위치를 True로 설정

        # entity에 대해 나누는 경우
        # pri_train은 fact와 동일한 index를 가지기 때문에 base_idxs를 그대로 사용 가능
        seen_ent[self.pri_train[base_idxs, 0]] = True # base_idxs에 해당하는 primary triplet의 head entity 위치를 True로 설정
        seen_ent[self.pri_train[base_idxs, 2]] = True # base_idxs에 해당하는 primary triplet의 tail entity 위치를 True로 설정
        if len(self.qual_train) > 0: # qualifier가 존재하는 경우
            # qual2fact_train: qualifier가 등장하는 fact의 index
            # qual_train: (qualifier_relation, qualifier_entity) 쌍을 담고 있는 list -> 1번째 column이 qualifier_entity
            seen_ent[self.qual_train[base_mask[self.qual2fact_train], 1]] = True # base_idxs에 해당하는 fact에 등장하는 qualifier entity 위치를 True로 설정

        # relation에 대해 나누는 경우
        seen_rel[self.pri_train[base_idxs, 1]] = True # primary triplet의 relation 위치를 True로 설정
        if len(self.qual_train) > 0: # qualifier가 존재하는 경우
            seen_rel[self.qual_train[base_mask[self.qual2fact_train], 0]] = True # base_idxs에 해당하는 fact에 등장하는 qualifier relation 위치를 True로 설정

        # query에 해당하는 데이터의 mask 생성
        query_mask = torch.ones(len(self.pri_train)).bool().cuda()
        query_mask[base_idxs] = False # base_idxs에 해당하는 위치를 False로 설정

        # logical_not: True -> False, False -> True
        # 여기선 seen_ent, seen_rel이 False인 위치를 찾아서 query_mask에서 False로 설정
        query_mask[torch.logical_not(seen_ent[self.pri_train[:, 0]])] = False       
        query_mask[torch.logical_not(seen_ent[self.pri_train[:, 2]])] = False
        if len(self.qual_train) > 0: # qualifier가 존재하는 경우
            query_mask[self.qual2fact_train[torch.logical_not(seen_ent[self.qual_train[:, 1]])]] = False

        query_mask[torch.logical_not(seen_rel[self.pri_train[:, 1]])] = False
        if len(self.qual_train) > 0: # qualifier가 존재하는 경우
            query_mask[self.qual2fact_train[torch.logical_not(seen_rel[self.qual_train[:, 0]])]] = False
        query_idxs = query_mask.nonzero(as_tuple = True)[0] # query에 해당하는 fact의 index 리스트

        # 원본 데이터를 보존하기 위해 복사본 생성
        # detach()를 하는 이유는 backpropagation을 막기 위해서 -> 복사본으로 하는 모든 작업은 학습과는 무관하다
        pri = self.pri_train.clone().detach()
        qual = self.qual_train.clone().detach()

        hpair = self.hpair_train.clone().detach()
        fact2hpair = self.fact2hpair_train.clone().detach()
        tpair = self.tpair_train.clone().detach()
        fact2tpair = self.fact2tpair_train.clone().detach()
        qpair = self.qpair_train.clone().detach()
        qual2qpair = self.qual2qpair_train.clone().detach()

        # index 재설정
        # base_idxs에 해당하는 fact의 index는 1부터 시작하도록 설정
        idx2idx = 0 * torch.arange(self.num_train).cuda() # 모든 index를 0으로 초기화
        idx2idx[base_idxs] = torch.arange(1, len(base_idxs) + 1).cuda() # base_idxs 위치에 대해 1부터 시작하는 index로 재설정

        # conv_ent: entity의 id를 재설정하는 tensor
        # conv_rel: relation의 id를 재설정하는 tensor
        conv_ent = -1 * torch.ones(self.num_ent_train).long().cuda() # 모든 entity id를 -1로 초기화 (나타나지 않음)
        conv_ent[seen_ent] = torch.arange(seen_ent.sum()).cuda() # seen_ent 위치에 대해 0부터 시작하는 id로 재설정
        conv_rel = -1 * torch.ones(self.num_rel_train).long().cuda() # 모든 relation id를 -1로 초기화 (나타나지 않음)
        conv_rel[seen_rel] = torch.arange(seen_rel.sum()).cuda() # seen_rel 위치에 대해 0부터 시작하는 id로 재설정

        conv_pri = pri # primary triplet 복사본
        # id 재설정하는 작업
        conv_pri[:, 0] = conv_ent[conv_pri[:, 0]]
        conv_pri[:, 1] = conv_rel[conv_pri[:, 1]]
        conv_pri[:, 2] = conv_ent[conv_pri[:, 2]]
        conv_qual2fact = idx2idx[self.qual2fact_train]
        conv_qual = qual
        if len(conv_qual) > 0:
            conv_qual[:, 0] = conv_rel[conv_qual[:, 0]]
            conv_qual[:, 1] = conv_ent[conv_qual[:, 1]]
        
        base_qual2fact_mask = (conv_qual2fact > 0) # base_idxs에 해당하는 fact에 등장하는 qualifier의 위치를 True로 설정하는 mask
        
        base_pri = conv_pri[idx2idx > 0] # base_idxs에 해당하는 primary triplet
        assert (base_pri[:, 0] == -1).sum() == 0
        base_qual2fact = conv_qual2fact[base_qual2fact_mask] - 1 # base_idxs에 해당하는 fact에 등장하는 qualifier의 fact index (0부터 시작하도록 재설정)
        base_qual = conv_qual[base_qual2fact_mask] # base_idxs에 해당하는 fact에 등장하는 qualifier
        
        # hpair의 id 재설정
        base_hpair_freq = torch.bincount(fact2hpair[idx2idx > 0], minlength = len(hpair)) # base 데이터에 등장하는 hpair의 빈도수
        base_hpair = hpair[base_hpair_freq > 0] # 한 번이라도 등장한 hpair
        base_hpair_idx2idx = -1 * torch.ones(len(hpair)).long().cuda() # 모든 hpair id를 -1로 초기화
        base_hpair_idx2idx[base_hpair_freq > 0] = torch.arange(len(base_hpair)).cuda() # 등장한 hpair 위치에 대해 0부터 시작하는 id로 재설정
        base_hpair_freq = base_hpair_freq[base_hpair_freq > 0] # 빈도수가 0이 아닌 hpair의 빈도수만 추출
        base_fact2hpair = base_hpair_idx2idx[fact2hpair[idx2idx > 0]] # base_idxs에 해당하는 fact에 등장하는 hpair의 id 재설정
        base_hpair[:, 0] = conv_ent[base_hpair[:, 0]] # hpair의 head entity id 재설정
        base_hpair[:, 1] = conv_rel[base_hpair[:, 1]] # hpair의 relation id 재설정

        # tpair의 id 재설정
        base_tpair_freq = torch.bincount(fact2tpair[idx2idx > 0], minlength = len(tpair))
        base_tpair = tpair[base_tpair_freq > 0]
        base_tpair_idx2idx = -1 * torch.ones(len(tpair)).long().cuda()
        base_tpair_idx2idx[base_tpair_freq > 0] = torch.arange(len(base_tpair)).cuda()
        base_tpair_freq = base_tpair_freq[base_tpair_freq > 0]
        base_fact2tpair = base_tpair_idx2idx[fact2tpair[idx2idx > 0]]
        base_tpair[:, 0] = conv_ent[base_tpair[:, 0]]
        base_tpair[:, 1] = conv_rel[base_tpair[:, 1]]

        # qpair의 id 재설정
        base_qpair_freq = torch.bincount(qual2qpair[conv_qual2fact > 0], minlength = len(qpair))
        base_qpair = qpair[base_qpair_freq > 0]
        base_qpair_idx2idx = -1 * torch.ones(len(qpair)).long().cuda()
        base_qpair_idx2idx[base_qpair_freq > 0] = torch.arange(len(base_qpair)).cuda()
        base_qpair_freq = base_qpair_freq[base_qpair_freq > 0]
        base_qual2qpair = base_qpair_idx2idx[qual2qpair[conv_qual2fact > 0]]
        if len(base_qpair) > 0: # qualifier가 존재하는 경우
            base_qpair[:, 0] = conv_rel[base_qpair[:, 0]]
            base_qpair[:, 1] = conv_ent[base_qpair[:, 1]]  

        num_base_ents = seen_ent.sum() # base_idxs에 등장하는 entity의 개수
        num_base_rels = seen_rel.sum() # base_idxs에 등장하는 relation의 개수

        return base_pri, base_qual, base_qual2fact, num_base_ents, num_base_rels, \
               base_hpair, base_hpair_freq, base_fact2hpair, \
               base_tpair, base_tpair_freq, base_fact2tpair, \
               base_qpair, base_qpair_freq, base_qual2qpair, \
               conv_ent, conv_rel, query_idxs

    def train_preds(self, conv_ent, conv_rel, query_idxs):
        query_idxs, _ = query_idxs.sort() # index 오름차순 정렬

        # query_idxs에 해당하는 fact에서 무작위로 entity를 선택하여 예측 대상으로 설정
        # 해당 fact에 있는 entity의 개수 * uniform(0, 1) 값을 소수점 버림하여 예측 대상 entity의 위치를 선택
        pred_locs = (self.fact2nument_train[query_idxs] * torch.rand(len(query_idxs)).cuda()).type(torch.LongTensor).cuda()

        # 현재 pred_locs는 fact에서의 위치가 아니라 몇 번째 entity인지 알려주는 것이므로, 실제 위치로 변환 (relation도 껴있기 때문에 한 번 더 작업을 해줘야 함)
        pred_locs = self.entloc_train[self.fact2entlocstart_train[query_idxs] + pred_locs]
        pred_idxs = query_idxs # query_idxs에 해당하는 fact의 index 리스트

        pri = self.pri_train.clone().detach()
        qual = self.qual_train.clone().detach()
        qual2fact = self.qual2fact_train.clone().detach()

        hpair = self.hpair_train.clone().detach()
        fact2hpair = self.fact2hpair_train.clone().detach()
        tpair = self.tpair_train.clone().detach()
        fact2tpair = self.fact2tpair_train.clone().detach()
        qpair = self.qpair_train.clone().detach()
        qual2qpair = self.qual2qpair_train.clone().detach()

        pred_ids = torch.arange(len(pred_locs)).cuda() + self.num_ent_train # 실제 entity id와 겹치지 않도록, num_ent_train을 더해서 새로운 id로 설정

        # 예측해야하는 head entity의 위치에 대해 새로운 id로 설정
        pri[pred_idxs[pred_locs == 0], 0] = pred_ids[pred_locs == 0] # 예측해야하는 head entity 위치에 대해, 새로운 id로 설정
        # fact2hpair에서 예측해야하는 head entity가 등장하는 fact의 위치에 대해, 새로운 hpair id로 설정
        # num_ent_train을 더하는 것 처럼 hpair의 id도 겹치지 않도록, len(hpair)를 더해서 새로운 id로 설정
        fact2hpair[pred_idxs[pred_locs == 0]] = len(hpair) + torch.arange((pred_locs == 0).sum()).cuda() 
        hpair = torch.cat([hpair, pri[pred_idxs[pred_locs == 0], :2]], dim = 0) # 실제 hpair에 (예측해야하는 head entity, relation) 쌍을 추가

        # 예측해야하는 tail entity의 위치에 대해 새로운 id로 설정
        pri[pred_idxs[pred_locs == 2], 2] = pred_ids[pred_locs == 2]
        fact2tpair[pred_idxs[pred_locs == 2]] = len(tpair) + torch.arange((pred_locs == 2).sum()).cuda()
        # flip을 통해 (relation, tail) 순서로 되어있는 tpair를 (tail, relation) 순서로 바꿔서 추가
        tpair = torch.cat([tpair, pri[pred_idxs[pred_locs == 2], 1:].flip(dims = (-1, ))], dim = 0)

        # 예측해야하는 qualifier entity의 위치에 대해 새로운 id로 설정

        # qual_pred_locs는 예측해야하는 qualifier의 qual에서의 위치
        # pred_locs > 2인 경우, qualifier entity의 위치이므로 qual에서 해당 위치를 찾아서 새로운 id로 설정
        # pred_locs[pred_locs > 2]//2 - 2: fact에서의 위치를 qualifier에서의 위치로 변환하는 작업
        # ex. fact: (h, r, t, qr1, qe1, qr2, qe2) -> pred_locs가 4인 경우, 4//2 - 2 = 0 -> qual에서 0번째 위치
        # 여러 개의 qualifier가 있을 수 있는데, pred_locs[pred_locs > 2]//2 - 2로 인해 원하는 위치가 정확히 선택됨
        qual_pred_locs = self.fact2qualstart_train[pred_idxs][pred_locs > 2] + pred_locs[pred_locs > 2]//2 - 2

        if len(qual) > 0:
            qual[qual_pred_locs, 1] = pred_ids[pred_locs > 2] # 예측해야하는 qualifier entity 위치에 대해, 새로운 id로 설정
            qual2qpair[qual_pred_locs] = len(qpair) + torch.arange((pred_locs > 2).sum()).cuda() # qual2qpair에서 예측해야하는 qualifier가 등장하는 위치에 대해, 새로운 qpair id로 설정
            qpair = torch.cat([qpair, qual[qual_pred_locs]], dim = 0) # 실제 qpair에 (qualifier_relation, 예측해야하는 qualifier_entity) 쌍을 추가

        # pred_locs//2를 통해 예측해야하는 entity가 fact에서 몇 번째 entity인지 알 수 있음
        # answers에 예측해야하는 entity의 실제 id를 저장
        answers = conv_ent[self.ent_train[self.fact2entstart_train[pred_idxs] + pred_locs//2]]

        # query에 해당하는 데이터들에 대해서도 index 재설정 (1부터 시작)
        idx2idx = 0 * torch.arange(self.num_train).cuda()
        idx2idx[query_idxs] = torch.arange(1, len(query_idxs) + 1).cuda()

        # query에 해당하는 데이터만 True로 설정
        query_mask = torch.zeros(len(self.pri_train)).bool().cuda()
        query_mask[query_idxs] = True
        
        conv_ent = torch.cat([conv_ent, pred_ids - self.num_ent_train + (conv_ent != -1).sum()], dim = 0) # 예측해야하는 entity에 대해서 conv_ent 뒤에 id가 추가되도록 설정하여, conv_ent로만 entity에 접근할 수 있음

        # 옛날 ID를 새로운 ID로 변환
        conv_pri = pri
        conv_pri[:, 0] = conv_ent[conv_pri[:, 0]]
        conv_pri[:, 1] = conv_rel[conv_pri[:, 1]]
        conv_pri[:, 2] = conv_ent[conv_pri[:, 2]]
        conv_qual2fact = idx2idx[self.qual2fact_train]
        conv_qual = qual
        if len(conv_qual) > 0:
            conv_qual[:, 0] = conv_rel[conv_qual[:, 0]]
            conv_qual[:, 1] = conv_ent[conv_qual[:, 1]]

        # conv_qual2fact는 query에 해당하는 fact는 1부터 시작, base에 해당하는 fact는 0
        query_qual2fact_mask = (conv_qual2fact > 0) # query에 해당하는 fact에 등장하는 qualifier의 위치를 True로 설정하는 mask

        query_pri = conv_pri[idx2idx > 0] # query에 해당하는 primary triplet
        query_qual2fact = conv_qual2fact[query_qual2fact_mask] - 1 # query에 해당하는 fact에 등장하는 qualifier의 fact index (0부터 시작하도록 재설정)
        query_qual = conv_qual[query_qual2fact_mask] # query에 해당하는 fact에 등장하는 qualifier

        # 이번 batch에서 query를 처리할 때 사용되는 hpair만 추출하여 ID 재설정
        query_hpair_freq = torch.bincount(fact2hpair[idx2idx > 0], minlength = len(hpair)) # query 데이터에 등장하는 hpair의 빈도수
        query_hpair = hpair[query_hpair_freq > 0] # 한 번이라도 등장한 hpair
        query_hpair_idx2idx = -1 * torch.ones(len(hpair)).long().cuda() # 모든 hpair id를 -1로 초기화
        query_hpair_idx2idx[query_hpair_freq > 0] = torch.arange(len(query_hpair)).cuda() # 등장한 hpair 위치에 대해 0부터 시작하는 id로 재설정
        query_hpair_freq = query_hpair_freq[query_hpair_freq > 0] # 빈도수가 0이 아닌 hpair의 빈도수만 추출하여 query_hpair와 대응
        query_fact2hpair = query_hpair_idx2idx[fact2hpair[idx2idx > 0]] # query에 해당하는 fact에 등장하는 hpair의 id 재설정
        query_hpair[:, 0] = conv_ent[query_hpair[:, 0]] # hpair의 head entity id 재설정
        query_hpair[:, 1] = conv_rel[query_hpair[:, 1]] # hpair의 relation id 재설정

        # 이번 batch에서 query를 처리할 때 사용되는 tpair만 추출하여 ID 재설정
        query_tpair_freq = torch.bincount(fact2tpair[idx2idx > 0], minlength = len(tpair))
        query_tpair = tpair[query_tpair_freq > 0]
        query_tpair_idx2idx = -1 * torch.ones(len(tpair)).long().cuda()
        query_tpair_idx2idx[query_tpair_freq > 0] = torch.arange(len(query_tpair)).cuda()
        query_tpair_freq = query_tpair_freq[query_tpair_freq > 0]
        query_fact2tpair = query_tpair_idx2idx[fact2tpair[idx2idx > 0]]
        query_tpair[:, 0] = conv_ent[query_tpair[:, 0]]
        query_tpair[:, 1] = conv_rel[query_tpair[:, 1]]

        # 이번 batch에서 query를 처리할 때 사용되는 qpair만 추출하여 ID 재설정
        query_qpair_freq = torch.bincount(qual2qpair[conv_qual2fact > 0], minlength = len(qpair))
        query_qpair = qpair[query_qpair_freq > 0]
        query_qpair_idx2idx = -1 * torch.ones(len(qpair)).long().cuda()
        query_qpair_idx2idx[query_qpair_freq > 0] = torch.arange(len(query_qpair)).cuda()
        query_qpair_freq = query_qpair_freq[query_qpair_freq > 0]
        query_qual2qpair = query_qpair_idx2idx[qual2qpair[conv_qual2fact > 0]]
        if len(query_qpair) > 0:
            query_qpair[:, 0] = conv_rel[query_qpair[:, 0]]
            query_qpair[:, 1] = conv_ent[query_qpair[:, 1]]

        return query_pri, query_qual, query_qual2fact, \
               query_hpair, query_hpair_freq, query_fact2hpair, \
               query_tpair, query_tpair_freq, query_fact2tpair, \
               query_qpair, query_qpair_freq, query_qual2qpair, answers


    def valid_inputs(self, idxs):
        fact_ids = torch.arange(len(idxs)).cuda() # validation으로 사용할 데이터가 val_size만큼으로 나누어서 들어오기 때문에, fact의 index를 0부터 시작하도록 설정
        pris = []
        qual2fact = []
        quals = []

        hpairs = []
        hpair2idx = {}
        hpair_freqs = []
        fact2hpairs = []

        tpairs = []
        tpair2idx = {}
        tpair_freqs = []
        fact2tpairs = []

        qpairs = []
        qpair2idx = {}
        qpair_freqs = []
        qual2qpairs = []

        all_answers = []
        locs = []

        # inf에 등장하는 entity의 개수 이후부터 새로운 entity id로 설정
        ent_idx = self.num_ent_inf # ent_idx: 예측해야하는 entity의 id
        for i, idx in enumerate(idxs):
            # valid_query[idx]는 ((h, r, t), (qr1, qe1), (qr2, qe2), ...) 형태로 되어있음
            # valid_query는 이런 튜플이 여러 개 들어있는 리스트
            query = [list(comp) for comp in self.valid_query[idx]] # 안에 있는 값을 바꾸기 위해 튜플 -> 리스트로 변환
            # sum(query, []): query 안에 있는 리스트들을 하나의 리스트로 합침 ([]라는 빈 리스트에서 시작해서 query 안에 있는 리스트들을 차례로 더함 -> 하나의 긴 리스트가 됨)
            pred_loc = sum(query, []).index(-1) # -1이 있는 위치를 찾아서, 예측해야하는 entity의 위치를 파악
            pri = query[0] # query의 구조 상 첫 번째 요소는 primary triplet
            if pred_loc == 0: # head entity를 예측해야 하는 경우
                pri[0] = ent_idx # head entity 위치에 -1을 새로운 id로 설정
                ent_idx += 1
            elif pred_loc == 2: # tail entity를 예측해야 하는 경우
                pri[2] = ent_idx # tail entity 위치에 -1을 새로운 id로 설정
                ent_idx += 1
            elif pred_loc % 2 == 0: # qualifier entity를 예측해야 하는 경우
                tmp_qual = query[pred_loc//2 - 1] # qualifier entity 위치에 -1을 새로운 id로 설정
                tmp_qual[1] = ent_idx # qualifier는 (relation, entity) 쌍으로 되어있기 때문에, 1번째 위치에 -1이 있음
                ent_idx += 1 
                query[pred_loc//2 - 1] = tmp_qual # 변경된 qualifier를 다시 query에 반영
            else: # 나머지는 relation 예측에 대한 것이므로 지원하지 않음
                raise NotImplementedError
            
            # 반복문을 다 돌면, 이번 배치에서 포함된 모든 문제들의 빈칸 위치 정보가 담김
            # ex) [2, 4, 0, 4, 2, 0, 2, 4, ...]: 첫 번째 문제는 tail entity, 두 번째 문제는 두 번째 qualifier entity, 세 번째 문제는 head entity, ...
            locs.append(pred_loc) # 예측해야하는 entity의 위치를 저장
            # 반복문을 다 돌면, pris에는 이번 배치에서 포함된 모든 문제들의 primary triplet 정보가 담김
            # ex) [[e1, r1, 15001], [e3, r2, e4], [15003, r3, e6], ...]: 각 문제의 primary triplet 정보
            pris.append(pri) # primary triplet 저장

            # hpair를 선언하고, idx와 매핑 및 빈도수 계산
            # 이 valid_inputs에서 쓰이는 hpair에 대한 처리
            hpair = (pri[0], pri[1])
            tpair = (pri[2], pri[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(hpairs)
                hpairs.append(list(hpair))
                hpair_freqs.append(0)
            fact2hpairs.append(hpair2idx[hpair])
            hpair_freqs[hpair2idx[hpair]] += 1

            # tpair를 선언하고, idx와 매핑 및 빈도수 계산
            # 이 valid_inputs에서 쓰이는 tpair에 대한 처리
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(tpairs)
                tpairs.append(list(tpair))
                tpair_freqs.append(0)
            fact2tpairs.append(tpair2idx[tpair])
            tpair_freqs[tpair2idx[tpair]] += 1

            # qpair를 선언하고, idx와 매핑 및 빈도수 계산
            # 이 valid_inputs에서 쓰이는 qpair에 대한 처리
            for qual in query[1:]:
                quals.append(qual)
                qual2fact.append(fact_ids[i])
                qpair = tuple(qual)
                if qpair not in qpair2idx:
                    qpair2idx[qpair] = len(qpairs)
                    qpairs.append(list(qpair))
                    qpair_freqs.append(0)
                qual2qpairs.append(qpair2idx[qpair])
                qpair_freqs[qpair2idx[qpair]] += 1

            # 해당 문제에 대한 모든 정답의 list를 all_answers에 추가
            # eval_filter_dict: 각 문제에 대한 정답의 집합을 담고 있는 dictionary
            # valid_query[idx]를 key로 사용하여, 해당 문제에 대한 정답의 집합을 가져옴
            all_answers.append(self.eval_filter_dict[self.valid_query[idx]]) 

        return torch.tensor(pris).long().cuda(), \
               torch.tensor(quals).long().cuda(), \
               torch.tensor(qual2fact).long().cuda(), \
               torch.tensor(hpairs).long().cuda(), torch.tensor(hpair_freqs).long().cuda(), torch.tensor(fact2hpairs).long().cuda(), \
               torch.tensor(tpairs).long().cuda(), torch.tensor(tpair_freqs).long().cuda(), torch.tensor(fact2tpairs).long().cuda(), \
               torch.tensor(qpairs).long().cuda(), torch.tensor(qpair_freqs).long().cuda(), torch.tensor(qual2qpairs).long().cuda(), \
               all_answers, locs

    def test_inputs(self, idxs):
        fact_ids = torch.arange(len(idxs)).cuda()
        pris = []
        qual2fact = []
        quals = []

        hpairs = []
        hpair2idx = {}
        hpair_freqs = []
        fact2hpairs = []

        tpairs = []
        tpair2idx = {}
        tpair_freqs = []
        fact2tpairs = []

        qpairs = []
        qpair2idx = {}
        qpair_freqs = []
        qual2qpairs = []

        all_answers = []
        locs = []

        ent_idx = self.num_ent_inf
        for i, idx in enumerate(idxs):
            query = [list(comp) for comp in self.test_query[idx]]
            pred_loc = sum(query, []).index(-1)
            pri = query[0]
            if pred_loc == 0:
                pri[0] = ent_idx
                ent_idx += 1
            elif pred_loc == 2:
                pri[2] = ent_idx
                ent_idx += 1
            elif pred_loc % 2 == 0:
                tmp_qual = query[pred_loc//2 - 1]
                tmp_qual[1] = ent_idx
                ent_idx += 1
                query[pred_loc//2 - 1] = tmp_qual
            else:
                raise NotImplementedError
            locs.append(pred_loc)
            pris.append(pri)

            hpair = (pri[0], pri[1])
            tpair = (pri[2], pri[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(hpairs)
                hpairs.append(list(hpair))
                hpair_freqs.append(0)
            fact2hpairs.append(hpair2idx[hpair])
            hpair_freqs[hpair2idx[hpair]] += 1
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(tpairs)
                tpairs.append(list(tpair))
                tpair_freqs.append(0)
            fact2tpairs.append(tpair2idx[tpair])
            tpair_freqs[tpair2idx[tpair]] += 1

            for qual in query[1:]:
                quals.append(qual)
                qual2fact.append(fact_ids[i])
                qpair = tuple(qual)
                if qpair not in qpair2idx:
                    qpair2idx[qpair] = len(qpairs)
                    qpairs.append(list(qpair))
                    qpair_freqs.append(0)
                qual2qpairs.append(qpair2idx[qpair])
                qpair_freqs[qpair2idx[qpair]] += 1

            all_answers.append(self.eval_filter_dict[self.test_query[idx]])

        return torch.tensor(pris).long().cuda(), \
               torch.tensor(quals).long().cuda(), \
               torch.tensor(qual2fact).long().cuda(), \
               torch.tensor(hpairs).long().cuda(), torch.tensor(hpair_freqs).long().cuda(), torch.tensor(fact2hpairs).long().cuda(), \
               torch.tensor(tpairs).long().cuda(), torch.tensor(tpair_freqs).long().cuda(), torch.tensor(fact2tpairs).long().cuda(), \
               torch.tensor(qpairs).long().cuda(), torch.tensor(qpair_freqs).long().cuda(), torch.tensor(qual2qpairs).long().cuda(), \
               all_answers, locs