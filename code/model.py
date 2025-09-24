import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Union, Annotated
from collections import OrderedDict

class Attn(nn.Module): # query에 관련성이 높은 것들만 골라서 attention
    def __init__(self, dim: int, num_head: int, dropout: float = 0):
        super(Attn, self).__init__() # pytorch의 nn.Module 상속

        self.dim = dim # 입력 데이터의 전체 차원
        self.num_head = num_head # 병렬로 수행할 head의 수
        # head는 어텐션을 수행하는 하나의 단위, 여러명 에게 물어보고 마지막에 종합하여 정보를 얻는 것에 비유할 수 있음.
        self.dim_per_head = dim // num_head # 각 head가 처리하는 차원
        self.drop = nn.Dropout(p = dropout) # overfitting을 방지할 dropout layer 초기화

        # 선형 변환 레이어
        # dim 차원의 텐서를 받아, 학습 가능한 가중치 행렬을 곱하고 bias를 더해 dim 차원의 텐서를 출력
        self.P = nn.Linear(dim, dim, bias = True) # 여러 헤드에서 찾은 정보를 모두 종합한 뒤, 출력하기 전에 한 번 더 가공하는 레이어
        self.V = nn.Linear(dim, dim, bias = True) # value를 만드는 레이어
        self.Q = nn.Linear(dim, dim, bias = True) # query를 만드는 레이어
        self.K = nn.Linear(dim, dim, bias = True) # key를 만드는 레이어
        # nn.Parameter: 텐서를 학습 가능한 파라미터로 등록
        self.a = nn.Parameter(torch.zeros((1, num_head, self.dim_per_head))) # attention 점수를 계산하는 데 사용되는 가중치 벡터 / 크기 : (1, num_head, dim_per_head)
        self.act = nn.PReLU() # 활성화 함수로 PReLU 사용
        self.emb_ln = nn.LayerNorm(dim) # 훈련을 안정시키기 위한 Layer Normalization 레이어

        self.param_init() # weight와 bias를 pytorch의 기본 방식이 아닌 Xavier 초기화 방식으로 초기화하는 함수 호출
        # weight를 아무렇게나 초기화하면 2가지 문제가 발생할 수 있음
        # 1. weight가 너무 크면, 신호가 점점 증폭되어 발산할 수 있음
        # 2. weight가 너무 작으면, 신호가 점점 작아져 소멸할 수 있음
        # Xavier 초기화는 weight가 너무 크거나 작지 않도록 조절하여, 신호가 적절한 범위 내에서 유지되도록 함

    def param_init(self):
        # Xavier 초기화는 해당 레이어에 들어오는 신호의 개수와 나가는 신호의 개수를 고려하여, 적절한 범위의 값으로 weight를 초기화하는 방법
        # 입력이 많으면 weight의 분산을 작게, 입력이 적으면 weight의 분산을 크게 설정
        nn.init.xavier_normal_(self.P.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.V.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.Q.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.K.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.a, gain = nn.init.calculate_gain('relu'))
        
        nn.init.zeros_(self.P.bias)
        nn.init.zeros_(self.V.bias)
        nn.init.zeros_(self.Q.bias)
        nn.init.zeros_(self.K.bias)

    def forward(self, query: torch.FloatTensor, keys: List[torch.FloatTensor], values: List[torch.FloatTensor],
                query_len: Union[int, torch.LongTensor], self_attn: bool = False,
                attn_idxs: Optional[List[torch.LongTensor]] = None,
                query_idxs: Optional[List[torch.LongTensor]] = None, 
                key_idxs: Optional[List[torch.LongTensor]] = None,
                value_idxs: Optional[List[torch.LongTensor]] = None,
                pair_weights: Optional[List[torch.FloatTensor]] = None) -> torch.FloatTensor: # pair_weights: 각 pair에 대한 가중치 (head pair, tail pair, qualifier pair중 어떤 것을 중요하게 둘지)
        # ent_attn 기준:
        # query: emb_ent(모든 entity의 임베딩)
        # len(query): 전체 entity 개수
        # keys: fact가 head, tail, qualifier entity로 보낸 메세지 리스트 / 이미 query에 대해 정렬이 되어있음. 0번째 행은 query_idxs[0]에 해당하는 entity가 받아야 하는 메세지
        # query_idxs: 어떤 entity가 어떤 종류의 메세지를 받아야 하는지
        # key_idxs: None
        # values: fact가 head, tail, qualifier entity로 보낸 메세지 리스트

        # 입력이 들어오지 않는 경우를 대비
        if query_idxs is None:
            query_idxs = [None for _ in range(len(keys))]
        if key_idxs is None:
            key_idxs = [None for _ in range(len(keys))]
        if value_idxs is None:
            value_idxs = [None for _ in range(len(values))]
        if pair_weights is None:
            pair_weights = [None for _ in range(len(keys))]
        if attn_idxs is None:
            attn_idxs = [None for _ in range(len(keys))]

        if self_attn: # self attention인 경우 query도 key, value 역할을 함
            # self.Q(query) + self.K(query)를 하여 자기 자신에 대한 어텐션 점수를 계산
            # 그 이후 activation 함수 (PReLU)를 거침
            # multi head attention을 위해 dim 차원의 임베딩을 (num_head * dim_per_head) 크기로 변환
            # reshape 된 텐서들에 self.a를 곱하여 각 head에 대한 attention 점수를 계산
            # 마지막으로 각 어텐션 점수들을 모두 합함 (각 query에 대해 하나의 점수)
            attn_query_per_query = (self.a * self.act(self.Q(query) + self.K(query)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True) # 크기: (query_len, num_head, 1)
        
        attn_query_per_key = []
        # 각 entity가 자신이 속한 fact로 부터 받아야하는 메세지를, 가중치의 합으로(어텐션) 표현
        for key, query_idx, key_idx, attn_idx in zip(keys, query_idxs, key_idxs, attn_idxs):
            if (query_idx is not None and len(query) > 1) and key_idx is not None: # 어텐션 계산에 필요한 query, key 둘 다 index가 주어진 경우
                if len(query) < len(query_idx) and len(key) < len(key_idx): # query, key 안에서 모두 중복이 많아, 먼저 선형 변환을 하고 index_select하는 것이 더 효율적임
                    attn_query_per_key.append((self.a * self.act(torch.index_select(self.Q(query), 0, query_idx) + torch.index_select(self.K(key), 0, key_idx)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
                elif len(query) < len(query_idx) and len(key) >= len(key_idx): # query 안에서만 중복이 많음
                    attn_query_per_key.append((self.a * self.act(torch.index_select(self.Q(query), 0, query_idx) + self.K(torch.index_select(key, 0, key_idx))).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
                elif len(query) >= len(query_idx) and len(key) < len(key_idx): # key 안에서만 중복이 많음
                    attn_query_per_key.append((self.a * self.act(self.Q(torch.index_select(query, 0, query_idx)) + torch.index_select(self.K(key), 0, key_idx)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
                else: # query, key 안에서 모두 중복이 적어, index_select하고 선형 변환하는 것이 더 효율적임
                    attn_query_per_key.append((self.a * self.act(self.Q(torch.index_select(query, 0, query_idx)) + self.K(torch.index_select(key, 0, key_idx))).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))

            elif (query_idx is None or len(query) == 1) and key_idx is not None:
                if len(key) < len(key_idx):
                    attn_query_per_key.append((self.a * self.act(self.Q(query) + torch.index_select(self.K(key), 0, key_idx)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
                else:
                    attn_query_per_key.append((self.a * self.act(self.Q(query) + self.K(torch.index_select(key, 0, key_idx))).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
            
            elif (query_idx is not None and len(query) > 1) and key_idx is None:
                if len(query) < len(query_idx):
                    attn_query_per_key.append((self.a * self.act(torch.index_select(self.Q(query), 0, query_idx) + self.K(key)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
                else:
                    attn_query_per_key.append((self.a * self.act(self.Q(torch.index_select(query, 0, query_idx)) + self.K(key)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
            
            else:
                attn_query_per_key.append((self.a * self.act(self.Q(query) + self.K(key)).reshape(-1, self.num_head, self.dim_per_head)).sum(dim = -1, keepdim = True))
            if attn_idx is not None: # 특정 어텐션 조합만 원하는 경우
                attn_query_per_key[-1] = attn_query_per_key[-1][attn_idx]

        # softmax 함수에서 숫자가 너무 커지는 것을 방지하기 위해 max값을 모든 값에 대해 다 빼줌.
        # softmax(a, b, c) == softmax(a-M, b-M, c-M)
        if self_attn: # self attention인 경우 자기 자신의 attention을 max attention으로 설정
            max_attn_per_query = attn_query_per_query
        else: # self attention이 아닌 경우, keys들 중 가장 값이 작은 값을 max attention으로 설정
            min_attn = min([torch.min(attn) for attn in attn_query_per_key if len(attn) > 0])
            max_attn_per_query = min_attn * torch.ones((query_len, self.num_head, 1)).cuda()
        for attn, query_idx in zip(attn_query_per_key, query_idxs): # max attention을 찾는 과정
            if query_idx is not None:
                # source의 i번째 원소가, query_idx[i]에 해당하는 위치에 있는 max_attn_per_query의 행을 업데이트해야함.
                # 'amax'는 두 값 중 더 큰 값을 선택
                # 'include_self'가 True이므로, 자기 자신도 포함하여 더 큰 값을 선택
                # .detach()를 통해, max_attn_per_query가 계산 그래프에서 분리되어, 이후의 역전파 과정에서 영향을 받지 않도록 함 (메모리를 효율적으로 사용)
                max_attn_per_query = max_attn_per_query.index_reduce(dim = 0, index = query_idx, source = attn, reduce = 'amax', include_self = True).detach()
            else:
                max_attn_per_query = torch.maximum(max_attn_per_query, attn).detach()

        # softmax 계산을 위한 분모(exp_attn_per_query) 변수 생성
        # self attention인 경우 자기 자신에 대한 어텐션 점수 - 최댓값 을 지수화하여 초기화
        # self attention이 아닌 경우 0으로 초기화
        if self_attn:
            exp_attn_query_per_query = torch.exp(attn_query_per_query - max_attn_per_query) # 크기 : (query_len, num_head, 1)
            exp_attn_per_query = exp_attn_query_per_query
        else:
            exp_attn_per_query = torch.zeros((query_len, self.num_head, 1)).cuda()

        exp_attn_query_per_key = [] # softmax의 분자(exp(attn - max_attn))를 저장하는 리스트
        for attn, query_idx, pair_weight in zip(attn_query_per_key, query_idxs, pair_weights):
            if pair_weight is None: # 특정 가중치가 주어지지 않은 경우, 모든 쌍에 대해 동일한 가중치(1)를 사용
                pair_weight = torch.ones_like(attn)
            if query_idx is not None:
                exp_attn_query_per_key.append(torch.exp(attn - max_attn_per_query[query_idx])) # softmax의 분자(exp(attn - max_attn)) 계산
                exp_attn_per_query = exp_attn_per_query.index_add(dim = 0, index = query_idx, source = pair_weight * exp_attn_query_per_key[-1]) # softmax의 분모에 방금 계산한 값을 더함
            else:
                exp_attn_query_per_key.append(torch.exp(attn - max_attn_per_query))
                exp_attn_per_query = exp_attn_per_query + pair_weight * exp_attn_query_per_key[-1]
        # 고립된 노드의 경우 + self attention도 없는 경우 분모가 0이 될 수 있음
        exp_attn_per_query = torch.where(exp_attn_per_query == 0, 1, exp_attn_per_query) # 분모가 0이 되는 것을 방지하기 위해, 0인 경우 1로 바꿈 (어차피 분자도 0이기 때문에 문제 없음)

        # msg_to_query의 크기: (query_len, num_head, dim_per_head)
        if self_attn:
            msg_to_query = exp_attn_query_per_query * self.V(query).reshape(-1, self.num_head, self.dim_per_head) # 실제 계산을 위해 내 자신의 메세지를 Value로 변환하여 곱함
        else:
            msg_to_query = torch.zeros((query_len, self.num_head, self.dim_per_head)).cuda() # 가중합 계산을 위한 변수 msg_to_query 초기화
        for exp_attn, query_idx, value, value_idx, pair_weight in zip(exp_attn_query_per_key, query_idxs, values, value_idxs, pair_weights): # 각 key, value에 대해 메세지를 받아서 가중합을 계산
            if pair_weight is None: # 특정 가중치가 주어지지 않은 경우, 모든 쌍에 대해 동일한 가중치(1)를 사용
                pair_weight = torch.ones_like(exp_attn)
            if query_idx is not None and value_idx is not None:
                if len(value) < len(value_idx):
                    msg_to_query = msg_to_query.index_add(dim = 0, index = query_idx, source = pair_weight * exp_attn * torch.index_select(self.V(value), 0, value_idx).reshape(-1, self.num_head, self.dim_per_head))
                else:
                    msg_to_query = msg_to_query.index_add(dim = 0, index = query_idx, source = pair_weight * exp_attn * self.V(torch.index_select(value, 0, value_idx)).reshape(-1, self.num_head, self.dim_per_head))
            elif query_idx is None and value_idx is not None:
                if len(value) < len(value_idx):
                    msg_to_query = msg_to_query + pair_weight * exp_attn * torch.index_select(self.V(value), 0, value_idx).reshape(-1, self.num_head, self.dim_per_head)
                else:
                    msg_to_query = msg_to_query + pair_weight * exp_attn * self.V(torch.index_select(value, 0, value_idx)).reshape(-1, self.num_head, self.dim_per_head)
            elif query_idx is not None and value_idx is None:
                msg_to_query = msg_to_query.index_add(dim = 0, index = query_idx, source = pair_weight * exp_attn * self.V(value).reshape(-1, self.num_head, self.dim_per_head))
            else:
                msg_to_query = msg_to_query + pair_weight * exp_attn * self.V(value).reshape(-1, self.num_head, self.dim_per_head)
    
        # msg_to_query / exp_attn_per_query: softmax를 통해 가중합을 계산한 결과
        # flatten: (크기: (query_len, num_head, dim_per_head)) -> (크기: (query_len, dim))
        # self.P: 여러 헤드에서 찾은 정보를 모두 종합한 뒤, 출력하기 전에 한 번 더 가공하는 레이어
        # drop: overfitting을 방지하기 위한 dropout 레이어
        # query + ... : residual connection -> 모델이 변화량을 학습하도록 유도
        # emb_ln: layer normalization을 통해 훈련을 안정화
        new_query = self.emb_ln(query + self.drop(self.P((msg_to_query / exp_attn_per_query).flatten(start_dim = 1))))

        return new_query

class Init_Layer(nn.Module): # Structure-driven Initialization
    def __init__(self, dim,logger, dropout = 0.1):
        super(Init_Layer, self).__init__()

        self.dim = dim

        self.drop = nn.Dropout(p = dropout)

        self.logger = logger
        # head, tail, relation, qualifier entity, qualifier relation 각각에 대해 독립적인 선형 변환 레이어를 사용
        # proj_A2B: A에서 B로 정보를 전달하는 역할
        # Linear -> y = Wx + b  
        # dim 차원의 입력을 받아 dim 차원의 출력을 생성
        # bias를 사용하겠다
        self.proj_he2e = nn.Linear(dim, dim, bias = True) # head entity to entity
        self.proj_te2e = nn.Linear(dim, dim, bias = True) # tail entity to entity
        self.proj_qe2e = nn.Linear(dim, dim, bias = True) # qualifier entity to entity

        self.proj_he2pr = nn.Linear(dim, dim, bias = True)
        self.proj_te2pr = nn.Linear(dim, dim, bias = True)
        self.proj_qe2qr = nn.Linear(dim, dim, bias = True)

        self.proj_pr2he = nn.Linear(dim, dim, bias = True)
        self.proj_pr2te = nn.Linear(dim, dim, bias = True)
        self.proj_qr2qe = nn.Linear(dim, dim, bias = True)

        self.proj_pr2r = nn.Linear(dim, dim, bias = True)
        self.proj_qr2r = nn.Linear(dim, dim, bias = True)

        self.proj_fe2he = nn.Linear(dim, dim, bias = True)
        self.proj_fe2te = nn.Linear(dim, dim, bias = True)
        self.proj_fe2qe = nn.Linear(dim, dim, bias = True)

        self.proj_fr2pr = nn.Linear(dim, dim, bias = True)
        self.proj_fr2qr = nn.Linear(dim, dim, bias = True)

        self.emb_ent_ln = nn.LayerNorm(dim) # entity에 대한 layer normalization
        self.emb_rel_ln = nn.LayerNorm(dim) # relation에 대한 layer normalization

        self.param_init() # Xavier 초기화 함수 호출

    def param_init(self):
        nn.init.xavier_normal_(self.proj_he2e.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_te2e.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qe2e.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_he2pr.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_te2pr.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qe2qr.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_pr2he.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_pr2te.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qr2qe.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_pr2r.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qr2r.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_fe2he.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fe2te.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fe2qe.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_fr2pr.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fr2qr.weight, gain = nn.init.calculate_gain('relu'))
        

        nn.init.zeros_(self.proj_he2e.bias)
        nn.init.zeros_(self.proj_te2e.bias)
        nn.init.zeros_(self.proj_qe2e.bias)

        nn.init.zeros_(self.proj_he2pr.bias)
        nn.init.zeros_(self.proj_te2pr.bias)
        nn.init.zeros_(self.proj_qe2qr.bias)

        nn.init.zeros_(self.proj_pr2he.bias)
        nn.init.zeros_(self.proj_pr2te.bias)
        nn.init.zeros_(self.proj_qr2qe.bias)

        nn.init.zeros_(self.proj_pr2r.bias)
        nn.init.zeros_(self.proj_qr2r.bias)

        nn.init.zeros_(self.proj_fe2he.bias)
        nn.init.zeros_(self.proj_fe2te.bias)
        nn.init.zeros_(self.proj_fe2qe.bias)

        nn.init.zeros_(self.proj_fr2pr.bias)
        nn.init.zeros_(self.proj_fr2qr.bias)

    def forward(self, emb_ent, emb_rel, pri, qual2fact, qual):
        heads = pri[:, 0] # primary triple의 head entity 인덱스
        rels = pri[:, 1] # primary triple의 relation 인덱스
        tails = pri[:, 2] # primary triple의 tail entity 인덱스
        if len(qual) > 0: 
            qual_rels = qual[:, 0] # qualifier relation 인덱스
            qual_ents = qual[:, 1] # qualifier entity 인덱스
        else: # qualifier가 없는 경우 빈 텐서 생성
            qual_rels = torch.tensor([]).long().cuda() # 빈 텐서 생성
            qual_ents = torch.tensor([]).long().cuda() # 빈 텐서 생성

        # base graph에 있는 모든 fact에 대해, 각 fact의 head entity가 보낼 메세지를 생성
        # emb_ent가 proj_he2e를 거치면서 선형 변환되고, num_ent * dim 텐서가 됨
        # index_select를 통해, pri에 있는 head entity 인덱스에 해당하는 행들만 선택
        # index_select의 2번째 인자를 0으로 하여 0번째 차원, 즉 행을 기준으로 선택하라는 뜻
        # heads에 있는 인덱스들에 해당하는 행들만 선택 -> msg_he2e는 (num_fact, dim) 크기의 텐서
        msg_he2e = torch.index_select(self.proj_he2e(emb_ent), 0, heads) # head entity가 entity로 보내는 메세지 (num_fact, dim)
        msg_te2e = torch.index_select(self.proj_te2e(emb_ent), 0, tails) # tail entity가 entity로 보내는 메세지 (num_fact, dim)
        msg_qe2e = torch.index_select(self.proj_qe2e(emb_ent), 0, qual_ents) # qualifier entity가 entity로 보내는 메세지 (num_qual, dim)

        # msg_he2e + msg_te2e: 요소별로 더함
        # index_add: dim 차원에 대해, index에 해당하는 위치에 source를 더함
        # index는 qual2fact의 value를 보고 얻어짐 (qualifier가 속한 fact의 index)
        # 결과적으로 msg_qe2e의 각 행에 대해 해당 qualifier가 속한 fact의 index에 해당하는 행에 더함
        # (msg_he2e + msg_te2e)에다가 msg_qe2e를 qual2fact에 따라 더하는 방식
        # msg_qe2e와 qual2fact가 index 정렬 되어있기 때문에 가능
        # qualifier가 없다면 단순 (msg_he2e + msg_te2e) 반환
        msg_fe2e = (msg_he2e + msg_te2e).index_add(dim = 0, index = qual2fact, source = msg_qe2e) # fact가 entity로 보내는 메세지 (num_fact, dim)

        # 각 fact에 해당 fact에 몇개의 entity가 있는지 세는 과정
        # qual2fact는 하나의 qualifier가 어느 fact에 속해있는지를 나타내므로, bincount를 통해 fact에 속한 qualifier의 개수를 세고, head + tail entity(2개) 를 더함
        fe_cnt = torch.bincount(qual2fact, minlength = len(pri)).float() + 2

        msg_he2pr = torch.index_select(self.proj_he2pr(emb_ent), 0, heads)     # head entity가 primary relation로 보내는 메세지 / 크기: (num_fact, dim)
        msg_te2pr = torch.index_select(self.proj_te2pr(emb_ent), 0, tails)     # tail entity가 primary relation로 보내는 메세지 / 크기: (num_fact, dim)
        msg_qe2qr = torch.index_select(self.proj_qe2qr(emb_ent), 0, qual_ents) # qualifier entity가 qualifier relation로 보내는 메세지 / 크기: (num_qual, dim)

        msg_pr2he = torch.index_select(self.proj_pr2he(emb_rel), 0, rels)      # primary relation이 head entity로 보내는 메세지 / 크기: (num_fact, dim)
        msg_pr2te = torch.index_select(self.proj_pr2te(emb_rel), 0, rels)      # primary relation이 tail entity로 보내는 메세지 / 크기: (num_fact, dim)
        msg_qr2qe = torch.index_select(self.proj_qr2qe(emb_rel), 0, qual_rels) # qualifier relation이 qualifier entity로 보내는 메세지 / 크기: (num_qual, dim)

        msg_pr2r = torch.index_select(self.proj_pr2r(emb_rel), 0, rels)        # primary relation이 relation로 보내는 메세지 / 크기: (num_fact, dim)
        msg_qr2r = torch.index_select(self.proj_qr2r(emb_rel), 0, qual_rels)   # qualifier relation이 relation로 보내는 메세지 / 크기: (num_qual, dim)

        # ### 지울 것

        # msg_he2pr = torch.index_select(self.proj_he2pr(emb_ent), 0, heads)     # head entity가 primary relation로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_he2qr = torch.index_select(self.proj_he2qr(emb_ent), 0, heads)     # head entity가 qualifier relation로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_te2pr = torch.index_select(self.proj_te2pr(emb_ent), 0, tails)     # tail entity가 primary relation로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_te2qr = torch.index_select(self.proj_te2qr(emb_ent), 0, tails)     # tail entity가 qualifier relation로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_qe2pr = torch.index_select(self.proj_qe2pr(emb_ent), 0, qual_ents) # qualifier entity가 primary relation로 보내는 메세지 / 크기: (num_qual, dim)
        # msg_qe2qr = torch.index_select(self.proj_qe2qr(emb_ent), 0, qual_ents) # qualifier entity가 qualifier relation로 보내는 메세지 / 크기: (num_qual, dim)

        # msg_pr2he = torch.index_select(self.proj_pr2he(emb_rel), 0, rels)      # primary relation이 head entity로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_pr2te = torch.index_select(self.proj_pr2te(emb_rel), 0, rels)      # primary relation이 tail entity로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_pr2qe = torch.index_select(self.proj_pr2qe(emb_rel), 0, rels)      # primary relation이 qualifier entity로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_qr2he = torch.index_select(self.proj_qr2he(emb_rel), 0, qual_rels) # qualifier relation이 head entity로 보내는 메세지 / 크기: (num_qual, dim)
        # msg_qr2te = torch.index_select(self.proj_qr2te(emb_rel), 0, qual_rels) # qualifier relation이 tail entity로 보내는 메세지 / 크기: (num_qual, dim)
        # msg_qr2qe = torch.index_select(self.proj_qr2qe(emb_rel), 0, qual_rels) # qualifier relation이 qualifier entity로 보내는 메세지 / 크기: (num_qual, dim)

        # msg_pr2r = torch.index_select(self.proj_pr2r(emb_rel), 0, rels)        # primary relation이 relation로 보내는 메세지 / 크기: (num_fact, dim)
        # msg_qr2r = torch.index_select(self.proj_qr2r(emb_rel), 0, qual_rels)   # qualifier relation이 relation로 보내는 메세지 / 크기: (num_qual, dim)

        # ###

        # msg_pr2r에다가 msg_qr2r를 qual2fact에 따라 더하는 방식
        msg_fr2r = msg_pr2r.index_add(dim = 0, index = qual2fact, source = msg_qr2r) # fact가 relation로 보내는 메세지 (num_fact, dim)
        fr_cnt = fe_cnt - 1 # fact에 속한 relation의 개수 (primary relation 1개 + qualifier relation 개수) = (fact에 속한 entity 개수 - 1)

        # 빈 캔버스를 만들어 메세지를 받을 준비
        zero4ent = torch.zeros_like(emb_ent) # emb_ent와 동일한 크기의 텐서지만, 모든 값이 0인 텐서

        idx4e = torch.cat([heads, tails], dim = 0) # head와 tail entity의 인덱스를 모두 모음 (2 * num_fact, )

        # 자기 자신의 메세지는 제외
        # head entity가 받을 메세지, ... , tail entity가 받을 메세지, ... 순으로 되어있음
        src4e2e = torch.cat([self.proj_fe2he(msg_fe2e-msg_he2e), self.proj_fe2te(msg_fe2e-msg_te2e)], dim = 0) # head와 tail entity가 각각 fact로 부터 받는 메세지 (2 * num_fact, dim)
        src4e2ecnt = torch.cat([fe_cnt-1, fe_cnt-1], dim = 0) # exclude self
        src4r2e = torch.cat([msg_pr2he, msg_pr2te], dim = 0) # head와 tail entity가 각각 primary relation로 부터 받는 메세지 (2 * num_fact, dim)

        if len(qual2fact) > 0: # qualifier가 있는 경우
            idx4e = torch.cat([idx4e, qual_ents], dim = 0) # idx4e에 qualifier entity 인덱스 추가 (2 * num_fact + num_qual, )
            # 마찬가지로 자신이 보낸 메세지는 제외
            src4e2e = torch.cat([src4e2e, self.proj_fe2qe(torch.index_select(msg_fe2e, 0, qual2fact)-msg_qe2e)], dim = 0) # src4e2e에 qualifier entity가 fact로 부터 받는 메세지 추가 (2 * num_fact + num_qual, dim)
            src4e2ecnt = torch.cat([src4e2ecnt, torch.index_select(fe_cnt-1, 0, qual2fact)], dim = 0) # fact안에서 자신을 제외한 entity 개수
            src4r2e = torch.cat([src4r2e, msg_qr2qe], dim = 0) # src4r2e에 qualifier entity가 qualifier relation로 부터 받는 메세지 추가 (2 * num_fact + num_qual, dim)

        e2e = zero4ent.index_add(dim = 0, index = idx4e, source = src4e2e) # entity id 위치에 메세지를 다 더함 / 크기: (num_ent, dim)
        e2e_cnt = torch.zeros(len(emb_ent)).cuda().index_add(dim = 0, index = idx4e, source = src4e2ecnt).unsqueeze(dim = 1) # entity id 위치에 몇개의 메세지가 더해졌는지 셈 / 크기: (num_ent, 1)
        e2e_cnt = torch.where(e2e_cnt == 0, 1, e2e_cnt) # entity가 없는 경우 0인 경우 1로 바꿈 (어차피 e2e가 0이기 때문에 문제 없음)
        
        r2e = zero4ent.index_reduce(dim = 0, index = idx4e, source = src4r2e, reduce = 'mean', include_self = False) # 해당 entity가 속한 fact의 relation이 보낸 메세지의 평균 / 크기: (num_ent, dim)
        # primary entity는 primary relation으로부터, qualifier entity는 qualifier relation으로부터 메세지를 받음
        
        # 기존 임베딩(emb_ent)에 dropout을 적용한 e2e와 r2e를 더한 후, layer normalization 적용
        # entity 에 대해서는 단순 합만 하였기 때문에 평균을 내어 더함
        new_emb_ent = self.emb_ent_ln(emb_ent + self.drop(e2e/e2e_cnt) + self.drop(r2e)) 

        # relation에 대해서도 빈 캔버스를 생성
        zero4rel = torch.zeros_like(emb_rel)

        if len(qual2fact) > 0: # qualifier가 있는 경우
            # relation이 relation들로부터 받을 메세지
            r2r = zero4rel.index_add(dim = 0, index = torch.cat([rels, qual_rels], dim = 0), \
                                    source = torch.cat([self.proj_fr2pr(msg_fr2r-msg_pr2r), self.proj_fr2qr(torch.index_select(msg_fr2r, 0, qual2fact)-msg_qr2r)], dim = 0))

            r2r_cnt = torch.zeros(len(emb_rel)).cuda().index_add(dim = 0, index = torch.cat([rels, qual_rels], dim = 0), \
                                                                 source = torch.cat([fr_cnt-1, torch.index_select(fr_cnt-1, 0, qual2fact)], dim = 0)).unsqueeze(dim = 1) #exclude self
            r2r_cnt = torch.where(r2r_cnt == 0, 1, r2r_cnt) # relation id 위치에 몇개의 relation 메세지가 더해졌는지 센다 / 크기: (num_rel, 1)

            e2r = zero4rel.index_reduce(dim = 0, index = torch.cat([rels, rels, qual_rels], dim = 0),
                                        source = torch.cat([msg_he2pr, msg_te2pr, msg_qe2qr], dim = 0), reduce = 'mean', include_self = False) # 해당 relation이 속한 fact의 entity가 보낸 메세지의 평균
            
            new_emb_rel = self.emb_rel_ln(emb_rel + self.drop(r2r/r2r_cnt) + self.drop(e2r)) # 기존 임베딩에 dropout을 적용한 r2r와 e2r를 더한 후, layer normalization 적용
        
        else:
            e2r = zero4rel.index_reduce(dim = 0, index = torch.cat([rels, rels], dim = 0),
                                        source = torch.cat([msg_he2pr, msg_te2pr], dim = 0), reduce = 'mean', include_self = False) # entity가 relation에 보낸 메세지의 평균
            new_emb_rel = self.emb_rel_ln(emb_rel + self.drop(e2r)) # 기존 임베딩에 dropout을 적용한 e2r를 더한 후, layer normalization 적용

        return new_emb_ent, new_emb_rel

class ANMP_Layer(nn.Module): # Attentive Neural Message Passing
    def __init__(self, dim, num_head, logger, dropout = 0.2): # dropout이 0.2 인데, Init layer보다 더 강한 규제를 하기 위함?
        super(ANMP_Layer, self).__init__()

        self.dim = dim
        self.dim_per_head = dim // num_head
        assert self.dim_per_head * num_head == dim
        self.num_head = num_head

        self.drop = nn.Dropout(p = dropout)

        self.init_emb_fact = nn.Parameter(torch.zeros(1, dim)) # 모든 fact의 초기 임베딩을 동일한 값으로 설정
        self.init_emb_fact_ln = nn.LayerNorm(dim) # 초기 fact에 대한 layer normalization

        self.fact_ln = nn.LayerNorm(dim) # entity와 relation에서 메시지를 받은 후의 fact에 대한 layer normalization

        ## Parameters for hyper-relational fact representation

        # 각각의 entity, relation을 메세지로 가공
        self.proj_head_ent = nn.Linear(dim, dim, bias = True)
        self.proj_tail_ent = nn.Linear(dim, dim, bias = True)
        self.proj_qual_ent = nn.Linear(dim, dim, bias = True)

        self.proj_head_rel = nn.Linear(dim, dim, bias = True)
        self.proj_tail_rel = nn.Linear(dim, dim, bias = True)
        self.proj_qual_rel = nn.Linear(dim, dim, bias = True)

        # 가공된 메세지를 바탕으로 pair를 만들어 fact에 전달
        self.proj_hpair_to_fact = nn.Linear(dim, dim, bias = True)
        self.proj_tpair_to_fact = nn.Linear(dim, dim, bias = True)
        self.proj_qpair_to_fact = nn.Linear(dim, dim, bias = True)

        # fact에서 받은 메세지를 바탕으로 entity, relation에 전달할 메세지 가공
        self.proj_head_rel2 = nn.Linear(dim, dim, bias = True)
        self.proj_tail_rel2 = nn.Linear(dim, dim, bias = True)
        self.proj_qual_rel2 = nn.Linear(dim, dim, bias = True)
        self.proj_head_ent2 = nn.Linear(dim, dim, bias = True)
        self.proj_tail_ent2 = nn.Linear(dim, dim, bias = True)
        self.proj_qual_ent2 = nn.Linear(dim, dim, bias = True)

        # fact에서 entity, relation으로 메세지 전달
        self.proj_fact_to_head_ent = nn.Linear(dim, dim, bias = True)
        self.proj_fact_to_tail_ent = nn.Linear(dim, dim, bias = True)
        self.proj_fact_to_pri_rel = nn.Linear(dim, dim, bias = True)
        self.proj_fact_to_qual_ent = nn.Linear(dim, dim, bias = True)
        self.proj_fact_to_qual_rel = nn.Linear(dim, dim, bias = True)

        # entity, relation 각각에 대해 서로의 messege에만 집중할 수 있도록 attention layer를 별도로 둠
        self.ent_attn = Attn(dim, num_head, dropout = dropout)
        self.rel_attn = Attn(dim, num_head, dropout = dropout)

        self.to_fact_ln = nn.LayerNorm(dim)
        self.to_ent_ln = nn.LayerNorm(dim)
        self.to_rel_ln = nn.LayerNorm(dim)

        self.logger = logger

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.init_emb_fact, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_ent.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_ent.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_ent.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_rel.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_rel.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_rel.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_ent2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_ent2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_ent2.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_rel2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_rel2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_rel2.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_hpair_to_fact.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tpair_to_fact.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qpair_to_fact.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_fact_to_head_ent.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_tail_ent.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_pri_rel.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_qual_ent.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_qual_rel.weight, gain = nn.init.calculate_gain('relu'))

        nn.init.zeros_(self.proj_head_ent.bias)
        nn.init.zeros_(self.proj_tail_ent.bias)
        nn.init.zeros_(self.proj_qual_ent.bias)

        nn.init.zeros_(self.proj_head_rel.bias)
        nn.init.zeros_(self.proj_tail_rel.bias)
        nn.init.zeros_(self.proj_qual_rel.bias)

        nn.init.zeros_(self.proj_head_ent2.bias)
        nn.init.zeros_(self.proj_tail_ent2.bias)
        nn.init.zeros_(self.proj_qual_ent2.bias)

        nn.init.zeros_(self.proj_head_rel2.bias)
        nn.init.zeros_(self.proj_tail_rel2.bias)
        nn.init.zeros_(self.proj_qual_rel2.bias)

        nn.init.zeros_(self.proj_hpair_to_fact.bias)
        nn.init.zeros_(self.proj_tpair_to_fact.bias)
        nn.init.zeros_(self.proj_qpair_to_fact.bias)

        nn.init.zeros_(self.proj_fact_to_head_ent.bias)
        nn.init.zeros_(self.proj_fact_to_tail_ent.bias)
        nn.init.zeros_(self.proj_fact_to_pri_rel.bias)
        nn.init.zeros_(self.proj_fact_to_qual_ent.bias)
        nn.init.zeros_(self.proj_fact_to_qual_rel.bias)


    def forward(self, emb_ent, emb_rel, \
                pri, qual2fact, qual, \
                hpair_ent, hpair_rel, hpair_freq, fact2hpair, \
                tpair_ent, tpair_rel, tpair_freq, fact2tpair, \
                qpair_ent, qpair_rel, qpair_freq, qual2qpair):
        heads = pri[:, 0] # primary triple의 head entity 인덱스
        rels = pri[:, 1] # primary triple의 relation 인덱스
        tails = pri[:, 2] # primary triple의 tail entity 인덱스
        if len(qual) > 0: 
            qual_rels = qual[:, 0] # qualifier relation 인덱스
            qual_ents = qual[:, 1] # qualifier entity 인덱스
        else: # qualifier가 없는 경우 빈 텐서 생성
            qual_rels = torch.tensor([]).long().cuda() # 빈 텐서 생성
            qual_ents = torch.tensor([]).long().cuda() # 빈 텐서 생성

        # init_emb_fact는 (1, dim) 크기의 학습 가능한 파라미터 (fact의 초기 임베딩)
        # xavier 초기화한 init_emb_fact를 layer normalization과 dropout을 거쳐 emb_fact로 사용
        emb_fact = self.drop(self.init_emb_fact_ln(self.init_emb_fact)) # fact의 초기 임베딩 (1, dim)

        emb_ent_head = self.proj_head_ent(emb_ent) # head entity 역할을 위한 메세지 (num_ent, dim)
        emb_ent_tail = self.proj_tail_ent(emb_ent) # tail entity 역할을 위한 메세지 (num_ent, dim)
        if len(qual2fact) > 0:
            emb_ent_qual = self.proj_qual_ent(emb_ent) # qualifier entity 역할을 위한 메세지 (num_ent, dim)
        else:
            emb_ent_qual = torch.empty((0, self.dim)).cuda() # qualifier가 없는 경우 빈 텐서 생성

        # relation이 head냐 tail이냐에 따라 역할(임베딩)이 달라짐
        emb_rel_head = self.proj_head_rel(emb_rel) # head와 연결된 relation의 역할을 위한 메세지 (num_rel, dim)
        emb_rel_tail = self.proj_tail_rel(emb_rel) # tail과 연결된 relation의 역할을 위한 메세지 (num_rel, dim)
        if len(qual2fact) > 0:
            emb_rel_qual = self.proj_qual_rel(emb_rel) # qualifier와 연결된 relation의 역할을 위한 메세지 (num_rel, dim)
        else:
            emb_rel_qual = torch.empty((0, self.dim)).cuda() # qualifier가 없는 경우 빈 텐서 생성
    
        # torch.index_select(emb_ent_head, 0, hpair_ent): hpair_ent 인덱스에 해당하는 head entity 메세지 선택 (num_hpair, dim)
        # torch.index_select(emb_rel_head, 0, hpair_rel): hpair_rel 인덱스에 해당하는 head와 연결된 relation 메세지 선택 (num_hpair, dim)
        # 두 개를 요소 곱 한 후 proj_hpair_to_fact를 거쳐 메세지 가공
        # layer nomarlization을 거쳐 emb_hpair_to_fact로 사용
        emb_hpair_to_fact = self.to_fact_ln(self.proj_hpair_to_fact(torch.index_select(emb_rel_head, 0, hpair_rel) * torch.index_select(emb_ent_head, 0, hpair_ent)))
        emb_tpair_to_fact = self.to_fact_ln(self.proj_tpair_to_fact(torch.index_select(emb_rel_tail, 0, tpair_rel) * torch.index_select(emb_ent_tail, 0, tpair_ent)))
        emb_qpair_to_fact = self.to_fact_ln(self.proj_qpair_to_fact(torch.index_select(emb_rel_qual, 0, qpair_rel) * torch.index_select(emb_ent_qual, 0, qpair_ent)))

        # ### 지울 것

        # h_rels = torch.index_select(emb_rel_head, 0, hpair_rel)
        # h_ents = torch.index_select(emb_ent_head, 0, hpair_ent)
        # h_cat = torch.cat([h_rels, h_ents], dim = 0)
        # emb_hpair_to_fact = self.to_fact_ln(self.proj_hpair_to_fact(h_cat))

        # ###

        # fact의 임베딩을 구하는 과정
        new_emb_fact = torch.index_select(emb_hpair_to_fact, 0, fact2hpair) + torch.index_select(emb_tpair_to_fact, 0, fact2tpair) # fact에 속한 head-pair, tail-pair 메세지를 더함
        new_emb_fact = new_emb_fact.index_add(dim = 0, index = qual2fact, source = torch.index_select(emb_qpair_to_fact, 0, qual2qpair)) # fact에 속한 qualifier-pair 메세지를 더함
        new_emb_fact = new_emb_fact / (torch.bincount(qual2fact, minlength = len(pri)) + 2).unsqueeze(dim = 1) # fact에 합한 수 만큼 나눠서 평균을 embedding으로 사용

        new_emb_fact = self.fact_ln(emb_fact + self.drop(new_emb_fact)) # 초기 fact 임베딩과 더한 후, dropout과 layer normalization 적용
   
        # 구한 fact 임베딩을 바탕으로 entity, relation에 메세지를 보내는 과정
        # proj_head_rel2를 통해 emb_rel에서 head에게 보낼 메세지용으로 가공
        # fact의 정보를 head entity가 받을 수 있게 '번역기' 역할을 하는 emb_rel을 이용함
        # index_select를 통해 primary relation 인덱스에 해당하는 행들만 선택
        # fact의 전체 문맥과 relation의 정보를 결합하여 메세지의 내용을 풍부하게 만듦 (* new_emb_fact)
        # proj_fact_to_head_ent를 통해 fact에서 head entity로 보낼 메세지용으로 가공
        # layer normalization을 거쳐 emb_fact_to_head로 사용
        emb_fact_to_head = self.to_ent_ln(self.proj_fact_to_head_ent(torch.index_select(self.proj_head_rel2(emb_rel), 0, rels) * new_emb_fact))
        emb_fact_to_tail = self.to_ent_ln(self.proj_fact_to_tail_ent(torch.index_select(self.proj_tail_rel2(emb_rel), 0, rels) * new_emb_fact))
        emb_fact_to_qual = self.to_ent_ln(self.proj_fact_to_qual_ent(torch.index_select(self.proj_qual_rel2(emb_rel), 0, qual_rels) * new_emb_fact[qual2fact]))

        new_emb_ent = self.ent_attn(query = emb_ent, keys = [emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual], \
                                    values = [emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual], query_len = len(emb_ent), self_attn = True, \
                                    query_idxs = [heads, tails, qual_ents])
        
        emb_fact_to_pri_rel = self.to_rel_ln(self.proj_fact_to_pri_rel((torch.index_select(self.proj_head_ent2(emb_ent), 0, heads) + torch.index_select(self.proj_tail_ent2(emb_ent), 0, tails)) * new_emb_fact))
        emb_fact_to_qual_rel = self.to_rel_ln(self.proj_fact_to_qual_rel(torch.index_select(self.proj_qual_ent2(emb_ent), 0, qual_ents) * torch.index_select(new_emb_fact, 0, qual2fact)))

        new_emb_rel = self.rel_attn(query = emb_rel, keys = [emb_fact_to_pri_rel, emb_fact_to_qual_rel], \
                                    values = [emb_fact_to_pri_rel, emb_fact_to_qual_rel], query_len = len(emb_rel), self_attn = True, \
                                    query_idxs = [rels, qual_rels])

        return new_emb_ent, new_emb_rel

    def pred(self, num_ent, num_rel, emb_ent, emb_rel, pri, qual2fact, qual, \
             hpair_ent, hpair_rel, hpair_freq, fact2hpair, \
             tpair_ent, tpair_rel, tpair_freq, fact2tpair, \
             qpair_ent, qpair_rel, qpair_freq, qual2qpair):
        heads = pri[:, 0] # 이번 batch 내 fact에서 primary triple의 head entity 인덱스를 모아둠 [4, 10000, 52, 10001] 이런식
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()

        pred_heads = (heads >= num_ent).nonzero(as_tuple = True)[0] # heads >= num_ent인 index 추출 -> 예측해야 하는 entity의 index
        pred_tails = (tails >= num_ent).nonzero(as_tuple = True)[0] # tails >= num_ent인 index 추출 -> 예측해야 하는 entity의 index
        if len(qual) > 0:
            pred_qual_ents = (qual_ents >= num_ent).nonzero(as_tuple = True)[0] # qual_ents >= num_ent인 index 추출 -> 예측해야 하는 entity의 index
        else:
            qual = qual.long()
            qual2fact = qual2fact.long()
            pred_qual_ents = torch.tensor([]).long().cuda()

        pred_ent = torch.zeros_like(heads) # 이번 batch에서 포함된 fact의 총 개수를 heads로 대신함
        pred_ent[pred_heads] = 1 # head 예측이 일어나는 fact위치를 1로 표현
        pred_ent[pred_tails] = 1 # tail 예측이 일어나는 fact위치를 1로 표현
        pred_ent[qual2fact[pred_qual_ents]] = 1 # qualifier 예측이 일어나는 fact위치를 1로 표현
        pred_ent = pred_ent.nonzero(as_tuple = True)[0] # pred_ent가 1인 index 추출 -> 예측해야 하는 fact의 index

        # self.init_emb_fact는 (1, dim) 크기의 학습 가능한 파라미터 (fact의 초기 임베딩)
        # xavier 초기화한 init_emb_fact를 layer normalization과 dropout을 거쳐 emb_fact로 사용
        # dropout을 거치는 이유는 init_emb_fact라는 '기본값'에 너무 의존하는 것을 막기 위함
        emb_fact = self.drop(self.init_emb_fact_ln(self.init_emb_fact))

        # 모든 entity 임베딩(emb_ent)을 각각의 역할에 맞는 버전으로 변환해두는 사전 작업
        emb_ent_head = self.proj_head_ent(emb_ent) # head entity 역할을 위한 메세지
        emb_ent_tail = self.proj_tail_ent(emb_ent) # tail entity 역할을 위한 메세지
        if len(qual2fact) > 0:
            emb_ent_qual = self.proj_qual_ent(emb_ent) # qualifier entity 역할을 위한 메세지
        else:
            emb_ent_qual = torch.empty((0, self.dim)).cuda() # qualifier가 없는 경우 빈 텐서 생성

        # 모든 relation 임베딩(emb_rel)을 각각의 역할에 맞는 버전으로 변환해두는 사전 작업
        emb_rel_head = self.proj_head_rel(emb_rel) # head와 연결된 relation의 역할을 위한 메세지
        emb_rel_tail = self.proj_tail_rel(emb_rel) # tail과 연결된 relation의 역할을 위한 메세지
        if len(qual2fact) > 0:
            emb_rel_qual = self.proj_qual_rel(emb_rel) # qualifier와 연결된 relation의 역할을 위한 메세지
        else:
            emb_rel_qual = torch.empty((0, self.dim)).cuda() # qualifier가 없는 경우 빈 텐서 생성

        # distmult 방식으로 각 pari의 임베딩을 결합하고 fact에 전달할 메세지로 가공
        # layer normalization을 거쳐 emb_hpair_to_fact, emb_tpair_to_fact, emb_qpair_to_fact로 사용
        emb_hpair_to_fact = self.to_fact_ln(self.proj_hpair_to_fact(torch.index_select(emb_rel_head, 0, hpair_rel) * torch.index_select(emb_ent_head, 0, hpair_ent)))
        emb_tpair_to_fact = self.to_fact_ln(self.proj_tpair_to_fact(torch.index_select(emb_rel_tail, 0, tpair_rel) * torch.index_select(emb_ent_tail, 0, tpair_ent)))
        emb_qpair_to_fact = self.to_fact_ln(self.proj_qpair_to_fact(torch.index_select(emb_rel_qual, 0, qpair_rel) * torch.index_select(emb_ent_qual, 0, qpair_ent)))

        # 각 fact에 해당하는 pair의 임베딩을 모아 더하는 과정
        # fact2hpair에는 fact에 속한 head-pair의 인덱스가, fact2tpair에는 tail-pair의 인덱스가, qual2qpair에는 qualifier-pair의 인덱스가 담겨있음
        # fact2~를 보고 해당 fact에 속한 pair 임베딩을 emb_~pair_to_fact에서 찾아 더함
        new_emb_fact = torch.index_select(emb_hpair_to_fact, 0, fact2hpair) + torch.index_select(emb_tpair_to_fact, 0, fact2tpair)
        new_emb_fact = new_emb_fact.index_add(dim = 0, index = qual2fact, source = torch.index_select(emb_qpair_to_fact, 0, qual2qpair))
        new_emb_fact = new_emb_fact / (torch.bincount(qual2fact, minlength = len(pri)) + 2).unsqueeze(dim = 1) # fact에 합한 수만큼 나눠서 평균을 구함 (qualifier가 더 많은 fact는 값이 커질 수 있기 때문에)
        new_emb_fact = self.fact_ln(emb_fact + self.drop(new_emb_fact)) # dropout한 값을 초기 emb_fact와 더한 후, layer normalization 적용

        # 예측해야하는 엔티티가 포함된 fact의 embedding을 뽑고, 해당 fact의 relation과 곱하여 entity로 보낼 메세지를 만듦
        # proj_head_rel2를 통해 emb_rel에서 head에게 보낼 메세지용으로 가공 (relation의 정보를 '번역'하는 역할)
        # proj_fact_to_head_ent를 통해 fact에서 head entity로 보낼 메세지용으로 가공
        # layer normalization을 거쳐 emb_fact_to_head로 사용
        emb_fact_to_head = self.to_ent_ln(self.proj_fact_to_head_ent(torch.index_select(self.proj_head_rel2(emb_rel), 0, rels[pred_heads]) * torch.index_select(new_emb_fact, 0, pred_heads)))
        emb_fact_to_tail = self.to_ent_ln(self.proj_fact_to_tail_ent(torch.index_select(self.proj_tail_rel2(emb_rel), 0, rels[pred_tails]) * torch.index_select(new_emb_fact, 0, pred_tails)))
        emb_fact_to_qual = self.to_ent_ln(self.proj_fact_to_qual_ent(torch.index_select(self.proj_qual_rel2(emb_rel), 0, qual_rels[pred_qual_ents]) * torch.index_select(new_emb_fact, 0, qual2fact[pred_qual_ents])))

        # arrange(start, end): start부터 end-1까지의 숫자를 차례대로 담은 1차원 텐서 생성
        pred_ent_idxs = torch.arange(num_ent, len(emb_ent)).cuda() # 예측해야 하는 entity의 인덱스 (num_ent ~ len(emb_ent)-1)

        # 예측해야하는 entity가 두 종류의 정보 소스에 대해 어텐션을 수행하여 최종 임베딩을 계산 (emb_ent + fact로부터 받은 메세지)
        new_emb_ent = self.ent_attn(query = emb_ent, keys = [emb_ent, emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual], \
                                     values = [emb_ent, emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual], query_len = len(emb_ent), self_attn = False, \
                                     query_idxs = [pred_ent_idxs, heads[pred_heads], tails[pred_tails], qual_ents[pred_qual_ents]], \
                                     key_idxs = [pred_ent_idxs, None, None, None], value_idxs = [pred_ent_idxs, None, None, None])
        return new_emb_ent

class MAYPL(nn.Module):
    def __init__(self, dim, num_init_layer, num_layer, num_head, logger, model_dropout = 0.2):
        super(MAYPL, self).__init__()
        init_layers = [] # Init_Layer 쌓기 위한 리스트, Init_Layer는 Structure-driven Initialization 담당
        layers = [] # ANMP_Layer 쌓기 위한 리스트, ANMP_Layer는 Attentive Neural Message Passing 담당

        for _ in range(num_init_layer): # layer 수 만큼 Init_Layer 쌓기
            init_layers.append(Init_Layer(dim, logger, dropout = model_dropout))

        for _ in range(num_layer): # layer 수 만큼 ANMP_Layer 쌓기
            layers.append(ANMP_Layer(dim, num_head, logger, dropout = model_dropout))
        
        # pyTorch에서 사용할 수 있도록 ModuleList로 변환
        self.init_layers = nn.ModuleList(init_layers)
        self.layers = nn.ModuleList(layers)

        self.dim = dim
        self.drop = nn.Dropout(p = model_dropout)

        self.init_emb_ent = nn.Parameter(torch.zeros(1, dim)) # 모든 entity의 초기 임베딩을 동일한 값으로 설정
        self.init_emb_rel = nn.Parameter(torch.zeros(1, dim)) # 모든 relation의 초기 임베딩을 동일한 값으로 설정

        self.logger = logger

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.init_emb_ent, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.init_emb_rel, gain = nn.init.calculate_gain('relu'))

    def forward(self, pri, qual, qual2fact, num_ent, num_rel, \
                hpair, hpair_freq, fact2hpair, \
                tpair, tpair_freq, fact2tpair, \
                qpair, qpair_freq, qual2qpair):

        hpair_ent = hpair[:, 0] # hpair의 entity index들
        hpair_rel = hpair[:, 1] # hpair의 relation index들
        hpair_freq = hpair_freq.unsqueeze(dim = 1).unsqueeze(dim = 2) # broadcast를 위해 unsqueeze -> 3차원으로 만듦   

        tpair_ent = tpair[:, 0] # tpair의 entity index들
        tpair_rel = tpair[:, 1] # tpair의 relation index들
        tpair_freq = tpair_freq.unsqueeze(dim = 1).unsqueeze(dim = 2) # broadcast를 위해 unsqueeze -> 3차원으로 만듦

        if len(qual) > 0: # qual이 존재하는 경우
            qpair_ent = qpair[:, 1] # qpair의 entity index들
            qpair_rel = qpair[:, 0] # qpair의 relation index들
        else: # qual이 존재하지 않는 경우
            qpair_ent = torch.tensor([]).long().cuda() # 빈 텐서 생성
            qpair_rel = torch.tensor([]).long().cuda() # 빈 텐서 생성
        qpair_freq = qpair_freq.unsqueeze(dim = 1).unsqueeze(dim = 2) # broadcast를 위해 unsqueeze -> 3차원으로 만듦
        
        # repeat: init_emb_ent의 0번째 차원을 num_ent, 1번째 차원은 1로 복제 -> 행은 num_ent, 열은 dim * 1 크기의 텐서 생성
        emb_ent = self.init_emb_ent.repeat(num_ent, 1) # num_ent 개수만큼 init_emb_ent 복제 -> num_ent * dim 크기의 텐서
        emb_rel = self.init_emb_rel.repeat(num_rel, 1) # num_rel 개수만큼 init_emb_rel 복제 -> num_rel * dim 크기의 텐서

        init_embs_ent = [emb_ent] # 각 Init_Layer를 거치면서의 entity 임베딩들을 저장하기 위한 리스트
        init_embs_rel = [emb_rel] # 각 Init_Layer를 거치면서의 relation 임베딩들을 저장하기 위한 리스트
        for init_layer in self.init_layers: # Init_Layer들을 순차적으로 거치면서 entity, relation 임베딩 업데이트
            # emb_ent, emb_rel은 바로 직전 layer를 거친 후의 임베딩
            emb_ent, emb_rel = init_layer(emb_ent, emb_rel, pri, qual2fact, qual) 
            init_embs_ent.append(emb_ent) # 각 layer를 거치면서의 entity 임베딩들을 리스트에 추가
            init_embs_rel.append(emb_rel) # 각 layer를 거치면서의 relation 임베딩들을 리스트에 추가

        init_emb_ent = emb_ent # Init_Layer를 모두 거친 후의 entity 임베딩
        init_emb_rel = emb_rel # Init_Layer를 모두 거친 후의 relation 임베딩

        emb_ents = [init_emb_ent] # 각 ANMP_Layer를 거치면서의 entity 임베딩들을 저장하기 위한 리스트
        emb_rels = [init_emb_rel] # 각 ANMP_Layer를 거치면서의 relation 임베딩들을 저장하기 위한 리스트

        for layer in self.layers: # ANMP_Layer들을 순차적으로 거치면서 entity, relation 임베딩 업데이트
            emb_ent, emb_rel = layer(emb_ent, emb_rel, \
                                     pri, qual2fact, qual, \
                                     hpair_ent, hpair_rel, hpair_freq, fact2hpair, \
                                     tpair_ent, tpair_rel, tpair_freq, fact2tpair, \
                                     qpair_ent, qpair_rel, qpair_freq, qual2qpair)
            emb_ents.append(emb_ent) # 각 layer를 거치면서의 entity 임베딩들을 리스트에 추가
            emb_rels.append(emb_rel) # 각 layer를 거치면서의 relation 임베딩들을 리스트에 추가
        return emb_ents, emb_rels, init_embs_ent, init_embs_rel
    
    def pred(self, pri, qual, qual2fact, \
             hpair, hpair_freq, fact2hpair, \
             tpair, tpair_freq, fact2tpair, \
             qpair, qpair_freq, qual2qpair, \
             emb_ents, emb_rels, init_embs_ent, init_embs_rel):
        #assumption: former indices are known entities, and latter indices are unknown entities

        # fact2hpair: 해당 fact의 head-pair 인덱스
        # fact2tpair: 해당 fact의 tail-pair 인덱스
        # qual2qpair: 해당 qualifier의 qualifier-pair 인덱스

        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]
        hpair_freq = hpair_freq.unsqueeze(dim = 1).unsqueeze(dim = 2) # 추후 계산을 위해 차원을 3차원으로 늘림 / 크기: (num_hpair, 1, 1)

        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]
        tpair_freq = tpair_freq.unsqueeze(dim = 1).unsqueeze(dim = 2) # 추후 계산을 위해 차원을 3차원으로 늘림 / 크기: (num_tpair, 1, 1)

        if len(qual) > 0:
            qpair_ent = qpair[:, 1]
            qpair_rel = qpair[:, 0]
        else: # qual이 존재하지 않는 경우
            qpair_ent = torch.tensor([]).long().cuda() # 빈 텐서 생성
            qpair_rel = torch.tensor([]).long().cuda() # 빈 텐서 생성
            qual2qpair = qual2qpair.long() # long 타입 보장
        qpair_freq = qpair_freq.unsqueeze(dim = 1).unsqueeze(dim = 2) # 추후 계산을 위해 차원을 3차원으로 늘림 / 크기: (num_qpair, 1, 1)

        # emb_ents, emb_rels: MAYPL의 forward 함수를 거친 후의 entity, relation 임베딩 리스트
        # emb_ents: (layer 수 + 1) 개수의 각 entity의 임베딩이 담긴 리스트
        # emb_ents[0]을 하면 Init_Layer를 모두 거친 후의 entity 임베딩을 의미함 -> len으로 총 entity 개수를 알 수 있음
        num_ent = len(emb_ents[0])
        num_rel = len(emb_rels[0])

        # max_ent_cands: 이번 query batch에서 등장하는 가장 높은 entity ID를 찾기 위한 후보 리스트
        # max(pri[:,0]): primary triple의 head entity 인덱스 중 최대값
        # num_ent - 1: 전체 entity 개수 - 1 (0부터 시작하는 인덱스이므로)
        # max(pri[:, 2]): primary triple의 tail entity 인덱스 중 최대값
        # max(qual[:, 1]): qualifier entity 인덱스 중 최대값
        max_ent_cands = [max(pri[:,0]), num_ent - 1]
        max_ent_cands.append(max(pri[:, 2]))
        if len(qual) > 0:
            max_ent_cands.append(max(qual[:, 1]))

        num_ent_preds = max(max_ent_cands) - num_ent + 1 # 이번 query batch에서 예측해야 하는 entity 개수

        # init_emb_ent: Init_Layer를 모두 거친 후의 entity 임베딩 / 크기: (num_ent, dim)
        # self.init_emb_ent: 모든 entity의 초기 임베딩을 동일한 값으로 설정 / 크기: (1, dim)
        # num_ent_preds 개수만큼 self.init_emb_ent를 복제하여 이어붙임 / 크기: (num_ent_preds, dim)
        # emb_ent: 이번 query batch에서 예측해야 하는 entity 개수를 포함한 전체 entity 임베딩 / 크기: (num_ent + num_ent_preds, dim)
        emb_ent = torch.cat([init_embs_ent[0], self.init_emb_ent.repeat(num_ent_preds, 1)], dim = 0)

        for l, init_layer in enumerate(self.init_layers):
            # init_layer 인자: (emb_ent, emb_rel, pri, qual2fact, qual)
                # emb_ent: 이 레이어가 업데이트 해야할 주요 대상 1, 모든 entity의 임베딩
                # emb_rel: 이 레이어가 업데이트 해야할 주요 대상 2, 모든 relation의 임베딩
                # pri, qual2fact, qual: 업데이트를 위한 설계도
            # 결국 3, 4, 5번 인자를 보고 1, 2번 인자들 사이에서 메세지를 주고받게 하여 최종적으로 업데이트된 새로운 1번과 2번 재료를 반환.
                # init_embs_rel[l]: l번째 Init_Layer를 거친 후의 relation 임베딩 / 크기: (num_rel, dim)
                # pri, qual2fact, qual: 예측해야할 query의 구조 정보
            emb_ent, _ = init_layer(emb_ent, init_embs_rel[l], pri, qual2fact, qual)
            # emb_ent[num_ent:]: 예측해야 하는 entity 임베딩 부분
            # init_embs_ent[l+1]: (l+1)번째 Init_Layer
            # l번째 layer 거친 이후의 예측해야하는 entity 임베딩을 기존 entity 임베딩 뒤에 이어붙임
            emb_ent = torch.cat([init_embs_ent[l+1], emb_ent[num_ent:]], dim = 0)

        input_emb_ent = emb_ent # 예측해야하는 entity의 임베딩도 포함된 entity 임베딩
        input_emb_rel = init_embs_rel[-1] # Init_Layer를 모두 거친 후의 relation 임베딩
        # emb_ents[0:], emb_rels[0:]은 init_layer 직후의 상태이므로 [1:]을 사용해 ANMP_Layer의 결과물만 가져옴
        for emb_ent, emb_rel, layer in zip(emb_ents[1:], emb_rels[1:], self.layers):
            
            # AMNP_Layer의 pred 함수의 역할: 이미 계산된 기존 entity, relation의 임베딩을 바탕으로 예측해야하는 entity 임베딩을 업데이트
            # 인자들:
                # num_ent, num_rel: 기존 entity, relation 개수
                # 3, 4번째 (emb_ent, emb_rel): model.forward()를 거친 후의 entity, relation 임베딩 (사전 지식 역할)
                # 5번째 인자부터 (pri, qual2fact, qual, hpair_ent, ...): 예측해야할 query의 구조 정보
            # 반환값: 예측해야하는 entity 임베딩이 업데이트된 전체 entity 임베딩
            output_emb_ent = layer.pred(num_ent, num_rel,
                                        input_emb_ent, input_emb_rel,
                                        pri, qual2fact, qual,
                                        hpair_ent, hpair_rel, hpair_freq, fact2hpair, 
                                        tpair_ent, tpair_rel, tpair_freq, fact2tpair, 
                                        qpair_ent, qpair_rel, qpair_freq, qual2qpair)

            input_emb_ent = torch.cat([emb_ent, output_emb_ent[num_ent:]], dim = 0) # 기존 entity 임베딩에 업데이트 된 예측해야하는 entity 임베딩을 이어붙임
            input_emb_rel = emb_rel # relation 임베딩은 그대로 놔둠

        # torch.inner: 두 텐서의 내적을 계산
        # input_emb_ent[:num_ent]: 기존 entity 임베딩
        # input_emb_ent[num_ent:]: 예측해야하는 entity 임베딩
        # ent_preds: (num_ent_preds, num_ent) 크기의 텐서, 각 예측해야하는 entity 임베딩과 기존 entity 임베딩 간의 유사도(내적) 계산 결과
        ent_preds = torch.inner(input_emb_ent[num_ent:], input_emb_ent[:num_ent])

        return ent_preds