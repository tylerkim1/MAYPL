from dataloader import HKG
import importlib
from tqdm import tqdm
from utils import calculate_rank, metrics
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import math
import random
import logging
from model import MAYPL
import copy

os.environ['OMP_NUM_THREADS']='8'
torch.set_num_threads(8) # CPU 연산 시 사용할 스레드 수 설정
torch.cuda.empty_cache() # GPU 메모리 캐시 비우기

torch.manual_seed(0) # 재현성을 위해 랜덤 시드 고정
random.seed(0) # 재현성을 위해 랜덤 시드 고정
np.random.seed(0) # 재현성을 위해 랜덤 시드 고정

def train(args, logger):

    for arg_name in vars(args).keys(): # 실험의 재현성을 위해 로그에 모든 인자를 기록함
        logger.info(f"{arg_name}:{vars(args)[arg_name]}")
    logger.info("Args Listed!")

    KG = HKG(args.data_dir, args.dataset_name, logger, setting = args.setting,  msg_add_tr = args.msg_add_tr)
    # data_dir, dataset_name: 데이터셋 경로 및 이름
    # logger: 로그 기록용 logger 객체
    # setting: "Transductive" or "Inductive"
    # msg_add_tr: training 데이터에 메시지 추가 여부

    # 정답으로 예측한 것보다 높은 순위에 유사 정답이 있는 경우 이들을 제외하고 순위를 메김
    default_answer = [] # 진짜 정답을 제외한 유사 정답을 담는 리스트
    if args.msg_add_tr: 
        orig_KG = HKG(args.data_dir, args.dataset_name, logger, setting = args.setting) # msg_add_tr이 False인 원본 KG
        for ent in orig_KG.ent2id_train:
            default_answer.append(KG.ent2id_inf[ent]) 

    model = MAYPL(
        dim = args.dim,
        num_head = args.num_head,
        num_init_layer = args.num_init_layer,
        num_layer = args.num_layer,
        logger = logger,
        model_dropout = args.model_dropout
    ).cuda()

    criterion = nn.CrossEntropyLoss(label_smoothing = args.smoothing, reduction = 'sum') # score와 정답 간의 Cross Entropy Loss 계산
    # label_smoothing: 정답은 1보다 작게, 오답은 0보다 크게 만들어 모델이 정답에 과도하게 집중하는 것을 방지
    # reduction: 'sum'으로 설정하여 배치 내 모든 샘플의 손실을 합산

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # criterion의 값을 최소화하는 방향으로 모델 파라미터를 업데이트
    # lr: learning rate, 파라미터 업데이트 시 이동하는 보폭의 크기

    if args.start_epoch != 0: # 지난 학습에서 이어서 학습할 경우
        model.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.start_epoch}.ckpt")["model_state_dict"])
        optimizer.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.start_epoch}.ckpt")["optimizer_state_dict"])

    train_facts = KG.train
    best_valid_mrr = 0
    best_valid_epoch = 0
    # for epoch in range(args.start_epoch, args.num_epoch):
    for epoch in tqdm(range(args.start_epoch, args.num_epoch)):
        epoch_loss = 0
        train_graph_idxs = (torch.rand(KG.num_train) < args.train_graph_ratio).nonzero(as_tuple = True)[0] # num_train 개의 fact 중 train_graph_ratio 비율만큼 무작위로 선택
        base_pri, base_qual, base_qual2fact, num_base_ents, num_base_rels, \
        base_hpair, base_hpair_freq, base_fact2hpair, \
        base_tpair, base_tpair_freq, base_fact2tpair, \
        base_qpair, base_qpair_freq, base_qual2qpair, \
        conv_ent, conv_rel, query_idxs = KG.train_split(train_graph_idxs) # train_graph_idxs에 해당하는 fact는 base fact로, 나머지는 query fact로 사용
        num_batch = args.batch_num
        for rand_idxs in tqdm(torch.tensor_split(torch.randperm(len(query_idxs)), args.batch_num)):
            batch = query_idxs[rand_idxs]
            query_pri, query_qual, query_qual2fact, \
            query_hpair, query_hpair_freq, query_fact2hpair, \
            query_tpair, query_tpair_freq, query_fact2tpair, \
            query_qpair, query_qpair_freq, query_qual2qpair, answers = KG.train_preds(conv_ent, conv_rel, batch)
            optimizer.zero_grad() # 이전 배치에서 계산된 기울기를 초기화
            # pytorch에서 model()로 모델을 실행하면 forward() 함수가 호출됨
            # entity와 relation에 대한 임베딩 구하기
            emb_ents, emb_rels, init_embs_ent, init_embs_rel = model(base_pri.clone().detach(), base_qual.clone().detach(), base_qual2fact, \
                                                                     num_base_ents, num_base_rels, \
                                                                     base_hpair.clone().detach(), base_hpair_freq, base_fact2hpair, \
                                                                     base_tpair.clone().detach(), base_tpair_freq, base_fact2tpair, \
                                                                     base_qpair.clone().detach(), base_qpair_freq, base_qual2qpair)

            # model.pred(): query에 대한 예측 수행
            # predictions: (len(batch), num_ent) 크기의 텐서, 각 예측해야하는 entity 임베딩과 기존 entity 임베딩 간의 유사도(내적) 계산 결과
            predictions = model.pred(query_pri, query_qual, query_qual2fact, \
                                    query_hpair, query_hpair_freq, query_fact2hpair, \
                                    query_tpair, query_tpair_freq, query_fact2tpair, \
                                    query_qpair, query_qpair_freq, query_qual2qpair, \
                                    emb_ents, emb_rels, init_embs_ent, init_embs_rel)
            loss = criterion(predictions, answers)/len(batch) # 예측한 값과 정답을 비교하여 loss를 계산 (기본적으로 sum으로 계산되므로 배치 크기로 나누어 평균을 구함)
            loss.backward() # backpropagation을 통해 각 파라미터에 대해 얼마나 많은 '책임'(기울기)이 있는지를 수학적으로 계산 -> 각 파라미터에 기울기 값이 계산되어 저장됨
            optimizer.step() # optimizer가 저장된 기울기 값을 보고 파라미터를 업데이트
            # loss에는 오차 점수 뿐만 아니라 어떤 계산 과정을 거쳐 나왔는지에 대한 정보가 다 있는 '영수증'같은 객체
            # .item()을 통해 loss 값만 뽑아서 epoch_loss에 더함
            epoch_loss += loss.item() # 이번 배치의 loss 값을 epoch_loss에 더함
        # epoch이 끝난 후 평균 loss 값을 로그에 기록
        # GPU 메모리를 가장 많이 썼을 때가 얼마인지도 보여주어 현재 메모리 사용량 확인 가능
        logger.info(f"Epoch {epoch+1} GPU:{torch.cuda.max_memory_allocated()} Loss:{epoch_loss/num_batch:.6f}")


        if (epoch+1) % args.val_dur == 0: # 주기적으로 validation 수행
            model.eval() # 모델에게 이제 평가 모드로 전환한다는 것을 알림 (스위치 기능)
            # dropout이 비활성화
            # maypl 에서는 batch normalization이 없지만, 보통은 훈련 과정에서 계산해 둔 고정된 평균과 분산을 사용해여 일관된 기준으로 정규화를 수행

            lp_head_list_rank = []
            lp_tail_list_rank = []
            lp_pri_list_rank = []
            lp_qual_list_rank = []
            lp_all_list_rank = []

            with torch.no_grad(): # 평가 시에는 기울기 계산을 하지 않음 -> 메모리 절약 및 계산 속도 향상
                # model()을 호출하면, 임베딩이 어떻게 '성장' 했는지 기록이 반환됨
                # init_~: Init_layer를 통과한 임베딩 ([0]: 모든 entity가 똑같은 상태의 초기 임베딩)
                # emb_~: ANMP_layer까지 통과한 임베딩 ([0]: 모든 Init_layer를 통과한 임베딩)
                emb_ents, emb_rels, init_embs_ent, init_embs_rel = model(KG.pri_inf.clone().detach(), KG.qual_inf.clone().detach(), KG.qual2fact_inf, \
                                                                         KG.num_ent_inf, KG.num_rel_inf, \
                                                                         KG.hpair_inf.clone().detach(), KG.hpair_freq_inf, KG.fact2hpair_inf, \
                                                                         KG.tpair_inf.clone().detach(), KG.tpair_freq_inf, KG.fact2tpair_inf, \
                                                                         KG.qpair_inf.clone().detach(), KG.qpair_freq_inf, KG.qual2qpair_inf)
                
                # torch.arange(len(KG.valid_query)): 0부터 총 검증해야하는 문제 수 만큼의 정수가 있는 텐서 생성
                # torch.split(..., args.val_size): val_size 크기만큼 자름 (0~10000까지 있었고 val_size가 100이면, 0~99, 100~199, ..., 9900~9999로 나눔)
                for idxs in tqdm(torch.split(torch.arange(len(KG.valid_query)), args.val_size)):
                    query_pri, query_qual, query_qual2fact, \
                    query_hpair, query_hpair_freq, query_fact2hpair, \
                    query_tpair, query_tpair_freq, query_fact2tpair, \
                    query_qpair, query_qpair_freq, query_qual2qpair, \
                    answers, pred_locs = KG.valid_inputs(idxs)

                    # model.pred()를 통해 예측 수행
                    preds = model.pred(query_pri, query_qual, query_qual2fact, \
                                       query_hpair, query_hpair_freq, query_fact2hpair, \
                                       query_tpair, query_tpair_freq, query_fact2tpair, \
                                       query_qpair, query_qpair_freq, query_qual2qpair, \
                                       emb_ents, emb_rels, init_embs_ent, init_embs_rel)
                    for i, idx in enumerate(idxs):
                        # pred_locs: 이번 배치 전체 문제들의 빈칸 위치 목록 ex. [2, 0, 4, ...]
                        pred_loc = pred_locs[i] # 빈칸이 어디에 있는지 정보를 저장
                        # answers[i]: 이번 배치의 i번째 문제의 진짜 정답 리스트
                        answer = answers[i] + default_answer # 이번 문제에서 정답으로 간주할 모든 entity의 목록
                        for valid_answer in KG.valid_answer[idx]: # valid_answer: 이번 배치의 i번째 문제의 진짜 정답 리스트 중 하나
                            # preds: (len(batch), num_ent) 크기의 텐서, 각 예측해야하는 entity 임베딩과 기존 entity 임베딩 간의 유사도(내적) 계산 결과
                            # tensor를 numpy 배열로 변환 후, i번째 문제에 대해 예측한 값과 valid_answer(진짜 정답) 간의 순위를 계산
                            rank = calculate_rank(preds.detach().cpu().numpy()[i], valid_answer, answer)

                            # 각각 에측하는 위치에 따라 별도의 리스트에 순위 기록
                            if pred_loc <= 2:
                                lp_pri_list_rank.append(rank)
                            if pred_loc == 0:
                                lp_head_list_rank.append(rank)
                            elif pred_loc == 2:
                                lp_tail_list_rank.append(rank)
                            else:
                                lp_qual_list_rank.append(rank)
                            lp_all_list_rank.append(rank)

                # head, tail, pri, qual, all에 대해 각각의 리스트에 기록된 순위를 바탕으로 MR, MRR, Hit@K 계산
                head_mr, head_mrr, head_hit10, head_hit3, head_hit1 = metrics(np.array(lp_head_list_rank))
                tail_mr, tail_mrr, tail_hit10, tail_hit3, tail_hit1 = metrics(np.array(lp_tail_list_rank))
                if len(lp_qual_list_rank) > 0:
                    qual_ent_mr, qual_ent_mrr, qual_ent_hit10, qual_ent_hit3, qual_ent_hit1 = metrics(np.array(lp_qual_list_rank))
                pri_ent_mr, pri_ent_mrr, pri_ent_hit10, pri_ent_hit3, pri_ent_hit1 = metrics(np.array(lp_pri_list_rank))
                all_ent_mr, all_ent_mrr, all_ent_hit10, all_ent_hit3, all_ent_hit1 = metrics(np.array(lp_all_list_rank))

                logger.info(f"Link Prediction (Head, {len(lp_head_list_rank)})\nMR:{head_mr}\nMRR:{head_mrr}\nHit1:{head_hit1}\nHit3:{head_hit3}\nHit10:{head_hit10}")
                logger.info(f"Link Prediction (Tail, {len(lp_tail_list_rank)})\nMR:{tail_mr}\nMRR:{tail_mrr}\nHit1:{tail_hit1}\nHit3:{tail_hit3}\nHit10:{tail_hit10}")
                if len(lp_qual_list_rank) > 0:
                    logger.info(f"Link Prediction (Qual, {len(lp_qual_list_rank)})\nMR:{qual_ent_mr}\nMRR:{qual_ent_mrr}\nHit1:{qual_ent_hit1}\nHit3:{qual_ent_hit3}\nHit10:{qual_ent_hit10}")
                logger.info(f"Link Prediction (Pri, {len(lp_pri_list_rank)})\nMR:{pri_ent_mr}\nMRR:{pri_ent_mrr}\nHit1:{pri_ent_hit1}\nHit3:{pri_ent_hit3}\nHit10:{pri_ent_hit10}")
                if len(lp_qual_list_rank) > 0:
                    logger.info(f"Link Prediction (All, {len(lp_all_list_rank)})\nMR:{all_ent_mr}\nMRR:{all_ent_mrr}\nHit1:{all_ent_hit1}\nHit3:{all_ent_hit3}\nHit10:{all_ent_hit10}")

                if not args.no_write: # 파일을 저장하는 경우
                    # model.state_dict(): 모델의 모든 파라미터와 버퍼를 포함하는 딕셔너리 (두뇌라고 생각하면 됨)
                    # optimizer.state_dict(): 옵티마이저의 상태와 하이퍼파라미터를 포함하는 딕셔너리 (학습률, 기울기들의 이동 평균 등)
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, \
                                f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{epoch+1}.ckpt")

                if all_ent_mrr > best_valid_mrr:
                    best_valid_mrr = all_ent_mrr
                    best_valid_epoch = epoch
                elif args.early_stop > 0: # 조기 종료 기능이 있는 경우
                    if epoch - best_valid_epoch >= args.early_stop:
                        break

            model.train() # 다시 훈련 모드로 전환 (dropout이 다시 활성화됨)

if __name__ == '__main__':
    
    logger = logging.getLogger() # 최상위 로거 객체 가져오기 (방송국 같은 역할)
    logger.setLevel(logging.INFO) # INFO 레벨 이상의 로그를 기록 (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # 로그 메시지의 형식 지정 (자막 형식)

    stream_handler = logging.StreamHandler() # 콘솔에 로그를 출력하는 핸들러 (송출 장치)
    stream_handler.setFormatter(log_format) # 자막 형식 적용
    logger.addHandler(stream_handler) # 방송국에 송출 장치 연결

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default = "WikiPeople--eval", type = str)
    parser.add_argument('--data_dir', default = "../data/", type = str)
    parser.add_argument('--exp', default = "ICML2025", type = str)
    parser.add_argument('--setting', default = "Transductive", type = str)
    parser.add_argument('--log_name', default = None, type = str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--start_epoch', default = 0, type = int)
    parser.add_argument('--num_epoch', default=2900, type=int)
    parser.add_argument('--val_dur', default = 2900, type = int)
    parser.add_argument('--val_size', default = 100, type = int)
    parser.add_argument('--num_init_layer', default=3, type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--num_head', default=32, type=int)
    parser.add_argument('--early_stop', default = 0, type = int)
    parser.add_argument('--batch_num', default = 20, type = int)
    parser.add_argument('--train_graph_ratio', default = 0.7, type = float)
    parser.add_argument('--model_dropout', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.0, type=float)
    parser.add_argument('--no_write', action = 'store_true')
    parser.add_argument('--msg_add_tr', action = 'store_true')
    args = parser.parse_args()

    # 파일 이름 지정
    if args.log_name is None:
        file_format = datetime.datetime.now()
    else:
        file_format = args.log_name

    # 저장할 파일의 디렉토리 생성
    if not args.no_write:
        # exist_ok=True: 이미 디렉토리가 존재해도 에러를 발생시키지 않음
        os.makedirs(f"./ckpt/{args.exp}/{args.dataset_name}", exist_ok = True)
        os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok = True)
    else:
        file_format = None

    # 파일로 로그를 저장하는 핸들러 (송출 장치 추가)
    if not args.no_write:
        file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}.log")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.info(f"{os.getpid()}")

    try:
        train(args, logger)
    except Exception as e:
        logging.critical(e, exc_info=True)

    logger.info("END")