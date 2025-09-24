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
from model import MAYPL
import logging
import copy

os.environ['OMP_NUM_THREADS']='8'
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only = True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--log_name')
parser.add_argument('--exp')
parser.add_argument('--dataset_name')
parser.add_argument('--test_epoch', type = int)
parser.add_argument('--msg_add_tr', action = 'store_true')
parser.add_argument('--data_dir', default = "../data/", type = str)
parser.add_argument('--setting', default = "Transductive", type = str)
parser.add_argument('--dim', default=256, type=int)
parser.add_argument('--val_size', default = 100, type = int)
parser.add_argument('--num_init_layer', default=3, type=int)
parser.add_argument('--num_layer', default=4, type=int)
parser.add_argument('--num_head', default=32, type=int)
args = parser.parse_args()


os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok = True)

file_format = args.log_name
if args.msg_add_tr:
    file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}_test_msg+tr_{args.test_epoch}.log")
else:
    file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.dataset_name}/{file_format}_test_trans_{args.test_epoch}.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"{os.getpid()}")
for arg_name in vars(args).keys():
    logger.info(f"{arg_name}:{vars(args)[arg_name]}")
logger.info("Args Listed!")


default_answer = []
KG = HKG(args.data_dir, args.dataset_name, logger, setting = args.setting, msg_add_tr = args.msg_add_tr)
if args.msg_add_tr:
    orig_KG = HKG(args.data_dir, args.dataset_name, logger, setting = args.setting)
    for ent in orig_KG.ent2id_train:
        default_answer.append(KG.ent2id_inf[ent])

model = MAYPL(
    dim = args.dim,
    num_head = args.num_head,
    num_init_layer = args.num_init_layer,
    num_layer = args.num_layer,
    logger = logger
).cuda()


model.load_state_dict(torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.test_epoch}_200.ckpt")["model_state_dict"])

model.eval()

with torch.no_grad():
    lp_head_list_rank = []
    lp_tail_list_rank = []
    lp_pri_list_rank = []
    lp_qual_list_rank = []
    lp_all_list_rank = []

    emb_ent, emb_rel, init_embs_ent, init_embs_rel = model(KG.pri_inf.clone().detach(), KG.qual_inf.clone().detach(), KG.qual2fact_inf, \
                                                           KG.num_ent_inf, KG.num_rel_inf, \
                                                           KG.hpair_inf.clone().detach(), KG.hpair_freq_inf, KG.fact2hpair_inf, \
                                                           KG.tpair_inf.clone().detach(), KG.tpair_freq_inf, KG.fact2tpair_inf, \
                                                           KG.qpair_inf.clone().detach(), KG.qpair_freq_inf, KG.qual2qpair_inf)
    for idxs in tqdm(torch.split(torch.arange(len(KG.test_query)), args.val_size)):
        query_pri, query_qual, query_qual2fact, \
        query_hpair, query_hpair_freq, query_fact2hpair, \
        query_tpair, query_tpair_freq, query_fact2tpair, \
        query_qpair, query_qpair_freq, query_qual2qpair, \
        answers, pred_locs = KG.test_inputs(idxs)
        preds = model.pred(query_pri, query_qual, query_qual2fact, \
                           query_hpair, query_hpair_freq, query_fact2hpair, \
                           query_tpair, query_tpair_freq, query_fact2tpair, \
                           query_qpair, query_qpair_freq, query_qual2qpair, \
                           emb_ent, emb_rel, init_embs_ent, init_embs_rel)
        for i, idx in enumerate(idxs):
            pred_loc = pred_locs[i]
            answer = answers[i] + default_answer
            for test_answer in KG.test_answer[idx]:
                rank = calculate_rank(preds.detach().cpu().numpy()[i], test_answer, answer)
                if pred_loc <= 2:
                    lp_pri_list_rank.append(rank)
                if pred_loc == 0:
                    lp_head_list_rank.append(rank)
                elif pred_loc == 2:
                    lp_tail_list_rank.append(rank)
                else:
                    lp_qual_list_rank.append(rank)
                lp_all_list_rank.append(rank)
    head_mr, head_mrr, head_hit10, head_hit3, head_hit1 = metrics(np.array(lp_head_list_rank))
    tail_mr, tail_mrr, tail_hit10, tail_hit3, tail_hit1 = metrics(np.array(lp_tail_list_rank))
    pri_ent_mr, pri_ent_mrr, pri_ent_hit10, pri_ent_hit3, pri_ent_hit1 = metrics(np.array(lp_pri_list_rank))
    if len(lp_qual_list_rank) > 0:
        qual_ent_mr, qual_ent_mrr, qual_ent_hit10, qual_ent_hit3, qual_ent_hit1 = metrics(np.array(lp_qual_list_rank))
    all_ent_mr, all_ent_mrr, all_ent_hit10, all_ent_hit3, all_ent_hit1 = metrics(np.array(lp_all_list_rank))
    logger.info(f"Link Prediction (Head, {len(lp_head_list_rank)})\nMR:{head_mr}\nMRR:{head_mrr}\nHit1:{head_hit1}\nHit3:{head_hit3}\nHit10:{head_hit10}")
    logger.info(f"Link Prediction (Tail, {len(lp_tail_list_rank)})\nMR:{tail_mr}\nMRR:{tail_mrr}\nHit1:{tail_hit1}\nHit3:{tail_hit3}\nHit10:{tail_hit10}")
    if len(lp_qual_list_rank) > 0:
        logger.info(f"Link Prediction (Qual, {len(lp_qual_list_rank)})\nMR:{qual_ent_mr}\nMRR:{qual_ent_mrr}\nHit1:{qual_ent_hit1}\nHit3:{qual_ent_hit3}\nHit10:{qual_ent_hit10}")
    logger.info(f"Link Prediction (Pri, {len(lp_pri_list_rank)})\nMR:{pri_ent_mr}\nMRR:{pri_ent_mrr}\nHit1:{pri_ent_hit1}\nHit3:{pri_ent_hit3}\nHit10:{pri_ent_hit10}")
    if len(lp_qual_list_rank) > 0:
        logger.info(f"Link Prediction (All, {len(lp_all_list_rank)})\nMR:{all_ent_mr}\nMRR:{all_ent_mrr}\nHit1:{all_ent_hit1}\nHit3:{all_ent_hit3}\nHit10:{all_ent_hit10}")