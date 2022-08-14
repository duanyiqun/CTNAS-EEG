# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for pretrain phase. """
from distutils.log import debug
import os.path as osp
import os
import tqdm
import numpy as np
import logging
import json
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import csv
import random
from mundus.dataset.dataloader.samplers_BCI_IV import CategoriesSampler
from mundus.models.head.MAML_fc import MtlLearner
from mundus.utils.misc import Averager, Timer, count_acc, ensure_path
# from mundus.models.backbone.DARTS.search_eeg_cnn import SearchCNNController
from mundus.models.backbone.DARTS.search_eeg_cnn_small_seed import FixCNNController, SearchCNNController
from mundus.models.backbone.DARTS.archetect import Architect
from tensorboardX import SummaryWriter
from mundus.dataset.dataloader.dataset_loader_seed_v import DatasetLoader_SEED_V_sep as Dataset
from mundus.visualization.search_visual import plot
from thop import clever_format
from thop import profile
from torchsummaryX import summary

criterion = nn.CrossEntropyLoss()

proto_model = SearchCNNController(62, 8, 7, 5, 2, criterion, n_nodes=2, single_path=False)
singlepath = False
if singlepath:                           
    model = FixCNNController(62, 8, 7, 5, 2, criterion, n_nodes=2, genotype_fix=None)
# self.model.print_alphas()
else:
    model = proto_model


if __name__ == "__main__":
    smaple_x = torch.randn(1,7,400,62)
    # train_raw_x = np.transpose(smaple_x, [0, 2, 1])
    logits = model(smaple_x)
    print(logits)