import argparse
import torch
import torch.nn as nn
from mudus.models.backbone.DARTS.search_eeg_cw_tw_cnn import SearchCNNController
# from mudus.models.backbone.DARTS.search_eeg_cnn import SearchCNNController
from mudus.models.backbone.DARTS.archetect import Architect


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search module test')
    parser.add_argument("--verbose", default=False, help='whether verbose each stage')
    parser.add_argument('--distributed', default=False, help='switch to distributed training on slurm')
    parser.add_argument('--input_channels', default=22, type=int)
    parser.add_argument('--init_stacks_channel', default=16, type=int)
    parser.add_argument('--init_stacks', default=7, type=int)
    parser.add_argument('--Search_layers', default=3, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--num_class', default=7, type=int)
    parser.add_argument('--w_lr', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=0.01, type=float)
    parser.add_argument('--w_momentum', default=0.9, type=float)
    parser.add_argument('--w_weight_decay', default=0.1, type=float)
    parser.add_argument('--alpha_weight_decay', default=0.1, type=float)
    parser.add_argument('--fp16', default=False, help='whether use apex quantization')
    args = parser.parse_args()

    criterion = nn.CrossEntropyLoss()
    eeg_darts = SearchCNNController(args.input_channels, args.init_stacks_channel, args.init_stacks, args.num_class,
                                args.Search_layers, criterion)
    w_optim = torch.optim.SGD(eeg_darts.weights(), args.w_lr, momentum=args.w_momentum,
                              weight_decay=args.w_weight_decay)
    alpha_optim = torch.optim.Adam(eeg_darts.alphas(), args.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=args.alpha_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, args.epochs, eta_min=0.0001)
    architect = Architect(eeg_darts, args.w_momentum, args.w_weight_decay, args)

    sample_input = torch.randn(16, 7, 400, 22)
    sample_output = eeg_darts(sample_input)

