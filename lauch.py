""" Main function for this repo. """
import argparse
import torch
from mudus.utils.misc import pprint
from mudus.utils.gpu_tools import set_gpu
from mudus.runners.meta import MetaTrainer
from mudus.runners.pre import PreTrainer
from mudus.runners.normal import Noraml_Trainer, Normal_Search_Trainer, Searched_ReTrainer
from mudus.runners.ind_search import PreTrainer as Ind_search
from mudus.runners.ind_search_cw_tw import PreTrainer as Ind_search_cw_tw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='EEGNet',
                        choices=['EEGNet', 'Search', 'Search_cw_tw', 'Search_retrain'])  # The network architecture
    parser.add_argument('--dataset', type=str, default='BCI_IV') # Dataset
    parser.add_argument('--data_folder', type=str, default='./data/bci_iv') # Dataset)
    parser.add_argument('--phase', type=str, default='meta_train',
                        choices=['pre_train', 'meta_train', 'meta_eval', 'independent', 'dependent'])  # Phase
    # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='1')  # GPU id
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/')  # Dataset folder

    # Parameters for meta-train phase
    # Epoch number for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=12)
    # The number for different tasks used for meta-train
    parser.add_argument('--num_batch', type=int, default=12)
    # Shot number, how many samples for one class in a task
    parser.add_argument('--shot', type=int, default=10)
    # Way number, how many classes in a task
    parser.add_argument('--way', type=int, default=3)
    # The number of training samples for each class in a task
    parser.add_argument('--train_query', type=int, default=10)
    # The number of test samples for each class in a task
    parser.add_argument('--val_query', type=int, default=10)
    # Learning rate for SS weights
    parser.add_argument('--meta_lr1', type=float, default=0.0001)
    # Learning rate for FC weights
    parser.add_argument('--meta_lr2', type=float, default=0.005)
    # Learning rate for the inner loop
    parser.add_argument('--base_lr', type=float, default=0.005)
    # The number of updates for the inner loop
    parser.add_argument('--update_step', type=int, default=20)
    # The number of epochs to reduce the meta learning rates
    parser.add_argument('--step_size', type=int, default=3)
    # Gamma for the meta-train learning rate decay
    parser.add_argument('--gamma', type=float, default=0.8)
    # The pre-trained weights for meta-train phase
    parser.add_argument('--init_weights', type=str, default=None)
    # The meta-trained weights for meta-eval phase
    parser.add_argument('--eval_weights', type=str, default=None)
    # Additional label for meta-train
    parser.add_argument('--meta_label', type=str, default='exp1')

    # Parameters for pretain phase
    # Epoch number for pre-train phase
    parser.add_argument('--pre_max_epoch', type=int, default=10)
    # Batch size for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=12)
    # embedding size
    parser.add_argument('--embed_size', type=int, default=200)
    # Learning rate for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.05)
    # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_gamma', type=float, default=0.5)
    # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_step_size', type=int, default=20)
    # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9)
    # Weight decay for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay',
                        type=float, default=0.0005)
    parser.add_argument('--lr_schedular', type=str, default='cosine',
                        choices=['cosine', 'multi-step', 'exp']) 
    parser.add_argument("--verbose", default=False, help='whether verbose each stage')
    parser.add_argument('--distributed', default=False, help='switch to distributed training on slurm')
    parser.add_argument('--input_channels', default=22, type=int)
    parser.add_argument('--init_stacks_channel', default=16, type=int)
    parser.add_argument('--init_stacks', default=7, type=int)
    parser.add_argument('--Search_layers', default=3, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--searched_weights', default='', type=str)
    parser.add_argument('--num_class', default=7, type=int)
    parser.add_argument('--w_lr', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=0.01, type=float)
    parser.add_argument('--w_momentum', default=0.9, type=float)
    parser.add_argument('--w_weight_decay', default=0.1, type=float)
    parser.add_argument('--alpha_weight_decay', default=0.1, type=float)
    parser.add_argument('--graph_plot_path', default=True, type=bool)
    parser.add_argument('--exp_spc', default='exp1', type=str)

    # Set the parameters
    args = parser.parse_args()
    # pprint(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase == 'meta_train':
        trainer = MetaTrainer(args)
        trainer.train()
    elif args.phase == 'meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()
    elif args.phase == 'pre_train':
        if args.model_type == 'EEGNet':
            trainer = PreTrainer(args)
            trainer.train()
    elif args.phase == 'independent':
        if args.model_type == 'Search':
            trainer = Ind_search(args)
            trainer.train()
        elif args.model_type == 'Search_cw_tw':
            trainer = Ind_search_cw_tw(args)
            trainer.train()
    elif args.phase == 'dependent':
        if args.model_type == 'Search':
            trainer = Normal_Search_Trainer(args)
            trainer.train()
        elif args.model_type == 'Search_cw_tw':
            trainer = Normal_Search_Trainer(args)
            trainer.train()
        elif args.model_type == 'Search_retrain':
            trainer = Searched_ReTrainer(args)
            trainer.train()
        else:
            trainer = Noraml_Trainer(args)
            trainer.train()
    else:
        raise ValueError('Please set correct phase.')
