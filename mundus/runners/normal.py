""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import logging
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

# from mundus.models.backbone.DARTS.search_eeg_cnn_small import FixCNNController, SearchCNNController
from mundus.models.backbone.DARTS.search_eeg_cnn_small_seed import FixCNNController, SearchCNNController
from mundus.models.backbone.DARTS.archetect import Architect
from tensorboardX import SummaryWriter
from mundus.dataset.dataloader.dataset_loader_BCI_IV_c import DatasetLoader_BCI_IV_mix as Dataset
from mundus.visualization.search_visual import plot


class Noraml_Trainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'normal')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(
            args.pre_gamma) + '_step' + \
                     str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) + '_' + str(args.exp_spc)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        print("Preparing dataset loader")
        self.trainset = Dataset('train', self.args, train_aug=False)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.valset_loader = DataLoader(dataset=self.valset, batch_size=args.pre_batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True)

        self.testset = Dataset('test', self.args)
        self.testset_loader = DataLoader(dataset=self.testset, batch_size=args.pre_batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True)

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_class

        # Build pretrain model
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain)
        # self.model=self.model.float()
        # Set optimizer
        params = list(self.model.encoder.parameters()) + list(self.model.pre_fc.parameters())
        
        self.optimizer = optim.Adam(params)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, eta_min=0.005)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Set the model to train mode

            print('Epoch {}'.format(epoch))
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader

            tqdm_gen = tqdm.tqdm(self.train_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # start the original evaluation
            self.model.eval()
            self.model.mode = 'origval'

            # _, valid_results = self.val_orig(self.valset.X_val, self.valset.y_val)
            # print('validation accuracy ', valid_results[0])

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'origval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test 

            val_tqdm_gen = tqdm.tqdm(self.valset_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(val_tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            print('validation accuracy ', val_acc_averager.item())
            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)


            # Set averager classes to record validation losses and accuracies
            test_loss_averager = Averager()
            test_acc_averager = Averager()

            # Generate the labels for test 

            self.model.eval()
            self.model.mode = 'origval'

            test_tqdm_gen = tqdm.tqdm(self.testset_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(test_tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                test_loss_averager.add(loss.item())
                test_acc_averager.add(acc)
            
            print('test accuracy ', test_acc_averager.item())
            # Update validation averagers
            test_loss_averager = test_loss_averager.item()
            test_acc_averager = test_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/test_loss', float(test_loss_averager), epoch)
            writer.add_scalar('data/test_acc', float(test_acc_averager), epoch)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        writer.close()


class Normal_Search_Trainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'normal_search')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_w_lr' + str(args.w_lr) + '_alpha_lr' + str(args.alpha_lr) + '_gamma' + str(
            args.pre_gamma) + '_step' + \
                     str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) + '_' + str(args.exp_spc)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

       # Load pretrain set
        print("Preparing dataset loader")
        self.trainset = Dataset('train', self.args, train_aug=False)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=int(args.pre_batch_size), shuffle=True,
                                       num_workers=8, pin_memory=False)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.valset_loader = DataLoader(dataset=self.valset, batch_size=int(args.pre_batch_size), shuffle=True,
                                       num_workers=8, pin_memory=False)

        self.testset = Dataset('test', self.args)
        self.testset_loader = DataLoader(dataset=self.testset, batch_size=int(args.pre_batch_size*0.5), shuffle=True,
                                       num_workers=8, pin_memory=False)

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_class

        # Build pretrain model
        criterion = nn.CrossEntropyLoss()
        self.model = SearchCNNController(args.input_channels, args.init_stacks_channel, args.init_stacks, args.num_class,
                                        args.Search_layers, criterion, n_nodes=args.Search_nodes)
        # self.model=self.model.float()
        # Set optimizer
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

        self.w_optim = torch.optim.SGD(self.model.weights(), args.w_lr, momentum=args.w_momentum,
                                       weight_decay=args.w_weight_decay)
        # self.w_optim_lr_schedular = lr_scheduler.CosineAnnealingLR(self.w_optim)
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), args.alpha_lr, betas=(0.5, 0.999),
                                            weight_decay=args.alpha_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, args.epochs, eta_min=0.005)
        self.architect = Architect(self.model, args.w_momentum, args.w_weight_decay, args)
        self.logger = logging.getLogger()
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        alpha_weights_path = os.path.join(self.args.save_path, 'alpha_wegiths_json.csv')
        alpha_weights_list_path = os.path.join(self.args.save_path, 'alpha_wegiths_list.csv')
        alpha_weights_file = open(alpha_weights_path, 'w') 
        alpha_weights_file_list = open(alpha_weights_list_path, 'w') 


        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Set the model to train mode

            print('Epoch {}'.format(epoch))
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_lr()[0]

            # Using tqdm to read samples from train loader
            # alpha_weights = self.model.alphas()
            alpha_norm , alpha_reduce, all_alphas = self.model.save_alphas()
            # outdata_alpha = json.dumps({'alpha_norm': alpha_norm, 'alpha_reduce':alpha_reduce})
            # alpha_weights_file.writelines(outdata_alpha+'\n')
            alpha_list_writer = csv.writer(alpha_weights_file_list)
            alpha_list_writer.writerow(all_alphas)

            tqdm_gen = tqdm.tqdm(self.train_loader)
            val_len = self.valset_loader.__len__()
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                """
                val_ind = []
                for i in range(4):
                    val_ind.append(random.randint(1,val_len))
                val_batch = self.valset.__getitem__(val_ind)
                print(torch.from_numpy(val_batch[1]).size())

                if torch.cuda.is_available():
                    val_data, _ = [torch.from_numpy(_).cuda() for _ in val_batch]
                else:
                    val_data = val_batch[0]
                val_label = torch.from_numpy(val_batch[1])
                if torch.cuda.is_available():
                    val_label = val_label.type(torch.cuda.LongTensor)
                else:
                    val_label = val_label.type(torch.LongTensor)
                """
                val_ind = random.randint(1,val_len)
                # print('val_ind', val_ind)
                for i, raw_batch in enumerate(self.valset_loader, 1):
                    # print('i', i)
                    if i != val_ind:
                        continue
                    elif i == val_ind:
                        val_batch = raw_batch
                        break
                    else:
                        print('the random val batch may have claws {}'.format(val_ind))
                        val_batch = raw_batch
                        break
                
                if torch.cuda.is_available():
                    val_data, _ = [_.cuda() for _ in val_batch]
                else:
                    val_data = val_batch[0]
                val_label = val_batch[1]
                if torch.cuda.is_available():
                    val_label = val_label.type(torch.cuda.LongTensor)
                else:
                    val_label = val_label.type(torch.LongTensor)

                self.alpha_optim.zero_grad()
                self.architect.unrolled_backward(data, label, val_data, val_label, lr, self.w_optim)
                self.alpha_optim.step()

                logits = self.model(data)
                loss = self.model.criterion(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                # Loss backwards and optimizer updates
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # start the original evaluation
            self.model.print_alphas(self.logger)
            self.model.eval()
            # self.model.mode = 'origval'

            # _, valid_results = self.val_orig(self.valset.X_val, self.valset.y_val)
            # print('validation accuracy ', valid_results[0])

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'origval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test 

            val_tqdm_gen = tqdm.tqdm(self.valset_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(val_tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            print('validation accuracy ', val_acc_averager.item())
            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Set averager classes to record validation losses and accuracies
            test_loss_averager = Averager()
            test_acc_averager = Averager()

            # Generate the labels for test 

            test_tqdm_gen = tqdm.tqdm(self.testset_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(test_tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                test_loss_averager.add(loss.item())
                test_acc_averager.add(acc)
            
            print('test accuracy ', test_acc_averager.item())
            # Update validation averagers
            test_loss_averager = test_loss_averager.item()
            test_acc_averager = test_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/test_loss', float(test_loss_averager), epoch)
            writer.add_scalar('data/test_acc', float(test_acc_averager), epoch)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            genotype = self.model.genotype()
            self.logger.info("genotype = {}".format(genotype))

            if self.args.graph_plot_path:
                plot_path = os.path.join(self.args.save_path, "EP{:02d}".format(epoch + 1))
                if not os.path.isdir(os.path.join(self.args.save_path)):
                    os.makedirs(os.path.join(self.args.save_path))
                caption = "Epoch {}".format(epoch + 1)
                plot(genotype.normal, plot_path + "-normal", caption)
                plot(genotype.reduce, plot_path + "-reduce", caption)
                # writer.add_image(plot_path + '.png')
                # writer.add_image('countdown', cv.cvtColor(cv.imread('{}.jpg'.format(i)), cv.COLOR_BGR2RGB), dataformats='HWC')

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                  timer.measure(epoch / self.args.max_epoch)))
        alpha_weights_file.close()
        alpha_weights_file_list.close()  
        writer.close()



class Searched_ReTrainer(object):
    """The class that contains the code for the pretrain phase."""

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'normal_search')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_w_lr' + str(args.w_lr) + '_alpha_lr' + str(args.alpha_lr) + '_gamma' + str(
            args.pre_gamma) + '_step' + \
                     str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch) + '_' + str(args.exp_spc)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args
        searched_path = args.searched_weights

       # Load pretrain set
        print("Preparing dataset loader")
        self.trainset = Dataset('train', self.args, train_aug=False)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=int(args.pre_batch_size), shuffle=True,
                                       num_workers=8, pin_memory=False)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.valset_loader = DataLoader(dataset=self.valset, batch_size=int(args.pre_batch_size), shuffle=True,
                                       num_workers=8, pin_memory=False)

        self.testset = Dataset('test', self.args)
        self.testset_loader = DataLoader(dataset=self.testset, batch_size=int(args.pre_batch_size), shuffle=True,
                                       num_workers=8, pin_memory=False)

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_class

        # Build pretrain model
        criterion = nn.CrossEntropyLoss()
        self.proto_model = SearchCNNController(args.input_channels, args.init_stacks_channel, args.init_stacks, args.num_class,
                                         args.Search_layers, criterion, n_nodes=args.Search_nodes, single_path=False)
        self.proto_model.load_state_dict(torch.load(searched_path)['params'])

        geno_prototype = self.proto_model.genotype()   
        print(geno_prototype)   
        # input()

        # singlepath = args.singlepath
        singlepath = True
        if singlepath:                           
            self.model = FixCNNController(args.input_channels, args.init_stacks_channel, args.init_stacks, args.num_class,
                                         args.Search_layers, criterion, n_nodes=args.Search_nodes, genotype_fix=geno_prototype)
        # self.model.print_alphas()
        else:
            self.model = self.proto_model

        # input()

        # self.model=self.model.float()
        # Set optimizer
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

        self.w_optim = torch.optim.SGD(self.model.weights(), args.w_lr, momentum=args.w_momentum,
                                       weight_decay=args.w_weight_decay)
        # self.w_optim_lr_schedular = lr_scheduler.CosineAnnealingLR(self.w_optim)
        # self.alpha_optim = torch.optim.Adam(self.model.alphas(), args.alpha_lr, betas=(0.5, 0.999),
        #                                     weight_decay=args.alpha_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.w_optim, args.epochs, eta_min=0.0005)
        # self.architect = Architect(self.model, args.w_momentum, args.w_weight_decay, args)
        self.logger = logging.getLogger()
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.state_dict(), genotype=self.model.genotype), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['test_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['test_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Set the model to train mode

            print('Epoch {}'.format(epoch))
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_lr()[0]

            # Using tqdm to read samples from train loader

            tqdm_gen = tqdm.tqdm(self.train_loader)
            val_len = self.valset_loader.__len__()
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                """
                val_ind = []
                for i in range(4):
                    val_ind.append(random.randint(1,val_len))
                val_batch = self.valset.__getitem__(val_ind)
                print(torch.from_numpy(val_batch[1]).size())

                if torch.cuda.is_available():
                    val_data, _ = [torch.from_numpy(_).cuda() for _ in val_batch]
                else:
                    val_data = val_batch[0]
                val_label = torch.from_numpy(val_batch[1])
                if torch.cuda.is_available():
                    val_label = val_label.type(torch.cuda.LongTensor)
                else:
                    val_label = val_label.type(torch.LongTensor)
                """
                """
                val_ind = random.randint(1,val_len) - 1
                for i, raw_batch in enumerate(self.valset_loader, 1):
                    if i != val_ind:
                        continue
                    elif i == val_ind:
                        val_batch = raw_batch
                        break
                    else:
                        print('the random val batch may have claws {}'.format(val_ind))
                        val_batch = raw_batch
                        break
                """
                """
                if torch.cuda.is_available():
                    val_data, _ = [_.cuda() for _ in val_batch]
                else:
                    val_data = val_batch[0]
                val_label = val_batch[1]
                if torch.cuda.is_available():
                    val_label = val_label.type(torch.cuda.LongTensor)
                else:
                    val_label = val_label.type(torch.LongTensor)
                """
                # self.alpha_optim.zero_grad()
                # self.architect.unrolled_backward(data, label, data, label, lr, self.w_optim)
                # self.alpha_optim.step()

                logits = self.model(data)
                loss = self.model.criterion(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                # Loss backwards and optimizer updates
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # start the original evaluation
            self.model.print_alphas(self.logger)
            self.model.eval()
            # self.model.mode = 'origval'

            # _, valid_results = self.val_orig(self.valset.X_val, self.valset.y_val)
            # print('validation accuracy ', valid_results[0])

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'origval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test 

            val_tqdm_gen = tqdm.tqdm(self.valset_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(val_tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            print('validation accuracy ', val_acc_averager.item())
            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Set averager classes to record validation losses and accuracies
            test_loss_averager = Averager()
            test_acc_averager = Averager()

            # Generate the labels for test 

            test_tqdm_gen = tqdm.tqdm(self.testset_loader)
            # for i, batch in enumerate(self.train_loader):
            for i, batch in enumerate(test_tqdm_gen, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                # Calculate train accuracy
                acc = count_acc(logits, label)
                test_loss_averager.add(loss.item())
                test_acc_averager.add(acc)
            
            print('test accuracy ', test_acc_averager.item())
            # Update validation averagers
            test_loss_averager = test_loss_averager.item()
            test_acc_averager = test_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/test_loss', float(test_loss_averager), epoch)
            writer.add_scalar('data/test_acc', float(test_acc_averager), epoch)
            trlog['test_loss'].append(test_loss_averager)
            trlog['test_acc'].append(test_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            genotype = self.model.genotype
            self.logger.info("genotype = {}".format(genotype))

            if self.args.graph_plot_path:
                plot_path = os.path.join(self.args.save_path, "EP{:02d}".format(epoch + 1))
                if not os.path.isdir(os.path.join(self.args.save_path)):
                    os.makedirs(os.path.join(self.args.save_path))
                caption = "Epoch {}".format(epoch + 1)
                plot(genotype.normal, plot_path + "-normal", caption)
                plot(genotype.reduce, plot_path + "-reduce", caption)
                # writer.add_image(plot_path + '.png')
                # writer.add_image('countdown', cv.cvtColor(cv.imread('{}.jpg'.format(i)), cv.COLOR_BGR2RGB), dataformats='HWC')

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        writer.close()
