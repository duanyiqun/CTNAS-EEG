""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from .search_cells import SearchCell, Cell
import mundus.models.backbone.DARTS.genotypes_seed as gt
# from torch.nn.parallel._functions import Broadcast
import logging

# def broadcast_list(l, device_ids):
#     """ Broadcasting list """
#     l_copies = Broadcast.apply(device_ids, *l)
#     l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

#     return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, C_in, C, S_c, n_classes, n_layers, n_nodes=2, stem_multiplier=1, single_path=False):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            S_c: # of depth of the stacked time series
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.S_c = S_c
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(S_c, C_cur, (1, 3), 1, (0, 1), bias=False),
            # nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        """
        self.channel_wise_mix = nn.Sequential(
            nn.Conv2d(C_in, C_cur, (33, 1), 1, (0, 0), bias=False),
            # nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        """
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_cur = C_in
        C_pp, C_p, C_cur = C_cur, C_cur, C_cur

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 4, 2 * n_layers // 4, 3 * n_layers // 4]:
                C_cur *= 1
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, single_path=single_path)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveAvgPool2d((32,2))
        self.gap = nn.AdaptiveMaxPool2d((24,2))
        self.linear = nn.Linear(C_p * 24 *2 , n_classes)

    def forward(self, x, weights_normal, weights_reduce, debug=False):
        # s_tw = x.view(x.size()[0], -1, x.size()[-1]) # used for double channel implementation
        s0 = s1 = self.stem(x).permute(0, 3, 2, 1)
        # s0 = self.channel_wise_mix(s0.permute(0, 3, 2, 1))
        # s1 = self.channel_wise_mix(s1.permute(0, 3, 2, 1))
        if debug:
            print(s0.size())
            print(s1.size())
        # print("stem part end")

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
            if debug:
                print(s0.size())
                print(s1.size())

        out = self.gap(s1)
        if debug:
            print(out.size())
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, C_in, C, S_c, n_classes, n_layers, criterion, n_nodes=2, stem_multiplier=2, single_path=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        #         if device_ids is None:
        #             device_ids = list(range(torch.cuda.device_count()))
        #         self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, S_c,  n_classes, n_layers, n_nodes, stem_multiplier, single_path=single_path)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        #         if len(self.device_ids) == 1:
        #             return self.net(x, weights_normal, weights_reduce)

        #         # scatter x
        #         xs = nn.parallel.scatter(x, self.device_ids)
        #         # broadcast weights
        #         wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        #         wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        #         # replicate modules
        #         replicas = nn.parallel.replicate(self.net, self.device_ids)
        #         outputs = nn.parallel.parallel_apply(replicas,
        #                                              list(zip(xs, wnormal_copies, wreduce_copies)),
        #                                              devices=self.device_ids)
        #         return nn.parallel.gather(outputs, self.device_ids[0])
        return self.net(x, weights_normal, weights_reduce)

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def save_alphas(self):
        # remove formats
        all_alphas = []
        print("####### ALPHA #######")
        print("# Alpha - normal")
        for alpha in self.alpha_normal:
            print(F.softmax(alpha, dim=-1))
            item_list = F.softmax(alpha, dim=-1).tolist()
            item_content = []
            for item in item_list:
                item_content.extend([66])
                item_content.extend(item)
            all_alphas.extend(item_content)
            all_alphas.extend([99])

        print("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            print(F.softmax(alpha, dim=-1))
            item_list = F.softmax(alpha, dim=-1).tolist()
            item_content = []
            for item in item_list:
                item_content.extend([66])
                item_content.extend(item)
            all_alphas.extend(item_content)
            all_alphas.extend([99])
        print("#####################")
        return self.alpha_normal, self.alpha_reduce, all_alphas

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p



class FixCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, C_in, C, S_c, n_classes, n_layers, n_nodes=2, stem_multiplier=1, genotype=None):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            S_c: # of depth of the stacked time series
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.S_c = S_c
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(S_c, C_cur, (1, 3), 1, (0, 1), bias=False),
            # nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        """
        self.channel_wise_mix = nn.Sequential(
            nn.Conv2d(C_in, C_cur, (33, 1), 1, (0, 0), bias=False),
            # nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        """
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_cur = C_in
        C_pp, C_p, C_cur = C_cur, C_cur, C_cur

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 4, 2 * n_layers // 4, 3 * n_layers // 4]:
                C_cur *= 1
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_pp, C_p, C_cur, reduction, reduction_p)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveAvgPool2d((32,2))
        self.gap = nn.AdaptiveMaxPool2d((24,2))
        self.linear = nn.Linear(C_p * 24 *2 , n_classes)

    def forward(self, x, debug=False):
        # s_tw = x.view(x.size()[0], -1, x.size()[-1]) # used for double channel implementation
        s0 = s1 = self.stem(x).permute(0, 3, 2, 1)
        # s0 = self.channel_wise_mix(s0.permute(0, 3, 2, 1))
        # s1 = self.channel_wise_mix(s1.permute(0, 3, 2, 1))
        if debug:
            print(s0.size())
            print(s1.size())

        for cell in self.cells:
            # weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, drop_prob=-1)
            if debug:
                print(s0.size())
                print(s1.size())

        out = self.gap(s1)
        if debug:
            print(out.size())
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class FixCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, C_in, C, S_c, n_classes, n_layers, criterion, n_nodes=2, stem_multiplier=2, genotype_fix=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        #         if device_ids is None:
        #             device_ids = list(range(torch.cuda.device_count()))
        #         self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
        
        self.genotype = genotype_fix

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        self.net = FixCNN(C_in, C, S_c,  n_classes, n_layers, n_nodes, stem_multiplier, genotype=genotype_fix)
    
    def _compile(self, C_in, C, S_c,  n_classes, n_layers, n_nodes, stem_multiplier):
        self.net = FixCNN(C_in, C, S_c,  n_classes, n_layers, n_nodes, stem_multiplier, genotype=self.genotype)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        #         if len(self.device_ids) == 1:
        #             return self.net(x, weights_normal, weights_reduce)

        #         # scatter x
        #         xs = nn.parallel.scatter(x, self.device_ids)
        #         # broadcast weights
        #         wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        #         wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        #         # replicate modules
        #         replicas = nn.parallel.replicate(self.net, self.device_ids)
        #         outputs = nn.parallel.parallel_apply(replicas,
        #                                              list(zip(xs, wnormal_copies, wreduce_copies)),
        #                                              devices=self.device_ids)
        #         return nn.parallel.gather(outputs, self.device_ids[0])
        return self.net(x)

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
    """
    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)
    """
    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
