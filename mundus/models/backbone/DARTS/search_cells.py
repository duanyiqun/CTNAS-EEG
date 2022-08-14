""" CNN cell for architecture search """
import torch
import torch.nn as nn
import mundus.models.backbone.DARTS.ops as ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, single_path=False):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                if single_path == False:
                    op = ops.MixedOp(C, stride)
                else:
                    op = ops.Single_Path_Op(C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    self.reduction = reduction
    if reduction_prev:
      self.preprocess0 = ops.FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ops.StdConv(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ops.StdConv(C_prev, C, 1, 1, 0)
    print(genotype.reduce)
    print(genotype.normal)
    reduce_cell = []
    for modular in genotype.reduce:
        reduce_cell.extend(modular)
    normal_cell = []
    for modular in genotype.normal:
        normal_cell.extend(modular)

    if reduction:
      op_names, indices = zip(*reduce_cell)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*normal_cell)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    print('OP_names', indices)
    print('indices', indices)
    self._indices = []
    for name, index in zip(op_names, indices):
      # index = name[1]
      # name = name[0]
      stride = 2 if reduction and index < 2 else 1
      print('name', name)
      print('index', index)
      op = ops.OPS[name](C, stride, True)
      self._ops += [op]
      self._indices.append(index)
    

  def forward(self, s0, s1, drop_prob, debug=False):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, ops.Identity):
          h1 = ops.drop_path(h1, drop_prob)
        if not isinstance(op2, ops.Identity):
          h2 = ops.drop_path(h2, drop_prob)
      s = h1 + h2
      if debug:
          print(s.size())
      states += [s]
    if debug:
        print(len(states))
        print(self._concat)
    
    return torch.cat([states[i] for i in self._concat], dim=1)