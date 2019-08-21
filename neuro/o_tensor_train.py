import math
import numpy as np

import torch
from torch.nn import Module, Parameter, ParameterList
from torch.nn import functional as F
from torch.nn import init


def init_ott_type_cores(in_modes, out_modes, OTTrank):
    dim = len(in_modes)

    params = []
    for i in range(0, dim):
        for j in range(0, out_modes[i]):
            for k in range(0, in_modes[i]):
                if i == 0 or i == dim-1 or OTTrank == 1:
                    params.append(Parameter(torch.Tensor(OTTrank, 1)))
                else:
                    params.append(Parameter(torch.Tensor(OTTrank, OTTrank)))
    myparams = ParameterList(params)
    return myparams

def getQ(params, in_modes, out_modes, OTTrank):

    dim = len(in_modes)

    Q = []
    s = 0
    for i in range(0, dim):
        for j in range(0, out_modes[i]):
            for k in range(0, in_modes[i]):
                if i == 0 or i == dim-1 or OTTrank == 1:
                    Q.append(params[s])
                else:
                    mytriu = params[s].triu(1)
                    # skew symmetric
                    A = mytriu - mytriu.t()

                    # Cayley transform to Orthogonal SO(r)
                    I = torch.eye(OTTrank).cuda()
                    # invapprox = I - OTTrank*A + torch.mm(OTTrank*A, OTTrank*A)
                    # tmp = torch.mm(I - A , invapprox)
                    tmp = torch.mm(I - A , torch.inverse(I + A))
                    Q.append(tmp)
                s = s + 1
    return Q

def getU(Q, input_modes, output_modes):

    U = []
    start = 0
    end = 0
    for i in range(0, len(input_modes)):
        tmp = []
        for j in range(0, output_modes[i]):
            end = end + input_modes[i]
            tmp.append(torch.stack(Q[start:end], dim=1))
            start = end
        tmp = torch.stack(tmp, dim=1)
        if i==0:
            tmp = torch.transpose(tmp, 0,3)
        U.append( tmp )
    return U

def getG(U, input_modes, output_modes, ranks):
    G = []
    for i in range(len(input_modes)):
        G.append(torch.reshape(U[i],
            (output_modes[i]*ranks[i+1], input_modes[i]*ranks[i])))
    return G

def tt_dot(in_modes, out_modes, ranks, input, weight, bias=None) :
    assert len(in_modes) == len(out_modes) == len(ranks)-1
    # print('ttdot inputshape', input.shape)
    # print('ttdot inmodes', in_modes)
    assert input.shape[1] == np.prod(in_modes)
    res = input
    res = res.view(-1, int(np.prod(in_modes)))
    res = res.transpose(1, 0)
    res = res.contiguous()
    dim = len(in_modes)
    for ii in range(dim) :
        res = res.view(ranks[ii] * in_modes[ii], -1)
        res = torch.matmul(weight[ii], res)
        res = res.view(out_modes[ii], -1)
        res = res.transpose(1, 0)
        res = res.contiguous()
    res = res.view(-1, int(np.prod(out_modes)))

    if bias is not None :
        res += bias
    return res

class OTTLinear(Module):

    def __init__(self, in_modes, out_modes, OTTrank, bias=True):
        super().__init__()
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.OTTrank = OTTrank
        self.ranks = [1] + 3*[OTTrank] + [1]
        dim = len(self.in_modes)

        assert len(self.in_modes) == len(self.out_modes)
        
        # self.weight = _create_tt_cores(self.in_modes, self.out_modes, self.OTTrank)
        self.params = init_ott_type_cores(self.in_modes, self.out_modes, self.OTTrank)

        if bias:
            self.bias = Parameter(torch.Tensor(int(np.prod(out_modes))))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_xavier(self) :
        for ii in range(len(self.params)) :
            init.xavier_normal(self.params[ii])

    def reset_normal(self) :
        CONST = ((((0.05**2)/np.prod(self.ranks)))**(1/(len(self.ranks)-1))) ** 0.5 
        for ii in range(len(self.params)) :
            init.normal_(self.params[ii], 0, CONST)

    def reset_parameters(self) :
        self.reset_normal()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        Q = getQ(self.params, self.in_modes, self.out_modes, self.OTTrank)
        U = getU(Q, self.in_modes, self.out_modes)
        cores = getG(U, self.in_modes, self.out_modes, self.ranks)
        return tt_dot(self.in_modes, self.out_modes, self.ranks, input, cores, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + 'in: ' \
                + str(self.in_modes) + ' -> out:' \
            + str(self.out_modes) + ' | ' \
            + 'rank: {}'.format(str(self.ranks))

