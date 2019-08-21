import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensor_train import TTLinear
from o_tensor_train import OTTLinear

from helper import tensorauto, torchauto

seqlen = 3

class adniNet(nn.Module):


    def __init__(self, ttRanks, rnn_bias=True):
        super(adniNet, self).__init__()

        self.adniConvInit()
        self.adniRNNInit(ttRanks, rnn_bias)
        self.adniDeconvInit()


    def forward(self, x):
        batch = x.shape[0]
        for tt in range(0, seqlen-1):
            inp = x[:, tt, :, :, :].view(batch, 1, 121, 145, 121)
            inp_mapped = self.adniConv(inp)
            if tt == 0:
                # h_p = Variable(torchauto(self).FloatTensor(batch, 512, 2, 2, 2).zero_())
                h_p = Variable(torchauto(self).FloatTensor(batch, self.hidden_size).zero_())
            else:
                h_p = self._state
            # print('input shape', inp_mapped.shape)
            # print('hidden state', h_p.shape)
            pre_ih = self.weight_ih(inp_mapped.view(batch, -1))
            pre_hh = self.weight_hh(h_p)
            # pre_ih = inp_mapped
            # pre_hh = h_p
            # print(pre_ih.shape)
            # print(pre_hh.shape)
            preact = pre_ih + pre_hh
            h_t = self.activation(preact)
            # h_t = self.activation(preact.view(batch, 512, 2, 2, 2))
            self._state = h_t.view(batch, -1)
            # print(h_t.shape)

        h_t = self.weight_ho(h_t)
        out_mapped = self.adniDeconv(h_t.view(batch, 512, 2, 2, 2))

        return out_mapped

    def adniConv(self, inp):

        hid = self.convA1(inp)
        hid = self.swishA1(hid)
        # print('after first conv', hid.shape)

        self.size1 = hid.size()
        hid, self.indices1 = self.maxpoolA1(hid)
        hid = self.convA2(hid)
        hid = self.swishA2(hid)
        # print('after second conv', hid.shape)
        
        self.size2 = hid.size()
        hid, self.indices2 = self.maxpoolA2(hid)
        hid = self.convA3(hid)
        hid = self.swishA3(hid)
        # print('after third conv', hid.shape)

        self.size3 = hid.size()
        hid, self.indices3 = self.maxpoolA3(hid)
        hid = self.convA4(hid)
        hid = self.swishA4(hid)
        # print('after fourth conv', hid.shape)

        self.size4 = hid.size()
        hid, self.indices4 = self.maxpoolA4(hid)
        hid = self.convA5(hid)
        hid = self.swishA5(hid)
        # print('after fifth conv', hid.shape)

        self.size5 = hid.size()
        hid, self.indices5 = self.maxpoolA5(hid)
        hid = self.convA6(hid)
        hid = self.swishA6(hid)
        # print('after sixth conv', hid.shape)

        self.size6 = hid.size()
        hid, self.indices6 = self.maxpoolA6(hid)
        hid = self.convA7(hid)
        hid = self.swishA7(hid)
        # print('after seventh conv', hid.shape)

        return hid


    def adniDeconv(self, hid):

        out = self.deconvB7(hid)
        out = self.swishB7(out)
        # # print('before first unpool', out.shape)

        out = self.maxunpoolB6(out, self.indices6, self.size6)
        out = self.deconvB6(out)
        out = self.swishB6(out)
        
        # # print('before second unpool', out.shape)
        out = self.maxunpoolB5(out, self.indices5, self.size5)
        out = self.deconvB5(out)
        out = self.swishB5(out)
        
        # print('before third unpool', out.shape)
        out = self.maxunpoolB4(out, self.indices4, self.size4)
        out = self.deconvB4(out)
        out = self.swishB4(out)
        
        # print('before fourth unpool', out.shape)
        out = self.maxunpoolB3(out, self.indices3, self.size3)
        out = self.deconvB3(out)
        out = self.swishB3(out)
        
        # print('before fifth unpool', out.shape)
        out = self.maxunpoolB2(out, self.indices2, self.size2)
        out = self.deconvB2(out)
        out = self.swishB2(out)
        
        # print('before sixth unpool', out.shape)
        out = self.maxunpoolB1(out, self.indices1, self.size1)
        out = self.deconvB1(out)
        out = self.swishB1(out)

        return out

    def adniConvInit(self):
        #Convolution 1
        self.convA1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.convA1.weight) #Xaviers Initialisation
        self.swishA1 = nn.ReLU()

        #Max Pool 1
        self.maxpoolA1 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 2
        self.convA2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=2, bias=False)
        nn.init.xavier_uniform_(self.convA2.weight)
        self.swishA2 = nn.ReLU()

        #Max Pool 2
        self.maxpoolA2 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 3
        self.convA3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.convA3.weight)
        self.swishA3 = nn.ReLU()

        #Max Pool 3
        self.maxpoolA3 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 4
        self.convA4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=2, bias=False)
        nn.init.xavier_uniform_(self.convA4.weight)
        self.swishA4 = nn.ReLU()

        #Max Pool 4
        self.maxpoolA4 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 5
        self.convA5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.convA5.weight)
        self.swishA5 = nn.ReLU()

        #Max Pool 5
        self.maxpoolA5 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 6
        self.convA6 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.convA6.weight)
        self.swishA6 = nn.ReLU()

        #Max Pool 6
        self.maxpoolA6 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 7
        self.convA7 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.convA7.weight)
        self.swishA7 = nn.ReLU()


    def adniRNNInit(self, ranks, bias=True):
        self.in_modes = [512, 2, 2, 2]
        self.out_modes = [8, 8, 8, 8]

        self.input_size = int(np.prod(self.in_modes))
        self.hidden_size = int(np.prod(self.out_modes))

        # self.compress_in = compress_in
        # self.compress_out = compress_out

        self.bias = bias
        self.ranks = [1] + 3*[ranks] + [1]
        # out_modes_3x = list(self.out_modes)
        # out_modes_3x[-1] *= 3
        # if compress_in :
        # self.weight_ih = OTTLinear(self.in_modes, self.out_modes, ranks, bias=self.bias)
        self.weight_ih = TTLinear(self.in_modes, self.out_modes, self.ranks, bias=self.bias)
        # self.weight_ih = nn.Conv3d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # else :
        # self.weight_ih = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        # if compress_out :
        # self.weight_hh = OTTLinear(self.out_modes, self.out_modes, ranks, bias=self.bias)
        self.weight_hh = TTLinear(self.out_modes, self.out_modes, self.ranks, bias=self.bias)
        # self.weight_hh = nn.Conv3d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # else :
        # self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        # self.activation = nn.Conv3d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # self.activation = TTLinear(self.out_modes, self.out_modes, self.ranks, bias=self.bias)
        self.activation = OTTLinear(self.out_modes, self.out_modes, ranks, bias=self.bias)

        self.weight_ho = TTLinear(self.out_modes, self.in_modes, self.ranks, bias=self.bias)
        # self.weight_ho = nn.Conv3d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # self.weight_ho = nn.Linear(self.hidden_size, self.input_size, bias=self.bias)

        self.reset_parameters()
        pass


    def adniDeconvInit(self):
        #De Convolution 7
        self.deconvB7 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.deconvB7.weight)
        self.swishB7 = nn.ReLU()

        #Max UnPool 6
        self.maxunpoolB6 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 6
        self.deconvB6 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.deconvB6.weight)
        self.swishB6 = nn.ReLU()

        #Max UnPool 5
        self.maxunpoolB5 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 5
        self.deconvB5 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.deconvB5.weight)
        self.swishB5 = nn.ReLU()

        #Max UnPool 4
        self.maxunpoolB4 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 4
        self.deconvB4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, padding=2)
        nn.init.xavier_uniform_(self.deconvB4.weight)
        self.swishB4 = nn.ReLU()

        #Max UnPool 3
        self.maxunpoolB3 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 3
        self.deconvB3 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.deconvB3.weight)
        self.swishB3 = nn.ReLU()

        #Max UnPool 2
        self.maxunpoolB2 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 2
        self.deconvB2 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, padding=2)
        nn.init.xavier_uniform_(self.deconvB2.weight)
        self.swishB2 = nn.ReLU()

        #Max UnPool 1
        self.maxunpoolB1 = nn.MaxUnpool3d(kernel_size=2)

        #DeConvolution 1
        self.deconvB1 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.deconvB1.weight)
        self.swishB1 = nn.ReLU()

    def reset_parameters(self) :
        self.weight_hh.reset_parameters()
        self.weight_ih.reset_parameters()

    def reset(self) :
        self._state = None