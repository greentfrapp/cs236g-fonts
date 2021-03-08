import functools

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    # print "classname",classname
    if classname.find('Conv') != -1:
        # print "in random conv"
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        # print "in random batchnorm"
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FontEncoder(nn.Module):
    def __init__(self, input_nc=52, output_nc=52, ngf=2, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, norm_type='batch'):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = conv_norm_relu_module(norm_type, norm_layer, input_nc, ngf, 7, 3) 
        n_downsampling = 4
        for i in range(n_downsampling):
            factor_ch = 2 #2**i : 3**i is a more complicated filter
            mult = factor_ch**i 
            model += conv_norm_relu_module(norm_type,norm_layer, ngf * mult, ngf * mult * factor_ch, 3,1, stride=2)
        mult = factor_ch**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout, norm_type=norm_type)]
        self.model = nn.Sequential(*model)

    def forward(self, input):  
        return self.model(input) 


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_nc = 52
        output_nc = 52
        self.netG_3d = ResnetGenerator_3d_conv(input_nc, output_nc)
        self.encoder = ResnetEncoder(input_nc, output_nc)
        self.decoder = ResnetDecoder(input_nc, output_nc)
        self.netG_3d.apply(weights_init)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def forward(self, input):
        return self.decode(self.encode(input))

    def encode(self, input):
        output = self.netG_3d(input.unsqueeze(2))
        output = self.encoder(output.squeeze(2))
        return output

    def decode(self, input):
        output = self.decoder(input)
        return torch.sigmoid(output)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        self.model = NLayerDiscriminator(
            input_nc=52,
            ndf=ndf,
            n_layers=n_layers,
            use_sigmoid=True,
            norm_layer=get_norm_layer(norm_type="batch"),
            norm_type="batch",
            postConv=True
        )
        self.model.apply(weights_init)

    def forward(self, input):
        return self.model(input)


class ResnetGenerator_3d_conv(nn.Module):
    def __init__(self, input_nc, output_nc, norm_type='batch', groups=26, ksize=3, padding=1):
        super(ResnetGenerator_3d_conv, self).__init__()
        self.input_nc = input_nc
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm3d,affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
        else:
            norm_layer = functools.partial(nn.BatchNorm3d,affine=True)
        model = [nn.Conv3d(input_nc, output_nc, kernel_size=ksize, padding=padding, groups=groups),
                 norm_layer(output_nc),
                 nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, norm_type='batch'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.encoder = ResnetEncoder(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, norm_type)
        self.decoder = ResnetDecoder(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, norm_type)

    def forward(self, input, decoder=True):  
        return self.decoder(self.encoder(input))


class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, norm_type='batch'):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = conv_norm_relu_module(norm_type, norm_layer, input_nc, ngf, 7, 3) 
        n_downsampling = 2
        for i in range(n_downsampling):
            factor_ch = 3 #2**i : 3**i is a more complicated filter
            mult = factor_ch**i 
            model += conv_norm_relu_module(norm_type,norm_layer, ngf * mult, ngf * mult * factor_ch, 3,1, stride=2)
        mult = factor_ch**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout, norm_type=norm_type)]
        self.model = nn.Sequential(*model)

    def forward(self, input):  
        return self.model(input) 


class ResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, norm_type='batch', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        n_downsampling = 2
        factor_ch = 3
        mult = factor_ch**n_downsampling
        model = []
        for i in range(n_downsampling):
            mult = factor_ch**(n_downsampling - i)
            model += convTranspose_norm_relu_module(norm_type,norm_layer, ngf * mult, int(ngf * mult / factor_ch), 3, 1,
                                        stride=2, output_padding=1)
        if norm_type=='batch' or norm_type=='instance':
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert('norm not defined')
        # model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):  
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, norm_type='batch'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, norm_type)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, norm_type):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1
        # TODO: InstanceNorm
        conv_block += conv_norm_relu_module(norm_type, norm_layer, dim,dim, 3, p)
        if use_dropout:
            conv_block += [nn.Dropout(0.2)]
        else:
            conv_block += [nn.Dropout(0.0)]
        if norm_type=='batch' or norm_type=='instance':
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim)]
        else:
            assert("norm not defined")
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        norm_layer = None
        print('normalization layer [%s] is not found' %norm_type)
    return norm_layer


def conv_norm_relu_module(norm_type, norm_layer, input_nc, ngf, kernel_size, padding, stride=1, relu='relu'):
    model = [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, padding=padding,stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]
    if relu=='relu':
        model += [nn.ReLU(True)]
    elif relu=='Lrelu':
        model += [nn.LeakyReLU(0.2, True)]
    return model


def convTranspose_norm_relu_module(norm_type, norm_layer, input_nc, ngf, kernel_size, padding, stride=1, output_padding=0):
    if norm_type=='batch' or norm_type=='instance':
        model = [nn.ConvTranspose2d(input_nc, ngf,
                    kernel_size=kernel_size, stride=stride,padding=padding, output_padding=output_padding),
                    norm_layer(int(ngf)),
                    nn.ReLU(True)]
    return model


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, norm_type='batch', postConv=True):
        super(NLayerDiscriminator, self).__init__()
        kw = 5
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += conv_norm_relu_module(norm_type, norm_layer, ndf * nf_mult_prev,
                        ndf * nf_mult, kw, padw, stride=2, relu='Lrelu')
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += conv_norm_relu_module(norm_type, norm_layer, ndf * nf_mult_prev,
                    ndf * nf_mult, kw, padw, stride=1, relu='Lrelu')
        if postConv:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
            if use_sigmoid:
                sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False).to(self.device)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False).to(self.device)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
