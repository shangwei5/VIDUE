# from torch.nn import init
import functools
import time

# from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import Sty_layer, CA_layer
# from .DCN_v2.modules.modulated_deform_conv import *

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_offset_quad(input_nc, nf, n_offset,offset_mode='quad', norm='batch'):
    # net_offset = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert (torch.cuda.is_available())
    net_offset = OffsetNet_quad(input_nc,nf,n_offset,offset_mode=offset_mode,norm_layer=norm_layer)

    # if use_gpu:
    #     # net_offset.cuda(gpu_ids[0])
    #     net_offset.to(gpu_ids[0])
    #     if len(gpu_ids)>1:
    #         net_offset = torch.nn.DataParallel(net_offset,gpu_ids)
    net_offset.apply(weights_init)
    return net_offset

def define_offset_quad_v2(input_nc, nf, n_offset,offset_mode='quad', norm='batch', vecdim=256):
    # net_offset = None
    # use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    # if use_gpu:
    #     assert (torch.cuda.is_available())
    net_offset = OffsetNet_quad_v2(input_nc,nf,n_offset,offset_mode=offset_mode,norm_layer=norm_layer, vecChannels=vecdim)

    # if use_gpu:
    #     # net_offset.cuda(gpu_ids[0])
    #     net_offset.to(gpu_ids[0])
    #     if len(gpu_ids)>1:
    #         net_offset = torch.nn.DataParallel(net_offset,gpu_ids)
    net_offset.apply(weights_init)
    return net_offset

def define_blur(): #gpu_ids=[]
    # net_blur = None
    # use_gpu = len(gpu_ids) > 0
    #
    # if use_gpu:
    #     assert (torch.cuda.is_available())
    net_blur = BlurNet()

    # if use_gpu:
    # net_blur.cuda(gpu_ids[0])
    return net_blur



def define_deblur_offset(input_nc, nf, n_offset, offset_mode, norm_layer=nn.BatchNorm2d,gpu_ids=[]):
    # net_deblur = None
    # use_gpu = len(gpu_ids) > 0
    #
    # if use_gpu:
    #     assert (torch.cuda.is_available())
    net_deblur = DMPHN_decoder_offset(input_nc,nf,n_offset, offset_mode, norm_layer=norm_layer, gpu_ids=[])
    # if use_gpu:
    #     net_deblur.to(gpu_ids[0])
    #     # import ipdb; ipdb.set_trace()
    #     if len(gpu_ids)>1:
    #         net_deblur = torch.nn.DataParallel(net_deblur,gpu_ids)
    net_deblur.apply(weights_init)
    return net_deblur


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

class ResnetBlock_woNorm(nn.Module):

    def __init__(self, dim, use_bias):
        super(ResnetBlock_woNorm, self).__init__()

        padAndConv_1 = [
                nn.ReplicationPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        padAndConv_2 = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        blocks = padAndConv_1 + [
            nn.ReLU(True)
        ]  + padAndConv_2 
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def TriResblock(input_nc, use_bias=True):
    Res1 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    Res2 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    Res3 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    return nn.Sequential(Res1,Res2,Res3)

def conv_TriResblock(input_nc,out_nc,stride, use_bias=True):
    Relu = nn.ReLU(True)
    if stride==1:
        pad = nn.ReflectionPad2d(2)
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=1,padding=0,bias=use_bias)
    elif stride==2:
        pad = nn.ReflectionPad2d((1,2,1,2))
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=2,padding=0,bias=use_bias)
    tri_resblock = TriResblock(out_nc)
    return nn.Sequential(pad,conv,Relu,tri_resblock)

class Bottleneck(nn.Module):
    def __init__(self,nChannels,kernel_size=3):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels*2, kernel_size=1, 
                                padding=0, bias=True)
        self.lReLU1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(nChannels*2, nChannels, kernel_size=kernel_size, 
                                padding=(kernel_size-1)//2, bias=True)
        self.lReLU2 = nn.LeakyReLU(0.2, True)
        self.model = nn.Sequential(self.conv1,self.lReLU1,self.conv2,self.lReLU2)
    def forward(self,x):
        out = self.model(x)
        return out

class OffsetNet_quad(nn.Module):
    # offset for Start and End Points, then calculate a quadratic function
    def __init__(self, input_nc, nf, n_offset, offset_mode='quad', norm_layer=nn.BatchNorm2d):
        super(OffsetNet_quad,self).__init__()
        self.input_nc = input_nc  #3
        self.nf = nf  #16
        self.n_offset = n_offset    # 15
        self.offset_mode = offset_mode
        if offset_mode == 'quad' or offset_mode == 'bilin':
            output_nc = 2 * 2
        elif offset_mode == 'lin':
            output_nc = 1 * 2
        else:
            output_nc = 2 * 2
        
        use_dropout = False
        use_bias=True

        self.pad_1 = nn.ReflectionPad2d((1,2,1,2))
        self.todepth = SpaceToDepth(block_size=2)
        self.conv_1 = conv_TriResblock(input_nc*4,nf,stride=1,use_bias=True)
        self.conv_2 = conv_TriResblock(nf,nf*2,stride=2,use_bias=True)
        self.conv_3 = conv_TriResblock(nf*2,nf*4,stride=2,use_bias=True)

        self.bottleneck_1 = Bottleneck(nf*4)
        self.uconv_1 = nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)

        self.bottleneck_2 = Bottleneck(nf*4)        
        self.uconv_2 = nn.ConvTranspose2d(nf*4, nf, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        self.bottleneck_3 = Bottleneck(nf*2)
        self.uconv_3 = nn.ConvTranspose2d(nf*2, nf*2, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        self.conv_out_0 = nn.Conv2d(nf*2,output_nc,kernel_size=5,stride=1,padding=2,bias=use_bias)

    def Quad_traj(self,offset10,offset12):
        B,C,H,W = offset10.size()
        N = self.n_offset//2
        t = torch.arange(1,N,step=1,dtype=torch.float32).cuda()
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = 0.5 * ((t + t**2)*offset12 - (t - t**2)*offset10)
        offset_10N = 0.5 * ((t + t**2)*offset10 - (t - t**2)*offset12)
        offset_12N = offset_12N.view(B,-1,H,W)   # b, (N-1)*c, h,w
        offset_10N = offset_10N.view(B,-1,H,W)

        return offset_10N,offset_12N

    def Bilinear_traj(self,offset10,offset12):
        B,C,H,W = offset10.size()
        N = self.n_offset//2
        t = torch.arange(1,N,step=1,dtype=torch.float32).cuda()
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = t * offset12
        offset_10N = t * offset10
        offset_12N = offset_12N.view(B,-1,H,W)
        offset_10N = offset_10N.view(B,-1,H,W)
        return offset_10N,offset_12N
    

    def forward(self, input):
        scale_0 = input
        B,N,H,W = input.size()
        scale_0_depth = self.todepth(scale_0)
        d_conv1 = self.conv_1(scale_0_depth)
        d_conv2 = self.conv_2(d_conv1)
        d_conv3 = self.conv_3(d_conv2)

        d_conv3 = self.bottleneck_1(d_conv3)
        u_conv1 = self.uconv_1(d_conv3)
        u_conv1 = F.leaky_relu(u_conv1,0.2,True) 
        u_conv1 = torch.cat((u_conv1 , d_conv2),dim=1)
        
        u_conv1 = self.bottleneck_2(u_conv1)
        u_conv2 = self.uconv_2(u_conv1)
        u_conv2 = F.leaky_relu(u_conv2,0.2,True)
        u_conv2 = torch.cat((u_conv2 , d_conv1),dim=1)

        u_conv2 = self.bottleneck_3(u_conv2)
        u_conv3 = self.uconv_3(u_conv2)

        out = self.conv_out_0(F.relu(u_conv3))
        # quadratic or bilinear
        if self.offset_mode == 'se':
            return out
        if self.offset_mode == 'quad' or self.offset_mode == 'bilin':
            offset_SPoint = out[:,:2,:,:]
            offset_EPoint = out[:,2:,:,:]
            if self.offset_mode == 'quad':
                offset_S_0, offset_0_E = self.Quad_traj(offset_SPoint,offset_EPoint)
            else:
                offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint,offset_EPoint)
        elif self.offset_mode == 'lin':
            # linear
            offset_SPoint = out
            offset_EPoint = 0 - out
            offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint,offset_EPoint)
        else:
            # return out[:, 2:, :, :] - out[:, :2, :, :]
            delta = torch.abs(out[:, 2:, :, :] - out[:, :2, :, :])
            return torch.sqrt(delta[:,0:1,:,:].pow(2)+delta[:,1:2,:,:].pow(2))

        zeros = torch.zeros(B,2,H,W).cuda()
        out = torch.cat((offset_SPoint,offset_S_0,zeros,offset_0_E,offset_EPoint),dim=1)  #b, c*(N-1+1+1+N-1+1),h,w
        return out


class OffsetNet_quad_v2(nn.Module):
    # offset for Start and End Points, then calculate a quadratic function
    def __init__(self, input_nc, nf, n_offset, offset_mode='quad', norm_layer=nn.BatchNorm2d, vecChannels=256):
        super(OffsetNet_quad_v2, self).__init__()
        self.input_nc = input_nc  # 3
        self.nf = nf  # 16
        self.n_offset = n_offset  # 15
        self.offset_mode = offset_mode
        if offset_mode == 'quad' or offset_mode == 'bilin':
            output_nc = 2 * 2
        elif offset_mode == 'lin':
            output_nc = 1 * 2
        else:
            output_nc = 2 * 2

        use_dropout = False
        use_bias = True

        self.pad_1 = nn.ReflectionPad2d((1, 2, 1, 2))
        self.todepth = SpaceToDepth(block_size=2)
        self.conv_1 = conv_TriResblock(input_nc * 4, nf, stride=1, use_bias=True)
        self.conv_2 = conv_TriResblock(nf, nf * 2, stride=2, use_bias=True)
        self.conv_3 = conv_TriResblock(nf * 2, nf * 4, stride=2, use_bias=True)

        self.bottleneck_1 = Bottleneck(nf * 4)
        self.uconv_1 = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1,
                                          bias=use_bias)

        self.bottleneck_2 = Bottleneck(nf * 4)
        self.uconv_2 = nn.ConvTranspose2d(nf * 4, nf, kernel_size=4, stride=2, padding=1,
                                          bias=use_bias)
        self.bottleneck_3 = Bottleneck(nf * 2)
        self.uconv_3 = nn.ConvTranspose2d(nf * 2, nf * 2, kernel_size=4, stride=2, padding=1,
                                          bias=use_bias)
        self.conv_out_0 = nn.Conv2d(nf * 2, output_nc, kernel_size=5, stride=1, padding=2, bias=use_bias)

        if vecChannels == 256:
            reduc = 4
        else:
            reduc = 1
        self.use3 = CA_layer(vecChannels, nf * 2, reduc)
        self.use2 = CA_layer(vecChannels, nf * 4, reduc)
        self.use1 = CA_layer(vecChannels, nf * 4, reduc)

    def Quad_traj(self, offset10, offset12):
        B, C, H, W = offset10.size()
        N = self.n_offset // 2
        t = torch.arange(1, N, step=1, dtype=torch.float32).cuda()
        t = t / N
        t = t.view(-1, 1, 1, 1)
        offset10 = offset10.view(B, 1, 2, H, W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = 0.5 * ((t + t ** 2) * offset12 - (t - t ** 2) * offset10)
        offset_10N = 0.5 * ((t + t ** 2) * offset10 - (t - t ** 2) * offset12)
        offset_12N = offset_12N.view(B, -1, H, W)  # b, (N-1)*c, h,w
        offset_10N = offset_10N.view(B, -1, H, W)

        return offset_10N, offset_12N

    def Bilinear_traj(self, offset10, offset12):
        B, C, H, W = offset10.size()
        N = self.n_offset // 2
        t = torch.arange(1, N, step=1, dtype=torch.float32).cuda()
        t = t / N
        t = t.view(-1, 1, 1, 1)
        offset10 = offset10.view(B, 1, 2, H, W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = t * offset12
        offset_10N = t * offset10
        offset_12N = offset_12N.view(B, -1, H, W)
        offset_10N = offset_10N.view(B, -1, H, W)
        return offset_10N, offset_12N

    def forward(self, input, vec):
        num = input.size()[0] // vec.size()[0]
        new_vec = []
        for i in range(vec.size()[0]):
            new_vec.append(vec[i:i+1, :].repeat(num,1))
        vec = torch.cat(new_vec, dim=0)
        scale_0 = input
        B, N, H, W = input.size()
        scale_0_depth = self.todepth(scale_0)
        d_conv1 = self.conv_1(scale_0_depth)
        d_conv2 = self.conv_2(d_conv1)
        d_conv3 = self.conv_3(d_conv2)

        d_conv3 = self.bottleneck_1(d_conv3)
        d_conv3 = self.use1(d_conv3, vec)
        u_conv1 = self.uconv_1(d_conv3)
        u_conv1 = F.leaky_relu(u_conv1, 0.2, True)
        u_conv1 = torch.cat((u_conv1, d_conv2), dim=1)

        u_conv1 = self.bottleneck_2(u_conv1)
        u_conv1 = self.use2(u_conv1, vec)
        u_conv2 = self.uconv_2(u_conv1)
        u_conv2 = F.leaky_relu(u_conv2, 0.2, True)
        u_conv2 = torch.cat((u_conv2, d_conv1), dim=1)

        u_conv2 = self.bottleneck_3(u_conv2)
        u_conv2 = self.use3(u_conv2, vec)
        u_conv3 = self.uconv_3(u_conv2)

        out = self.conv_out_0(F.relu(u_conv3))
        # quadratic or bilinear
        if self.offset_mode == 'se':
            return out
        if self.offset_mode == 'quad' or self.offset_mode == 'bilin':
            offset_SPoint = out[:, :2, :, :]
            offset_EPoint = out[:, 2:, :, :]
            if self.offset_mode == 'quad':
                offset_S_0, offset_0_E = self.Quad_traj(offset_SPoint, offset_EPoint)
            else:
                offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint, offset_EPoint)
        elif self.offset_mode == 'lin':
            # linear
            offset_SPoint = out
            offset_EPoint = 0 - out
            offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint, offset_EPoint)
        else:
            # return out[:, 2:, :, :] - out[:, :2, :, :]
            delta = torch.abs(out[:, 2:, :, :] - out[:, :2, :, :])
            return torch.sqrt(delta[:, 0:1, :, :].pow(2) + delta[:, 1:2, :, :].pow(2))

        zeros = torch.zeros(B, 2, H, W).cuda()
        out2 = torch.cat((offset_SPoint, offset_S_0, zeros, offset_0_E, offset_EPoint),
                        dim=1)  # b, c*(N-1+1+1+N-1+1),h,w
        return out, out2

class BlurNet(nn.Module):
    def __init__(self):
        super(BlurNet,self).__init__()
        self.Dcn = ModulatedDeformConv_blur(in_channels=1, out_channels=1, kernel_size=1,
						stride=1, padding=0, deformable_groups=1)
        # import torchvision
        # torchvision.ops.deform_conv2d()
    def forward(self,real_B,offset):

        o1, o2 = torch.chunk(offset, 2, dim=1)  ## These two operation may be neccessary for accurate BP in torch 1.10
        offset = torch.cat((o1, o2), dim=1)     ## suggest not to delete

        B,C,H,W = offset.size()
        mask = torch.ones(B,1,H,W).cuda()

        # real_B = self.pad(real_B)
        fake_A = self.Dcn(real_B,offset,mask)
        return fake_A
# import torchvision
# torchvision.ops.deform_conv2d()



def TriResblock_uconv(input_nc,out_nc,stride, use_bias=True):
    tri_resblock = TriResblock(input_nc)
    if stride ==1:
        uconv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=1,padding=2,bias=use_bias)
    elif stride==2:
        uconv = nn.ConvTranspose2d(input_nc, out_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
    lRelu = nn.ReLU(True)
    # lRelu = nn.LeakyReLU(0.2,True)
    return nn.Sequential(tri_resblock,uconv,lRelu)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x




def offset_modulation(offset,ratio):
    B,C,H,W = offset.shape

    offset_view = offset.view(B,-1,2,H,W)
    kernel_size = 3
    offset_choice = kernel_size **2
    offset_crop = offset_view[:,3:12,:,:,:]
    offset_x = offset_crop[:,:,0,:,:].contiguous()
    offset_y = offset_crop[:,:,1,:,:].contiguous()
    offset_xy = torch.cat([offset_x,offset_y], dim=1)
    offset_xy_scale = offset_xy * ratio

    xv, yv = torch.meshgrid([torch.arange(-1,2),torch.arange(-1,2)])
    xy_grid = torch.cat((xv.contiguous().view(-1),yv.contiguous().view(-1)),dim=0)
    index = xy_grid.view(1,-1,1,1)
    index = index.repeat(B,1,H,W)
    index = index.cuda().float()
    offset_modulate = offset_xy_scale - index
    # mask = torch.ones(B,offset_choice,H,W).cuda()
    return offset_modulate


class Decoder_offset001(nn.Module):
    def __init__(self,seq,in_channel,out_channel,mid_channel):
        super(Decoder_offset001, self).__init__()
        # Deconv3
        self.ratio_3 = nn.Parameter(torch.tensor(0.08),requires_grad=False)
        input_channel = seq*in_channel

        self.dcn_5_0 = ModulatedDeformConv(32, 32, kernel_size=3, stride=1, padding=1, deformable_groups=1)
        self.dcn_5_1 = ModulatedDeformConv(32, 32, kernel_size=3, stride=1, padding=1, deformable_groups=1)
        self.dcn_6_0 = ModulatedDeformConv(32, 32, kernel_size=3, stride=1, padding=1, deformable_groups=1)
        self.dcn_6_1 = ModulatedDeformConv(32, 32, kernel_size=3, stride=1, padding=1, deformable_groups=1)

        self.layer12 = nn.Conv2d(input_channel,mid_channel,kernel_size=3, padding=1)
        self.layer13 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1)
            )
        # self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # #Deconv2
        # self.layer17 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #     )
        # self.layer18 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #     )
        # self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        self.layer15 = nn.Conv2d(mid_channel, 32, kernel_size=3, padding=1)
        # self.layer19 = nn.Conv2d(32, mid_channel, kernel_size=3, padding=1)

        self.layer24 = nn.Conv2d(32, out_channel, kernel_size=3, padding=1)


    def forward(self,x, offset_0):

        # modulation
        offset_scale3 = offset_modulation(offset_0, self.ratio_3)

        #conv
        x = self.layer12(x)
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer15(x)

        res5 = self.dcn_5_0(x,offset_scale3)
        res5 = F.relu(res5)
        res5 = self.dcn_5_1(res5,offset_scale3)
        x = x + res5
        res6 = self.dcn_6_0(x,offset_scale3)
        res6 = F.relu(res6)
        res6 = self.dcn_6_1(res6,offset_scale3)
        x = x + res6

        x = self.layer24(x)
        return x


class DMPHN_decoder_offset_001(nn.Module):
    def __init__(self):
        super(DMPHN_decoder_offset_001,self).__init__()

        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()

        self.decoder_lv1 = Decoder_offset001()
        self.decoder_lv2 = Decoder_offset001()
        self.decoder_lv3 = Decoder_offset001()

    def forward(self, image, offset):
        images_lv1 = image
        offset_lv1 = offset
        H = images_lv1.size(2)
        W = images_lv1.size(3)

        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        offset_lv2_1 = offset_lv1[:,:,0:int(H/2),:]
        offset_lv2_2 = offset_lv1[:,:,int(H/2):H,:]
        offset_lv3_1 = offset_lv2_1[:,:,:,0:int(W/2)]
        offset_lv3_2 = offset_lv2_1[:,:,:,int(W/2):W]
        offset_lv3_3 = offset_lv2_2[:,:,:,0:int(W/2)]
        offset_lv3_4 = offset_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = self.encoder_lv3(images_lv3_1 )
        feature_lv3_2 = self.encoder_lv3(images_lv3_2 )
        feature_lv3_3 = self.encoder_lv3(images_lv3_3 )
        feature_lv3_4 = self.encoder_lv3(images_lv3_4 )
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = self.decoder_lv3(feature_lv3_top, offset_lv2_1)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot, offset_lv2_2)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = self.decoder_lv2(feature_lv2, offset_lv1)

        feature_lv1 = self.encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
        residual_lv1 = self.decoder_lv1(feature_lv1, offset_lv1)
        deblur_image = images_lv1 + residual_lv1
        return deblur_image



class DMPHN_decoder_offset(nn.Module):
    def __init__(self, input_nc, nf, n_offset, offset_mode, norm_layer=nn.BatchNorm2d,gpu_ids=[]):
        super(DMPHN_decoder_offset,self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.n_offset = n_offset

        self.offset_net = OffsetNet_quad(input_nc,nf,n_offset,offset_mode=offset_mode, norm_layer=norm_layer,gpu_ids=gpu_ids)
        self.deblur_net = DMPHN_decoder_offset_001()  #decoder 001


    def forward(self, img_in):
        self.offsets = self.offset_net(img_in)
        self.deblur_img = self.deblur_net(img_in,self.offsets)
        return self.offsets, self.deblur_img


class ETR_motion(nn.Module):
    def __init__(self, pre_trained, n_GPUs, offset_network_path, offset_mode='none'):
        super(ETR_motion,self).__init__()
        self.offset_net = define_offset_quad(input_nc=3, nf=16, n_offset=15, norm='batch', offset_mode=offset_mode)
        self.n_GPUs = n_GPUs
        self.n_offset = 15
        if pre_trained:
            # if self.n_GPUs > 1:
            #     self.offset_net.module.load_state_dict(torch.load(offset_network_path))
            # else:
            self.offset_net.load_state_dict(torch.load(offset_network_path))
            print('Loading Offset pretrain model from {}'.format(offset_network_path))
    def forward(self, img_in):

        offset = self.offset_net(img_in)   ##b, c*(1+N-1+1+N-1+1),h,w

        return offset


class ETR_motion_V2(nn.Module):
    def __init__(self, pre_trained, n_GPUs, offset_network_path, offset_mode='none', vecdim=256):
        super(ETR_motion_V2,self).__init__()
        self.offset_net = define_offset_quad_v2(input_nc=3, nf=16, n_offset=15, norm='batch', offset_mode=offset_mode, vecdim=vecdim)
        self.n_GPUs = n_GPUs
        self.n_offset = 15
        if pre_trained:
            pretrained_model = torch.load(offset_network_path)
            pretrained_dict = pretrained_model
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            model_dict = self.offset_net.state_dict()
            for k, v in pretrained_dict.items():
                if k not in model_dict:
                    print("Not in model", k)
                if v.shape != model_dict[k].shape:
                    print("Mismatch shape:", k)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                                  k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.offset_net.load_state_dict(model_dict)
            print('Loading Offset pretrain model from {}'.format(offset_network_path))

    def forward(self, img_in, vec):

        offset = self.offset_net(img_in, vec)   ##b, c*(1+N-1+1+N-1+1),h,w

        return offset

