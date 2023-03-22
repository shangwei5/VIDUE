import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.blocks as blocks
from model.blocks import Sty_layer
from model.time_prior_predict_weighted import TPP
from model.ETR import ETR_motion_V2
from model.refine.UNet2 import UNet2_V2 as Refine_V
# from model.pytorch_pwc.extract_flow import warp

def make_model(args):
    return ResNet(n_inputs=args.n_sequence, n_outputs_e=args.m, n_outputs_r=args.n, blocks=args.n_resblock, feats=args.n_feat, offset_network_path=args.offset_network_path, extractor_path=args.extractor_path, halve=args.halve)


class SynBlock(nn.Module):
    def __init__(self, nf, ks, exp=5, ro=3):
        super(SynBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf//4, 1,1,0)
        self.conv2 = nn.Conv2d(1 + nf // 4, exp + ro, 3, 1, 1)
        self.se1 = Sty_layer(256, (exp + ro)*2, 4)

        ref1 = [nn.Conv2d(2, 2, kernel_size=ks, stride=1, padding=ks // 2)
                for _ in range(exp + ro)]
        self.ref1 = nn.Sequential(*ref1)
        syn1 = [nn.Conv2d(nf//4, nf//4, kernel_size=ks, stride=1, padding=ks // 2)
                for _ in range(exp+ro)]
        self.syn1 = nn.Sequential(*syn1)
        syn2 = [blocks.ResBlock(nf//4, nf//4, kernel_size=ks, stride=1)
                          for _ in range(exp+ro)]
        self.syn2 = nn.Sequential(*syn2)
        syn3 = [blocks.ResBlock(nf // 4, nf // 4, kernel_size=ks, stride=1)
                for _ in range(exp + ro)]
        self.syn3 = nn.Sequential(*syn3)
        out = [nn.Conv2d(nf//4, 3, kernel_size=ks, stride=1, padding=ks // 2)
                for _ in range(exp+ro)]
        self.out = nn.Sequential(*out)

        # conv_ab = [nn.Conv2d(256, nf // 4, kernel_size=ks, stride=1, padding=ks // 2)
        #         for _ in range(exp + ro)]
        # self.conv_ab = nn.Sequential(*conv_ab)
        # conv_a = [nn.Conv2d(nf // 4, nf // 4, kernel_size=ks, stride=1, padding=ks // 2)
        #            for _ in range(exp + ro)]
        # self.conv_a = nn.Sequential(*conv_a)
        # conv_b = [nn.Conv2d(nf // 4, nf // 4, kernel_size=ks, stride=1, padding=ks // 2)
        #            for _ in range(exp + ro)]
        # self.conv_b = nn.Sequential(*conv_b)
        self.xN = exp+ro
        # self.lrelu = nn.LeakyReLU(0.2)
        # self.exp = exp

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(torch.device('cuda'))
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output #, mask

    def forward(self, fea, flows, mmap, vec, out_size):  #
        H, W = out_size
        B, C, _, _ = flows.size()
        in_fea = self.conv1(fea)
        results = []
        flows = F.interpolate(flows, size=out_size, mode='bilinear')
        ref_flows = self.se1(flows, vec)
        flows = flows.view(B, C // 2, 2, H, W)
        ref_flows = ref_flows.view(B, C // 2, 2, H, W)
        mmap = F.interpolate(mmap, size=out_size, mode='bilinear')
        mmap = self.conv2(torch.cat([mmap, in_fea], dim=1)) + mmap
        # map_m = mmap * vec.unsqueeze(-1).unsqueeze(-1)
        # b,c,h,w = mmap.size()
        # mmap = F.softmax(mmap.view(b, 1, -1), dim=-1).view(b, 1, h, w)
        mmap = F.sigmoid(mmap)

        f_flows = [ref_flows[:,j]*mmap[:,j:j+1]+flows[:,j]*(1-mmap[:,j:j+1]) for j in range(self.xN)]

        for i in range(self.xN):
            f_flows[i] = self.ref1(f_flows[i]) + f_flows[i]
            fea2 = in_fea*(1-mmap[:,i:i+1])+self.warp(in_fea, f_flows[i])*mmap[:,i:i+1]
            # alpha_belta = self.conv_ab[i](map_m)
            # alpha = self.conv_a[i](alpha_belta)
            # belta = self.conv_b[i](alpha_belta)
            # mu = fea2.mean(2, True).mean(3, True)
            # siga = torch.sqrt((fea2 - mu).pow(2) + 1e-12).mean(2, True).mean(3, True)
            # fea2 = (fea2-mu).pow(2) / siga
            # fea2 = belta + alpha*fea2
            fea3 = self.syn1[i](fea2) + fea2
            fea4 = self.syn2[i](fea3) + fea3
            fea5 = self.syn3[i](fea4) + fea4
            out = self.out[i](fea5)
            results.append(out)

        return results

class UNet(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=[3,3,9,3], n_feat=32, kernel_size=3,n_outputs_e=5, n_outputs_r=3):
        super(UNet, self).__init__()

        InBlock = []

        InBlock.extend([nn.Sequential(
            nn.Conv2d(in_channels * n_sequence, n_feat, kernel_size=7, stride=1,
                      padding=7 // 2),
            nn.LeakyReLU(0.1,inplace=True)
        )])

        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, se=False)
                        for _ in range(n_resblock[0])])

        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.LeakyReLU(inplace=True)
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, se=False)
                              for _ in range(n_resblock[1])])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.LeakyReLU(inplace=True)
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, se=False)
                               for _ in range(n_resblock[2])])

        # encoder3
        Encoder_third = [nn.Sequential(
            nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.LeakyReLU(inplace=True)
        )]
        Encoder_third.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, se=False)
                               for _ in range(n_resblock[3])])

        # decoder3
        Decoder_third = [nn.Sequential(
            nn.ConvTranspose2d(n_feat *4, n_feat * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True)
        )]
        Decoder_third.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, se=False)
                          for _ in range(n_resblock[3])])

        # decoder2
        Decoder_second = [nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True)
        )]
        Decoder_second.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, se=False)
                          for _ in range(n_resblock[2])])
        # decoder1
        Decoder_first = [nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True)
        )]
        Decoder_first.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, se=False)
                         for _ in range(n_resblock[1])])

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock[0])]
        OutBlock.append(nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))

        self.predict_ll = SynBlock(n_feat * 4, ks=3, exp=n_outputs_e, ro=n_outputs_r)
        self.predict_l = SynBlock(n_feat * 2, ks=3, exp=n_outputs_e, ro=n_outputs_r)
        self.predict = SynBlock(n_feat, ks=3, exp=n_outputs_e, ro=n_outputs_r)
        # self.seq = out_channels//3
        # self.split_size = n_feat // self.seq

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.encoder_third = nn.Sequential(*Encoder_third)
        self.decoder_third = nn.Sequential(*Decoder_third)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)

        self.se3 = Sty_layer(n_feat * 4, n_feat * 4, 4)
        self.se2 = Sty_layer(n_feat * 4, n_feat * 2, 4)
        self.se1 = Sty_layer(n_feat * 4, n_feat, 4)
        self.conv1 = nn.Conv2d(n_sequence, 1, 1,1,0)


    def forward(self, x, flows, vec, mm, mean_):
        # inblock_fea,encoder_first_fea,encoder_second_fea,encoder_third_fea = prior
        mm = mm.squeeze(2)
        b,c,h,w = mm.size()
        mm1 = mm.mean(dim=1, keepdim=True)
        mmap = self.conv1(mm) + mm1

        first_scale_inblock = self.inBlock(x)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock) #torch.cat([first_scale_inblock,inblock_fea],dim=1)
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)  #torch.cat([first_scale_encoder_first,encoder_first_fea],dim=1)
        first_scale_encoder_third = self.encoder_third(first_scale_encoder_second)   #torch.cat([first_scale_encoder_second,encoder_second_fea],dim=1)

        first_scale_decoder_third = self.decoder_third(first_scale_encoder_third)  #torch.cat([first_scale_encoder_third,encoder_third_fea],dim=1)
        first_scale_decoder_third = self.se3(first_scale_decoder_third, vec) + first_scale_decoder_third
        out_ll = self.predict_ll(first_scale_decoder_third, flows, mmap, vec, first_scale_decoder_third.size()[-2:])

        first_scale_decoder_second = self.decoder_second(first_scale_decoder_third + first_scale_encoder_second)
        first_scale_decoder_second = self.se2(first_scale_decoder_second, vec) + first_scale_decoder_second
        out_l = self.predict_l(first_scale_decoder_second, flows, mmap, vec, first_scale_decoder_second.size()[-2:])
        out_l = [F.interpolate(out_ll[i], size=out_l[0].size()[-2:], mode='bilinear') + out_l[i] for i in range(len(out_l))]

        first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
        first_scale_decoder_first = self.se1(first_scale_decoder_first, vec) + first_scale_decoder_first
        out_ = self.predict(first_scale_decoder_first, flows, mmap, vec, first_scale_decoder_first.size()[-2:])
        out_ = [F.interpolate(out_l[i], size=out_[0].size()[-2:], mode='bilinear') + out_[i] for i in
                 range(len(out_))]

        first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock)
        out = torch.split(first_scale_outBlock, dim=1, split_size_or_sections=3)
        out = list(out)
        fout = [out[i] + out_[i] for i in range(len(out))]
        mean_ = mean_.squeeze(2)
        fout = [o + mean_ for o in fout]

        return fout


class ResNet(nn.Module):
    def __init__(self, n_inputs, n_outputs_e, n_outputs_r, blocks=[3,3,9,3], feats=32, loading=True, offset_network_path=None, extractor_path=None, halve=False):
        super().__init__()
        self.UNet = UNet(in_channels=3, n_sequence=n_inputs, out_channels=3*(n_outputs_e+n_outputs_r), n_resblock=blocks, n_feat=feats,
                 kernel_size=3, n_outputs_e=n_outputs_e, n_outputs_r=n_outputs_r)
        if n_inputs % 2 !=0:
            tpp_inputs = n_inputs - 1
        else:
            tpp_inputs = n_inputs
        self.extractor = TPP(tpp_inputs, [2, 2, 2, 2], 64)

        self.refine = Refine_V(n_inputs*4, (n_outputs_e+n_outputs_r)*2)
        # self.flow_epoch = flow_epoch
        self.motion = ETR_motion_V2(offset_network_path is not None, 1, offset_network_path, offset_mode='se', vecdim=256)
        self.n_sequence = n_inputs
        if loading:
            print("Loading from ", extractor_path)
            self.extractor.load_state_dict(torch.load(extractor_path))

            for param in self.extractor.parameters():
                param.requires_grad = False
            # for param in self.motion.parameters():
            #     param.requires_grad = False


    def forward(self, images, epoch=None):
        images1 = torch.stack(images, dim=2)
        mean_ = images1.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        norm_x = images1 - mean_

        inputs = torch.cat(images,dim=1)
        if self.n_sequence % 2 !=0:
            tpp_inputs = torch.cat(images[1:],dim=1)
        else:
            tpp_inputs = inputs
        probas, vec1 = self.extractor(tpp_inputs, True)

        x = inputs
        b, c, h, w = x.size()
        norm_x = norm_x.view(b, c, h, w)
        imgs = x.view(b, self.n_sequence, 3, h, w)
        imgs_m = imgs.reshape(-1, 3, h, w)
        offsets = self.motion(imgs_m, vec1)

        delta = torch.abs(offsets[:, 2:, :, :] - offsets[:, :2, :, :])
        motion_map = torch.sqrt(delta[:, 0:1, :, :].pow(2) + delta[:, 1:2, :, :].pow(2))
        motion_map = motion_map.view(b, self.n_sequence, 1, h, w)
        # mm2 = mm1.view(b, self.n_sequence, h, w).mean(dim=1, keepdim=True)
        # mm2 = F.softmax(mm2.view(b, 1, -1), dim=-1).view(b, 1, h, w)
        # print(offsets.size())
        offsets = offsets.view(b, self.n_sequence, 4, h, w)
        offsets = offsets.view(b, self.n_sequence * 4, h, w)
        flows = self.refine(offsets, vec1)
        result = self.UNet(norm_x, flows, vec1, motion_map, mean_)

        return result, flows
