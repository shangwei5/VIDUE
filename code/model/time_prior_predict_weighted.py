import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import blocks
from model.ETR import ETR_motion

def make_model(args):
    return TPP(n_inputs=args.n_sequence, blocks=args.n_resblock, feats=args.n_feat, xN=args.n_outputs, offset_network_path=args.offset_network_path, halve=args.halve)

class Exractor(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, n_resblock=[3,3,9,3], n_feat=32, kernel_size=3, xN=8):
        super(Exractor, self).__init__()
        InBlock = []

        InBlock.extend([nn.Sequential(
            nn.Conv2d(in_channels * n_sequence, n_feat, kernel_size=7, stride=1,
                      padding=7 // 2),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(0.1,inplace=True)
        )])

        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1, bn=True, se=False)
                        for _ in range(n_resblock[0])])

        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm2d(n_feat*2),
            nn.LeakyReLU(inplace=True)
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1, bn=True, se=False)
                              for _ in range(n_resblock[1])])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm2d(n_feat*4),
            nn.LeakyReLU(inplace=True)
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, bn=True, se=False)
                               for _ in range(n_resblock[2])])

        # encoder3
        Encoder_third = [nn.Sequential(
            nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm2d(n_feat * 4),
            nn.LeakyReLU(inplace=True)
        )]
        Encoder_third.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1, bn=True, se=False)
                               for _ in range(n_resblock[3])])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.encoder_third = nn.Sequential(*Encoder_third)
        self.mlp = nn.Sequential(
            nn.Linear(n_feat * 4, n_feat * 4 * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_feat * 4 * 4, n_feat * 4)
        )
        self.conv1 = nn.Conv2d(n_sequence, 1, 1, 1, 0)
        self.num_classes = xN
        self.n_sequence = n_sequence

    def forward(self, x, offsets):
        # x = torch.cat(x, dim=1)
        b, t, c, h, w = offsets.size()
        mmap = offsets.view(b, self.n_sequence, h, w).mean(dim=1, keepdim=True)
        mmap = self.conv1(offsets.view(b, self.n_sequence, h, w)) + mmap
        mmap = torch.sigmoid(mmap)

        first_scale_inblock = self.inBlock(x * mmap)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock * mmap)
        mmap_lv1 = F.interpolate(mmap, size=first_scale_encoder_first.size()[-2:], mode='bilinear')
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first * mmap_lv1)
        mmap_lv2 = F.interpolate(mmap_lv1, size=first_scale_encoder_second.size()[-2:], mode='bilinear')
        first_scale_encoder_third = self.encoder_third(first_scale_encoder_second * mmap_lv2)
        # offsets_lv3 = F.interpolate(offsets_lv2, size=first_scale_encoder_third.size()[-2:], mode='bilinear')
        vec = self.avgpool(first_scale_encoder_third).squeeze(-1).squeeze(-1)
        vec = self.mlp(vec)
        # logits = logits.view(-1, (self.num_classes - 1), 2)
        # probas = torch.nn.functional.softmax(logits, dim=2)[:, :, 1]

        return vec  #, logits, probas


class TPP(nn.Module):
    def __init__(self, n_inputs, blocks=[3,3,9,3], feats=32, xN=8, halve=False, offset_network_path=None, fix=True):
        super().__init__()
        if halve:
            xN = xN // 2
            print("Halving xN!")
        self.extractor = Exractor(in_channels=3, n_sequence=n_inputs, n_resblock=blocks, n_feat=feats,
                 kernel_size=3, xN=xN)
        #state_dict = torch.load('/mnt/disk10T/shangwei/code/B2SNet/experiment/TIME_PRIOR_PREDICT_NIU_finetune2/model/model_best.pt')
        #from collections import OrderedDict
        #new_state_dict = OrderedDict()
       # for k, v in state_dict.items():
          #  name = k[10:]  # remove `extractor.`
          #  new_state_dict[name] = v
       # self.extractor.load_state_dict(new_state_dict, strict=False)
        #print("Loading extractor successfully !")
        self.n_sequence = n_inputs
        self.motion = ETR_motion(offset_network_path is not None, 1, offset_network_path)
        if fix:
            for param in self.motion.parameters():
                param.requires_grad = False


    def forward(self, images, Visual=False):
        x = images
        b, c, h, w = x.size()
        imgs = x.view(b, self.n_sequence, 3, h, w)
        imgs_m = imgs.reshape(-1, 3, h, w)
        offsets = self.motion(imgs_m)
        # print(offsets.size())
        offsets = offsets.view(b, self.n_sequence, 1, h, w)

        # compute query features
        vec = self.extractor(images, offsets.detach())  # , logits, probas

        embedding = []
        if Visual:
            return embedding, vec
        vec = nn.functional.normalize(vec, dim=1)
        return embedding, vec


