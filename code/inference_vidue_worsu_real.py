import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from model.vidue_worsu import ResNet
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.nn as nn

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.n_seq = args.n_sequence
        self.size_must_mode = 8
        self.device = 'cuda'
        self.GPUs = args.n_GPUs
        self.n_outputs = args.n_outputs
        self.m = args.m
        self.n = args.n

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.input_path = self.data_path

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = ResNet(self.n_seq, self.m, self.n, blocks=[3,6,6,6], feats=64, loading=False)  #, halve=True
        self.net.load_state_dict(torch.load(self.model_path))  # , strict=False
        self.net = self.net.to(self.device)
        if args.n_GPUs > 1:
            self.net = nn.DataParallel(self.net, range(args.n_GPUs))
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        # self.net.eval()
        # def set_bn_eval(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         # print("Fixing BN in the model..")
        #         m.eval()
        # self.net.apply(set_bn_eval)

    def infer(self):
        # self.net.eval()
        with torch.no_grad():
            ### for BSD data
            # kinds = sorted(os.listdir(self.input_path))[2:]
            # assert len(kinds) == 3
            # for k in kinds:
            ###
            videos = sorted(os.listdir(os.path.join(self.input_path))) #, k, 'blur'
            # result_path1 = os.path.join(self.result_path, k)
            # if not os.path.exists(result_path1):
            #     os.makedirs(result_path1)
            for v in videos:
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*")))  #for realblur in CDVD-TSP
                # input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "input", "*"))) #for all realblur
                # input_frames = sorted(glob.glob(os.path.join(self.input_path, k, 'blur', v, "*")))  #for all BSD
                input_seqs, _ = self.gene_seq(input_frames, n_seq=self.n_seq)
                for i in range(len(input_seqs)):   #, bm_seq, label_seq   , bm_seqs, label_seqs
                    start_time = time.time()
                    filename = os.path.basename(input_seqs[i][self.n_seq // 2-1]).split('.')[0]

                    # print(in_seq)
                    input = [imageio.imread(p) for p in input_seqs[i]]
                    inputs = input
                    # inputs = [cv2.resize(p, None, fx=0.5, fy=0.5) for p in input]

                    h, w, c = inputs[self.n_seq // 2].shape
                    # print(h, w, c)
                    in_tensor = self.numpy2tensor(inputs, self.device)
                    if h % self.size_must_mode !=0 or w % self.size_must_mode != 0:
                        in_tensor=[F.pad(in_ten,pad=[0,self.size_must_mode-w % self.size_must_mode,0,self.size_must_mode-h % self.size_must_mode,0,0],mode='replicate') for in_ten in in_tensor]
                    torch.cuda.synchronize()
                    preprocess_time = time.time()
                    # print(in_tensor.size(), bm_tensor.size(), label_tensor.size())

                    # if self.GPUs ==1:
                    output, _ = self.net(in_tensor)  #, bm_tensor, label_tensor
                    torch.cuda.synchronize()
                    forward_time = time.time()

                    for j in range(self.n_outputs):
                        output_img = self.tensor2numpy(output[j])
                        save_name = '%06d_%02d.png' % (int(filename),j)
                        print(filename, save_name)

                        if self.save_image:
                            if not os.path.exists(os.path.join(self.result_path, v)):
                                os.mkdir(os.path.join(self.result_path, v))  #self.result_path, k
                            imageio.imwrite(os.path.join(self.result_path, v, os.path.basename(save_name)), output_img)
                        postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}-{} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, os.path.basename(save_name).split('.')[0],
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))


    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[1:1+half]
            img_list_temp.reverse()
            img_list_temp.extend(img_list)
            end_list = img_list[-half - 1:-1]
            end_list.reverse()
            img_list_temp.extend(end_list)
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list, img_list

    def numpy2tensor(self, input_seq, device='cuda', rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor.unsqueeze(0).to(device))
        # stacked = torch.stack(tensor_list).unsqueeze(0)
        return tensor_list

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLAVR-Inference')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', default=False, help='restore border images of video if true')

    parser.add_argument('--default_data', type=str, default='GOPRO',
                        help='quick test, optional: REDS, GOPRO')
    parser.add_argument('--data_path', type=str, default='../dataset/DVD/test',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
                        help='the path of pretrain model')
    parser.add_argument('--result_path', type=str, default='../infer_results',
                        help='the path of deblur result')
    parser.add_argument('--m', type=int, default=7,
                        help='Number of exposure frames')
    parser.add_argument('--n', type=int, default=1,
                        help='Number of readout frames')
    # Model specifications
    model_choices = ["unet_18", "unet_34"]
    parser.add_argument('--submodel', choices=model_choices, type=str, default="unet_18")
    parser.add_argument('--joinType', choices=["concat", "add", "none"], default="concat")
    parser.add_argument('--upmode', choices=["transpose", "upsample"], type=str, default="transpose")
    parser.add_argument('--n_outputs', type=int, default=8,
                        help="For Kx FLAVR, use n_outputs k-1")
    args = parser.parse_args()

    if args.default_data == 'GOPRO':
        args.data_path = '/home/sw/sw/dataset/dataset/RealBlur/test/blur'
        args.model_path = '../experiment/VIDUE_GoPro8x/model_best.pt'
        args.result_path = '../infer_results_real/VIDUE_GoPro8x_RealBlur'
        args.n_sequence = 4
        args.n_GPUs = 1

    Infer = Inference(args)
    Infer.infer()

