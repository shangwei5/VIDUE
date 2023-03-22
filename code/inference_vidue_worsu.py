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
        self.args = args
        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.gt_path = args.gt_path
        self.result_path = args.result_path
        self.n_seq = args.n_sequence
        self.size_must_mode = 8
        self.device = 'cuda'
        self.GPUs = args.n_GPUs
        self.n_outputs = args.n_outputs
        self.m = args.m
        self.n = args.n
        self.blur_deg = args.blur_deg

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.input_path = self.data_path
        self.GT_path = self.gt_path

        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time))

        self.logger.write_log('Inference - {}'.format(now_time))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('gt_path: {}'.format(self.gt_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = ResNet(self.n_seq, self.m, self.n, blocks=[3,6,6,6], feats=64, loading=False)
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
            total_psnr = {}
            total_ssim = {}
            total_t = {}
            total_deblur_psnr = 0.
            total_deblur_ssim = 0.
            total_blur_num = 0
            total_exposure_psnr = 0.
            total_exposure_ssim = 0.
            total_exposure_num = 0
            total_readout_psnr = 0.
            total_readout_ssim = 0.
            total_readout_num = 0
            videos = sorted(os.listdir(self.input_path))

            for v in videos:
                video_psnr = []
                video_ssim = []
                total_time = 0
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*")))
                gt_frames_path = os.path.join(self.GT_path, v)
                input_seqs, _ = self.gene_seq(input_frames, n_seq=self.n_seq)
                total_blur_num += len(input_seqs)
                total_exposure_num += len(input_seqs)*self.m
                total_readout_num += len(input_seqs) * self.n
                for i in range(len(input_seqs)):   #, bm_seq, label_seq   , bm_seqs, label_seqs
                    start_time = time.time()
                    filename = os.path.basename(input_seqs[i][self.n_seq // 2-1]).split('.')[0]

                    # print(in_seq)
                    inputs = [imageio.imread(p) for p in input_seqs[i]]

                    h, w, c = inputs[self.n_seq // 2].shape

                    in_tensor = self.numpy2tensor(inputs, self.device)
                    if h % self.size_must_mode !=0 or w % self.size_must_mode != 0:
                        in_tensor=[F.pad(in_ten,pad=[0,self.size_must_mode-w % self.size_must_mode,0,self.size_must_mode-h % self.size_must_mode,0,0],mode='replicate') for in_ten in in_tensor]
                    torch.cuda.synchronize()
                    preprocess_time = time.time()
                    # print(in_tensor.size(), bm_tensor.size(), label_tensor.size())

                    # if self.GPUs == 1:
                    output, _ = self.net(in_tensor)  #, bm_tensor, label_tensor
                    # else:
                    #     output = self.forward_chop(in_tensor)
                    torch.cuda.synchronize()
                    forward_time = time.time()

                    for j in range(self.n_outputs):
                        output_img = self.tensor2numpy(output[j])
                        if self.args.default_data == 'Adobe':
                            save_name = os.path.join(gt_frames_path, '%04d.png' % (
                                        int(filename) - (self.m // 2) * self.blur_deg + j * self.blur_deg))
                        else:
                            save_name = os.path.join(gt_frames_path, '%06d.png' % (int(filename)-(self.m//2)+j))
                        gt = imageio.imread(save_name)
                        gt = cv2.resize(gt, None, fx=0.5, fy=0.5)
                        print(filename, save_name)
                        psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                        if j == self.m//2:
                            total_deblur_psnr += psnr
                            total_deblur_ssim += ssim
                        if j < self.m:
                            total_exposure_psnr += psnr
                            total_exposure_ssim += ssim
                        else:
                            total_readout_psnr += psnr
                            total_readout_ssim += ssim
                        video_psnr.append(psnr)
                        video_ssim.append(ssim)

                        if self.save_image:
                            if not os.path.exists(os.path.join(self.result_path, v)):
                                os.mkdir(os.path.join(self.result_path, v))
                            imageio.imwrite(os.path.join(self.result_path, v, os.path.basename(save_name)), output_img)
                        postprocess_time = time.time()

                        self.logger.write_log(
                            '> {}-{} PSNR={:.5}, SSIM={:.4} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                                .format(v, os.path.basename(save_name).split('.')[0], psnr, ssim,
                                        preprocess_time - start_time,
                                        forward_time - preprocess_time,
                                        postprocess_time - forward_time,
                                        postprocess_time - start_time))
                    if i != 0:
                        total_time += (forward_time - preprocess_time)

                    total_psnr[v] = video_psnr
                    total_ssim[v] = video_ssim
                    total_t[v] = total_time
                self.logger.write_log('> {} model_inference_time:{:.5}s'.format(v, total_time / (i - 1)))
            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            for k in total_psnr.keys():
                self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                n_img += len(total_psnr[k])
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))
            self.logger.write_log("# Total AVG-DEBLUR-PSNR={:.5}, AVG-DEBLUR-SSIM={:.4}".format(total_deblur_psnr / total_blur_num, total_deblur_ssim / total_blur_num))
            self.logger.write_log(
                "# Total AVG-exposure-PSNR={:.5}, AVG-exposure-SSIM={:.4}".format(total_exposure_psnr / total_exposure_num,
                                                                              total_exposure_ssim / total_exposure_num))
            self.logger.write_log(
                "# Total AVG-readout-PSNR={:.5}, AVG-readout-SSIM={:.4}".format(total_readout_psnr / total_readout_num,
                                                                              total_readout_ssim / total_readout_num))
            self.logger.write_log(
                "# Total AVG-Inference_time={:.5}s".format(sum(total_t) / len(total_t)))
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

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')

    def forward_chop(self, *args, shave_h=20, shave_w=20, min_size=160000):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = 1  #self.opt['scale']
        n_GPUs = min(torch.cuda.device_count(), 4)
        # print(n_GPUs)
        args = [a.squeeze().unsqueeze(0) for a in args]

        # height, width
        h, w = args[0].size()[-2:]
        # print('len(args)', len(args))
        # print('args[0].size()', args[0].size())

        top = slice(0, h//2 + shave_h)
        bottom = slice(h - h//2 - shave_w, h)
        left = slice(0, w//2 + shave_h)
        right = slice(w - w//2 - shave_w, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]
        # print('len(x_chops)', len(x_chops))
        # print('x_chops[0].size()', x_chops[0].size())

        y_chops = []
        if h * w < 6 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                # print(len(x))
                # print(x[0].size())
                y =  P.data_parallel(self.net.module, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:

            # print(x_chops[0].size())
            for p in zip(*x_chops):
                # print('len(p)', len(p))
                # print('p[0].size()', p[0].size())
                y = self.forward_chop(*p, shave_h=shave_h, shave_w=shave_w, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1:
            y = y[0]

        return y



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLAVR-Inference')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', default=False, help='restore border images of video if true')

    parser.add_argument('--default_data', type=str, default='GOPRO',
                        help='quick test, optional: Adobe, GOPRO')
    parser.add_argument('--data_path', type=str, default='../dataset/DVD/test',
                        help='the path of test data')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/CDVD_TSP_DVD_Convergent.pt',
                        help='the path of pretrain model')
    parser.add_argument('--result_path', type=str, default='../infer_results',
                        help='the path of deblur result')
    parser.add_argument('--m', type=int, default=5,
                        help='Number of exposure frames')
    parser.add_argument('--n', type=int, default=3,
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
        args.data_path = '/home/sw/sw/dataset/Event-data/GOPRO_large_all/LFR_Gopro_53'
        args.gt_path = '/home/sw/sw/dataset/Event-data/GOPRO_large_all/test/images'
        args.model_path = '../experiment/VIDUE_GoPro8x/model_best.pt'
        args.result_path = '../infer_results/VIDUE_GoPro8x_53'
        args.n_sequence = 4
        args.n_GPUs = 1
        args.blur_deg = 1
    elif args.default_data == 'Adobe':
        args.data_path = '/home/sw/sw/dataset/dataset/adobe240/LFR_Adobe_53'
        args.gt_path = '/home/sw/sw/dataset/dataset/adobe240/test'
        args.model_path = '../experiment/VIDUE_Adobe8x/model_best.pt'
        args.result_path = '../infer_results/VIDUE_Adobe8x_53'
        args.n_sequence = 4
        args.n_GPUs = 1
        args.blur_deg = 3

    Infer = Inference(args)
    Infer.infer()

