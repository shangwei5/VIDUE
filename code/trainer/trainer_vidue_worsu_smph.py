import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
# from loss.BDP import BalancedDataParallel
# from model.pytorch_pwc.extract_flow import *

class Trainer_UNet(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_UNet, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-VIDUE")
        self.m = args.m
        self.n = args.n
        self.blur_deg = args.blur_deg
        self.args = args
        # self.flow_model = PWCNet().to(torch.device('cuda'))
        # self.flow_epoch = args.flow_epochs
        # # self.midL2 = nn.MSELoss()
        # if not args.cpu and args.n_GPUs > 1:
        #     self.flow_model = nn.DataParallel(self.flow_model, range(args.n_GPUs), range(args.n_GPUs)[-1])
        # kwargs = {'lr': 1e-3, 'weight_decay': self.args.weight_decay}
        # self.optimizer2 = optim.Adam(self.model.get_model().refine.parameters(),**kwargs)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        # for k in self.model.get_model().state_dict():
        #     print(k)
        if self.args.opti == 'Adam':
            print("Using Adam...")
            return optim.Adam([{"params": self.model.get_model().UNet.parameters()},
                                 {"params": self.model.get_model().refine.parameters()}], **kwargs)
        else:
            print("Using Adamax...")
            return optim.Adamax([{"params": self.model.get_model().UNet.parameters()},
            {"params": self.model.get_model().refine.parameters()},
            {"params": self.model.get_model().motion.parameters()}],**kwargs)  #, "lr": 1e-6

    def charbonnier(self, x, alpha=0.25, epsilon=1.e-9):
        return torch.pow(torch.pow(x, 2) + epsilon ** 2, alpha)

    def smoothness_loss(self, flow):
        b, c, h, w = flow.size()
        v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
        h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
        s_loss = self.charbonnier(flow - v_translated) + self.charbonnier(flow - h_translated)
        s_loss = torch.sum(s_loss, dim=1) / 2

        return torch.sum(s_loss) / b

    def train(self):
        print("Now training")
        epoch = self.scheduler.last_epoch #+ 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']   #self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        #
        # def set_bn_eval(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         # print("Fixing BN in the model..")
        #         m.eval()
        # self.model.apply(set_bn_eval)
        self.ckp.start_log()
        mid_loss_sum = 0.
        mid_loss_sum_warp = 0.

        for batch, (input_list, gt_list, output_filenames, input_filenames, exp) in enumerate(self.loader_train):

            input = [inp.to(self.device) for inp in input_list]
            # print('input:',input_filenames)
            # print('output',output_filenames)
            # print(prior.size())
            # print(prior)
            gt_list = [gt.to(self.device) for gt in gt_list]
            # gt_front_e = torch.cat(gt_list[:self.m], dim=1)
            # gt_readout = torch.cat(gt_list[self.m:-self.m], dim=1)
            # gt_rear_e = torch.cat(gt_list[-self.m:], dim=1)
            # print(exp)
            # gt_warps = []
            # with torch.no_grad():
            #     for i in range(len(gt_list)):
            #         if i == self.m // 2:
            #             continue
            #         flow = extract_flow_torch(self.flow_model, gt_list[i], gt_list[self.m // 2])
            #         gt_warps.append(warp(gt_list[self.m // 2], flow, flow.get_device())[0])
            # gtwarp = torch.cat(gt_warps,dim=1)
            # b,t,c,h,w = gtflow.size()
            # gtflow = gtflow.view(b,-1,h,w)
            gt = torch.cat(gt_list, dim=1)
            # print(output.size(),gt.size())
            # if mid_loss:  # mid loss is the loss during the model
            # if epoch > self.flow_epoch:
            self.optimizer.zero_grad()
            out, flows = self.model(input)  #, epoch
            output = torch.cat(out, dim=1)
            loss = self.loss(output, gt)  # + loss_reblur + loss_pair/4
            b, c, h, w = flows.size()
            flows = flows.view(b, -1, 2, h, w)
            # warps = []
            # for i in range(flows.shape[1]):
            #     if i == self.m // 2:
            #         continue
            #     warps.append(warp(gt_list[self.m // 2], flows[:,i], flow.get_device())[0])
            # warps = torch.cat(warps,dim=1)
            # mid_loss_warp = self.loss(warps, gtwarp)
            flows = flows.view(-1, 2, h, w)
            mid_loss = self.smoothness_loss(flows)
            loss = loss + self.args.mid_loss_weight * mid_loss #+ mid_loss_warp
            mid_loss_sum = mid_loss_sum + mid_loss.item()
            # mid_loss_sum_warp = mid_loss_sum_warp + mid_loss_warp.item()
            loss.backward()
            self.optimizer.step()
            # else:
            #     self.optimizer2.zero_grad()
            #     flows = self.model(input, epoch)
            #     b, c, h, w = flows.size()
            #     flows = flows.view(b, -1, 2, h, w)
            #     warps = []
            #     for i in range(flows.shape[1]):
            #         warps.append(warp(gt_list[self.m // 2], flows[:, i], flow.get_device())[0])
            #     warps = torch.cat(warps, dim=1)
            #     loss = self.loss(warps, gtwarp)
            #     loss.backward()
            #     self.optimizer2.step()
            # print(self.model.get_model().refine.conv1.bias)
            # for k,v in self.model.named_parameters():
            #     print(k,v)
            #     break
            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}][mid_warp: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum / (batch + 1),
                    mid_loss_sum_warp / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))
        self.scheduler.step(self.ckp.loss_log[-1])
        self.loss.step()

    def test(self):
        epoch = self.scheduler.last_epoch #+ 1
        self.ckp.write_log('\nEvaluation:')
        # self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            total_num = 0.
            total_deblur_PSNR =0.
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input_list, gt_list, output_filenames, input_filenames, exp) in enumerate(tqdm_test):
                exp = exp//self.blur_deg
                input = [inp.to(self.device) for inp in input_list]
                # print('input:', input_filenames)
                # print('output', output_filenames)

                gt = [gti.to(self.device) for gti in gt_list]
                out, _ = self.model(input)
                # out = out[:-self.m]
                PSNR = 0.
                for i in range(len(out)):
                    # print(gt[i].size(),out[i].size())
                    PSNR_item = utils.calc_psnr(gt[i], out[i], rgb_range=self.args.rgb_range)
                    if i == exp//2:
                        deblur_PSNR = PSNR_item
                        # print(deblur_PSNR)
                    # print(PSNR_item)
                    PSNR += PSNR_item / len(out)
                    # print(PSNR)
                total_deblur_PSNR += deblur_PSNR
                total_num += 1
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    save_list = out
                    save_list.append(gt[exp//2])
                    save_list.append(input[self.args.n_sequence//2-1])
                    # print(len(out))
                    # print(len(gt[self.args.m//2]))
                    # print(len(save_list))
                    save_list = utils.postprocess(save_list,rgb_range=self.args.rgb_range,
                                                ycbcr_flag=False, device=self.device)
                    # print(len(save_list))
                    self.ckp.save_images(output_filenames, save_list, epoch, exp)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage Deblur_PSNR: {:.3f} Total_PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                total_deblur_PSNR / total_num,
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))  #
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))  #


