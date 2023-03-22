import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
# from loss.BDP import BalancedDataParallel

class Trainer_Prior_predict(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Prior_predict, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer-Prior_predict")
        self.m = args.m
        self.n = args.n
        self.min = 0

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}

        return optim.Adam(self.model.get_model().parameters(),**kwargs)



    def train(self):
        print("Now training")
        epoch = self.scheduler.last_epoch #+ 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']   #self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.

        for batch, (input_list, label_list, input_filenames, exp) in enumerate(self.loader_train):
            images = torch.cat([input_list[0], input_list[1]], dim=0)
            # print(images.size(),label_list.size())
            input = images.to(self.device)  #[inp.to(self.device) for inp in images]
            # print('input:',input_filenames)
            # print('output',output_filenames)
            # print(exp)
            # print(label_list)
            # print(input.size())

            _, out = self.model(input)

            bsz = label_list.shape[0]
            f1, f2 = torch.split(out, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            self.optimizer.zero_grad()
            loss = self.loss(features, label_list) #+ loss_reblur + loss_pair/4
            # if mid_loss:  # mid loss is the loss during the model
            # loss = loss + self.args.mid_loss_weight * mid_loss
            # mid_loss_sum = mid_loss_sum + mid_loss.item()

            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))
        self.min = self.ckp.loss_log[-1]
        self.scheduler.step(self.ckp.loss_log[-1])
        self.loss.step()

    def test(self):
        epoch = self.scheduler.last_epoch #+ 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            total_num = 0.
            total_deblur_PSNR =0.
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input_list, label_list, input_filenames, exp) in enumerate(tqdm_test):

                images = torch.cat([input_list[0], input_list[1]], dim=0)
                # print(images.size(),label_list.size())
                input = images.to(self.device)  # [inp.to(self.device) for inp in images]
                # print('input:',input_filenames)
                # print('output',output_filenames)
                # print(exp)
                # print(label_list)
                # print(input.size())

                _, out = self.model(input)

                bsz = label_list.shape[0]
                f1, f2 = torch.split(out, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                loss = self.loss(features, label_list)
                # print(loss)
                # input = [inp.to(self.device) for inp in input_list]
                # # print('input:', input_filenames)
                # # print('output', output_filenames)
                #
                # gt = [gti.to(self.device) for gti in gt_list]
                # out = self.model(input)
                # # out = out[:-self.m]
                # PSNR = 0.
                # for i in range(len(out)):
                #     # print(gt[i].size(),out[i].size())
                #     PSNR_item = utils.calc_psnr(gt[i], out[i], rgb_range=self.args.rgb_range)
                #     if i == exp//2:
                #         deblur_PSNR = PSNR_item
                #         # print(deblur_PSNR)
                #     # print(PSNR_item)
                #     PSNR += PSNR_item / len(out)
                #     # print(PSNR)
                # total_deblur_PSNR += deblur_PSNR
                # total_num += 1
                self.ckp.report_log(self.min, train=False)
            #
            #     if self.args.save_images:
            #         save_list = out
            #         save_list.append(gt[exp//2])
            #         save_list.append(input[self.args.n_sequence//2-1])
            #         # print(len(out))
            #         # print(len(gt[self.args.m//2]))
            #         # print(len(save_list))
            #         save_list = utils.postprocess(save_list,rgb_range=self.args.rgb_range,
            #                                     ycbcr_flag=False, device=self.device)
            #         # print(len(save_list))
            #         self.ckp.save_images(output_filenames, save_list, epoch, exp)
            #
            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.min(0)
            self.ckp.write_log('[{}]\taverage SupCon: {:.5f} (Best: {:.5f} @epoch {})'.format(
                self.args.data_test,
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))  #
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))  #


