import os
from importlib import import_module

import torch
import torch.nn as nn
from loss.BDP import BalancedDataParallel

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if not args.cpu and args.n_GPUs > 1:
            # self.model = BalancedDataParallel(args.first_cuda_batch_size, self.model, dim=0).to(self.device)
            # gpu0_bsz：第一个GPU的batch_size;
            # model：模型；
            # dim：batch所在维度
            self.model = nn.DataParallel(self.model, range(args.n_GPUs), range(args.n_GPUs)[-1])

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu,
            args=args
        )
        print(self.get_model(), file=ckp.log_file)

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        if not self.cpu and self.n_GPUs > 1:
            return self.model.module
        else:
            return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        if self.save_middle_models:
            if epoch % 1 == 0:
                torch.save(
                    target.state_dict(),
                    os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
                )

    def load(self, apath, pre_train='.', resume=False, cpu=False, args=None):  #
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            if args.model == "TIME_PRIOR_PREDICT_WEIGHTED":
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs), strict=False  #
                )
            elif args.model == '.':  #"VIDUE_WORSU"
                print('Loading model from {}'.format(pre_train))
                print("Excluding mismatching params....")
                pretrained_model = torch.load(pre_train, **kwargs)
                pretrained_dict = pretrained_model
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in pretrained_dict.items():
                    name = 'module.' + k[:]  # remove `module.`
                    new_state_dict[name] = v

                model_dict = self.model.state_dict()
                # 将pretrained_dict里不属于model_dict的键剔除掉
                pretrained_dict = {k: v for k, v in new_state_dict.items() if
                                   k in model_dict and v.shape == model_dict[k].shape}
                # 更新现有的model_dict
                model_dict.update(pretrained_dict)
                new_state_dict = OrderedDict()
                for k, v in model_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # 加载我们真正需要的state_dict
                self.get_model().load_state_dict(new_state_dict)
            else:
                print('Loading model from {}'.format(pre_train))
                print("Excluding time prior predictor....")
                # module = import_module('model.' + 'unet')
                # pretrained_model = module.make_model(args).to(self.device)
                pretrained_model = torch.load(pre_train, **kwargs)
                pretrained_dict = pretrained_model
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in pretrained_dict.items():
                    name = 'module.' + k[:]  # remove `module.`
                    new_state_dict[name] = v

                model_dict = self.model.state_dict()
                # for k, v in model_dict.items():
                #     print(k)
                #     break
                # for k, v in new_state_dict.items():
                #     print(k)
                #     break
                # for k, v in new_state_dict.items():
                #     if k in model_dict and v.shape != model_dict[k].shape:
                #         print("Shape of weights mismatch:")
                #         print(k)
                #     if k not in model_dict:
                #         print("Unexpected keys: ")
                #         print(k)

                # 将pretrained_dict里不属于model_dict的键剔除掉
                # pretrained_dict = {k: v for k, v in new_state_dict.items() if
                #                    k in model_dict and v.shape == model_dict[k].shape}
                for name, param in new_state_dict.items():
                    # if name not in model_dict:
                    #     print('Not in model:', name)
                    #     continue
                    if 'extractor' not in name and model_dict[name].shape == param.shape:   #and 'mlp' not in name
                        model_dict[name].copy_(param)
                    else:
                        print('Not in name or mismatching', name)

                # 更新现有的model_dict
                # model_dict.update(pretrained_dict)
                new_state_dict = OrderedDict()
                for k, v in model_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # 加载我们真正需要的state_dict
                self.get_model().load_state_dict(new_state_dict)

        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_best.pt'), **kwargs),
                strict=False
            )
        else:
            pass
