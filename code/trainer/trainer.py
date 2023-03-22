import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda'+':'+str(args.n_GPUs-1))
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step(ckp.loss_log[-1])

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.opti == 'Adam':
            return optim.Adam(self.model.parameters(), **kwargs)
        else:
            return optim.Adamax(self.model.parameters(), **kwargs)  #ax

    def make_scheduler(self):
        # kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        # return lr_scheduler.StepLR(self.optimizer, **kwargs)
        return lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.args.patience, verbose=True)

    def train(self):
        pass

    def test(self):
        pass

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
