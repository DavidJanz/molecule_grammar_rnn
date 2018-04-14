import numpy as np
import torch
from torch.autograd import Variable

from base_classes.trainer import TrainerBase, TrainerArgParser
from grammar_model.alphabets import MoleculeAlphabet
from utilities import ops


class GrammarTrainer(TrainerBase):
    def __init__(self, args, model, optimiser, scheduler, train_dataloader, test_dataloader):
        super().__init__(args.logdir, args.tag)
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = args.num_epochs

        self.alphabet = MoleculeAlphabet()
        self.epoch = None

    def run_epoch(self):
        epoch_losses, epoch_accuracies = [], []
        for x, y in self.train_dataloader:
            self.model.zero_grad()
            x, y = Variable(x).cuda(), Variable(y).cuda().float()

            p_full = self.model.forward(x, 1)
            p = torch.mean(ops.corresponding(p_full, x.data), 0)

            loss = ops.qfun_loss(y, p)
            loss.backward()
            self.optimiser.step()

            y_hat = torch.FloatTensor([(ps > 0.5).all() for ps in p])
            accuracy = (y_hat == y.cpu().data).float().mean()

            epoch_losses.append(loss.data)
            epoch_accuracies.append(accuracy)

        return np.mean(epoch_losses), np.mean(epoch_accuracies)

    def train(self):
        for self.epoch in range(1, self.num_epochs + 1):
            loss, accuracy = self.run_epoch()
            frac_valid = self.validate()

            self.tensorboard.add_scalar('train/loss', loss, self.epoch)
            self.tensorboard.add_scalar('train/accuracy', accuracy, self.epoch)
            self.tensorboard.add_scalar('test/validity', frac_valid * 100, self.epoch)

            self.logger.info("Epoch {} :: loss {:.3f} :: accuracy {:.3f} :: validity {:.1f}%"
                             .format(self.epoch, loss, accuracy, frac_valid * 100))

    def validate(self):
        n_seq = valid = 0
        for sample in self.test_dataloader:
            output = self.model.forward_cells(Variable(sample.cuda()))
            valid += self.alphabet.validate([t for t in output.cpu().data]).sum()
            n_seq += len(sample)
        return valid / n_seq


class GrammarArgParser(TrainerArgParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--test-mode', action='store_true')
        self.add_argument('--grad-clip', type=float, default=0.0)
        self.add_argument('--batch-size', type=int, default=50)
        self.add_argument('--vae-data', action='store_true')
