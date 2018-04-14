import shutil
import time

import numpy as np
import torch
from torch.autograd import Variable

from base_classes import schedulers
from base_classes.trainer import TrainerBase, TrainerArgParser
from utilities import ops, settings


class VAETrainer(TrainerBase):
    def __init__(self, args, encoder, decoder, optimiser, scheduler, train_dataloader, test_dataloader):
        super().__init__(args.logdir, args.tag)
        self.encoder = encoder
        self.decoder = decoder
        self.optimiser = optimiser
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.loss_function = ops.VAELoss()
        self.grad_clip = args.grad_clip
        self.num_epochs = args.num_epochs

        self.epoch = 0

    def run_epoch(self):
        total_data = total_loss = 0
        for x in self.train_dataloader:
            total_data += len(x)
            self.encoder.zero_grad(), self.decoder.zero_grad()
            x1h = Variable(ops.one_hot(x.cuda()).float())
            z_mean, z_log_var = self.encoder.forward(x1h)
            z_batch = ops.sample(z_mean, z_log_var, (len(z_mean), settings.latent_size))
            x_hat = self.decoder.forward(z_batch)

            loss = self.loss_function.forward(x1h, x_hat, z_mean, z_log_var) * len(x)
            total_loss += loss.data.cpu().numpy()[0]

            loss.backward()
            ops.clip_grads(list(self.decoder.parameters()), self.grad_clip)
            self.optimiser.step()

        return total_loss / total_data

    def validate(self):
        total_data = reconstructed = 0
        for x in self.test_dataloader:
            total_data += len(x)
            x1h = Variable(ops.one_hot(x.cuda()).float(), volatile=True)

            z_mean, z_log_var = self.encoder.forward(x1h)
            z_batch = ops.sample(z_mean, z_log_var, (len(z_mean), settings.latent_size))

            p_hat = self.decoder.forward(z_batch)
            x_hat = ops.preds2seqs(p_hat).cpu()
            reconstructed += np.sum(ops.seqs_equal(x, x_hat))

        return reconstructed / total_data

    def train(self):
        results_fmt = ("{} :: {} :: loss {:.3f} reconstruction {:.2f}" + " " * 30).format

        for self.epoch in range(1, self.num_epochs):
            loss = self.run_epoch()

            try:
                self.scheduler.step(loss)
            except schedulers.OptimiserFailed:
                self.epoch -= 1
                self.load('autosave')
                continue

            self.save(self.scheduler.is_best, 'autosave')

            if self.epoch % 25:
                self.save(False, self.epoch)

            recon = self.validate()

            self.tensorboard.add_scalar('train/loss', loss, self.epoch)
            self.tensorboard.add_scalar('test/reconstruction', recon, self.epoch)
            self.tensorboard.add_scalar('train/learning_rate', ops.get_lr(self.optimiser), self.epoch)

            self.logger.info(results_fmt(time.strftime("%H:%M:%S"), self.epoch, loss, recon))

    def save(self, is_best, name=None):
        state = {'encoder': self.encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'optimiser': self.optimiser.state_dict(),
                 'scheduler': self.scheduler.state_dict()}

        if name is None:
            path = self.checkpoint_path(self.epoch)
        else:
            path = self.checkpoint_path(name)
        torch.save(state, path)

        if is_best:
            shutil.copyfile(path, self.checkpoint_path('top'))

    def load(self, step):
        self.load_raw(self.checkpoint_path(step))

    def load_raw(self, path):
        state = torch.load(path)

        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.optimiser.load_state_dict(state['optimiser'])
        self.scheduler.load_state_dict(state['scheduler'])

    def sample_prior(self, n_samples):
        z_samples = torch.normal(torch.cuda.FloatTensor(n_samples, 56) * 0,
                                 torch.cuda.FloatTensor(n_samples, 56) * 0 + 1)
        return self.decoder(Variable(z_samples)).data


class VAEArgParser(TrainerArgParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--test-mode', action='store_true')
        self.add_argument('--grad-clip', type=float, default=2.0)
        self.add_argument('--wd', type=float, default=1e-4)
        self.add_argument('--batch-size', type=int, default=500)
        self.add_argument('--generate-samples', action='store_true')
