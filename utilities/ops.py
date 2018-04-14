from itertools import product

import torch
from torch import nn
from torch.autograd import Variable


def clip_grads(params, clip_value):
    if not clip_value > 0:
        return
    for param in params:
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)


def scale_grads(params, threshold):
    if not threshold > 0:
        return
    for param in params:
        l2 = torch.norm(param, 2).data
        if (l2 > threshold).any():
            param.grad.data *= threshold / l2


def get_lr(optimiser):
    for pg in optimiser.param_groups:
        lr = pg['lr']
    return lr


def match_weights(n_layers):
    rnn_fmt = "{}_{}_l{}".format
    cells_fmt = "{}.{}_{}".format

    n = range(n_layers)
    ltype = ['ih', 'hh']
    wtype = ['bias', 'weight']
    matchings = []
    for n, l, w in product(n, ltype, wtype):
        matchings.append((rnn_fmt(w, l, n), cells_fmt(n, w, l)))

    return matchings


def sample(z_mean, z_log_var, size, epsilon_std=0.01):
    epsilon = Variable(torch.cuda.FloatTensor(*size).normal_(0, epsilon_std))
    return z_mean + torch.exp(z_log_var / 2.0) * epsilon


def make_safe(x):
    return x.clamp(1e-7, 1 - 1e-7)


def binary_entropy(x):
    return - (x * x.log() + (1 - x) * (1 - x).log())


def info_gain(x):
    marginal = binary_entropy(x.mean(0))
    conditional = binary_entropy(x).mean(0)
    return marginal - conditional


def init_params(m):
    for module_name, module in m.named_modules():
        for param_name, param in module.named_parameters():
            if 'weight' in param_name:
                if 'conv' in param_name or 'lin' in param_name or 'ih' in param_name:
                    nn.init.xavier_uniform(param)
                elif 'hh' in param_name:
                    nn.init.orthogonal(param)
            elif param_name == 'bias':
                nn.init.constant(param, 0.0)


def qfun_loss(y, p):
    log_p = torch.log(make_safe(p))
    positive = torch.sum(log_p, 1)
    neg_prod = torch.exp(positive)
    negative = torch.log1p(-make_safe(neg_prod))

    return - torch.sum(y * positive + (1 - y) * negative)


class VAELoss(torch.autograd.Function):
    def __init__(self):
        self.binary_xentropy = nn.BCELoss()

    def forward(self, x, x_decoded_mean, z_mean, z_log_var):
        xent_loss = 85 * self.binary_xentropy.forward(x_decoded_mean.view(-1), x.view(-1))
        kl_loss = - 0.5 * torch.mean(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var))
        return xent_loss + kl_loss


def one_hot(x, alphabet_size=27):
    x_one_hot = torch.cuda.LongTensor(x.numel(), alphabet_size).zero_()
    x_one_hot.scatter_(1, x.view(-1).unsqueeze(-1), 1)

    return x_one_hot.view(*x.size(), alphabet_size)


def corresponding(values, idxs, dim=-1):
    idxs = Variable(idxs)
    if len(values.size()) == 4:
        if len(idxs.size()) == 2:
            idxs = idxs.unsqueeze(0)
        idxs = idxs.repeat(values.size()[0], 1, 1)
    return values.gather(dim, idxs.unsqueeze(dim)).squeeze(dim)


def preds2seqs(preds):
    seqs = [torch.cat([torch.multinomial(char_preds, 1)
                       for char_preds in seq_preds])
            for seq_preds in preds]
    return torch.stack(seqs).data


def seqs_equal(seqs1, seqs2):
    return [torch.eq(s1, s2).all() for s1, s2 in zip(seqs1, seqs2)]


def sample_prior(n_samples, dec, model=None):
    samples = Variable(torch.cuda.FloatTensor(n_samples, 56).normal_(0, 1))
    p_hat = dec.forward(samples)
    if model:
        decoded = to_numpy(model.forward_cells(p_hat))
    else:
        decoded = decode(p_hat)
    return decoded
