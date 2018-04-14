import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from utilities import ops, settings


class BayesRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = settings.layer_size
        self.embedding_size = settings.embedding_size
        self.alphabet_size = settings.alphabet_size
        self.n_layers = settings.n_layers
        self.max_seq_len = settings.max_len
        self.keep_prob = settings.keep_prob

        self._embedding = nn.Embedding(num_embeddings=self.alphabet_size, embedding_dim=self.embedding_size,
                                       padding_idx=0)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.n_layers, batch_first=True,
                          dropout=self.keep_prob)
        self._cells = nn.ModuleList(self._construct_cells())
        self._weight_matches = ops.match_weights(self.n_layers)

        self._projection = nn.Linear(self.hidden_size, self.alphabet_size)

        self._hidden, self._dropout = None, None

        ops.init_params(self)
        self.cuda()

    def _construct_cells(self):
        cells = []
        for n in range(self.n_layers):
            in_size = self.embedding_size if n == 0 else self.hidden_size
            cells.append(nn.GRUCell(in_size, self.hidden_size))
        return cells

    def _cell(self, x):
        cell_input = self._embedding(x)
        for n, cell in enumerate(self._cells):
            self._hidden[n] = cell(cell_input, self._hidden[n])
            cell_input = self._hidden[n].clone()
        return self._hidden[n].clone()

    def _sync_weights(self):
        rnn_state_dict = self.rnn.state_dict()
        cell_state_dict = {}
        for rnn_w_name, cell_w_name in self._weight_matches:
            cell_state_dict[cell_w_name] = rnn_state_dict[rnn_w_name]

        self._cells.load_state_dict(cell_state_dict)

    def _init_dropout(self, n_samples, batch_size):
        keep_prob = torch.cuda.FloatTensor(batch_size * n_samples, self.hidden_size).zero_() + self.keep_prob
        self._dropout = [Variable(torch.bernoulli(keep_prob)) for _ in self._cells]

    def _init_hidden(self, size):
        self._hidden = Variable(torch.cuda.FloatTensor(self.n_layers, size, self.hidden_size))

    def forward_cells(self, p_mod, n_samples=1):
        max_len, batch_size = 85, len(p_mod)
        self._sync_weights(), self._init_hidden(batch_size * n_samples)

        x_in = Variable(torch.cuda.LongTensor(batch_size).zero_(), volatile=True)
        x_out = Variable(torch.cuda.LongTensor(batch_size, max_len))

        for t in range(max_len):
            mask = F.sigmoid(self._projection(self._cell(x_in.repeat(n_samples)))) \
                .view(n_samples, batch_size, self.alphabet_size).mean(0).round()
            x_in = x_out[:, t] = (mask * p_mod[:, t, :] + 1e-7).multinomial().squeeze(-1)

        return x_out

    def forward(self, x, n_samples=16):
        batch_size, max_len = len(x), 85

        self._init_hidden(batch_size * n_samples)

        padding = Variable(torch.cuda.LongTensor(batch_size * n_samples, 1).zero_())
        x = torch.cat([padding, x], 1)

        output, _ = self.rnn(self._embedding(x), self._hidden)
        output = output[:, :self.max_seq_len, :]

        p = F.sigmoid(self._projection(output))
        return p.view(n_samples, batch_size, self.max_seq_len, self.alphabet_size)

    def save(self):
        state = self.state_dict()
        file_path = os.path.join('log', 'grammar')
        torch.save({'grammar_state_dict': state}, file_path)

    def load(self):
        file_path = os.path.join('log', 'grammar')
        state = torch.load(file_path)
        self.load_state_dict(state['grammar_state_dict'])
