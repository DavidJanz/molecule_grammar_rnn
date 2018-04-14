import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utilities import ops


class Encoder(nn.Module):
    def __init__(self, alphabet_size=27, max_len=85, latent_size=56):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(alphabet_size, 9, 9, padding=0)
        self.conv2 = torch.nn.Conv1d(9, 9, 9, padding=0)
        self.conv3 = torch.nn.Conv1d(9, 10, 11, padding=0)

        self.linear = torch.nn.Linear((max_len - 26) * 10, 435)
        self.mean_lin = torch.nn.Linear(435, latent_size)
        self.var_lin = torch.nn.Linear(435, latent_size)
        ops.init_params(self)
        self.cuda()

    def forward(self, batch):
        batch = batch.transpose(2, 1)

        h = F.relu(self.conv1(batch))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = h.transpose(2, 1).contiguous()

        h_flat = h.view(len(batch), -1)

        h_out = F.relu(self.linear(h_flat))
        return self.mean_lin(h_out), self.var_lin(h_out)


class Decoder(nn.Module):
    def __init__(self, alphabet_size=27, max_len=85, latent_size=56):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.max_len = max_len
        self.hidden_size = 512

        self.linear = torch.nn.Linear(latent_size, latent_size)
        self.lin_out = torch.nn.Linear(self.hidden_size, alphabet_size)
        self.gru = nn.GRU(input_size=latent_size, hidden_size=self.hidden_size, num_layers=3, batch_first=True)

        ops.init_params(self)
        self.cuda()

    def forward(self, batch):
        h = F.relu(self.linear(batch)).unsqueeze(1).repeat(1, self.max_len, 1)
        initial_state = Variable(torch.zeros(3, len(batch), self.hidden_size).cuda())

        out, _ = self.gru(h, initial_state)
        out = self.lin_out(out.contiguous().view(-1, out.size(2)))
        return F.softmax(out, -1).view(len(batch), self.max_len, -1)
