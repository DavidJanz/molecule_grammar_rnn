import torch
from torch.utils.data import DataLoader

from base_classes.schedulers import Scheduler
from data.molecule_dataset import PerturbedZincDataset
from grammar_model.alphabets import MoleculeAlphabet
from grammar_model.grammar_rnn import BayesRNN
from grammar_model.grammar_trainer import GrammarTrainer, GrammarArgParser

if __name__ == '__main__':
    args = GrammarArgParser().parse_args()
    alphabet = MoleculeAlphabet()

    train_dataset = PerturbedZincDataset('data/zinc.npz', alphabet)
    if args.test_mode:
        train_dataset.idxs = train_dataset.idxs[:1000]

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(torch.load('prior_samples.pkl')[:10000], batch_size=10000, shuffle=True,
                                 drop_last=True)

    model = BayesRNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = Scheduler(optimiser, 1e-8, 0.1)

    trainer = GrammarTrainer(args, model, optimiser, scheduler, train_dataloader, test_dataloader)
    trainer.train()
