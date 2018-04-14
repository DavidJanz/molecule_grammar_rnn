import torch
from torch.utils.data import DataLoader

from base_classes import schedulers
from data.molecule_dataset import ZincDataset
from vae import vae_models
from vae.vae_trainer import VAETrainer, VAEArgParser
import tqdm
import os


if __name__ == '__main__':
    args = VAEArgParser().parse_args()
    encoder, decoder = vae_models.Encoder(), vae_models.Decoder()

    if not os.path.isfile('data/zinc.npz'):
        raise ValueError('Please generate zinc.npz data file first by running data/preprocess_data!')

    train_dataset = ZincDataset('data/zinc.npz', fname='train')

    if args.test_mode:
        train_dataset.idxs = train_dataset.idxs[:1000]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(ZincDataset('data/zinc.npz', fname='test'), shuffle=True,
                                 batch_size=args.batch_size * 10)

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(all_params, lr=0.001)
    scheduler = schedulers.Scheduler(optimiser, 0.5, 1e-8)

    trainer = VAETrainer(args, encoder, decoder, optimiser, scheduler, train_dataloader, test_dataloader)

    if args.generate_samples:
        if not args.restore:
            raise ValueError('argument --restore with trained vae path required to generate samples!')

        trainer.load_raw(args.restore)
        samples = []
        for _ in tqdm.tqdm(range(1000)):
            samples.append(trainer.sample_prior(250).cpu())
        samples = torch.cat(samples, 0)

        torch.save(samples, 'prior_samples.pkl')
    else:
        trainer.train()
