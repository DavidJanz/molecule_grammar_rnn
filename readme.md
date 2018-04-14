Code for paper "Learning a Generative Model for Validity in Complex Discrete Structures" by
David Janz, Jos van der Westhuizen, Brooks Paige, Matt Kusner and José Miguel Hernández-Lobato. Arxiv: https://arxiv.org/abs/1712.01664

Steps to reproduce results in table 4:

1) Install requirements.
rdkit (chemical informatics, http://www.rdkit.org/docs/Install.html) can be tricky; we recommend using Anaconda.

2) First prepare data by running
python3 data/preprocess_data.py data/zinc.smi

3) Generate samples from pretrained VAE for testing the grammar model. See optional) for training your own VAE.
python3 vae --restore vae-pretrained --generate-samples

4) Train grammar_rnn model, the main paper contribution. This takes approximately ten minutes.
python3 grammar_model

Results are displayed to terminal + plotted to tensorboard (default location /tmp/models)

Citation bibtex:

@article{janz2018learning,
    title={Learning a Generative Model for Validity in Complex Discrete Structures},
    author={Janz, David and van der Westhuizen, Jos and Paige, Brooks and Kusner, Matt and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
    journal={International Conference on Learning Representations},
    year={2018}
}

Optional: Train your own character VAE for molecules. Enclosed implementation beats the original. Training takes a couple hours.
If you get a cuda out of memory error, consider reducing --batch-size (defaults at 500).
python3 vae
