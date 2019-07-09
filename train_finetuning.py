import torch
print("DEVICE NAME:", torch.cuda.get_device_name(0))
print("DEVICE COUNT:", torch.cuda.device_count())
print("CURRENT DEVICE:", torch.cuda.current_device())

from fastai import *
from fastai.text import *
from fastai.core import *
from pathlib import Path
import pandas as pd
import numpy as np
from ulmfit.pretrain_lm import *
from fastai.callbacks import CSVLogger, SaveModelCallback

wiki_data_path = Path('data/wiki/hr-100/')
#data_lm = TextLMDataBunch.from_csv(wiki_data_path, 'unsupervised_big.csv', text_cols=0, bs=30)
data_lm = load_data(wiki_data_path, 'lm_finetuned', bs=45)
itos, stoi, data_path = data_lm.vocab.itos, data_lm.vocab.stoi, data_lm.path
pretrained_fnames = ['hr-100-best', 'itos']

learner = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=pretrained_fnames, drop_mult=0.9, 
                                 model_dir='./models')

learner.freeze()
learner.opt_func = partial(optim.Adam, betas=(0.8, 0.99))
print("LEARN HEAD")
learner.fit_one_cycle(1, 1e-2)

learner.unfreeze()
print("LEARN EVERYTHING")
learner.fit_one_cycle(15, 1e-3, moms=(0.8,0.7))
print("FINISHIED")
learner.save('lm_fine_tuned_c4')
learner.save_encoder('ft_enc_c4')
