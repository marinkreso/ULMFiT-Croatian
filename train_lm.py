import platform
import torch

from fastai import *
from fastai.text import *
from fastai.core import *
from pathlib import Path
import pandas as pd
import numpy as np
from ulmfit.pretrain_lm import *
from fastai.callbacks import CSVLogger, SaveModelCallback


print(platform.python_version())
print("DEVICE NAME:", torch.cuda.get_device_name(0))
print("DEVICE COUNT:", torch.cuda.device_count())
print("CURRENT DEVICE:", torch.cuda.current_device())
print(torch.cuda.is_available())


wiki_data_path = Path('data/wiki/hr-100/')
trn_path = wiki_data_path/'hr.wiki.train.tokens'
val_path = wiki_data_path/'hr.wiki.valid.tokens'
#batch_size
bs = 30

data_lm = TextLMDataBunch.from_df(path=wiki_data_path, train_df=read_wiki_articles(trn_path),
                                  valid_df=read_wiki_articles(val_path), 
                                  classes=None, bs=bs, text_cols='texts')
data_lm.save('data_lm')

learner = language_model_learner(data=data_lm, arch=AWD_LSTM, drop_mult=0.9)
learner.opt_func = partial(optim.Adam, betas=(0.8, 0.99))
learner.callback_fns += [partial(SaveModelCallback, every='epoch', name='lm'),
                         partial(CSVLogger, filename=f"{learner.model_dir}/lm-history")]
learner.unfreeze()
learner.lr_find()
moms=(0.8,0.7)
print("TRAINING START")
learner.fit_one_cycle(15, 1e-3, moms=moms)
learner.save('hr-100-best')
