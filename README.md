# MASTER THESIS - ULMFiT experiments

### INTRODUCTION
- This is project for master thesis
- Main idea is to use ULMFiT method to solve problem of sentiment analysis and word sense disambiguation (WSD)
- For sentiment analysis use Croatian dataset from master thesis: "Analiza sentimenta u tvitovima na
hrvatskom jeziku" (Luka Krajcar, 2014), that contains tweets and their sentiments
- For WSD I use Croatian dataset from http://takelab.fer.hr/data/cro6wsd/ that contains six words and their annotations
- There are two different labeling techniques for sentiment analysis: distant (automatized labeling) and hand (human annotators)

### DESCRIPTION
- experiments/Performance_WSD_ULMFiT.ipynb	- ULMFiT experiments for WSD dataset for master thesis
- experiments/TWITTER_CLASSIFICATION_DISTANT.ipynb - ULMFiT experiments for sentiment analysis dataset with distant labels for master thesis
- experiments/TWITTER_CLASSIFICATION_HAND.ipynb - ULMFiT experiments for sentiment analysis dataset with hand labels for master thesis
- experiments/SAVE_MODEL_ITOS.ipynb - load data bunch used for language model pretraining and create itos pickle file
- experiments/labeled-distant.csv - dataset for Tweeter distant labels
- experiments/labeled-hand.csv - dataset for Tweeter hand labels
- train_lm.py - Language model pretraining on Croatian wikipedia dump
- train_finetuning.py - fine-tune of language model for Twitter sentiment analysis task

### GUIDE