# Modeling and Analysis

To recreate the python environment in which the experiments were run, use the `environ.yml` file. You can [recreate the environment]() on your local setup using conda:

```
conda env create -f environ.yml
```

## Interpersonal Group Relationship Prediction

The script `train_inout_classifier.py` takes in a tweet with target identity masked and trains a `BERTweet` model to output 1(in-group) or 0(out-group).`test_inout_classifier` is for evaluation of trained models.

## Emotion labelling
The script `train_emot_labeller.py` takes in a tweet with target identity masked and trains a `BERTweet` model to output 8 label scores corresponding to the 8 emotions in the Plutchik Wheel. `test_emot_labeller.py` is for evaluation of trained models.

## Multitask model

The script `train_multitask_classifier.py` takes in a tweet with target identity masked and trains a `BERTweet` model to output both interpersonal group relationship and emotion labels. `test_inout_classifier.py` is for evaluation of trained models.

The `Notebooks/` folder contains Jupyter notebooks used for analyses in the paper including NB-SVM, EMOLEX, UMAP clustering, data aggregation and analysis, and bootstrap significance testing.
