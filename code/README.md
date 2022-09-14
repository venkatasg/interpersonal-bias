`emot_labeller.py` takes in a tweet with target identity masked and trains a transformer based model to output 8 label scores corresponding to the 8 emotions in the Plutchik Wheel. `test_emot_labeller.py` is for evaluation of trained models.

`inout_classifier.py` takes in a tweet with target identity masked and trains a transformer based model to output +1(in-group) or 0(out-group). `test_inout_classifier` is for evaluation of trained models.

`multitask_classifier.py` takes in a tweet with target identity masked and trains a transformer based model to output both in/out-group and emotion labels. `test_inout_classifier.py` is for evaluation of trained models.

The `Notebooks/` folder contains Jupyter notebooks used for analyses in the paper, including NB-SVM code, UMAP clustering, and data aggregation and analysis.