# Interpersonal-Dynamics

This repository contains code and data for the paper [How people talk about each other: Modeling Generalized Intergroup Bias and Emotions](https://arxiv.org/abs/2209.06687), which will be presented at [EACL 2023](https://2023.eacl.org).

Our goal with this paper was to situate bias in language use through the lens of interpersonal relationships between the speaker and target of an utterance anchored with the interpersonal emotion expressed by the speaker towards the target.

## Code

Scripts and notebooks used for modeling and analysis are present in the `code` folder.


## Data

The dataset of tweets, with annotations detailed in the paper, is in the `data` folder. The full annotation protocol is also present in this folder (`cong-emotion.html`). Only tweet IDs are listed here &mdash; contact us if you need access to the full dataset.

## Citation

Please cite our work as follows, or use the corresponding `bib` entry:

```
@inproceedings{govindarajan-etal-2023-people,
    title = "How people talk about each other: Modeling Generalized Intergroup Bias and Emotion",
    author = "Govindarajan, Venkata Subrahmanyan  and
      Atwell, Katherine  and
      Sinno, Barea  and
      Alikhani, Malihe  and
      Beaver, David I.  and
      Li, Junyi Jessy",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.183",
    pages = "2496--2506",
    abstract = "Current studies of bias in NLP rely mainly on identifying (unwanted or negative) bias towards a specific demographic group. While this has led to progress recognizing and mitigating negative bias, and having a clear notion of the targeted group is necessary, it is not always practical. In this work we extrapolate to a broader notion of bias, rooted in social science and psychology literature. We move towards predicting interpersonal group relationship (IGR) - modeling the relationship between the speaker and the target in an utterance - using fine-grained interpersonal emotions as an anchor. We build and release a dataset of English tweets by US Congress members annotated for interpersonal emotion - the first of its kind, and {`}found supervision{'} for IGR labels; our analyses show that subtle emotional signals are indicative of different biases. While humans can perform better than chance at identifying IGR given an utterance, we show that neural models perform much better; furthermore, a shared encoding between IGR and interpersonal perceived emotion enabled performance gains in both tasks.",
}

```
