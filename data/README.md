`data.tsv` contains the tweet IDs along with interpersonal emotions annotations. It contains the following columns:

- *Split*: train, test or dev split
- *TweetID*: unique tweet ID 
- *username*: username of person who composed tweet
- *mentname*: username of target entity in tweet
- *Date*: date of tweet
- *group*: in-group(1) or out-group(-1)
- *party*: Democrat or Republican

and the following emotion label colums, which have either True or False:

- *Admiration*
- *Anger*
- *Disgust*
- *Fear*
- *Interest*
- *Joy*
- *Sadness*
- *Surprise*