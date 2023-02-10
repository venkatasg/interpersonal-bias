`data.tsv` contains the tweet IDs along with interpersonal emotions annotations. It contains the following columns:

- *Split*: train, test or dev split
- *TweetID*: unique tweet ID
- *username*: username of person who composed tweet
- *mentname*: username of target entity in tweet
- *Date*: date of tweet
- *group*: in-group(1) or out-group(-1)
- *party*: Democrat or Republican

and the following emotion label colums, which have either True or False: *Admiration*, *Anger*, *Disgust*, *Fear*, *Interest*, *Joy*, *Sadness*, *Surprise*

The Twitter API had a free tier for research when we prepared this dataset. This has since been [discontinued](https://twitter.com/TwitterDev/status/1621026986784337922) on 9th February 2023. If you are unable to access this dataset because Twitter's API and developer relations are a hot mess, you can [email me](mailto: gvenkata1994@gmail.com) and I will help you out.
