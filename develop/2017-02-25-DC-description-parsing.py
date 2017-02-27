import pandas as pd
import nltk

from nltk.tokenize import word_tokenize

with open("../data/processed_train.json") as f:
    data = pd.read_json(f)


d = data.description
d_words = d.apply(word_tokenize)
d_null = d.apply(lambda x: len(x) > 0)

percent_null = sum(~d_null)/float(sum(d_null))

dist = data.features.apply(
        lambda x: pd.Series(map(lambda z: 1 if (z in x) else 0, distinct_features) +
                            [list(np.setdiff1d(x, distinct_features))]))
