import pandas as pd

data = pd.read_json("../data/train.json")

random.seed(0)
train_data = data.sample(n=data.shape[0] * 7 / 10)
test_data = data.drop(train_data.index)
