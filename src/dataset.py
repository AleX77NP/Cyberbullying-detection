from torch.utils.data import Dataset


# Wrapper for pandas datasets
class TweetDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        sample = self.dataframe.iloc[index]
        tweet_str = sample["clean_tweet"]
        label = sample["category"]

        return label, tweet_str
