import pandas as pd
from torch.utils.data import Dataset, DataLoader

import tokenizer


# we can do all of this in pytorch but we will need to convert the data to numpy arrays at training time 
class ChinchillaDataset(Dataset):
    def __init__(self, problems_file):
        self.problems = pd.read_csv(problems_file)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem_statement = self.problems.loc[idx, "problems"]
        print(problem_statement)
        problem, answer = tokenizer.tokenizer(problem_statement)
        return problem, answer



dataset = ChinchillaDataset("data/val.csv")
dataloader = DataLoader(dataset, batch_size=10)
print(next(iter(dataloader)))