import numpy as np
import pandas as pd

low, high = 0, 1000 # [0, 1000) 

def generate_dataset(filename: str,size: int, destination: str = "data") -> str:
    data = []
    for _ in range(size):
        addend1, addend2 = np.random.randint(low=low, high=high, size=2)
        data.append(f"{addend1} + {addend2} = {addend1 + addend2}")
    pd.DataFrame(data, columns=["problems"]).to_csv(f"{destination}/{filename}")
    

train_size = 1000
val_size = 200
test_size = 200

generate_dataset("train.csv", train_size)
generate_dataset("val.csv", val_size)
generate_dataset("test.csv", test_size)