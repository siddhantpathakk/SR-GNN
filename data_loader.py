from torch.utils.data import Dataset, DataLoader
import pandas as pd

def load_data(path):
    # Load data from the given path
    
    buy_path = path + "yoochoose-buys.dat"
    click_path = path + "yoochoose-clicks.dat"
    test_path = path + "yoochoose-test.dat"
    
    buys = pd.read_csv(buy_path,sep=",",header=None)
    clicks = pd.read_csv(click_path,sep=",",header=None)
    test = pd.read_csv(test_path,sep=",",header=None)
    
    columns_list = ["session_id", "timestamp", "item_id", "price", "quantity"]
    
    buys.columns = columns_list
    clicks.columns = columns_list
    test.columns = columns_list[:-1]
    
    return {"buys": buys, "clicks": clicks, "test": test}

def split_dataset(data, split_ratio, validation):
    pass


def make_dataset(data):
    pass


def make_dataloaders(batch_size, train, test, val=None):
    pass


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)