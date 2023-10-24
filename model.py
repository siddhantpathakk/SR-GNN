from torch import nn, optim
import time
import torch

class MyModel(nn.Module):
    def __init__(self, parameters):
        super().__init__()
    
    def forward(self, x):
        pass

    
def get_model(parameters, resume):
    model = MyModel(parameters=parameters)
    if resume is not None:
        model = torch.load(resume)
    return model

def get_criterion(loss_fn):
    pass

def get_optimizer(optimizer, model, lr):
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)


def train_step(model, dataloader, optimizer, loss_fn, epochs):
    pass


def validate_step(model, dataloader, loss_fn):
    pass


def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs):
    pass


def test_model(model, test_dataloader, loss_fn):
    pass