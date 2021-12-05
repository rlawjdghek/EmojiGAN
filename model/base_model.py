import torch
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
    @abstractmethod
    def set_input(self): pass
    @abstractmethod
    def forward_G(self): pass
    @abstractmethod
    def train(self): pass
    @abstractmethod
    def validation(self): pass
    @staticmethod
    def set_requires_grad(models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad