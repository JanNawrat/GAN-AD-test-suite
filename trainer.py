import torch

class ReverseMapTrainer():
    def __init__(self, model_settings, generator, discriminator, train_loader):
        self.model_settings = model_settings
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader

    def train(self):
        pass