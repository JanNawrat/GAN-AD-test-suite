import torch
import torch.nn as nn
import json

class ReverseMapTrainer():
    def __init__(self, settings, generator, discriminator, train_loader, device, state_dir):
        self.settings = settings
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.device = device
        self.state_dir = state_dir # at this point directory isn't created yet

        self.lr_g = settings['model']['lr_g']
        self.lr_d = settings['model']['lr_d']

    def save_model_checkpoints(self, optimizerG, optimizerD, epoch):
        torch.save(self.generator.state_dict(), self.state_dir / 'generator' / f'G_{epoch}.pth')
        torch.save(self.discriminator.state_dict(), self.state_dir / 'discriminator' / f'D_{epoch}.pth')
        torch.save(optimizerG.state_dict(), self.state_dir / 'optim_generator' / f'G_optim_{epoch}.pth')
        torch.save(optimizerD.state_dict(), self.state_dir / 'optim_discriminator' / f'D_optim_{epoch}.pth')

    def train(self, n_epochs, model_save_frequency):
        criterion = nn.BCELoss()
        optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g)
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)

        seq_list = []
        G_losses = []
        D_losses = []
        iters = 0
        frame_length = self.settings['params']['frame_length']
        zdim = self.settings['model']['zdim']

        # preparing checkpoint directories
        # at this point the main directory should be created
        (self.state_dir / 'generator').mkdir(exist_ok=True)
        (self.state_dir / 'discriminator').mkdir(exist_ok=True)
        (self.state_dir / 'optim_generator').mkdir(exist_ok=True)
        (self.state_dir / 'optim_discriminator').mkdir(exist_ok=True)

        print("Starting training loop...")
        for epoch in range(n_epochs):
            for i, data in enumerate(self.train_loader):
                X, _ = data
                real_sequences = X.to(self.device)
                batch_size = real_sequences.shape[0]
                fake_labels = torch.zeros(batch_size * frame_length, dtype=torch.float, device=self.device)
                real_labels = torch.ones(batch_size * frame_length, dtype=torch.float, device=self.device)

                ############
                # D training
                ############

                # fake data
                self.discriminator.zero_grad()
                z = torch.randn(batch_size, frame_length, zdim, device=self.device)
                fake_sequences = self.generator(z)
                predictions = self.discriminator(fake_sequences.detach()).view(-1)
                loss_D_fake = criterion(predictions, fake_labels)
                loss_D_fake.backward()
                D_G_z1 = predictions.mean().item()

                # real data
                predictions = self.discriminator(real_sequences).view(-1)
                loss_D_real = criterion(predictions, real_labels)
                loss_D_real.backward()
                optimizerD.step()
                D_x = predictions.mean().item()

                loss_D = loss_D_fake + loss_D_real

                ############
                # G training
                ############

                # fake data
                self.generator.zero_grad()
                z = torch.randn(batch_size, frame_length, zdim, device=self.device)
                fake_sequences = self.generator(z)
                predictions = self.discriminator(fake_sequences).view(-1)
                loss_G = criterion(predictions, real_labels)
                loss_G.backward()
                optimizerG.step()
                D_G_z2 = predictions.mean().item()

                # output status
                if i % 50 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.train_loader)}]\tLoss_D: {loss_D.item():.4f}\tLoss_G: {loss_G.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

                # save losses
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

                # save model checkpoints
                if model_save_frequency and (epoch+1) % model_save_frequency == 0:
                    self.save_model_checkpoints(optimizerG, optimizerD, epoch+1)

        # saving final models
        if not model_save_frequency or (n_epochs) % model_save_frequency != 0:
            self.save_model_checkpoints(optimizerG, optimizerD, n_epochs)

        # save training history
        loss_history = {
            'g_losses': G_losses,
            'd_losses': D_losses
        }

        with open(self.state_dir / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f)
