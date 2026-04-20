import torch
import torch.nn as nn
import json

from ts_gan_bench.visualization import plot_tsne

class ReverseMapTrainer():
    def __init__(
            self,
            settings,
            generator,
            discriminator,
            train_loader,
        ):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader

        self.loss = settings.model.loss
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.disc_real_label = settings.model.disc_real_label # used for label smoothing
        self.clip_grad_g = settings.model.clip_grad_g
        self.clip_grad_d = settings.model.clip_grad_d

        self.device = torch.device(settings.device_name)
        self.state_dir = settings.paths.state_dir # at this point directory isn't created yet

        self.window_size = settings.params.window_size
        self.zdim = settings.model.generator.in_dim

        self.lr_g = settings.model.lr_g
        self.lr_d = settings.model.lr_d
        self.betas_g = settings.model.betas_g
        self.betas_d = settings.model.betas_d
        self.generator_rounds = settings.model.generator_rounds
        self.discriminator_rounds = settings.model.discriminator_rounds

    def save_model_checkpoints(self, optimizerG, optimizerD, epoch):
        self.generator.save(
            self.state_dir / 'generator' / f'G_{epoch}.pth'
        )
        self.discriminator.save(
            self.state_dir / 'discriminator' / f'D_{epoch}.pth',
        )
        torch.save(
            optimizerG.state_dict(),
            self.state_dir / 'optim_generator' / f'G_optim_{epoch}.pth'
        )
        torch.save(
            optimizerD.state_dict(),
            self.state_dir / 'optim_discriminator' / f'D_optim_{epoch}.pth',
        )

    def save_tsne(self, filename, z_shape, n=1000, from_hidden_states=False):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, z_shape[0], z_shape[1], device=self.device)
            fake_samples = self.generator(z).cpu().numpy()
        samples = []
        current_count = 0
        for X, _ in self.train_loader:
            samples.append(X)
            current_count += X.size(0)
            if current_count >= n:
                break
        real_samples = torch.cat(samples, dim=0)[:n]
        
        plot_tsne(real_samples, fake_samples, filename)

    
    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def discriminator_step(self, optimizerD, predictions_fake, predictions_real):
        optimizerD.zero_grad()

        if self.loss == 'bce':
            # fake data
            fake_labels = torch.zeros_like(predictions_fake)
            loss_fake = self.criterion_bce(predictions_fake, fake_labels)
            loss_fake.backward()
            predictions_fake_log = torch.sigmoid(predictions_fake).mean()
            # real data
            real_labels = torch.full_like(predictions_real, self.disc_real_label)
            loss_real = self.criterion_bce(predictions_real, real_labels)
            loss_real.backward()
            predictions_real_log = torch.sigmoid(predictions_real).mean()

            total_loss = loss_fake + loss_real

        if self.clip_grad_d:
            nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.clip_grad_d,
            )
        optimizerD.step()
        return total_loss.item(), predictions_fake_log.item(), predictions_real_log.item()

    def generator_step(self, optimizerG, predictions):
        optimizerG.zero_grad()

        # BCE loss (from logits)
        if self.loss == 'bce':
            real_labels = torch.ones_like(predictions)
            loss = self.criterion_bce(predictions, real_labels)
            predictions_log = torch.sigmoid(predictions).mean()

        loss.backward()
        if self.clip_grad_g:
            nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=self.clip_grad_g,
            )
        optimizerG.step()
        return loss.item(), predictions_log.item()    

    def train(self, n_epochs, model_save_frequency):
        optimizerG = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=self.betas_g,
        )
        optimizerD = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=self.betas_d,
        )

        seq_list = []
        G_losses = []
        D_losses = []
        iters = 0

        # preparing checkpoint directories
        # at this point the main directory should be created
        (self.state_dir / 'generator').mkdir(exist_ok=True)
        (self.state_dir / 'discriminator').mkdir(exist_ok=True)
        (self.state_dir / 'optim_generator').mkdir(exist_ok=True)
        (self.state_dir / 'optim_discriminator').mkdir(exist_ok=True)
        (self.state_dir / 'tsne').mkdir(exist_ok=True)

        print("Starting training loop...")
        for epoch in range(n_epochs):
            for i, data in enumerate(self.train_loader):
                X, _ = data
                real_sequences = X.to(self.device)
                batch_size = real_sequences.shape[0]

                ############
                # D training
                ############

                self.set_requires_grad(self.discriminator, True)
                
                for j in range(self.discriminator_rounds):
                    z = torch.randn(batch_size, self.window_size, self.zdim, device=self.device)
                    fake_sequences = self.generator(z)
                    predictions_fake = self.discriminator(fake_sequences.detach()).view(-1)
                    predictions_real = self.discriminator(real_sequences).view(-1)
                    loss_D, D_G_z1, D_x = self.discriminator_step(optimizerD, predictions_fake, predictions_real)

                ############
                # G training
                ############

                self.set_requires_grad(self.discriminator, False)

                for j in range(self.generator_rounds):
                    z = torch.randn(batch_size, self.window_size, self.zdim, device=self.device)
                    fake_sequences = self.generator(z)
                    predictions = self.discriminator(fake_sequences).view(-1)
                    loss_G, D_G_z2 = self.generator_step(optimizerG, predictions)

                # output status
                if i % 50 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.train_loader)}]', end='\t')
                    print(f'Loss_D: {loss_D:.4f}', end='\t')
                    print(f'Loss_G: {loss_G:.4f}', end='\t')
                    print(f'D(x): {D_x:.4f}', end='\t')
                    print(f'D_G_z1: {D_G_z1:.4f}', end='\t')
                    print(f'D_G_z2: {D_G_z2:.4f}')
                
                # save losses
                G_losses.append(loss_G)
                D_losses.append(loss_D)

            # save model checkpoints
            if model_save_frequency and (epoch+1) % model_save_frequency == 0:
                self.save_model_checkpoints(optimizerG, optimizerD, epoch+1)
                self.save_tsne(self.state_dir / 'tsne' / f'{epoch+1}.png', (self.window_size, self.zdim))
                self.generator.train()

        # saving final models
        if not model_save_frequency or (n_epochs) % model_save_frequency != 0:
            self.save_model_checkpoints(optimizerG, optimizerD, n_epochs)
            self.save_tsne(self.state_dir / 'tsne' / f'{n_epochs}.png', (self.window_size, self.zdim))
            self.generator.train()

        # save training history
        loss_history = {
            'g_losses': G_losses,
            'd_losses': D_losses
        }

        with open(self.state_dir / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f)
