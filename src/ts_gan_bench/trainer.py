import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from ts_gan_bench.utils import set_requires_grad, add_bounded_dequantization, map_anomaly_score_to_sequence
from ts_gan_bench.visualization import plot_tsne

class Trainer():
    def __init__(
            self,
            settings,
            train_loader,
            feature_names=None,
            actuator_idx=None,
        ):
        self.device = torch.device(settings.device_name)
        self.state_dir = settings.paths.state_dir # at this point directory isn't created yet
        self.window_size = settings.params.window_size
        self.stride = settings.params.stride
        self.train_loader = train_loader
        self.feature_names = feature_names
        self.time_last = settings.params.time_last
        self.bounded_dequantization = settings.model.bounded_dequantization
        self.actuator_idx = actuator_idx
        self.use_automatic_precision = settings.params.use_automatic_precision

    def save_sample_sequences(self, sample_sequences, epoch):
        # preprogrammed for 64 batch size
        samples_dir = self.state_dir / 'sample_sequences' / str(epoch)
        samples_dir.mkdir(exist_ok=True)
        if self.time_last:
            sample_sequences = np.permute_dims(sample_sequences, (0, 2, 1))
        # setting up the plot
        matplotlib.use('Agg')
        fig, axs = plt.subplots(8, 8)
        fig.set_size_inches(20, 20)
        plots = []
        for k in range(8):
            for l in range(8):
                ax = axs[k,l]
                plot = ax.plot([0. for i in range(sample_sequences.shape[1])])
                plots.append(plot)
                ax.set_ylim([-1.1, 1.1])
                ax.set_yticks([-1, 0, 1])
                ax.set_xticks([0, sample_sequences.shape[1]/2, sample_sequences.shape[1]])
        # reusing the plot
        for i in range(sample_sequences.shape[2]):
            name = self.feature_names[i] if self.feature_names is not None else f'Feature_{i}'
            fig.suptitle(name, fontsize=32)
            for j in range(64):
                plots[j][0].set_ydata(sample_sequences[j,:,i])
            plt.savefig(samples_dir / f'{name}.png')
        plt.close(fig)
        

class ReverseMapTrainer(Trainer):
    def __init__(
            self,
            settings,
            generator,
            discriminator,
            train_loader,
            feature_names=None,
            actuator_idx=None,
        ):
        super().__init__(settings, train_loader, feature_names, actuator_idx)
        self.generator = generator
        self.discriminator = discriminator

        if settings.params.compile_models:
            self.generator = torch.compile(
                generator,
                mode=settings.params.compilation_mode,
            )
            self.discriminator = torch.compile(
                discriminator,
                mode=settings.params.compilation_mode,
            )

        self.loss = settings.model.loss
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.gp_weight = settings.model.gp_weight
        self.disc_real_label = settings.model.disc_real_label # used for label smoothing
        self.clip_grad_g = settings.model.clip_grad_g
        self.clip_grad_d = settings.model.clip_grad_d

        z_channels = settings.model.generator.in_dim
        z_time = settings.params.window_size
        self.z_shape = (z_channels, z_time) if self.time_last else (z_time, z_channels)

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

        if self.time_last:
            real_samples = torch.permute(real_samples, (0, 2, 1))
            fake_samples = np.permute_dims(fake_samples, (0, 2, 1))
        
        plot_tsne(real_samples, fake_samples, filename)

    def compute_gradient_penalty(self, samples_fake, samples_real):
        batch_size = samples_real.size(0)
        alpha_shape = [batch_size] + [1] * (samples_real.dim() - 1)
        alpha = torch.rand(alpha_shape, device=self.device)
        interpolates = (alpha * samples_real + ((1 - alpha) * samples_fake)).requires_grad_(True)

        with torch.backends.cudnn.flags(enabled=False):
            d_interpolates = self.discriminator(interpolates)
        # d_interpolates = self.discriminator(interpolates)
        fake_outputs = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def discriminator_step(self, optimizerD, real_sequences, z):
        # concatenates real and fake data
        # this won't work if using batch norm
        optimizerD.zero_grad(set_to_none=True)

        # BCE loss (with logits)
        if self.loss == 'bce':
            batch_size = real_sequences.shape[0]
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_automatic_precision,
            ):
                fake_sequences = self.generator(z).detach()
                combined_sequences = torch.concat(
                    [fake_sequences, real_sequences],
                    dim=0,
                )
                predictions = self.discriminator(combined_sequences).view(-1)
                fake_labels = torch.zeros_like(predictions[:batch_size])
                real_labels = torch.full_like(predictions[batch_size:], self.disc_real_label)
                combined_labels = torch.concat(
                    [fake_labels, real_labels],
                    dim=0,
                )
                loss = self.criterion_bce(predictions, combined_labels)

            # logging
            predictions_fake_log = torch.sigmoid(predictions[:batch_size].detach().float()).mean()
            predictions_real_log = torch.sigmoid(predictions[batch_size:].detach().float()).mean()

        elif self.loss == 'wasserstein':
            # TODO
            pass

        loss.backward()
        if self.clip_grad_d:
            nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.clip_grad_d,
            )
        optimizerD.step()
        return loss.item(), predictions_fake_log.item(), predictions_real_log.item()


    def _discriminator_step(self, optimizerD, predictions, samples_fake, samples_real):
        # predictions - batch of precitions on fake data
        #   + batch of predictions on real data concatenated 
        optimizerD.zero_grad(set_to_none=True)

        if self.loss == 'bce':
            batch_size = predictions.shape[0] // 2
            fake_labels = torch.zeros_like(predictions[:batch_size])
            real_labels = torch.full_like(predictions[:batch_size], self.disc_real_label)
            combined_labels = torch.cat([fake_labels, real_labels], dim=0)
            total_loss = self.criterion_bce(predictions, combined_labels)
            total_loss.backward()

            # logging
            predictions_fake_log = torch.sigmoid(predictions[:batch_size]).mean()
            predictions_real_log = torch.sigmoid(predictions[batch_size:]).mean()

            # # fake data
            # fake_labels = torch.zeros_like(predictions_fake)
            # loss_fake = self.criterion_bce(predictions_fake, fake_labels)
            # loss_fake.backward()
            # predictions_fake_log = torch.sigmoid(predictions_fake).mean()
            # # real data
            # real_labels = torch.full_like(predictions_real, self.disc_real_label)
            # loss_real = self.criterion_bce(predictions_real, real_labels)
            # loss_real.backward()
            # predictions_real_log = torch.sigmoid(predictions_real).mean()

            # total_loss = loss_fake + loss_real
        # elif self.loss == 'wasserstein':
        #     total_loss = -torch.mean(predictions_real) + torch.mean(predictions_fake)
        #     if self.gp_weight != 0:
        #         total_loss += self.gp_weight * self.compute_gradient_penalty(samples_fake, samples_real)
        #     total_loss.backward()
        #     predictions_fake_log = torch.mean(predictions_fake)
        #     predictions_real_log = torch.mean(predictions_real)

        if self.clip_grad_d:
            nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.clip_grad_d,
            )
        optimizerD.step()
        return total_loss.item(), predictions_fake_log.item(), predictions_real_log.item()
    
    def generator_step(self, optimizerG, z):
        optimizerG.zero_grad(set_to_none=True)

        # BCE loss (from logits)
        if self.loss == 'bce':
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_automatic_precision,
            ):
                fake_sequences = self.generator(z)
                predictions = self.discriminator(fake_sequences)
                real_labels = torch.ones_like(predictions)
                loss = self.criterion_bce(predictions, real_labels)
                # loss = -nn.functional.logsigmoid(predictions).mean()
            predictions_log = torch.sigmoid(predictions.detach().float()).mean()
        elif self.loss == 'wasserstein':
            # TODO
            pass

        loss.backward()
        if self.clip_grad_g:
            nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=self.clip_grad_g,
            )
        optimizerG.step()
        return loss.item(), predictions_log.item()


    def _generator_step(self, optimizerG, predictions):
        optimizerG.zero_grad(set_to_none=True)

        # BCE loss (from logits)
        if self.loss == 'bce':
            real_labels = torch.ones_like(predictions)
            loss = self.criterion_bce(predictions, real_labels)
            predictions_log = torch.sigmoid(predictions).mean()
        elif self.loss == 'wasserstein':
            loss = -torch.mean(predictions)
            predictions_log = torch.mean(predictions)

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

        total_steps = 0
        history_G_losses = []
        history_D_losses = []
        # used to save sample sequences
        static_z = torch.randn(64, *self.z_shape, device=self.device)

        # preparing checkpoint directories
        # at this point the main directory should be created
        (self.state_dir / 'generator').mkdir(exist_ok=True)
        (self.state_dir / 'discriminator').mkdir(exist_ok=True)
        (self.state_dir / 'optim_generator').mkdir(exist_ok=True)
        (self.state_dir / 'optim_discriminator').mkdir(exist_ok=True)
        (self.state_dir / 'tsne').mkdir(exist_ok=True)
        (self.state_dir / 'sample_sequences').mkdir(exist_ok=True)

        # temporary solution
        # TODO: fix
        loss_G = 0.
        D_G_z2 = 0.

        print("Starting training loop...")
        for epoch in range(n_epochs):
            for i, data in enumerate(self.train_loader):
                X, _ = data
                real_sequences = X.to(self.device)
                batch_size = real_sequences.shape[0]

                # ==================================
                # Bounded dequantization
                # ==================================
                if self.bounded_dequantization:
                    real_sequences = add_bounded_dequantization(
                        real_sequences,
                        self.bounded_dequantization,
                        self.actuator_idx
                    )

                # ==================================
                # D training
                # ==================================

                set_requires_grad(self.discriminator, True)

                z = torch.randn(batch_size, *self.z_shape, device=self.device)
                loss_D, D_G_z1, D_x = self.discriminator_step(optimizerD, real_sequences, z)

                # ==================================
                # G training
                # ==================================

                set_requires_grad(self.discriminator, False)
                # setting default values in case generator doesn't run in this iteration
                loss_G = history_G_losses[-1] if len(history_G_losses) > 0 else 0.
                D_G_z2 = D_G_z1

                if (i + 1) & self.discriminator_rounds == 0:
                    for _ in range(self.generator_rounds):
                        z = torch.randn(batch_size, *self.z_shape, device=self.device)
                        loss_G, D_G_z2 = self.generator_step(optimizerG, z)

                # output status
                if i % 50 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.train_loader)}]', end='\t')
                    print(f'Loss_D: {loss_D:.4f}', end='\t')
                    print(f'Loss_G: {loss_G:.4f}', end='\t')
                    print(f'D(x): {D_x:.4f}', end='\t')
                    print(f'D(G(z)): {(D_G_z1+D_G_z2)/2:.4f}')
                
                # save losses
                history_G_losses.append(loss_G)
                history_D_losses.append(loss_D)

            # save model checkpoints
            if model_save_frequency and (epoch+1) % model_save_frequency == 0:
                self.generator.eval()
                self.save_model_checkpoints(optimizerG, optimizerD, epoch+1)
                self.save_tsne(self.state_dir / 'tsne' / f'{epoch+1}.png', self.z_shape)
                self.save_sample_sequences(self.generator(static_z).detach().cpu().numpy(), epoch+1)
                self.generator.train()

        # saving final models
        if not model_save_frequency or (n_epochs) % model_save_frequency != 0:
            self.generator.eval()
            self.save_model_checkpoints(optimizerG, optimizerD, n_epochs)
            self.save_tsne(self.state_dir / 'tsne' / f'{n_epochs}.png', self.z_shape)
            self.save_sample_sequences(self.generator(static_z).detach().cpu().numpy(), n_epochs+1)
            self.generator.train()

        # save training history
        loss_history = {
            'g_losses': history_G_losses,
            'd_losses': history_D_losses
        }

        with open(self.state_dir / 'loss_history.json', 'w') as f:
            json.dump(loss_history, f)

    def invert_latent_vector(self, generator, target, z_shape, num_epochs, lr=0.01):
        generator.eval()
        self.set_requires_grad(generator, False)
        generator.to(self.device)
        target = target.to(self.device)

        batch_size = target.size(0)
        z = torch.randn(batch_size, *z_shape, device=self.device)
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            generated_sequence = generator(z)
            loss = criterion(generated_sequence, target)
            prior_loss = torch.mean(z ** 2)
            total_loss = loss+  0.1 * prior_loss
            total_loss.backward()
            optimizer.step()
        
        return z.detach()
    
    def test(self, test_loader, test_name):
        tests_dir = self.state_dir / 'tests' / test_name
        tests_dir.mkdir(parents=True, exist_ok=True)

        mse = nn.MSELoss(reduction='none')
        # mae = nn.L1Loss(reduction='none')
        reduction_dim = 1 if self.time_last else 2
        reconstruction_mse = []
        discriminator_predictions = []

        for X, _ in test_loader:
            X = X.to(self.device)
            z = torch.randn(X.shape[0], *self.z_shape, device=self.device)
            # get z

            reconstruction = self.generator(z)
            pred = self.generator(X)
            if self.loss == 'bce':
                pred = nn.functional.sigmoid(pred)

            reconstruction_mse.append(
                mse(reconstruction, X).mean(dim=reduction_dim),
            )
            discriminator_predictions.append(pred)

        mse_map = map_anomaly_score_to_sequence(
            reconstruction_mse,
            self.window_size,
            self.stride,
        )

        np.save(tests_dir / 'mse.npy', mse_map)

    def test_stats(self, test_loader, test_settings):
        stats_dir = self.state_dir / 'stats'
        stats_dir.mkdir(exist_ok=True)

        stats = {}

        data = []
        labels = []
        all_reconstructions = []
        all_predictions = []

        for backprogapation_steps in [10, 20, 30, 40, 50]:
            for X, y in self.train_loader:
                X = X.to(self.device)
                z = self.invert_latent_vector(
                    self.generator,
                    X,
                    self.z_shape,
                    backprogapation_steps
                )
                data.append(X)
                labels.append(y)
                all_reconstructions.append(self.generator(z))
                all_predictions.append(self.discriminator(X))

        # TODO
        # very basic setup


