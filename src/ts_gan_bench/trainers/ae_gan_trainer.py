import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ts_gan_bench.trainers.base_trainer import BaseTrainer
from ts_gan_bench.utils import set_requires_grad, add_bounded_dequantization
from ts_gan_bench.visualization import plot_tsne

class AEGANTrainer(BaseTrainer):
    def __init__(
            self,
            settings,
            encoder,
            decoder,
            discriminator,
            train_loader,
            feature_names=None,
            actuator_idx=None,
        ):
        super().__init__(settings, train_loader, feature_names, actuator_idx)
        # networks
        if settings.params.compile_models:
            self.encoder = torch.compile(
                encoder,
                mode=settings.params.compilation_mode,
            )
            self.decoder = torch.compile(
                decoder,
                mode=settings.params.compilation_mode,
            )
            self.discriminator = torch.compile(
                discriminator,
                mode=settings.params.compilation_mode,
            )
        else:
            self.encoder = encoder
            self.decoder = decoder
            self.discriminator = discriminator

        # optimizers
        self.optimizer_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=settings.model.lr_g,
            betas=settings.model.betas_g,
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=settings.model.lr_d,
            betas=settings.model.betas_d,
        )

        # training parameters
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_mse = nn.MSELoss()
        self.adversarial_weight = settings.model.adversarial_loss_weight
        self.reconstruction_weight = settings.model.reconstruction_loss_weight
        self.disc_real_label = settings.model.disc_real_label # used for label smoothing
        self.clip_grad_g = settings.model.clip_grad_g
        self.clip_grad_d = settings.model.clip_grad_d
        self.generator_rounds = settings.model.generator_rounds
        self.discriminator_rounds = settings.model.discriminator_rounds

        # data dims (might not use them in AE)
        z_channels = settings.model.decoder.in_dim
        z_time = settings.params.window_size
        self.z_shape = (z_channels, z_time) if self.time_last else (z_time, z_channels)

        # preparing directories
        for dir in ['weights/encoder', 'weights/decoder', 'weights/discriminator', 'weights/optimizer_ae', 'weights/optimizer_d', 'tsne', 'samples', 'logs']:
            (self.state_dir / dir).mkdir(parents=True, exist_ok=True)

        # initializing tensorboard
        self.writer = SummaryWriter(log_dir=self.state_dir / 'logs')

    def train(self, n_epochs, save_freq, log_freq=100, print_freq=50, skip_saving=False):
        print("Starting training loop...")
        global_step = 0
        for epoch in range(n_epochs):
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
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
                d_metrics = self._discriminator_step(real_sequences)

                # ==================================
                # G training
                # ==================================

                set_requires_grad(self.discriminator, False)
                # setting default values in case generator doesn't run in this iteration
                # loss_G = history_G_losses[-1] if len(history_G_losses) > 0 else 0.
                # D_G_z2 = D_G_z1

                if (i + 1) % self.discriminator_rounds == 0:
                    for _ in range(self.generator_rounds):
                        ae_metrics = self._autoencoder_step(real_sequences)

                # ==================================
                # Logging
                # ==================================

                if global_step % log_freq == 0:
                    self._log_metrics(global_step, ae_metrics, d_metrics)
                global_step += 1

                if i % print_freq == 0:
                    self._print_metrics(epoch, n_epochs, i, ae_metrics, d_metrics)

            # ==================================
            # Scheduled saving
            # ==================================

            if not skip_saving and save_freq and (epoch+1) % save_freq == 0:
                self._save_checkpoint(epoch+1)
                
        # ==================================
        # Saving final checkpoint
        # ==================================

        if not skip_saving and (not save_freq or (n_epochs) % save_freq != 0):
            self._save_checkpoint(n_epochs)

    def _discriminator_step(self, real_sequences):
        self.optimizer_d.zero_grad(set_to_none=True)

        batch_size = real_sequences.shape[0]
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_automatic_precision,
        ):
            fake_sequences = self.decoder(self.encoder(real_sequences)).detach()
            combined_sequences = torch.concat(
                [fake_sequences, real_sequences],
                dim=0,
            )
            combined_predictions = self.discriminator(combined_sequences).view(-1)
            fake_labels = torch.zeros_like(combined_predictions[:batch_size])
            real_labels = torch.full_like(combined_predictions[batch_size:], self.disc_real_label)
            combined_labels = torch.concat(
                [fake_labels, real_labels],
                dim=0,
            )
            loss = self.criterion_bce(combined_predictions, combined_labels)

        loss.backward()
        if self.clip_grad_d:
            nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.clip_grad_d,
            )
        self.optimizer_d.step()
        return {
            'd_loss': loss.item(),
            'pred_fake': F.sigmoid(combined_predictions[:batch_size].detach().float()).mean().item(),
            'pred_real': F.sigmoid(combined_predictions[batch_size:].detach().float()).mean().item(),
        }

    def _autoencoder_step(self, real_sequences):
        self.optimizer_ae.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_automatic_precision,
        ):
            fake_sequences = self.decoder(self.encoder(real_sequences))
            reconstruction_loss = self.criterion_mse(fake_sequences, real_sequences)
            predictions = self.discriminator(fake_sequences)
            real_labels = torch.ones_like(predictions)
            adversarial_loss = self.criterion_bce(predictions, real_labels)
            # adversarial_loss = -nn.functional.logsigmoid(predictions).mean()
            total_loss = (
                reconstruction_loss * self.reconstruction_weight +
                adversarial_loss * self.adversarial_weight
            )
        
        total_loss.backward()
        if self.clip_grad_g:
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=self.clip_grad_g,
            )
        self.optimizer_ae.step()
        return {
            'total_loss': total_loss.item(),
            'raw_rec': reconstruction_loss.item(),
            'raw_adv': adversarial_loss.item(),
            'weighted_rec': reconstruction_loss.item() * self.reconstruction_weight,
            'weighted_adv': adversarial_loss.item() * self.adversarial_weight,
        }
    
    def _log_metrics(self, global_step, ae_metrics, d_metrics):
        self.writer.add_scalar('Loss/Discriminator', d_metrics['d_loss'], global_step)
        self.writer.add_scalar('Loss/Autoencoder', ae_metrics['total_loss'], global_step)
        self.writer.add_scalar('Loss/Reconstruction(Raw)', ae_metrics['raw_rec'], global_step)
        self.writer.add_scalar('Loss/Adversarial(Raw)', ae_metrics['raw_adv'], global_step)
        self.writer.add_scalar('Loss/Adversarial(Weighted)', ae_metrics['weighted_adv'], global_step)
        self.writer.add_scalar('Loss/Reconstruction(Weighted)', ae_metrics['weighted_rec'], global_step)
        self.writer.add_scalar('Predictions/Fake', d_metrics['pred_fake'], global_step)
        self.writer.add_scalar('Predictions/Real', d_metrics['pred_real'], global_step)

    def _print_metrics(self, epoch, n_epochs, step, ae_metrics, d_metrics):
        n_epochs = str(n_epochs)
        epoch = f'{epoch:0{len(n_epochs)}}'
        n_steps = str(len(self.train_loader))
        step = f'{step:0{len(n_steps)}}'
        print(f'[{epoch}/{n_epochs}][{step}/{n_steps}]', end=', ')
        print(f'Loss_D: {d_metrics['d_loss']:.4f}', end=', ')
        print(f'Loss_AE: {ae_metrics['total_loss']:.4f}', end=', ')
        print(f'Loss_adv: {ae_metrics['raw_adv']:.4f}', end=', ')
        print(f'Loss_rec: {ae_metrics['raw_rec']:.4f}', end=', ')
        print(f'Loss_adv(w): {ae_metrics['weighted_adv']:.4f}', end=', ')
        print(f'Loss_rec(w): {ae_metrics['weighted_rec']:.4f}', end=', ')
        print(f'D(X): {d_metrics['pred_real']:.4f}', end=', ')
        print(f'D(AE(X)): {d_metrics['pred_fake']:.4f}')

    def _save_checkpoint(self, epoch, n_tsne=1000):
        # saving states
        self.encoder.save(self.state_dir / f'weights/encoder/{epoch}.pth')
        self.decoder.save(self.state_dir / f'weights/decoder/{epoch}.pth')
        self.discriminator.save(self.state_dir / f'weights/discriminator/{epoch}.pth')
        torch.save(self.optimizer_ae.state_dict(), self.state_dir / f'weights/optimizer_ae/{epoch}.pth')
        torch.save(self.optimizer_d.state_dict(), self.state_dir / f'weights/optimizer_d/{epoch}.pth')
        # tsne TODO might move to logging instead
        self.encoder.eval()
        self.decoder.eval()
        real_samples = []
        fake_samples = []
        sample_count = 0
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_automatic_precision,
        ):
            for X, _ in self.train_loader:
                torch.compiler.cudagraph_mark_step_begin()
                
                real_samples.append(X.clone().cpu())
                fake_samples.append(self.decoder(self.encoder(X.to(self.device))).detach().cpu().float())
                sample_count += X.shape[0]
                if sample_count >= n_tsne:
                    break
        real_samples = torch.cat(real_samples, dim=0)[:n_tsne]
        fake_samples = torch.cat(fake_samples, dim=0)[:n_tsne]
        plot_tsne(
            torch.permute(real_samples, (0, 2, 1)),
            torch.permute(fake_samples, (0, 2, 1)),
            self.state_dir / f'tsne/{epoch}.png'
        )
        # sample sequences TODO