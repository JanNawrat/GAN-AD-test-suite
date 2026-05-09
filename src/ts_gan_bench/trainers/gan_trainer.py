import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ts_gan_bench.trainers.base_trainer import BaseTrainer
from ts_gan_bench.utils import set_requires_grad, add_bounded_dequantization, map_anomaly_score_to_sequence
from ts_gan_bench.visualization import plot_tsne

class GANTrainer(BaseTrainer):
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
        # networks
        self.generator = generator
        self.discriminator = discriminator

        # optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=settings.model.lr_g,
            betas=settings.model.betas_g,
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=settings.model.lr_d,
            betas=settings.model.betas_d,
        )

        # training parameters
        self.loss = settings.model.loss
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.gp_weight = settings.model.gp_weight
        self.disc_real_label = settings.model.disc_real_label # used for label smoothing
        self.clip_grad_g = settings.model.clip_grad_g
        self.clip_grad_d = settings.model.clip_grad_d
        self.generator_rounds = settings.model.generator_rounds
        self.discriminator_rounds = settings.model.discriminator_rounds

        # data dims
        z_channels = settings.model.generator.in_dim
        z_time = settings.params.window_size
        self.z_shape = (z_channels, z_time) if self.time_last else (z_time, z_channels)

        # preparing directories
        for dir in ['weights/generator', 'weights/discriminator', 'weights/optimizer_g', 'weights/optimizer_d', 'tsne', 'samples', 'logs']:
            (self.state_dir / dir).mkdir(parents=True, exist_ok=True)

        # initializing tensorboard
        self.writer = SummaryWriter(log_dir=self.state_dir / 'logs')

    def train(self, n_epochs, save_freg, log_freq=100, print_freq=50, skip_plotting=False):
        print('Starting training loop...')
        global_step = 0
        for epoch in range(n_epochs):
            self.generator.train()
            self.discriminator.train()
            for i, data in enumerate(self.train_loader):
                (X,) = data
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
                if (i + 1) % self.discriminator_rounds == 0:
                    for _ in range(self.generator_rounds):
                        g_metrics = self._generator_step(real_sequences)
                
                # ==================================
                # Logging
                # ==================================

                if global_step % log_freq == 0:
                    self._log_metrics(global_step, g_metrics, d_metrics)
                global_step += 1

                if i % print_freq == 0:
                    self._print_metrics(epoch, n_epochs, i, g_metrics, d_metrics)

            # ==================================
            # Shceduled saving
            # ==================================

            if save_freg and (epoch+1) % save_freg == 0:
                self._save_checkpoint(epoch+1, skip_plotting=skip_plotting)
        
        # ==================================
        # Saving final checkpoint
        # ==================================

        if (not save_freg or (n_epochs) % save_freg != 0):
            self._save_checkpoint(n_epochs, skip_plotting=skip_plotting)

    def _discriminator_step(self, real_sequences):
        self.optimizer_g.zero_grad(set_to_none=True)

        batch_size = real_sequences.shape[0]
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_automatic_precision,
        ):
            z = torch.randn(batch_size, *self.z_shape, device=self.device)
            fake_sequences = self.generator(z).detach()
            combined_sequences = torch.concat(
                [fake_sequences, real_sequences],
                dim=0,
            )
            combined_predictions = self.discriminator(combined_sequences).view(-1)
            combined_labels = torch.cat([
                torch.zeros_like(combined_predictions[:batch_size]),
                torch.full_like(combined_predictions[batch_size:], self.disc_real_label)
            ], dim=0)
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
            'pred_real' : F.sigmoid(combined_predictions[batch_size:].detach().float()).mean().item(),
        }
    
    def _generator_step(self, real_sequences):
        self.optimizer_g.zero_grad(set_to_none=True)

        batch_size = real_sequences.shape[0]
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_automatic_precision,
        ):
            z = torch.randn(batch_size, *self.z_shape, device=self.device)
            fake_sequences = self.generator(z)
            predictions = self.discriminator(fake_sequences)
            real_labels = torch.ones_like(predictions)
            loss = self.criterion_bce(predictions, real_labels)

        loss.backward()
        if self.clip_grad_g:
            nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=self.clip_grad_g,
            )
        self.optimizer_g.step()
        return {
            'g_loss': loss.item(),
        }
    
    def _log_metrics(self, global_step, g_metrics, d_metrics):
        self.writer.add_scalar('Loss/Discriminator', d_metrics['d_loss'], global_step)
        self.writer.add_scalar('Loss/Generator', g_metrics['g_loss'], global_step)
        self.writer.add_scalar('Predictions/Fake', d_metrics['pred_fake'], global_step)
        self.writer.add_scalar('Predictions/Real', d_metrics['pred_real'], global_step)

    def _print_metrics(self, epoch, n_epochs, step, g_metrics, d_metrics):
        n_epochs = str(n_epochs)
        epoch = f'{epoch:0{len(n_epochs)}}'
        n_steps = str(len(self.train_loader))
        step = f'{step:0{len(n_steps)}}'
        print(f'[{epoch}/{n_epochs}][{step}/{n_steps}]', end=', ')
        print(f'Loss_D: {d_metrics['d_loss']:.4f}', end=', ')
        print(f'Loss_AE: {g_metrics['g_loss']:.4f}', end=', ')
        print(f'D(X): {d_metrics['pred_real']:.4f}', end=', ')
        print(f'D(G(X)): {d_metrics['pred_fake']:.4f}')

    def _save_checkpoint(self, epoch, n_tsne=1000, skip_plotting=False):
        # saving states
        self.generator.save(self.state_dir / f'weights/generator/{epoch}.pth')
        self.discriminator.save(self.state_dir / f'weights/discriminator/{epoch}.pth')
        torch.save(self.optimizer_g.state_dict(), self.state_dir / f'weights/optimizer_g/{epoch}.pth')
        torch.save(self.optimizer_d.state_dict(), self.state_dir / f'weights/optimizer_d/{epoch}.pth')
        if skip_plotting:
            return
        # tsne TODO might move to logging instead
        self.generator.eval()
        real_samples = []
        fake_samples = []
        sample_count = 0
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_automatic_precision,
        ):
            for (X,) in self.train_loader:
                torch.compiler.cudagraph_mark_step_begin()

                real_samples.append(X.clone().cpu())
                z = torch.randn(X.shape[0], *self.z_shape, device=self.device)
                fake_samples.append(self.generator(z).detach().cpu().float())
                sample_count += X.shape[0]
                if sample_count >= n_tsne:
                    break
        real_samples = (torch.cat(real_samples, dim=0)[:n_tsne])
        fake_samples = torch.cat(fake_samples, dim=0)[:n_tsne]
        plot_tsne(
            torch.permute(real_samples, (0, 2, 1)),
            torch.permute(fake_samples, (0, 2, 1)),
            self.state_dir / f'tsne/{epoch}.png'
        )
        # sample sequences TODO
