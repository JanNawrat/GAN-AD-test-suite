import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

class BaseTrainer():
    def __init__(
            self,
            settings,
            train_loader,
            feature_names=None,
            actuator_idx=None,
        ):
        self.device = torch.device(settings.device_name)
        self.state_dir = settings.paths.state_dir
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