import torch


class GunshotTrainingSettings:
    """
    This object encapsulates various training settings.
    """
    device: torch.device
    initial_lr: float
    weight_decay: float
    weights_out_dir: str
    train_log_frequency: int  # log training metrics to tensorboard after this many batches
    val_log_frequency: int  # log val metrics to tensorboard after this many batches
    lr_decay_factor: float  # after lr_step_size epochs, lr = lr_decay_factor * lr
    lr_step_size: int  # after this many epochs, reduce lr
    max_epochs_since_best_val: int  # stop training if it has been this many epochs since the best validation loss

    def __init__(self, device: torch.device, initial_lr: float, weight_decay: float, weights_out_dir: str,
                 train_log_frequency: int, val_log_frequency: int, lr_decay_factor: float, lr_step_size: int,
                 max_epochs_since_best_val: int = 5):
        self.device = device
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.weights_out_dir = weights_out_dir
        self.train_log_frequency = train_log_frequency
        self.val_log_frequency = val_log_frequency
        self.lr_decay_factor = lr_decay_factor
        self.lr_step_size = lr_step_size
        self.max_epochs_since_best_val = max_epochs_since_best_val

