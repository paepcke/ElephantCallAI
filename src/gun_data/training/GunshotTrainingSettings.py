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
    max_epochs_since_best_val: int  # stop training if it has been this many epochs since the best validation loss

    def __init__(self, device: torch.device, initial_lr: float, weight_decay: float, weights_out_dir: str,
                 train_log_frequency: int, val_log_frequency: int,
                 max_epochs_since_best_val: int = 5):
        self.device = device
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.weights_out_dir = weights_out_dir
        self.train_log_frequency = train_log_frequency
        self.val_log_frequency = val_log_frequency
        self.max_epochs_since_best_val = max_epochs_since_best_val

