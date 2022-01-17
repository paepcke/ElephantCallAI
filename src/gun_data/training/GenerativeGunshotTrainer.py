from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from typing import Dict, Callable

from gun_data.training.GunshotTrainingSettings import GunshotTrainingSettings


# trainer object for a *generative* gunshot model
class GenerativeGunshotTrainer:
    train_loader: DataLoader
    val_loader: DataLoader
    tbx_writer: SummaryWriter
    model: nn.Module
    training_settings: GunshotTrainingSettings

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader,
                 tbx_writer: SummaryWriter, model: nn.Module,
                 training_settings: GunshotTrainingSettings):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tbx_writer = tbx_writer
        self.model = model
        self.training_settings = training_settings

    def train(self, max_epochs: int) -> nn.Module:
        self.model.to(self.training_settings.device)

        self.model.train()

        # TODO: use weight decay?
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_settings.initial_lr,
                                     weight_decay=0.)  # for now, don't use weight decay

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.training_settings.lr_step_size,
                                                       gamma=self.training_settings.lr_decay_factor)

        best_val_loss = float("inf")
        best_val_metrics = None
        epochs_since_best_val_loss = 0

        for epoch in range(max_epochs):
            # log learning rate to tbx
            self.tbx_writer.add_scalar("lr/lr", lr_scheduler.get_lr()[0], epoch * len(self.train_loader))

            num_batches_this_epoch = 0
            train_examples_this_reporting_cycle = 0
            running_loss = 0

            for batch in self.train_loader:
                optimizer.zero_grad()
                data, time_len = batch[0], batch[2]
                data = data.to(self.training_settings.device)
                time_len = time_len.to(self.training_settings.device)

                logits = self.model.forward(data[:, :-1, :], time_len)

                loss = self.model.loss(data, time_len, logits)

                running_loss += len(data)*loss.item()

                loss.backward()
                optimizer.step()

                num_batches_this_epoch += 1
                train_examples_this_reporting_cycle += len(data)

                total_batches_so_far = epoch * len(self.train_loader) + num_batches_this_epoch
                if num_batches_this_epoch % self.training_settings.train_log_frequency == 0:
                    # log metrics about the training set to tbx
                    self.tbx_writer.add_scalar("train/loss", running_loss/train_examples_this_reporting_cycle, total_batches_so_far)

                    # TODO: consider displaying a few spectrograms to tensorboard?

                    running_loss = 0
                    train_examples_this_reporting_cycle = 0
                if num_batches_this_epoch % self.training_settings.val_log_frequency == 0:
                    self.log_val_metrics(total_batches_so_far)

            # End of epoch
            total_batches_so_far = (epoch + 1) * len(self.train_loader)

            if train_examples_this_reporting_cycle > 0:
                # Only log training metrics if there has been at least one batch since the last time we logged them
                self.tbx_writer.add_scalar("train/loss", running_loss / train_examples_this_reporting_cycle,
                                           total_batches_so_far)

            val_metrics = self.log_val_metrics(total_batches_so_far)
            print(f"End of epoch {epoch + 1} of {max_epochs}.   {self.val_metrics_as_string(val_metrics)}")
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                best_val_metrics = val_metrics
                torch.save(self.model, f"{self.training_settings.weights_out_dir}/best.pt")
                epochs_since_best_val_loss = 0
            else:
                # Early stopping
                epochs_since_best_val_loss += 1
                if epochs_since_best_val_loss >= self.training_settings.max_epochs_since_best_val:
                    print(f"Target validation metric has not reached its best value so far ({best_val_loss})" +
                          f" for {epochs_since_best_val_loss} epochs," +
                          " invoking early stopping and completing training process.")
                    print(f"Best validation metrics for this training run are   {self.val_metrics_as_string(best_val_metrics)}")
                    torch.save(self.model, f"{self.training_settings.weights_out_dir}/last.pt")
                    return self.model

            # step LR scheduler
            lr_scheduler.step()
        # end of training
        print(f"Training complete! Best validation metrics are   {self.val_metrics_as_string(best_val_metrics)}")
        torch.save(self.model, f"{self.training_settings.weights_out_dir}/last.pt")
        return self.model

    def log_val_metrics(self, log_index: int) -> Dict[str, float]:
        num_examples = 0

        cumu_loss = 0

        for batch in self.val_loader:
            with torch.no_grad():
                data, time_len = batch[0], batch[2]
                data = data.to(self.training_settings.device)
                time_len = time_len.to(self.training_settings.device)

                logits = self.model.forward(data[:, :-1, :], time_len)

                loss = self.model.loss(data, time_len, logits)

                cumu_loss += len(data) * loss.item()

                num_examples += len(data)

        val_metrics = {}

        self.tbx_writer.add_scalar("val/loss", cumu_loss/num_examples, log_index)
        val_metrics["val/loss"] = cumu_loss/num_examples

        return val_metrics

    def val_metrics_as_string(self, val_metrics: Dict[str, float]) -> str:
        # sort keys to enforce same order across different invocations
        out = ""
        for key in sorted(val_metrics.keys()):
            out = out + f"{key}: {round(val_metrics[key], 5)}     "
        return out

    def eval(self, data_loader: DataLoader):
        self.model.to(self.training_settings.device)
        self.model.eval()

        running_loss = 0.
        num_examples = 0
        with torch.no_grad():
            for batch in data_loader:
                data, time_len = batch[0], batch[2]
                data = data.to(self.training_settings.device)
                time_len = time_len.to(self.training_settings.device)

                logits = self.model.forward(data[:, :-1, :], time_len)

                loss = self.model.loss(data, time_len, logits)

                running_loss += len(data)*loss.item()
                num_examples += len(data)

        avg_loss = running_loss/num_examples

        return avg_loss



