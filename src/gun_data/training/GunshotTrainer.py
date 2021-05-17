from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from typing import Dict

from gun_data.GunshotDataUtils import compute_acc
from gun_data.training.GunshotTrainingSettings import GunshotTrainingSettings


class GunshotTrainer:
    train_loader: DataLoader
    val_loader: DataLoader
    tbx_writer: SummaryWriter
    model: nn.Module
    training_settings: GunshotTrainingSettings

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader,
                 tbx_writer: SummaryWriter, model: nn.Module, training_settings: GunshotTrainingSettings):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tbx_writer = tbx_writer
        self.model = model
        self.training_settings = training_settings

    def train(self, max_epochs: int) -> nn.Module:
        self.model.to(self.training_settings.device)

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_settings.initial_lr,
                                     weight_decay=self.training_settings.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        loss_func = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_val_metrics = None
        epochs_since_best_val_loss = 0

        for epoch in range(max_epochs):
            # log learning rate to tbx
            self.tbx_writer.add_scalar("lr/lr", lr_scheduler.get_lr()[0], epoch * len(self.train_loader))

            num_batches_this_epoch = 0
            train_examples_this_reporting_cycle = 0
            running_loss = 0
            running_train_acc = 0
            for batch in self.train_loader:
                optimizer.zero_grad()
                data, labels = batch[0], batch[1]
                data = data.to(self.training_settings.device)
                labels = labels.to(self.training_settings.device)

                logits = self.model.forward(data)

                loss = loss_func(logits, labels)

                running_loss += len(labels)*loss.item()
                running_train_acc += len(labels)*compute_acc(labels.detach().cpu().numpy(), logits.detach().cpu().numpy())

                loss.backward()
                optimizer.step()

                num_batches_this_epoch += 1
                train_examples_this_reporting_cycle += len(labels)

                total_batches_so_far = epoch * len(self.train_loader) + num_batches_this_epoch
                if num_batches_this_epoch % self.training_settings.train_log_frequency == 0:
                    # log metrics about the training set to tbx
                    self.tbx_writer.add_scalar("train/loss", running_loss/train_examples_this_reporting_cycle, total_batches_so_far)
                    self.tbx_writer.add_scalar("train/acc", running_train_acc/train_examples_this_reporting_cycle, total_batches_so_far)

                    running_loss = 0
                    running_train_acc = 0
                    train_examples_this_reporting_cycle = 0
                if num_batches_this_epoch % self.training_settings.val_log_frequency == 0:
                    self.log_val_metrics(total_batches_so_far)

            # End of epoch
            total_batches_so_far = (epoch + 1) * len(self.train_loader)

            if train_examples_this_reporting_cycle > 0:
                # Only log training metrics if there has been at least one batch since the last time we logged them
                self.tbx_writer.add_scalar("train/loss", running_loss / train_examples_this_reporting_cycle,
                                           total_batches_so_far)
                self.tbx_writer.add_scalar("train/acc", running_train_acc / train_examples_this_reporting_cycle,
                                           total_batches_so_far)

            val_metrics = self.log_val_metrics(total_batches_so_far)
            print(f"End of epoch {epoch + 1} of {max_epochs}.   {self.val_metrics_as_string(val_metrics)}")
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                best_val_metrics = val_metrics
                torch.save(self.model.state_dict(), f"{self.training_settings.weights_out_dir}/best.pt")
                epochs_since_best_val_loss = 0
            else:
                # Early stopping
                epochs_since_best_val_loss += 1
                if epochs_since_best_val_loss >= self.training_settings.max_epochs_since_best_val:
                    print(f"Target validation metric has not reached its best value so far ({best_val_loss})" +
                          f" for {epochs_since_best_val_loss} epochs," +
                          " invoking early stopping and completing training process.")
                    print(f"Best validation metrics for this training run are   {self.val_metrics_as_string(best_val_metrics)}")
                    return self.model

            # step LR scheduler
            lr_scheduler.step()
        # end of training
        print(f"Training complete! Best validation metrics are   {self.val_metrics_as_string(best_val_metrics)}")
        return self.model


    def log_val_metrics(self, log_index: int) -> Dict[str, float]:
        num_examples = 0
        loss_func = nn.CrossEntropyLoss()

        cumu_loss = 0
        cumu_acc = 0
        for batch in self.val_loader:
            with torch.no_grad():
                data, labels = batch[0], batch[1]
                data = data.to(self.training_settings.device)
                labels = labels.to(self.training_settings.device)

                logits = self.model.forward(data)

                loss = loss_func(logits, labels)

                cumu_loss += len(labels) * loss.item()
                cumu_acc += len(labels) * compute_acc(labels.detach().cpu().numpy(),
                                                               logits.detach().cpu().numpy())
                num_examples += len(labels)

        self.tbx_writer.add_scalar("val/loss", cumu_loss/num_examples, log_index)
        self.tbx_writer.add_scalar("val/acc", cumu_acc/num_examples, log_index)

        return {"val/loss": cumu_loss/num_examples, "val/acc": cumu_acc/num_examples}

    def val_metrics_as_string(self, val_metrics: Dict[str, float]) -> str:
        # sort keys to enforce same order across different invocations
        out = ""
        for key in sorted(val_metrics.keys()):
            out = out + f"{key}: {round(val_metrics[key], 5)}     "
        return out



