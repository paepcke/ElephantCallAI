from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from typing import Dict
import numpy as np

from gun_data.utils.GunshotDataUtils import compute_acc
from gun_data.training.GunshotTrainingSettings import GunshotTrainingSettings
from gun_data.DataMaker import CLASS_LABELS
from gun_data.utils import PerformanceMetrics


# trainer object for a *discriminative* gunshot model
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

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.training_settings.lr_step_size,
                                                       gamma=self.training_settings.lr_decay_factor)

        loss_func = nn.CrossEntropyLoss()  # TODO: consider using focal loss as well

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

        all_labels = np.zeros((len(self.val_loader.dataset),))

        all_predictions = np.zeros_like(all_labels)

        for batch in self.val_loader:
            with torch.no_grad():
                data, labels = batch[0], batch[1]
                data = data.to(self.training_settings.device)
                labels = labels.to(self.training_settings.device)

                logits = self.model.forward(data)

                loss = loss_func(logits, labels)

                cumu_loss += len(labels) * loss.item()

                all_predictions[num_examples:(num_examples + len(labels))] = np.argmax(logits.detach().cpu().numpy(),
                                                                                       axis=-1)
                all_labels[num_examples:(num_examples + len(labels))] = labels.detach().cpu().numpy()

                num_examples += len(labels)

        val_metrics = {}

        all_class_acc = PerformanceMetrics.all_class_accuracy(all_predictions, all_labels)

        self.tbx_writer.add_scalar("val/loss", cumu_loss/num_examples, log_index)
        val_metrics["val/loss"] = cumu_loss/num_examples
        self.tbx_writer.add_scalar("val/all_class_acc", all_class_acc, log_index)
        val_metrics["val/all_class_acc"] = all_class_acc

        for label, classname in CLASS_LABELS.items():
            metric_prefix = f"val/{classname}"

            precision = PerformanceMetrics.precision(all_predictions, all_labels, label)
            self.tbx_writer.add_scalar(f"{metric_prefix}/precision", precision, log_index)
            val_metrics[f"{metric_prefix}/precision"] = precision

            recall = PerformanceMetrics.recall(all_predictions, all_labels, label)
            self.tbx_writer.add_scalar(f"{metric_prefix}/recall", recall, log_index)
            val_metrics[f"{metric_prefix}/recall"] = recall

            f1 = 2*precision*recall/(precision + recall)
            self.tbx_writer.add_scalar(f"{metric_prefix}/f1_score", f1, log_index)
            val_metrics[f"{metric_prefix}/f1_score"] = recall

            class_acc = PerformanceMetrics.class_accuracy(all_predictions, all_labels, label)
            self.tbx_writer.add_scalar(f"{metric_prefix}/class_accuracy", class_acc, log_index)
            val_metrics[f"{metric_prefix}/class_acc"] = class_acc

        return val_metrics

    def val_metrics_as_string(self, val_metrics: Dict[str, float]) -> str:
        # sort keys to enforce same order across different invocations
        out = ""
        for key in sorted(val_metrics.keys()):
            out = out + f"{key}: {round(val_metrics[key], 5)}     "
        return out



