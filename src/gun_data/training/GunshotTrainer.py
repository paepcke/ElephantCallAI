from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from typing import Dict
import numpy as np

from gun_data.utils.GunshotDataUtils import compute_acc, sigmoid
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
    two_class_setting: bool

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader,
                 tbx_writer: SummaryWriter, model: nn.Module, training_settings: GunshotTrainingSettings):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tbx_writer = tbx_writer
        self.model = model
        self.training_settings = training_settings
        self.two_class_setting = not self.train_loader.dataset.use_rapidfire_category

    def train(self, max_epochs: int) -> nn.Module:
        self.model.to(self.training_settings.device)

        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_settings.initial_lr,
                                     weight_decay=self.training_settings.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.training_settings.lr_step_size,
                                                       gamma=self.training_settings.lr_decay_factor)

        if self.two_class_setting:
            # use BCEWithLogitsLoss in 2-class case
            loss_func = nn.BCEWithLogitsLoss()
        else:
            loss_func = nn.CrossEntropyLoss()  # TODO: consider using focal loss as well

        best_val_metric: float  # the single scalar metric that dictates early stopping behavior
        optimizing_val_metric_name: str  # the name of the above metric in the val_metrics dictionary
        if self.two_class_setting:
            # use f1 score in two class setting
            optimizing_val_metric_name = "val/f1_score"
            best_val_metric = 0.
        else:
            # use loss in multiclass setting
            optimizing_val_metric_name = "val/loss"
            best_val_metric = float("inf")

        def compare_optimizing_metric(best: float, current: float) -> bool:
            """
            This function is needed because we may try to minimize loss or maximize f1 score depending on which
            metric we're using.

            :param best: the best value of the metric seen so far
            :param current: the current value of the metric
            :return: True if the current value of the metric is the best seen so far
            """
            if self.two_class_setting:
                return current > best
            else:
                return current < best

        # all of the val metrics associated with the 'best' model will be stored in best_val_metrics as a Dict[str, float]
        best_val_metrics = None
        epochs_since_best_val_metric = 0

        for epoch in range(max_epochs):
            # log learning rate to tbx
            self.tbx_writer.add_scalar("lr/lr", lr_scheduler.get_lr()[0], epoch * len(self.train_loader))

            num_batches_this_epoch = 0
            train_examples_this_reporting_cycle = 0
            running_loss = 0
            running_train_acc = 0

            if self.two_class_setting:
                running_train_true_pos = 0
                running_train_false_pos = 0
                running_train_false_neg = 0

            for batch in self.train_loader:
                optimizer.zero_grad()
                data, labels = batch[0], batch[1]
                data = data.to(self.training_settings.device)
                labels = labels.to(self.training_settings.device)

                logits = self.model.forward(data)

                loss = loss_func(logits, labels)

                running_loss += len(labels)*loss.item()
                running_train_acc += len(labels)*compute_acc(labels.detach().cpu().numpy(), logits.detach().cpu().numpy())

                if self.two_class_setting:
                    # collect additional training metrics for 2-class setting
                    train_predictions = np.round(sigmoid(logits.detach().cpu().numpy()))

                    # use '1' as target class label to obtain precision and recall wrt actual gunshots
                    running_train_true_pos += PerformanceMetrics.num_true_positives(train_predictions,
                                                                            labels.detach().cpu().numpy())
                    running_train_false_pos += PerformanceMetrics.num_false_positives(train_predictions,
                                                                      labels.detach().cpu().numpy())
                    running_train_false_neg += PerformanceMetrics.num_false_negatives(train_predictions,
                                                                                      labels.detach().cpu().numpy())

                loss.backward()
                optimizer.step()

                num_batches_this_epoch += 1
                train_examples_this_reporting_cycle += len(labels)

                total_batches_so_far = epoch * len(self.train_loader) + num_batches_this_epoch
                if num_batches_this_epoch % self.training_settings.train_log_frequency == 0:
                    # log metrics about the training set to tbx
                    self.log_train_metrics(running_loss, running_train_acc,
                                           train_examples_this_reporting_cycle, total_batches_so_far)

                    if self.two_class_setting:
                        # log additional metrics
                        self.log_additional_train_metrics_for_two_class_setting(running_train_true_pos,
                                                                                running_train_false_pos,
                                                                                running_train_false_neg,
                                                                                train_examples_this_reporting_cycle,
                                                                                total_batches_so_far)

                        running_train_true_pos = 0
                        running_train_false_pos = 0
                        running_train_false_neg = 0

                    running_loss = 0
                    running_train_acc = 0
                    train_examples_this_reporting_cycle = 0
                if num_batches_this_epoch % self.training_settings.val_log_frequency == 0:
                    self.log_val_metrics(total_batches_so_far)

            # End of epoch
            total_batches_so_far = (epoch + 1) * len(self.train_loader)

            self.log_train_metrics(running_loss, running_train_acc,
                                   train_examples_this_reporting_cycle, total_batches_so_far)
            if self.two_class_setting:
                # log additional metrics
                self.log_additional_train_metrics_for_two_class_setting(running_train_true_pos,
                                                                        running_train_false_pos,
                                                                        running_train_false_neg,
                                                                        train_examples_this_reporting_cycle,
                                                                        total_batches_so_far)

            val_metrics = self.log_val_metrics(total_batches_so_far)
            print(f"End of epoch {epoch + 1} of {max_epochs}.   {self.val_metrics_as_string(val_metrics)}")

            cur_opt_metric = val_metrics[optimizing_val_metric_name]
            if compare_optimizing_metric(best_val_metric, cur_opt_metric):
                best_val_metric = cur_opt_metric
                best_val_metrics = val_metrics
                torch.save(self.model.state_dict(), f"{self.training_settings.weights_out_dir}/best.pt")
                epochs_since_best_val_metric = 0
            else:
                # Early stopping
                epochs_since_best_val_metric += 1
                if epochs_since_best_val_metric >= self.training_settings.max_epochs_since_best_val:
                    print(f"Target validation metric has not reached its best value so far ({best_val_metric})" +
                          f" for {epochs_since_best_val_metric} epochs," +
                          " invoking early stopping and completing training process.")
                    print(f"Best validation metrics for this training run are   {self.val_metrics_as_string(best_val_metrics)}")
                    return self.model

            # step LR scheduler
            lr_scheduler.step()
        # end of training
        print(f"Training complete! Best validation metrics are   {self.val_metrics_as_string(best_val_metrics)}")
        return self.model

    def log_train_metrics(self, running_loss: float, running_train_acc: float,
                          train_examples_this_reporting_cycle: int, total_batches_so_far: int):
        if train_examples_this_reporting_cycle == 0:
            # skip logging if there is no new information to report
            return
        self.tbx_writer.add_scalar("train/loss", running_loss / train_examples_this_reporting_cycle,
                                   total_batches_so_far)
        self.tbx_writer.add_scalar("train/acc", running_train_acc / train_examples_this_reporting_cycle,
                                   total_batches_so_far)

    def log_additional_train_metrics_for_two_class_setting(self, running_train_true_pos: int,
                                                           running_train_false_pos: int,
                                                           running_train_false_neg: int,
                                                           train_examples_this_reporting_cycle: int,
                                                           total_batches_so_far: int):
        if train_examples_this_reporting_cycle == 0:
            # skip logging if there is no new information to report
            return
        train_precision = running_train_true_pos/(running_train_true_pos + running_train_false_pos)
        train_recall = running_train_true_pos / (running_train_true_pos + running_train_false_neg)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall)

        self.tbx_writer.add_scalar("train/precision", train_precision,
                                   total_batches_so_far)
        self.tbx_writer.add_scalar("train/recall", train_recall,
                                   total_batches_so_far)
        self.tbx_writer.add_scalar("train/f1", train_f1,
                                   total_batches_so_far)

    def log_val_metrics(self, log_index: int) -> Dict[str, float]:
        num_examples = 0
        if self.two_class_setting:
            # use BCEWithLogitsLoss in 2-class case
            loss_func = nn.BCEWithLogitsLoss()
        else:
            loss_func = nn.CrossEntropyLoss()  # TODO: consider using focal loss as well

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

                if not self.two_class_setting:
                    all_predictions[num_examples:(num_examples + len(labels))] = \
                        np.argmax(logits.detach().cpu().numpy(), axis=-1)
                else:
                    # predictions for 2-class setting
                    all_predictions[num_examples:(num_examples + len(labels))] = \
                        np.squeeze(np.round(sigmoid(logits.detach().cpu().numpy())))

                all_labels[num_examples:(num_examples + len(labels))] = np.squeeze(labels.detach().cpu().numpy())

                num_examples += len(labels)

        val_metrics = {}

        all_class_acc = PerformanceMetrics.all_class_accuracy(all_predictions, all_labels)

        self.tbx_writer.add_scalar("val/loss", cumu_loss/num_examples, log_index)
        val_metrics["val/loss"] = cumu_loss/num_examples
        self.tbx_writer.add_scalar("val/all_class_acc", all_class_acc, log_index)
        val_metrics["val/all_class_acc"] = all_class_acc

        if not self.two_class_setting:
            for label, classname in CLASS_LABELS.items():
                metric_prefix = f"val/{classname}"

                self.populate_val_metrics(val_metrics, metric_prefix, log_index,
                                     all_predictions, all_labels, target_class_label=label)
        else:
            # val metrics for 2-class setting
            metric_prefix = "val"

            # use target_class_label = 1 to specify gunshot examples
            self.populate_val_metrics(val_metrics, metric_prefix, log_index,
                                      all_predictions, all_labels, target_class_label=1)

        return val_metrics

    def populate_val_metrics(self, val_metrics: Dict[str, float], metric_prefix: str, log_index: int,
                             all_predictions: np.ndarray, all_labels: np.ndarray, target_class_label: int):
        """
        populates the input 'val_metrics' dictionary and reports populated values to Tensorboard

        :return: the updated dictionary val_metrics
        """
        precision = PerformanceMetrics.precision(all_predictions, all_labels, target_class_label)
        self.tbx_writer.add_scalar(f"{metric_prefix}/precision", precision, log_index)
        val_metrics[f"{metric_prefix}/precision"] = precision

        recall = PerformanceMetrics.recall(all_predictions, all_labels, target_class_label)
        self.tbx_writer.add_scalar(f"{metric_prefix}/recall", recall, log_index)
        val_metrics[f"{metric_prefix}/recall"] = recall

        f1 = 2 * precision * recall / (precision + recall)
        self.tbx_writer.add_scalar(f"{metric_prefix}/f1_score", f1, log_index)
        val_metrics[f"{metric_prefix}/f1_score"] = recall

    def val_metrics_as_string(self, val_metrics: Dict[str, float]) -> str:
        # sort keys to enforce same order across different invocations
        out = ""
        for key in sorted(val_metrics.keys()):
            out = out + f"{key}: {round(val_metrics[key], 5)}     "
        return out



