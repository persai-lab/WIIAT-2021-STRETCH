import pickle
from mvtf.mf import MF
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import shutil
import logging
from mvtf.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


class MFAgent:
    def __init__(self, config):
        self.config = config
        np.random.seed(config.seed)
        self.logger = logging.getLogger("Agent")
        self.pred_type = config.pred_type
        self.n_users = config.n_users
        self.n_attempts = config.n_attempts
        self.n_items = config.n_items
        self.n_factors = config.n_factors
        self.train_data = config.train_data
        self.val_data = None
        self.test_data = config.test_data
        self.model = MF(self.n_users, self.n_attempts, self.n_items, self.n_factors,
                        seed=config.seed)
        self.mse_loss = nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=config.learning_rate,
        #                                  weight_decay=config.weight_decay)
        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.SGD([
            {"params": self.model.user_factors.weight},
            {"params": self.model.item_factors.weight},
            {"params": self.model.time_factors.weight},
            {"params": self.model.user_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.time_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.item_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.stress_item_biases.weight, "lr": config.bias_learning_rate},
        ], lr=config.learning_rate, weight_decay=config.weight_decay)
        self.current_epoch = 0
        self.train_loss = 0
        self.val_loss = 0
        self.val_loss_list = []
        self.test_loss = 0
        self.early_stop = False
        self.smooth_weight = config.smooth_weight

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        self.diff = []

    def run(self):
        all_test_mse = 0
        all_count = 0
        pred_list = []
        true_list = []
        for test_attempt in range(self.config.min_test_index, self.config.max_test_index + 1):
            self.early_stop = False
            test_loss = 0
            count = 0
            self.val_loss_list = []
            print("-" * 100)
            self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.2)
            prPurple("train: {}, val: {}".format(len(self.train_data), len(self.val_data)))
            for epoch in range(1, self.config.max_epoch + 1):
                print("Test Attempt: {} Epoch: {}".format(test_attempt, epoch))
                self.train_one_epoch()
                self.validate()
                self.current_epoch += 1
                if epoch >= self.config.min_epoch and self.early_stop:
                    print("stop epoch at {}".format(epoch))
                    break
            self.train_data += self.val_data

            self.model.eval()
            with torch.no_grad():
                for (u, t, i, v) in self.config.test_data:
                    if t == test_attempt:
                        if self.pred_type == "stress":
                            view = torch.tensor(1).long()
                        elif self.pred_type == "rate":
                            view = torch.tensor(2).long()
                        elif self.pred_type == "done":
                            view = torch.tensor(3).long()
                        else:
                            raise ValueError
                        count += 1
                        user = torch.Tensor([u]).long()
                        attempt = torch.Tensor([t]).long()
                        item = torch.Tensor([i]).long()
                        value = torch.Tensor([v])
                        pred = self.model(user, attempt, item, view)
                        if self.pred_type in ["stress", "rate"]:
                            loss = self.mse_loss(pred, value)
                        else:
                            loss = self.bce_loss(pred, value)
                        test_loss += loss.item()
                        pred_list.append(pred)
                        true_list.append(value)
                        print("pred: {} true: {}, mse: {}".format(pred, value, loss))
                        self.train_data.append([u, t, i, v])
            all_count += count
            all_test_mse += test_loss
            if count != 0:
                print("current test rmse: {}".format(np.sqrt(test_loss / count)))
        if self.pred_type in ["stress", "rate"]:
            self.metric1 = mean_squared_error(true_list, pred_list, squared=False)  # rmse
            self.metric2 = mean_absolute_error(true_list, pred_list)
        else:
            self.metric1 = roc_auc_score(true_list, pred_list)
            fpr, tpr, thresholds = roc_curve(true_list, pred_list)
            J = tpr - fpr
            idx = np.argmax(J)
            best_threshold = thresholds[idx]
            pred_list = np.array(pred_list)
            pred_list[pred_list > best_threshold] = 1
            pred_list[pred_list <= best_threshold] = 0
            self.metric2 = accuracy_score(true_list, pred_list)
        print("metric 1 {}, {}".format(self.metric1, np.sqrt(all_test_mse / all_count)))
        print("metric 2 {}".format(self.metric2))
        # print("difference: {}".format(self.diff[-1] - self.diff[0]))
        self.save_checkpoint(file_name="checkpoint_{}.pth.tar".format(self.config.data))
        return all_count, self.metric1, self.metric2

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        count = 0
        np.random.shuffle(self.train_data)

        for idx, (u, t, i, v) in enumerate(self.train_data):
            if self.pred_type == "stress":
                view = torch.tensor(1).long()
            elif self.pred_type == "rate":
                view = torch.tensor(2).long()
            elif self.pred_type == "done":
                view = torch.tensor(3).long()
            else:
                raise ValueError
            user = torch.Tensor([u]).long()
            attempt = torch.Tensor([t]).long()
            prev_attempt = torch.Tensor([t - 1]).long()
            next_attempt = torch.Tensor([t + 1]).long()
            item = torch.Tensor([i]).long()
            value = torch.Tensor([v])
            self.optimizer.zero_grad()
            pred = self.model(user, attempt, item, view)
            if self.pred_type in ["stress", "rate"]:
                loss = self.mse_loss(pred, value)
            else:
                loss = self.bce_loss(pred, value)
            if self.pred_type in ["stress", "done"]:
                # print("index {}".format(idx))
                # print(self.model.time_factors.weight[1:, :])
                # print(self.model.time_factors.weight[0:self.n_attempts - 1, :])
                # self.diff.append(self.model.user_factors.weight[user])
                # self.diff.append(self.model.time_factors.weight[3, :])
                loss += self.smooth_weight * torch.linalg.norm(
                    self.model.time_factors.weight[1:, :] -
                    self.model.time_factors.weight[0:self.n_attempts - 1, :]
                )
                # if prev_attempt >= 0:
                #     # print(self.model.time_factors.weight[prev_attempt])
                #     # print(self.model.time_factors(prev_attempt))
                #     # prev_time_factor = self.model.time_factors(prev_attempt)
                #     # curr_time_factor = self.model.time_factors(attempt)
                #     prev_time_factor = self.model.time_factors.weight[prev_attempt, :]
                #     curr_time_factor = self.model.time_factors.weight[attempt, :]
                #     loss += self.smooth_weight * torch.linalg.norm(
                #         prev_time_factor - curr_time_factor)
                #     # print(loss.item(), self.smooth_weight, l2_norm)
                #     # loss += self.smooth_weight * l2_norm
                # if next_attempt < self.n_attempts:
                #     # next_time_factor = self.model.time_factors(next_attempt)
                #     # curr_time_factor = self.model.time_factors(attempt)
                #     next_time_factor = self.model.time_factors.weight[next_attempt, :]
                #     curr_time_factor = self.model.time_factors.weight[attempt, :]
                #     # l2_norm = torch.linalg.norm(next_time_factor - curr_time_factor)
                #     # loss += self.smooth_weight * l2_norm
                #     loss += self.smooth_weight * torch.linalg.norm(
                #         next_time_factor - curr_time_factor)
            train_loss += loss.item()
            count += 1
            # print("pred: {} rate: {}, loss: {}".format(pred, rate, loss))
            loss.backward()
            # self.diff.append(self.model.time_factors.weight.clone())
            # self.diff.append(self.model.user_factors.weight)
            # print("grad: {}".format(self.model.time_factors.weight))
            self.optimizer.step()
        self.train_loss = np.sqrt(train_loss / count)
        print("average train loss: {}".format(self.train_loss))

    def validate(self):
        self.model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for idx, (u, t, i, v) in enumerate(self.val_data):
                if self.pred_type == "stress":
                    view = torch.tensor(1).long()
                elif self.pred_type == "rate":
                    view = torch.tensor(2).long()
                elif self.pred_type == "done":
                    view = torch.tensor(3).long()
                else:
                    raise ValueError
                user = torch.Tensor([u]).long()
                attempt = torch.Tensor([t]).long()
                item = torch.Tensor([i]).long()
                value = torch.Tensor([v])
                pred = self.model(user, attempt, item, view)
                if self.pred_type in ["stress", "rate"]:
                    val_loss += self.mse_loss(pred, value).item()
                else:
                    val_loss += self.bce_loss(pred, value).item()
                count += 1
            self.val_loss = np.sqrt(val_loss / count)
            print("average val loss: {}".format(self.val_loss))
            if len(self.val_loss_list) != 0 and self.val_loss > np.mean(self.val_loss_list[-3:]):
                self.early_stop = True
            else:
                self.val_loss_list.append(self.val_loss)

    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"Checkpoint loaded successfully from '{self.config.checkpoint_dir}' "
                             f"at (epoch {checkpoint['epoch']}\n")
        except OSError as e:
            self.logger.info(f"No checkpoint exists from '{self.config.checkpoint_dir}'. "
                             f"Skipping...")
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is
            the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')
