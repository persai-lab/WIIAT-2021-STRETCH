import pickle
from mvtf.mvtf_torch import MVTF_Torch
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import shutil
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from mvtf.constraint import WeightClipper


class MVTFAgent:
    def __init__(self, config):
        self.config = config
        np.random.seed(config.seed)
        # self.logger = logging.getLogger("Agent")
        self.pred_type = config.pred_type
        self.n_users = config.n_users
        self.n_attempts = config.n_attempts
        self.n_items = config.n_items
        self.n_factors = config.n_factors
        self.train_data = None
        self.val_data = None  # used for early stop not for hyperparameters tuning
        self.stress_train_data = config.stress_train_data
        self.rate_train_data = config.rate_train_data
        self.done_train_data = config.done_train_data
        self.test_data = config.test_data
        self.model = MVTF_Torch(self.n_users, self.n_attempts, self.n_items, self.n_factors,
                                seed=config.seed)
        self.clipper = WeightClipper()

        self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.SGD([
            {"params": self.model.user_factors.weight},
            {"params": self.model.time_factors.weight},
            {"params": self.model.item_factors.weight},
            {"params": self.model.time_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.stress_user_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.stress_item_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.rate_user_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.rate_item_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.done_user_biases.weight, "lr": config.bias_learning_rate},
            {"params": self.model.done_item_biases.weight, "lr": config.bias_learning_rate},
        ], lr=config.learning_rate, weight_decay=config.weight_decay)
        self.current_epoch = 0
        self.stress_train_loss = 0
        self.rate_train_loss = 0
        self.done_train_loss = 0
        self.val_loss = 0
        self.val_loss_list = []
        self.metric1 = 0
        self.metric2 = 0
        self.early_stop = False
        self.smooth_weight = config.smooth_weight

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

    def run(self):
        all_test_mse = 0
        all_count = 0
        pred_list = []
        true_list = []
        for test_attempt in range(self.config.min_test_index, self.config.max_test_index + 1):
            self.early_stop = False
            test_mse = 0
            count = 0
            self.val_loss_list = []
            print("-" * 100)

            if self.pred_type == "rate":
                self.rate_train_data, self.val_data = train_test_split(
                    self.rate_train_data, test_size=0.2)
            elif self.pred_type == "stress":
                self.stress_train_data, self.val_data = train_test_split(
                    self.stress_train_data, test_size=0.2)
            elif self.pred_type == "done":
                self.done_train_data, self.val_data = train_test_split(
                    self.done_train_data, test_size=0.2)
            else:
                raise ValueError

            for epoch in range(1, self.config.max_epoch + 1):
                print("Test Attempt: {} Epoch: {}".format(test_attempt, epoch))
                self.train_one_epoch()
                self.validate()
                self.current_epoch += 1
                if epoch >= self.config.min_epoch and self.early_stop:
                    print("stop epoch".format(epoch))
                    break
            if self.pred_type == "rate":
                self.rate_train_data += self.val_data
            elif self.pred_type == "stress":
                self.stress_train_data += self.val_data
            else:
                self.done_train_data += self.val_data

            self.model.eval()
            with torch.no_grad():
                for (user, time_index, resource, item, value) in self.config.test_data:
                    if time_index == test_attempt and resource == self.pred_type:
                        if self.pred_type == "stress":
                            view = torch.tensor(1).long()
                        elif self.pred_type == "rate":
                            view = torch.tensor(2).long()
                        elif self.pred_type == "done":
                            view = torch.tensor(3).long()
                        else:
                            raise ValueError
                        count += 1
                        user = torch.Tensor([user]).long()
                        attempt = torch.Tensor([time_index]).long()
                        item = torch.Tensor([item]).long()
                        value = torch.Tensor([value])
                        pred = self.model(user, attempt, item, view)
                        if self.pred_type in ["stress", "rate"]:
                            loss = self.criterion(pred, value)
                        else:
                            loss = self.bce_loss(pred, value)
                        pred_list.append(pred)
                        true_list.append(value)
                        test_mse += loss.item()
                        print("pred: {} rate: {}, mse: {}".format(pred, value, loss))

                    if resource == "stress":
                        self.stress_train_data.append([user, time_index, resource, item, value])
                    elif resource == "rate":
                        self.rate_train_data.append([user, time_index, resource, item, value])
                    elif resource == "done":
                        self.done_train_data.append([user, time_index, resource, item, value])
                    else:
                        raise ValueError

            all_count += count
            all_test_mse += test_mse
            if count != 0:
                print("current test rmse: {}".format(np.sqrt(test_mse / count)))
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
        print("metric 1 {}".format(self.metric1))
        print("metric 2 {}".format(self.metric2))
        self.save_checkpoint(file_name=self.config.checkpoint)
        return all_count, self.metric1, self.metric2

    def train_one_epoch(self):
        self.model.train()
        stress_train_loss = 0
        rate_train_loss = 0
        done_train_loss = 0
        stress_count = 0
        rate_count = 0
        done_count = 0
        self.train_data = self.stress_train_data + self.rate_train_data + self.done_train_data
        np.random.shuffle(self.train_data)
        for idx, (user, time_index, resource, item, value) in enumerate(self.train_data):
            # print(user, time_index, resource, item, value)
            if resource == "stress":
                view = torch.tensor(1).long()
            elif resource == "rate":
                view = torch.tensor(2).long()
            elif resource == "done":
                view = torch.tensor(3).long()
            else:
                raise ValueError
            user = torch.Tensor([user]).long()
            attempt = torch.Tensor([time_index]).long()
            item = torch.Tensor([item]).long()
            value = torch.Tensor([value])
            self.optimizer.zero_grad()
            pred = self.model(user, attempt, item, view)
            if resource in ["stress", "rate"]:
                loss = self.criterion(pred, value)
            else:
                loss = self.bce_loss(pred, value)
            if resource == "stress":
                stress_train_loss += loss.item()
                stress_count += 1
            elif resource == "rate":
                rate_train_loss += loss.item()
                rate_count += 1
            elif resource == "done":
                done_train_loss += loss.item()
                done_count += 1
            else:
                raise ValueError

            if resource in ["stress"]:
                loss += self.smooth_weight * torch.linalg.norm(
                    self.model.time_factors.weight[1:, :] -
                    self.model.time_factors.weight[0:self.n_attempts - 1, :]
                )

            # print("pred: {} rate: {}, loss: {}".format(pred, rate, loss))
            loss.backward()
            self.optimizer.step()
            if "user" in self.config.clipping:
                self.model.user_factors.apply(self.clipper)
            if "time" in self.config.clipping:
                self.model.time_factors.apply(self.clipper)
            if "item" in self.config.clipping:
                self.model.item_factors.apply(self.clipper)
            if "all" in self.config.clipping:
                self.model.apply(self.clipper)

        # self.stress_train_loss = np.sqrt(stress_train_loss / stress_count)
        # self.rate_train_loss = np.sqrt(rate_train_loss / rate_count)
        # self.done_train_loss = np.sqrt(done_train_loss / done_count)
        # print("average stress train loss: {}".format(self.stress_train_loss))
        # print("average rate train loss: {}".format(self.rate_train_loss))
        # print("average done train loss: {}".format(self.done_train_loss))

    def validate(self):
        self.model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for idx, (user, time_index, resource, item, value) in enumerate(self.val_data):
                if resource == "stress":
                    view = torch.tensor(1)
                elif resource == "rate":
                    view = torch.tensor(2)
                elif resource == "done":
                    view = torch.tensor(3)
                else:
                    raise ValueError
                user = torch.Tensor([user]).long()
                attempt = torch.Tensor([time_index]).long()
                item = torch.Tensor([item]).long()
                value = torch.Tensor([value])
                pred = self.model(user, attempt, item, view)
                if resource in ["stress", "rate"]:
                    val_loss += self.criterion(pred, value).item()
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
            # self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.logger.info(f"Checkpoint loaded successfully from '{self.config.checkpoint_dir}' "
            #                  f"at (epoch {checkpoint['epoch']}\n")
        except OSError as e:
            # self.logger.info(f"No checkpoint exists from '{self.config.checkpoint_dir}'. "
            #                  f"Skipping...")
            # self.logger.info("**First time to train**")
            pass

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
        data = {
            "user_factors": self.model.user_factors.weight,
            "item_factors": self.model.item_factors.weight,
            "stress_item_factor": self.model.stress_item_factor.weight,
            "rate_user_biases": self.model.rate_user_biases.weight,
            "stress_user_biases": self.model.stress_user_biases.weight,
            "done_user_biases": self.model.done_user_biases.weight,
            "rate_item_biases": self.model.rate_item_biases.weight,
            "stress_item_biases": self.model.stress_item_biases.weight,
            "done_item_biases": self.model.done_item_biases.weight,
            "time_factors": self.model.time_factors.weight,
            "time_biases": self.model.time_biases.weight
        }
        torch.save(data, self.config.out_dir + self.config.output)

        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')
