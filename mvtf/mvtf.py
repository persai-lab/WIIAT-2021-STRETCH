import numpy as np
from numpy import linalg as LA
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import warnings
import copy
import pandas as pd
from scipy.stats import ttest_rel
from mvtf.utils import *
import matplotlib.pyplot as plt

warnings.filterwarnings("error")


class MVTF(object):
    """
    AGRTF knowledge modeling + Proximal recommendation strategy
    """

    def __init__(self, config, **kwargs):
        """
        :param config:
        :var
        """
        self.random_state = 1
        np.random.seed(self.random_state)
        self.train_data = config['train']
        # self.stress_records = config['stress_records']
        self.train_data = sorted(self.train_data, key=lambda x: x[1])
        # rate_list = []
        # stress_list = []
        # done_list = []
        # for user, time, resource, item, value in self.train_data:
        #     if resource == "rate":
        #         rate_list.append(value)
        #     if resource == "stress":
        #         stress_list.append(value)
        #     if resource == "done":
        #         done_list.append(value)
        # self.avg_rate = np.mean(rate_list)
        # self.avg_stress = np.mean(stress_list)
        # self.avg_done = np.mean(done_list)

        self.stress_list = []
        self.done_list = []
        self.rate_list = []
        self.user_stress_dict = {}
        self.user_rate_dict = {}
        self.user_done_dict = {}
        self.item_rate_dict = {}
        self.item_done_dict = {}
        for (user, time_index, resource, item, obs) in self.train_data:
            if resource == "stress":
                self.stress_list.append(obs)
                if user not in self.user_stress_dict:
                    self.user_stress_dict[user] = []
                self.user_stress_dict[user].append(obs)
            elif resource == "done":
                self.done_list.append(obs)
                if user not in self.user_done_dict:
                    self.user_done_dict[user] = []
                self.user_done_dict[user].append(obs)
                if item not in self.item_done_dict:
                    self.item_done_dict[item] = []
                self.item_done_dict[item].append(obs)
            elif resource == "rate":
                self.rate_list.append(obs)
                if user not in self.user_rate_dict:
                    self.user_rate_dict[user] = []
                self.user_rate_dict[user].append(obs)
                if item not in self.item_rate_dict:
                    self.item_rate_dict[item] = []
                self.item_rate_dict[item].append(obs)
            else:
                raise ValueError

        self.num_users = config['num_users']
        self.num_time_index = config['num_time_index']
        self.num_items = config['num_items']
        self.user_latent_dim = config['user_latent_dim']
        self.item_latent_dim = config['item_latent_dim']
        self.penalty_weight = config['penalty_weight']
        self.trade_off_stress = config['trade_off_stress']
        self.trade_off_rate = config['trade_off_rate']
        self.trade_off_done = config['trade_off_done']
        self.metrics = config["metrics"]
        self.pred = config["pred"]

        self.lr = config['lr']
        self.bias_s_lr = config['bias_s_lr']
        self.bias_t_lr = config['bias_t_lr']
        self.bias_q_lr = config['bias_q_lr']
        self.lambda_s = config['lambda_s']
        self.lambda_t = config['lambda_t']
        self.lambda_q = config['lambda_q']
        self.lambda_p = config['lambda_p']
        self.max_iter = config['max_iter']
        self.tol = config['tol']

        self.test_obs_list = []
        self.test_pred_list = []
        self.test_running_global_avg_pred_list = []
        self.test_running_user_avg_pred_list = []
        self.test_running_item_avg_pred_list = []
        # True if apply sigmoid for final output value
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        :return:
        """
        self.S = np.random.random_sample((self.num_users, self.user_latent_dim))
        self.T = np.random.random_sample(
            (self.user_latent_dim, self.num_time_index, self.item_latent_dim))
        self.Q = np.random.random_sample((self.item_latent_dim, self.num_items))
        self.P = np.random.random_sample((self.item_latent_dim, 1))
        self.stress_bias_s = np.zeros(self.num_users)
        self.done_bias_s = np.zeros(self.num_users)
        self.rate_bias_s = np.zeros(self.num_users)
        self.bias_t = np.zeros(self.num_time_index)
        self.done_bias_q = np.zeros(self.num_items)
        self.rate_bias_q = np.zeros(self.num_items)
        self.bias_p = np.zeros(1)
        # print("user:{}, time: {}, item: {}".format(
        #     self.num_users, self.num_time_index, self.num_items))

    def __getstate__(self):
        """
        since the logger cannot be pickled,
        to avoid the pickle error, we should add this
        :return:
        """
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_stress_prediction(self, user, time_index):
        """
        :return:
        """
        pred = np.dot(np.dot(self.S[user, :], self.T[:, time_index, :]), self.P[:, 0])
        pred += self.stress_bias_s[user] + self.bias_t[time_index]
        # pred += self.stress_bias_s[user] + self.bias_t[time_index]  # the best
        # pred += self.stress_bias_s[user] + self.avg_stress
        return pred

    def _get_done_prediction(self, user, time_index, item):
        pred = np.dot(np.dot(self.S[user, :], self.T[:, time_index, :]), self.Q[:, item])
        pred += self.done_bias_s[user] + self.bias_t[time_index] + self.done_bias_q[item]
        # pred += self.done_bias_s[user] + self.bias_t[time_index]
        return sigmoid(pred)

    def _get_rate_prediction(self, user, item):
        pred = np.dot(self.S[user, :], self.Q[:, item])
        pred += self.rate_bias_s[user] + self.rate_bias_q[item]
        # pred += self.avg_rate
        return pred

    def _get_loss(self):
        """
        compute the loss, which is RMSE of observed records + regularization
        + penalty of temporal non-smoothness
        :return: loss
        """
        loss, square_loss, bias_reg = 0., 0., 0.
        stress_loss = 0
        done_loss = 0
        rate_loss = 0
        stress_count = 0
        done_count = 0
        rate_count = 0
        for (user, time_index, resource, item, obs) in self.train_data:
            if resource == "stress":
                pred = self._get_stress_prediction(user, time_index)
                stress_loss += (obs - pred) ** 2
                stress_count += 1
            elif resource == "done":
                pred = self._get_done_prediction(user, time_index, item)
                done_loss += (obs - pred) ** 2
                done_count += 1
            elif resource == "rate":
                pred = self._get_rate_prediction(user, item)
                rate_loss += (obs - pred) ** 2
                rate_count += 1
            else:
                raise ValueError

        # regularization
        reg_S = LA.norm(self.S) ** 2
        reg_T = LA.norm(self.T) ** 2
        reg_Q = LA.norm(self.Q) ** 2
        reg_P = LA.norm(self.P) ** 2
        reg_loss = (self.trade_off_stress + self.trade_off_rate + self.trade_off_done) * \
                   self.lambda_s * reg_S + (self.trade_off_rate + self.trade_off_done) * \
                   self.lambda_q * reg_Q + (self.trade_off_stress + self.trade_off_done) * \
                   self.lambda_t * reg_T + self.trade_off_stress * self.lambda_p * reg_P

        # reg_loss = self.lambda_s * reg_S + self.lambda_q * reg_Q + self.lambda_t * reg_T + \
        #            self.lambda_p * reg_P

        # if self.lambda_bias:
        #     bias_reg = self.lambda_bias * (LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_q) ** 2)
        penalty = self._get_penalty()

        # avg_stress_loss = stress_loss / stress_count
        # avg_rate_loss = np.sqrt(rate_loss / rate_count)
        # avg_done_loss = done_loss /done_count
        # print("rate count: {}, avg rate loss: {}".format(rate_count, avg_rate_loss))
        # print("stress count {}, avg loss: {}, done count: {}, avg loss: {}, rate count: {}, "
        #       "avg loss: {}".format(stress_count, avg_stress_loss, done_count, avg_done_loss,
        #                             rate_count, avg_rate_loss))
        # print(np.mean(stress_list))
        # plt.hist(stress_list)
        # plt.title("stress")
        # plt.show()
        # plt.clf()
        # plt.hist(done_list)
        # plt.title("done")
        # plt.show()
        # plt.clf()
        # plt.hist(rate_list)
        # plt.title("rating")
        # plt.show()
        # plt.clf()
        loss = self.trade_off_stress * stress_loss + \
               self.trade_off_done * done_loss + \
               self.trade_off_rate * rate_loss + \
               reg_loss + self.penalty_weight * penalty
        return loss, stress_loss, done_loss, rate_loss, penalty, reg_loss

    def _get_penalty(self):
        """
        smooth the transition on tensor T
        :return:
        """
        p = LA.norm(self.T[:, 1:self.num_time_index, :] - self.T[:, 0:self.num_time_index - 1, :])
        return p

    def _grad_S_k(self, user, time_index, item, res=None, obs=None):
        """
        :param user:
        :param time_index:
        :param question:
        :param obs:
        :return:
        """
        try:
            grad = np.zeros_like(self.S[user, :])
            if res == "stress":
                pred = self._get_stress_prediction(user, time_index)
                grad += -2. * (obs - pred) * np.dot(self.T[:, time_index, :], self.P[:, 0])
            elif res == "done":
                pred = self._get_done_prediction(user, time_index, item)
                grad += -2. * (obs - pred) * np.dot(self.T[:, time_index, :], self.Q[:, item])
            elif res == "rate":
                pred = self._get_rate_prediction(user, item)
                grad += -2. * (obs - pred) * self.Q[:, item]
            grad += 2. * self.lambda_s * self.S[user, :]
        except Warning:
            grad = 0.
        return grad

    def _grad_T_ij(self, user, time_index, item, res=None, obs=None):
        """
        compute the gradient of loss w.r.t a specific student j's knowledge at
        a specific attempt i: T_{i,j,:},
        :param time_index: index
        :param user: index
        :param obs: observation
        :return:
        """
        try:
            grad = np.zeros_like(self.T[:, time_index, :])
            if res == "stress":
                pred = self._get_stress_prediction(user, time_index)
                grad += -2. * (obs - pred) * np.outer(self.S[user, :], self.P[:, 0])
                grad += 2. * self.lambda_t * self.T[:, time_index, :]
            elif res == "done":
                pred = self._get_done_prediction(user, time_index, item)
                grad = -2. * (obs - pred) * np.outer(self.S[user, :], self.Q[:, item])
                grad += 2. * self.lambda_t * self.T[:, time_index, :]

            last_slice_index = self.num_time_index - 1
            if time_index == 0:
                diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
                penalty_val = -2 * np.sum(diff)
                grad += self.penalty_weight * penalty_val
            elif time_index == last_slice_index:
                diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
                penalty_val = 2 * np.sum(diff)
                grad += self.penalty_weight * penalty_val
            else:
                diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
                penalty_val = 2 * np.sum(diff)
                grad += self.penalty_weight * penalty_val

                diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
                penalty_val = -2 * np.sum(diff)
                grad += self.penalty_weight * penalty_val
        except Warning:
            grad = 0.
        return grad

    def _grad_Q_k(self, user, time_index, item, res=None, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept
        of a question in Q-matrix,
        :param user:
        :param item:
        :param obs:
        :return:
        """
        try:
            grad = np.zeros_like(self.Q[:, item])
            if res == "done":
                pred = self._get_done_prediction(user, time_index, item)
                grad += -2. * (obs - pred) * pred * (1. - pred) * np.dot(self.S[user, :],
                                                                         self.T[:, time_index, :])
            elif res == "rate":
                pred = self._get_rate_prediction(user, item)
                grad += -2 * (obs - pred) * self.S[user, :]
            grad += 2. * self.lambda_q * self.Q[:, item]
        except Warning:
            grad = 0.
        return grad

    def _grad_P_k(self, user, time_index, stress_obs=None):
        """
        compute the gradient of loss w.r.t a specific concept
        of a question in Q-matrix,
        :param user:
        :param item:
        :param obs:
        :return:
        """
        try:
            grad = np.zeros_like(self.P[:, 0])
            pred = self._get_stress_prediction(user, time_index)
            grad += -2 * (stress_obs - pred) * np.dot(self.S[user, :], self.T[:, time_index, :])
            grad += 2. * self.lambda_p * self.Q[:, 0]
        except Warning:
            grad = 0.

        return grad

    def _grad_bias_s(self, user, time_index, item, res=None, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param time_index:
        :param user:
        :param item:
        :param obs:
        :return:
        """
        try:
            grad = 0.
            if res == "stress":
                pred = self._get_stress_prediction(user, time_index)
                grad -= 2. * (obs - pred)
            elif res == "done":
                pred = self._get_done_prediction(user, time_index, item)
                grad -= 2. * (obs - pred) * pred * (1. - pred)
            elif res == "rate":
                pred = self._get_rate_prediction(user, item)
                grad -= 2. * (obs - pred)
        except Warning:
            grad = 0.
        return grad

    def _grad_bias_t(self, user, time_index, item, res=None, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_a
        :param time_index:
        :param user:
        :param item:
        :return:
        """
        try:
            grad = 0.
            if res == "stress":
                pred = self._get_stress_prediction(user, time_index)
                grad -= 2. * (obs - pred)
            elif res == "done":
                pred = self._get_done_prediction(user, time_index, item)
                grad -= 2. * (obs - pred) * pred * (1. - pred)
            # grad += 2.0 * self.lambda_bias * self.bias_t[time_index]
        except Warning:
            grad = 0.
        return grad

    def _grad_bias_q(self, user, time_index, item, res=None, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param time_index:
        :param user:
        :param item:
        :param obs:
        :return:
        """
        try:
            grad = 0.
            if res == "done":
                pred = self._get_done_prediction(user, time_index, item)
                grad -= 2. * (obs - pred) * pred * (1. - pred)
            elif res == "rate":
                pred = self._get_rate_prediction(user, item)
                grad -= 2. * (obs - pred)
        except Warning:
            grad = 0.
        return grad

    def _optimize_sgd(self, user, time_index, item, res=None, obs=None):
        """
        train the S, T and Q with stochastic gradient descent
        :param item:
        :param time_index:
        :param item:
        :return:
        """
        if res == "stress":
            grad_s = self.trade_off_stress * self._grad_S_k(user, time_index, item, res, obs)
            self.S[user, :] -= self.lr * grad_s
            grad_t = self.trade_off_stress * self._grad_T_ij(user, time_index, item, res, obs)
            self.T[:, time_index, :] -= self.lr * grad_t
            grad_p = self.trade_off_stress * self._grad_P_k(user, time_index, obs)
            self.P[:, 0] -= self.lr * grad_p
            self.stress_bias_s[user] -= self.bias_s_lr * self.trade_off_stress * self._grad_bias_s(
                user, time_index, item, res, obs)
            self.bias_t[item] -= self.bias_t_lr * self.trade_off_stress * self._grad_bias_t(
                user, time_index, item, res, obs)
        elif res == "done":
            grad_s = self.trade_off_done * self._grad_S_k(user, time_index, item, res, obs)
            self.S[user, :] -= self.lr * grad_s
            grad_t = self.trade_off_done * self._grad_T_ij(user, time_index, item, res, obs)
            self.T[:, time_index, :] -= self.lr * grad_t
            grad_q = self.trade_off_done * self._grad_Q_k(user, time_index, item, res, obs)
            self.Q[:, item] -= self.lr * grad_q
            self.done_bias_s[user] -= self.bias_s_lr * self.trade_off_done * self._grad_bias_s(
                user, time_index, item, res, obs)
            self.done_bias_q[item] -= self.bias_q_lr * self.trade_off_done * self._grad_bias_q(
                user, time_index, item, res, obs)
            self.bias_t[time_index] -= self.bias_t_lr * self.trade_off_done * self._grad_bias_t(
                user, time_index, item, res, obs)
        else:
            grad_s = self.trade_off_rate * self._grad_S_k(user, time_index, item, res, obs)
            self.S[user, :] -= self.lr * grad_s
            grad_q = self.trade_off_rate * self._grad_Q_k(user, time_index, item, res, obs)
            self.Q[:, item] -= self.lr * grad_q
            self.rate_bias_s[user] -= self.bias_s_lr * self.trade_off_rate * self._grad_bias_s(
                user, time_index, item, res, obs)
            self.rate_bias_q[item] -= self.bias_q_lr * self.trade_off_rate * self._grad_bias_q(
                user, time_index, item, res, obs)

    def training(self):
        """
        minimize the loss until converged or reach the maximum iterations
        with stochastic gradient descent
        :return:
        """
        print(strBlue("*" * 40 + "[ Start   Training ]" + "*" * 40))

        train_perf = []
        start_time = time.time()
        converge = False
        iter_num = 0
        best_S, best_T, best_Q = [0] * 3
        best_bias_s, best_bias_t, best_bias_q = [0] * 3
        self._initialize_parameters()

        loss, stress_loss, done_loss, rate_loss, penalty, reg_loss = self._get_loss()
        print(strBlue(
            "initial: lr: {:.4f}, loss: {:.2f}, penalty: {:.5f}, reg: {:.5f}, "
            "stress-loss: {:.5f}, done loss: {:.5f}, rate loss: {:.5f}".format(
                self.lr, loss, penalty, reg_loss, stress_loss, done_loss, rate_loss)))
        loss_list = [loss]
        stress_loss_list = [stress_loss]
        done_loss_list = [done_loss]
        rate_loss_list = [rate_loss]
        penalty_list = [penalty]
        reg_loss_list = [reg_loss]
        print(strBlue("*" * 40 + "[ Training Outputs ]" + "*" * 40))

        while not converge:
            np.random.shuffle(self.train_data)
            best_S = np.copy(self.S)
            best_T = np.copy(self.T)
            best_Q = np.copy(self.Q)
            best_stress_bias_s = np.copy(self.stress_bias_s)
            best_done_bias_s = np.copy(self.done_bias_s)
            best_rate_bias_s = np.copy(self.rate_bias_s)
            best_bias_t = np.copy(self.bias_t)
            best_rate_bias_q = np.copy(self.rate_bias_q)
            best_done_bias_q = np.copy(self.done_bias_q)

            count = 0
            for (user, time_index, res, item, obs) in self.train_data:
                self._optimize_sgd(user, time_index, item, res, obs)
                count += 1

            loss, stress_loss, done_loss, rate_loss, penalty, reg_loss = self._get_loss()
            print(strBlue(
                "initial: lr: {:.4f}, loss: {:.2f}, penalty: {:.5f}, reg: {:.5f}, "
                "stress-loss: {:.5f}, done loss: {:.5f}, rate loss: {:.5f}".format(
                    self.lr, loss, penalty, reg_loss, stress_loss, done_loss, rate_loss)))
            stress_loss_list.append(stress_loss)
            done_loss_list.append(done_loss)
            rate_loss_list.append(rate_loss)
            penalty_list.append(penalty)
            reg_loss_list.append(reg_loss)

            if iter_num == self.max_iter:
                converge = True
            elif loss == np.nan:
                self.lr *= 0.1
            elif loss > loss_list[-1]:
                loss_list.append(loss)
                self.lr *= 0.5
                iter_num += 1
            else:
                loss_list.append(loss)
                iter_num += 1

        # reset to previous S, T, Q
        self.S = best_S
        self.T = best_T
        self.Q = best_Q
        self.stress_bias_s = best_stress_bias_s
        self.done_bias_s = best_done_bias_s
        self.rate_bias_s = best_rate_bias_s
        self.bias_t = best_bias_t
        self.done_bias_q = best_done_bias_q
        self.rate_bias_q = best_rate_bias_q

        # print(list(range(len(stress_loss_list))))
        # print(stress_loss_list)
        # plt.plot(list(range(len(stress_loss_list))), stress_loss_list, "r-")
        # plt.show()

    def testing(self, test_data, validation=False):
        """
        student performance prediction
        """
        if not validation:
            print(
                strGreen("*" * 40 + "[ Testing Results ]" + "*" * 40))
            # print(
            #     strGreen("Current testing time index: {}, Test size: {}".format(
            #         self.current_test_time, len(test_data))))

        curr_pred_list = []
        curr_obs_list = []

        for (student, time_index, resource, item, obs) in test_data:
            curr_obs_list.append(obs)
            if self.pred == "stress":
                pred = self._get_stress_prediction(student, time_index)
                avg_pred = np.mean(self.stress_list)
                if student in self.user_stress_dict:
                    running_user_avg_pred = np.mean(self.user_stress_dict[student])
                else:
                    running_user_avg_pred = np.mean(self.stress_list)
                # print("student: {}, time_index: {}, pred: {:.5f}, stress bias_s: {:.5f}, "
                #       "avg_pred: {:.5f}, running_avg_pred: {:.5f}, obs: {:.5f}, "
                #       "stress records: {}".format(
                #     student, time_index, pred, self.stress_bias_s[student], avg_pred,
                #     running_avg_pred, obs, self.stress_records[student]))
                running_item_avg_pred = np.mean(self.stress_list)
            elif self.pred == "rate":
                pred = self._get_rate_prediction(student, item)
                avg_pred = np.mean(self.rate_list)
                if student in self.user_rate_dict:
                    running_user_avg_pred = np.mean(self.user_rate_dict[student])
                else:
                    running_user_avg_pred = np.mean(self.rate_list)
                if item in self.item_rate_dict:
                    running_item_avg_pred = np.mean(self.item_rate_dict[item])
                else:
                    running_item_avg_pred = np.mean(self.rate_list)
            elif self.pred == "done":
                pred = self._get_done_prediction(student, time_index, item)
                avg_pred = np.mean(self.done_list)
                if student in self.user_done_dict:
                    running_user_avg_pred = np.mean(self.user_done_dict[student])
                else:
                    running_user_avg_pred = np.mean(self.done_list)
                if item in self.item_done_dict:
                    running_item_avg_pred = np.mean(self.item_done_dict[item])
                else:
                    running_item_avg_pred = np.mean(self.done_list)

            curr_pred_list.append(pred)
            self.test_obs_list.append(obs)
            self.test_pred_list.append(pred)
            self.test_running_global_avg_pred_list.append(avg_pred)
            self.test_running_user_avg_pred_list.append(running_user_avg_pred)
            self.test_running_item_avg_pred_list.append(running_item_avg_pred)
        return self.eval(curr_obs_list, curr_pred_list)

    def eval(self, obs_list, pred_list):
        """
        evaluate the prediction performance on different metrics
        :param obs_list:
        :param pred_list:
        :return:
        """
        assert len(pred_list) == len(obs_list)

        count = len(obs_list)
        # plt.hist(obs_list)
        # plt.title("stress test")
        # plt.show()
        perf_dict = {}
        # print(strGreen("Test Attempt: {}".format(self.current_test_time)))
        if len(pred_list) == 0:
            return perf_dict
        else:
            print(strGreen("Test Size: {}".format(count)))
            perf_dict["count"] = count

        for metric in self.metrics:
            if metric == "rmse":
                rmse = mean_squared_error(obs_list, pred_list, squared=False)
                perf_dict[metric] = rmse
                print(strGreen("RMSE: {:.5f}".format(rmse)))
            elif metric == 'mae':
                mae = mean_absolute_error(obs_list, pred_list)
                perf_dict[metric] = mae
                print(strGreen("MAE: {:.5f}".format(mae)))
            elif metric == "auc":
                if np.sum(obs_list) == count or np.sum(obs_list) == 0:
                    print(strGreen("AUC: None (all ones or all zeros in true y)"))
                    perf_dict[metric] = None
                else:
                    auc = roc_auc_score(obs_list, pred_list)
                    perf_dict[metric] = auc
                    print(strGreen("AUC: {:.5f}".format(auc)))
            elif metric == "acc":
                if np.sum(obs_list) == count or np.sum(obs_list) == 0:
                    print(strGreen("ACC: None (all ones or all zeros in true y)"))
                    perf_dict[metric] = None
                else:
                    fpr, tpr, thresholds = roc_curve(obs_list, pred_list)
                    J = tpr - fpr
                    idx = np.argmax(J)
                    best_threshold = thresholds[idx]
                    pred_list = np.array(pred_list)
                    pred_list[pred_list > best_threshold] = 1
                    pred_list[pred_list <= best_threshold] = 0
                    acc = accuracy_score(obs_list, pred_list)
                    perf_dict[metric] = acc
                    print(strGreen("ACC: {:.5f}".format(acc)))
        print("")
        return perf_dict
