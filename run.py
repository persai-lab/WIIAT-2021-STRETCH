import json
from multiprocessing import Pool, Lock

import pandas as pd

output_lock = Lock()
from mvtf.data_helper import *
from mvtf.mvtf import MVTF
from mvtf.data_helper import config


# 1. Don't set global average offset, it is not helpful
# 2. Data is too sparse, should avoid overfitting. So the training iterations should not be large.
# 3. Should avoid overfitting.


def sequential_recommendation(fold, user_latent_dim, item_latent_dim, penalty_weight, lr, bias_s_lr,
                              bias_t_lr, bias_q_lr, lambda_s, lambda_t, lambda_q, lambda_p,
                              max_iter, metrics, pred, mode, trade_off_stress, trade_off_rate,
                              trade_off_done):
    """
    pipeline of single run of experiment
    :para: a list of parameters for a single case of experiment
    :return:
    """

    model_config = config(fold, user_latent_dim, item_latent_dim, penalty_weight, lr, bias_s_lr,
                          bias_t_lr, bias_q_lr, lambda_s, lambda_t, lambda_q, lambda_p, max_iter,
                          metrics, pred, trade_off_stress, trade_off_rate, trade_off_done,
                          mode=mode)
    max_test_index = model_config["max_test_index"]
    min_test_index = model_config["min_test_index"]
    model = MVTF(model_config)
    test_data = model_config['test']
    print("=" * 100)

    for test_attempt in range(min_test_index, max_test_index + 1):
        print("test attempt: {}".format(test_attempt))
        SS = np.dot(model.S, model.S.T)
        model.current_test_attempt = test_attempt
        model.lr = lr
        model.training()
        test_set = []
        for (student, attempt, resource, item, obs) in test_data:
            if attempt == test_attempt:
                if pred == "stress" and resource == "stress":
                    test_set.append((student, attempt, resource, item, obs))
                elif pred == "rate" and resource == "rate":
                    test_set.append((student, attempt, resource, item, obs))
                elif pred == "done" and resource == "done":
                    test_set.append((student, attempt, resource, item, obs))
                model.train_data.append((student, attempt, resource, item, obs))
        model.testing(test_set)

    print("avg:")
    avg_result = model.eval(model.test_obs_list, model.test_running_global_avg_pred_list)
    print("runnning avg:")
    running_avg_result = model.eval(model.test_obs_list, model.test_running_user_avg_pred_list)
    print("our model:")
    result = model.eval(model.test_obs_list, model.test_pred_list)
    # print(model.stress_bias_s)
    # print(model.rate_bias_s)
    # print(model.done_bias_s)
    # print(model.rate_bias_q)
    # print(model.done_bias_q)
    # print(model.bias_t)
    # print(model.bias_p)
    print("observation     : {}".format(list(np.round(model.test_obs_list, 1))))
    print("avg pred        : {}".format(list(np.round(model.test_running_global_avg_pred_list, 1))))
    print("running avg pred: {}".format(list(np.round(model.test_running_user_avg_pred_list, 1))))
    print("our model pred  : {}".format(list(np.round(model.test_pred_list, 1))))
    if "rmse" in metrics and "mae" in metrics:
        print(fold, result["count"], result["rmse"], result["mae"],
              avg_result["rmse"], avg_result["mae"],
              running_avg_result["rmse"], running_avg_result["mae"])
        output = (fold, result["count"], result["rmse"], result["mae"],
                  avg_result["rmse"], avg_result["mae"],
                  running_avg_result["rmse"], running_avg_result["mae"])
    elif "auc" in metrics and "acc" in metrics:
        print(fold, result["count"], result["auc"], result["acc"],
              avg_result["auc"], avg_result["acc"],
              running_avg_result["auc"], running_avg_result["acc"])
        output = (fold, result["count"], result["auc"], result["acc"],
                  avg_result["auc"], avg_result["acc"],
                  running_avg_result["auc"], running_avg_result["acc"])
    return output


def single_run(hidden_dim, penalty_weight, lr, bias_s_lr, bias_t_lr, bias_q_lr, lambda_s, lambda_t,
               lambda_q, lambda_p, max_iter, metrics, pred, mode, trade_off_stress, trade_off_rate,
               trade_off_done):
    all_results = []
    for fold in range(1, 6):
        print("Fold: {}".format(fold))
        output = sequential_recommendation(fold=fold,
                                           user_latent_dim=hidden_dim,
                                           item_latent_dim=hidden_dim,
                                           penalty_weight=penalty_weight,
                                           lr=lr,
                                           bias_s_lr=bias_s_lr,
                                           bias_t_lr=bias_t_lr,
                                           bias_q_lr=bias_q_lr,
                                           lambda_s=lambda_s,
                                           lambda_t=lambda_t,
                                           lambda_q=lambda_q,
                                           lambda_p=lambda_p,
                                           max_iter=max_iter,
                                           metrics=metrics,
                                           pred=pred,
                                           mode=mode,
                                           trade_off_stress=trade_off_stress,
                                           trade_off_rate=trade_off_rate,
                                           trade_off_done=trade_off_done
                                           )
        all_results.append(output)
    if "rmse" in metrics and "mae" in metrics:
        for fold, count, rmse1, mae1, rmse2, mae2, rmse3, mae3 in all_results:
            print(fold, count, rmse1, mae1, rmse2, mae2, rmse3, mae3)
    elif "auc" in metrics and "acc" in metrics:
        for fold, count, auc1, acc1, auc2, acc2, auc3, acc3 in all_results:
            print(fold, count, auc1, acc1, auc2, acc2, auc3, acc3)
    avg_result = np.mean(all_results, axis=0)
    print("average of 5 folds results: {}".format(avg_result))
    output_lock.acquire()
    with open("cv_{}_output.csv".format(pred), "a") as f:
        f.write("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},"
                "{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
            hidden_dim, penalty_weight, lr, bias_s_lr, bias_t_lr, bias_q_lr, lambda_s, lambda_t,
            lambda_q, lambda_p, max_iter, trade_off_stress, trade_off_rate, trade_off_done,
            avg_result[1], avg_result[2], avg_result[3], avg_result[4], avg_result[5],
            avg_result[6], avg_result[7]))
    output_lock.release()


def check_progress(pred):
    file_name = "cv_{}_output.csv".format(pred)
    # df = pd.read_csv(file_name)
    f = open(file_name, "r")

    if pred in ["stress", "rate"]:
        min_rmse = 10.
        best_para = None
        for line in f:
            fields = line.strip().split(",")
            if len(fields) != 21:
                continue
            rmse = float(fields[-6])
            if rmse <= min_rmse:
                min_rmse = rmse
                best_para = [float(i) for i in fields]
        print(list(np.round(best_para, 3)))
    else:
        max_auc = 0.
        best_para = None
        for line in f:
            fields = line.strip().split(",")
            if len(fields) != 21:
                continue
            auc = float(line[-6])
            if auc >= max_auc:
                max_auc = auc
                best_para = [float(i) for i in fields]
        print(list(np.round(best_para, 3)))
    f.close()


if __name__ == '__main__':
    pred = "stress"
    metrics = ["rmse", "mae"]
    hidden_dim = 4
    penalty_weight = 0
    lr = 0.001
    bias_s_lr = 0.05
    bias_t_lr = 0.005
    bias_q_lr = 0.01
    lambda_s = 1.
    lambda_t = 0.
    lambda_q = 2.
    lambda_p = 0.
    max_iter = 8
    trade_off_stress = 1.
    trade_off_rate = 0.
    trade_off_done = 0.

    # pred = "rate"
    # metrics = ["rmse", "mae"]
    # hidden_dim = 4
    # penalty_weight = 0.
    # lr = 0.001
    # bias_s_lr = 0.3
    # bias_t_lr = 0.005
    # bias_q_lr = 0.01
    # lambda_s = 5.
    # lambda_t = 0.
    # lambda_q = 10.
    # lambda_p = 10.
    # max_iter = 4
    # trade_off_stress = 0.
    # trade_off_rate = 1.
    # trade_off_done = 0.

    # hidden_dim = 10
    # penalty_weight = 0.1
    # lr = 0.001
    # bias_s_lr = 0.01
    # bias_t_lr = 0.001
    # bias_q_lr = 0.001
    # lambda_s = 1.
    # lambda_t = 1.
    # lambda_q = 1.
    # lambda_p = 1.
    # max_iter = 2
    # trade_off_stress = 0.
    # trade_off_rate = 1.
    # trade_off_done = 0.

    # pred = "done"
    # metrics = ["auc", "acc"]
    # hidden_dim = 5
    # penalty_weight = 0.1
    # lr = 0.001
    # bias_s_lr = 0.5
    # bias_t_lr = 0.001
    # bias_q_lr = 0.001
    # lambda_s = 0.
    # lambda_t = 0.
    # lambda_q = 0
    # lambda_p = 1.
    # max_iter = 7.

    # para_list = []
    # for hidden_dim in [1]:
    #     for penalty_weight in [0.1, 1.0]:
    #         for lambda_s in [0, 0.5, 1]:
    #             for lambda_t in [0, 0.5, 1]:
    #                 for lambda_q in [0, 0.5, 1]:
    #                     for lambda_p in [0, 0.5, 1]:
    #                         for max_iter in range(2, 7):
    #                             for lr in [0.001, 0.0001]:
    #                                 for bias_q_lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #                                     for bias_s_lr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    #                                         for bias_t_lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #                                             for trade_off_stress in [0, 0.1, 0.5, 0.9]:
    #                                                 for trade_off_done in [0, 0.1, 0.5, 0.9, 5.]:
    #                                                     # trade_off_stress = 0.
    #                                                     trade_off_rate = 1.
    #                                                     # trade_off_done = 0.
    #                                                     mode = 'val'
    #                                                     para = (hidden_dim, penalty_weight, lr,
    #                                                             bias_s_lr, bias_t_lr, bias_q_lr,
    #                                                             lambda_s, lambda_t, lambda_q,
    #                                                             lambda_p, max_iter, metrics, pred,
    #                                                             mode, trade_off_stress,
    #                                                             trade_off_rate, trade_off_done)
    #                                                     para_list.append(para)
    # pool = Pool(processes=60)
    # pool.starmap(single_run, para_list)
    # pool.close()

    mode = 'test'
    single_run(hidden_dim, penalty_weight, lr, bias_s_lr, bias_t_lr, bias_q_lr, lambda_s, lambda_t,
               lambda_q, lambda_p, max_iter, metrics, pred, mode, trade_off_stress, trade_off_rate,
               trade_off_done)
    # check_progress(pred)
