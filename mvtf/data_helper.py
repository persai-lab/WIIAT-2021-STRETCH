# some helper functions for generating suitable data or configuration on running experiments
from mvtf.utils import *
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt


def config(fold, user_latent_dim, item_latent_dim, penalty_weight, lr, bias_s_lr, bias_t_lr,
           bias_q_lr, lambda_s, lambda_t, lambda_q, lambda_p, max_iter, metrics, pred="stress",
           trade_off_stress=0, trade_off_rate=1, trade_off_done=0, single_view=False, mode="test"):
    """
    generate model configurations for training and testing
    such as initialization of each parameters and hyperparameters
    :return: config dict
    """

    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)

    config = {
        'num_users': data['num_users'],
        'num_time_index': data['num_time_index'] + 1,
        'num_items': data['num_items'],
        'user_latent_dim': user_latent_dim,
        'item_latent_dim': item_latent_dim,
        'penalty_weight': penalty_weight,
        'lr': lr,
        'bias_s_lr': bias_s_lr,
        'bias_t_lr': bias_t_lr,
        'bias_q_lr': bias_q_lr,
        'lambda_s': lambda_s,
        'lambda_t': lambda_t,
        'lambda_q': lambda_q,
        'lambda_p': lambda_p,
        'max_iter': max_iter,
        'tol': 1e-3,
        'metrics': metrics,
        'pred': pred,
        'trade_off_stress': trade_off_stress,
        'trade_off_rate': trade_off_rate,
        'trade_off_done': trade_off_done
    }
    for key in config.keys():
        print(key, config[key])

    activity = data["activity"]
    user_records_size = data["user_records_size"]

    train_users = data["fold_{}".format(fold)]["train"]
    val_users = data["fold_{}".format(fold)]["val"]
    if mode == "test":
        train_users += val_users
        test_users = data["fold_{}".format(fold)]["test"]
    else:
        test_users = val_users

    # train_users, test_users = train_test_split(list(activity.keys()), test_size=0.3, random_state=0)
    stress_records = {}

    print("number of train user:{} test users: {}".format(len(train_users), len(test_users)))
    for user in train_users:
        print("train user_id: {}, length: {}, stress: {}, rate: {}, done: {}".format(
            user, len(activity[user]), user_records_size[user]["stress"],
            user_records_size[user]["rate"], user_records_size[user]["done"]))
    for user in test_users:
        print("test user_id: {}, length: {}, stress: {}, rate: {}, done: {}".format(
            user, len(activity[user]), user_records_size[user]["stress"],
            user_records_size[user]["rate"], user_records_size[user]["done"]))

    # generate config, train_set, test_set for general train and test
    train_data = []
    item_rating_dict = {}
    for user in train_users:
        for (user, time_index, resource, item, value) in activity[user]:
            if single_view and resource != pred:
                continue
            train_data.append([user, time_index, resource, item, value])

    test_data = []
    max_test_index = 0
    min_test_index = float('inf')
    for user in test_users:
        pred_record_size = user_records_size[user][pred]
        start_test_index = pred_record_size / 2
        count = 0
        for (user, time_index, resource, item, value) in sorted(activity[user], key=lambda x: x[1]):
            if single_view and resource != pred:
                continue
            if count < start_test_index:
                train_data.append([user, time_index, resource, item, value])
            else:
                test_data.append([user, time_index, resource, item, value])
                if time_index > max_test_index and resource == pred:
                    max_test_index = time_index
                if time_index < min_test_index and resource == pred:
                    min_test_index = time_index
            if resource == pred:
                count += 1
    config["train"] = train_data
    config["test"] = test_data
    config["stress_records"] = stress_records
    config["max_test_index"] = max_test_index
    config["min_test_index"] = min_test_index

    # for user in sorted(stress_records.keys()):
    #     print(user, stress_records[user], np.mean(stress_records[user]))
    true_test_users = {}
    for (user, time_index, resource, item, value) in sorted(test_data, key=lambda x: x[1]):
        if resource == pred:
            prRed("user {}: {}, {}, {}, {}".format(user, time_index, resource, item, value))
            if user not in true_test_users:
                true_test_users[user] = True

    for user in true_test_users:
        for (user, time_index, resource, item, value) in sorted(activity[user], key=lambda x: x[1]):
            prBlue("user {}: {}, {}, {}, {}".format(user, time_index, resource, item, value))

    return config
