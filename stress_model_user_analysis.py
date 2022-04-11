import numpy as np
import torch
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy import stats
from mvtf.utils import *
from rate_model_item_analysis import scatter_plot, heatmap_plot
from sklearn.manifold import TSNE


def user_analysis(clustering, n_clusters, fold, random_state=1024):
    prBlue("\n\nLoading Data...fold: {}, n_clusters: {}".format(fold, n_clusters))
    data = pickle.load(open("data/data.pkl", "rb"))
    user_name_id_mapping = data["user_name_id_mapping"]
    output_data = torch.load("experiments/MVTFAgent/out/stress_output_fold_{}.pth.tar".format(fold))
    print(output_data.keys())
    user_df = pd.read_csv("data/users.csv")
    user_df = user_df[["user_id", "user_group", "age", "gender"]]
    item_difficulty_dict = pickle.load(open("data/item_difficulty.pkl", "rb"))

    # step 0. read final score of each user
    pss_df = pd.read_csv("data/pss_coping.csv")
    pss_df = pss_df[['user_id', 'final_score']]
    user_pss_score_dict = {}
    for user_name, final_score in pss_df.to_numpy():
        if user_name in user_name_id_mapping:
            user_id = user_name_id_mapping[user_name]
            user_pss_score_dict[user_id] = final_score

    coping = pd.read_csv("data/ways_coping.csv")
    coping_df = coping[['user_id', 'final_score']]
    user_coping_score_dict = {}
    for user_name, final_score in coping_df.to_numpy():
        if user_name in user_name_id_mapping:
            user_id = user_name_id_mapping[user_name]
            user_coping_score_dict[user_id] = final_score

    age_list = list(user_coping_score_dict.values())
    print("user age: size: {}, mean: {}, st: {}, min: {}, med: {}, max: {}".format(
        len(user_coping_score_dict), np.mean(age_list), np.std(age_list),
        np.min(age_list), np.median(age_list), np.max(age_list)
    ))

    # step 1. create mappings
    user_age_dict = {}
    user_gender_dict = {}
    user_gender_id_dict = {}
    user_group_dict = {}
    user_group_id_dict = {}
    male_count = 0
    female_count = 0
    for user_name, user_group, age, gender in user_df.to_numpy():
        if user_name in user_name_id_mapping:
            user_id = user_name_id_mapping[user_name]
            user_age_dict[user_id] = age
            user_group_dict[user_id] = user_group
            user_gender_dict[user_id] = gender
            if gender == "m":
                user_gender_id_dict[user_id] = 0
                male_count += 1
            elif gender == "f":
                user_gender_id_dict[user_id] = 1
                female_count += 1
            else:
                user_gender_id_dict[user_id] = 2
            if user_group == "A":
                user_group_id_dict[user_id] = 0
            elif user_group == "B":
                user_group_id_dict[user_id] = 1
            else:
                raise ValueError
    print("male: {} female: {}".format(male_count, female_count))
    age_list = list(user_age_dict.values())
    print("user age: size: {}, mean: {}, st: {}, min: {}, med: {}, max: {}".format(
        len(user_age_dict), np.mean(age_list), np.std(age_list),
        np.min(age_list), np.median(age_list), np.max(age_list)
    ))

    # step 3. build item-id and rating-list mapping from user data
    user_rate_list_dict = {}
    user_difficulty_list_dict = {}
    user_stress_list_dict = {}
    user_done_list_dict = {}
    coping_strategy_data = pickle.load(open("data/coping_strategy_data.pkl", "rb"))
    strategy_name_id_mapping = coping_strategy_data["strategy_name_id_mapping"]
    item_strategy_dict = coping_strategy_data["item_strategy_dict"]
    user_strategy_list_dict = {}
    type_data = pickle.load(open("data/type_data.pkl", "rb"))
    type_name_id_mapping = type_data["type_name_id_mapping"]
    item_type_dict = type_data["item_type_dict"]
    user_type_list_dict = {}
    rate_list = []
    stress_list = []
    done_list = []
    for user_id in data['activity'].keys():
        records = data['activity'][user_id]
        for (user_id, time_id, resource, item_id, value) in records:
            if resource == "rate":
                rate_list.append(value)
                if user_id not in user_rate_list_dict:
                    user_rate_list_dict[user_id] = []
                    user_difficulty_list_dict[user_id] = []
                user_rate_list_dict[user_id].append(value)
                user_difficulty_list_dict[user_id].append(item_difficulty_dict[item_id])
            elif resource == "stress":
                stress_list.append(value)
                if user_id not in user_stress_list_dict:
                    user_stress_list_dict[user_id] = []
                user_stress_list_dict[user_id].append(value)
            elif resource == "done":
                done_list.append(value)
                if user_id not in user_done_list_dict:
                    user_done_list_dict[user_id] = []
                user_done_list_dict[user_id].append(value)
                if user_id not in user_strategy_list_dict:
                    user_strategy_list_dict[user_id] = np.zeros(len(strategy_name_id_mapping))
                strategy_id = strategy_name_id_mapping[item_strategy_dict[item_id]]
                user_strategy_list_dict[user_id][strategy_id] += value

                if user_id not in user_type_list_dict:
                    user_type_list_dict[user_id] = np.zeros(len(type_name_id_mapping))
                type_id = type_name_id_mapping[item_type_dict[item_id]]
                user_type_list_dict[user_id][type_id] += value

    rate_list = [np.sum(user_done_list_dict[i]) for i in user_done_list_dict]
    print("rate: size: {}, mean: {}, std: {}, min: {}, med.: {}, max: {}".format(
        len(rate_list), np.mean(rate_list), np.std(rate_list), np.min(rate_list),
        np.median(rate_list), np.max(rate_list)
    ))
    prRed("******************************** clustering results ***********************************")
    user_factors = output_data["user_factors"].detach().numpy()
    rate_user_biases = output_data["rate_user_biases"].detach().numpy()
    stress_user_biases = output_data["stress_user_biases"].detach().numpy()
    done_user_biases = output_data["done_user_biases"].detach().numpy()
    X = []
    for user_name in user_name_id_mapping.keys():
        user_id = user_name_id_mapping[user_name]
        user_factor = user_factors[user_id]
        rate_user_bias = rate_user_biases[user_id]
        stress_user_bias = stress_user_biases[user_id]
        done_user_bias = done_user_biases[user_id]
        x = np.append(np.append(np.append(user_factor, rate_user_bias),
                                stress_user_bias), done_user_bias)
        X.append(x)

    if clustering == "kmeans":
        clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    elif clustering == "spectral":
        clusters = SpectralClustering(n_clusters=n_clusters,
                                      assign_labels="discretize",
                                      random_state=random_state).fit(X)
    elif clustering == "hc":
        clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    prRed("n_clusters:{}, cluster scores -------------------------------------".format(n_clusters))
    cluster_score1 = metrics.silhouette_score(X, clusters.labels_, metric='euclidean')
    cluster_score2 = metrics.calinski_harabasz_score(X, clusters.labels_)
    cluster_score3 = metrics.davies_bouldin_score(X, clusters.labels_)
    print(cluster_score1)
    print(cluster_score2)
    print(cluster_score3)

    prRed("********************************* clusters analysis ***********************************")
    cluster_user_list = {}
    cluster_age_list = {}
    cluster_gender_id_list = {}
    cluster_group_id_list = {}
    cluster_avg_user_rate_list = {}
    cluster_user_rate_size_list = {}
    cluster_avg_user_difficulty_list = {}
    cluster_avg_user_stress_list = {}
    cluster_user_stress_var_list = {}
    cluster_user_stress_size_list = {}
    cluster_user_done_ratio_list = {}
    cluster_user_done_count_list = {}
    cluster_user_undone_count_list = {}
    cluster_user_pss_score_list = {}
    cluster_user_coping_score_list = {}
    cluster_user_strategy_list = {}
    cluster_user_type_list = {}
    for user, label in enumerate(clusters.labels_):
        if label not in cluster_user_list:
            cluster_user_list[label] = []
            cluster_age_list[label] = []
            cluster_gender_id_list[label] = []
            cluster_group_id_list[label] = []
            cluster_avg_user_rate_list[label] = []
            cluster_user_rate_size_list[label] = []
            cluster_avg_user_difficulty_list[label] = []
            cluster_avg_user_stress_list[label] = []
            cluster_user_stress_var_list[label] = []
            cluster_user_stress_size_list[label] = []
            cluster_user_done_ratio_list[label] = []
            cluster_user_done_count_list[label] = []
            cluster_user_undone_count_list[label] = []
            cluster_user_pss_score_list[label] = []
            cluster_user_coping_score_list[label] = []
            cluster_user_strategy_list[label] = []
            cluster_user_type_list[label] = []
        cluster_user_list[label].append(user)
        if user in user_age_dict:
            cluster_age_list[label].append(user_age_dict[user])
        if user in user_gender_id_dict:
            cluster_gender_id_list[label].append(user_gender_id_dict[user])
        if user in user_group_id_dict:
            cluster_group_id_list[label].append(user_group_id_dict[user])
        if user in user_rate_list_dict:
            cluster_avg_user_rate_list[label].append(np.mean(user_rate_list_dict[user]))
            cluster_user_rate_size_list[label].append(len(user_rate_list_dict[user]))
            cluster_avg_user_difficulty_list[label].append(np.mean(user_difficulty_list_dict[user]))
        else:
            cluster_user_rate_size_list[label].append(0)
        if user in user_stress_list_dict:
            cluster_avg_user_stress_list[label].append(np.mean(user_stress_list_dict[user]))
            cluster_user_stress_var_list[label].append(np.std(user_stress_list_dict[user]) ** 2)
            cluster_user_stress_size_list[label].append(len(user_stress_list_dict[user]))
        if user in user_done_list_dict:
            cluster_user_done_ratio_list[label].append(np.mean(user_done_list_dict[user]))
            cluster_user_done_count_list[label].append(np.sum(user_done_list_dict[user]))
            cluster_user_undone_count_list[label].append(
                len(user_done_list_dict[user]) - np.sum(user_done_list_dict[user])
            )
        if user in user_pss_score_dict:
            cluster_user_pss_score_list[label].append(user_pss_score_dict[user])
        if user in user_coping_score_dict:
            cluster_user_coping_score_list[label].append(user_coping_score_dict[user])
        if user in user_strategy_list_dict:
            cluster_user_strategy_list[label].append(user_strategy_list_dict[user])
        if user in user_type_list_dict:
            cluster_user_type_list[label].append(user_type_list_dict[user])

    print("cluster -> user list: {}".format(cluster_user_list))
    print("cluster -> age list: {}".format(cluster_age_list))
    print("cluster -> gender list: {}".format(cluster_gender_id_list))
    print("cluster -> group list: {}".format(cluster_group_id_list))
    print("cluster -> avg. rate list: {}".format(cluster_avg_user_rate_list))
    print("cluster -> rate count list: {}".format(cluster_user_rate_size_list))
    print("cluster -> avg. difficulty list: {}".format(cluster_avg_user_difficulty_list))
    print("cluster -> avg. stress list: {}".format(cluster_avg_user_stress_list))
    print("cluster -> stress var list: {}".format(cluster_user_stress_var_list))
    print("cluster -> stress count list: {}".format(cluster_user_stress_size_list))
    print("cluster -> done ratio list: {}".format(cluster_user_done_ratio_list))
    print("cluster -> done count list: {}".format(cluster_user_done_count_list))
    print("cluster -> undone count list: {}".format(cluster_user_undone_count_list))
    print("cluster -> pss score list: {}".format(cluster_user_pss_score_list))
    print("cluster -> coping score list: {}".format(cluster_user_coping_score_list))
    for cluster in sorted(cluster_age_list.keys()):
        value_list = cluster_age_list[cluster]
        print("{} & {:.2f} & {:.2f} ".format(cluster, np.mean(value_list), np.std(value_list)),
              end="")
        value_list = cluster_avg_user_stress_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)), end="")
        value_list = cluster_user_stress_var_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)), end="")
        value_list = cluster_user_pss_score_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)))
    for cluster in sorted(cluster_avg_user_rate_list.keys()):
        value_list = cluster_avg_user_rate_list[cluster]
        print("{} & {:.2f} & {:.2f} ".format(cluster, np.mean(value_list), np.std(value_list)),
              end="")
        value_list = cluster_user_rate_size_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)), end="")
        value_list = cluster_user_done_ratio_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)), end="")
        value_list = cluster_user_done_count_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)))

    prBlue("Cluster Type Analysis")
    print(type_name_id_mapping)
    observed = np.zeros((n_clusters, len(type_name_id_mapping)))
    for cluster in sorted(cluster_user_type_list.keys()):
        observed[cluster] = np.sum(cluster_user_type_list[cluster], axis=0)
    print(observed)
    expected = np.zeros((n_clusters, len(type_name_id_mapping)))
    for cluster in sorted(cluster_user_type_list.keys()):
        for type_id in range(len(type_name_id_mapping)):
            total = np.sum(observed)
            exp = np.sum(observed[:, type_id]) * np.sum(observed[cluster, :]) / total
            expected[cluster, type_id] = exp
    print(expected)
    m = np.zeros((n_clusters, len(type_name_id_mapping)))
    all_labels = []
    for cluster in sorted(cluster_user_type_list.keys()):
        labels = []
        obs = observed[cluster, :]
        exp = expected[cluster, :]
        chisq, p_val = stats.chisquare(obs, exp)
        # print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        print("{} & ".format(cluster), end="")
        for type_id in range(len(type_name_id_mapping)):
            obs_val = obs[type_id]
            exp_val = exp[type_id]
            normalized_diff = (obs_val - exp_val) / exp_val
            m[cluster, type_id] = normalized_diff
            la = "obs={:.2f}\nexp={:.2f}".format(obs_val, exp_val)
            labels.append(la)
            print("{:.2f} & {:.2f} &".format(obs_val, exp_val), end="")
        print("{:.2f} & {:.3f}".format(chisq, p_val))
        all_labels.append(labels)

    xlabels = ["Social Engagement", "Physical Activity", "Mindfulness", "Positive Thinking",
               "Enjoyable Activities"]
    plt.figure(figsize=(10, 5))
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    # sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(m, linewidths=1., ax=ax, cbar=True, annot=all_labels, fmt="", cmap="vlag",
                vmin=-1., vmax=1.,
                cbar_kws={"shrink": 0.95, 'label': 'Normalized Difference:\n (Obs-Exp) / Exp'},
                xticklabels=xlabels)
    ax.tick_params(axis='x', rotation=15)
    plt.xlabel("Item Type", fontsize=20)
    plt.ylabel("Cluster", fontsize=20)
    plt.tight_layout(pad=0.)
    plt.savefig("figures/user_chisquare_item_type.pdf")
    plt.show()
    plt.clf()

    prBlue("Cluster Strategy Type Analysis")
    print(strategy_name_id_mapping)
    m = np.zeros((n_clusters, len(strategy_name_id_mapping)))
    all_labels = []
    observed = np.zeros((n_clusters, len(strategy_name_id_mapping)))
    for cluster in sorted(cluster_user_strategy_list.keys()):
        observed[cluster] = np.sum(cluster_user_strategy_list[cluster], axis=0)
    print(observed)
    expected = np.zeros((n_clusters, len(strategy_name_id_mapping)))
    for cluster in sorted(cluster_user_strategy_list.keys()):
        for strategy_id in range(len(strategy_name_id_mapping)):
            total = np.sum(observed)
            exp = np.sum(observed[:, strategy_id]) * np.sum(observed[cluster, :]) / total
            expected[cluster, strategy_id] = exp
    print(expected)
    for cluster in sorted(cluster_user_strategy_list.keys()):
        # sta, p_val = stats.power_divergence(obs, exp)
        # print("power div test, cluster: {}, stat val: {}, p-val: {}".format(cluster, sta, p_val))
        labels = []
        obs = observed[cluster, :]
        exp = expected[cluster, :]
        chisq, p_val = stats.chisquare(obs, exp)
        # print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        print("{} & ".format(cluster), end="")
        for type_id in range(len(strategy_name_id_mapping)):
            obs_val = obs[type_id]
            exp_val = exp[type_id]
            normalized_diff = (obs_val - exp_val) / exp_val
            m[cluster, type_id] = normalized_diff
            la = "obs={:.2f}\nexp={:.2f}".format(obs_val, exp_val)
            labels.append(la)
            print("{:.2f} & {:.2f} &".format(obs_val, exp_val), end="")
        # print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        print("{:.2f} & {:.3f}".format(chisq, p_val))
        all_labels.append(labels)

    xlabels = ["Distancing", "Seeking Social Support", "Planful Problem Solving",
               "Accepting Responsibility", "Confront the Problem", "Positive Reappraisal",
               "Self-controlling", "Accepting of the Problem"]
    plt.figure(figsize=(20, 5))
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 5))
    # sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(m, linewidths=1., ax=ax, cbar=True, annot=all_labels, fmt="", cmap="vlag",
                cbar_kws={"shrink": 0.95, 'label': 'Normalized Difference:\n (Obs-Exp) / Exp'},
                xticklabels=xlabels)
    ax.tick_params(axis='x', rotation=15)
    plt.xlabel("Coping Strategy Type", fontsize=20)
    plt.ylabel("User Cluster", fontsize=20)
    plt.tight_layout(pad=0.)
    plt.savefig("figures/user_chisquare_cs_type.pdf")
    plt.show()
    plt.clf()

    # use PCA for visualization

    # use item factor and item bias to plot 2-dimensional scatter plots
    # model_df = pd.DataFrame({"X": np.squeeze(user_factors), "Y": np.squeeze(rate_user_biases),
    #                          "Cluster": clusters.labels_})
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=7, init="random")
    tsne_obj = tsne.fit_transform(X)
    model_df = pd.DataFrame({"X": tsne_obj[:, 0], "Y": tsne_obj[:, 1],
                             "Cluster": clusters.labels_})
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'slategray', 'darkcyan',
              'brown', 'dodgerblue', 'gold']
    colors = colors[:n_clusters]
    # plot scatter plots, colored with cluster label, tagged with user id
    scatter_plot(data_frame=model_df, colors=colors, tags=list(range(model_df.shape[0])),
                 title="Clusters with Tagged User ID",
                 fig_name="user_cluster_with_user_id")

    # prBlue("Cluster Age Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_age_dict,
    #              title="Clusters with Tagged User Age", x_label="User Factor", y_label="User Bias")
    heatmap_plot(n_clusters, cluster_age_list, equal_var=False,
                 title="User Age Comparison",
                 fig_name="user_heatmap_age")

    # prBlue("Cluster Gender Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_gender_dict,
    #              title="Clusters with Tagged User Gender",
    #              x_label="User Factor",
    #              y_label="User Bias")
    # heatmap_plot(n_clusters, cluster_gender_id_list, equal_var=False,
    #              title="User Gender Comparison")

    # prBlue("Cluster Group Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_group_dict,
    #              title="Clusters with Tagged User Group",
    #              x_label="User Factor",
    #              y_label="User Bias")
    # heatmap_plot(n_clusters, cluster_group_id_list, equal_var=False,
    #              title="User Group Comparison")

    prBlue("Cluster Rate Analysis")
    user_avg_rate_dict = {}
    user_rate_count_dict = {}
    user_avg_difficulty_dict = {}
    user_avg_stress_dict = {}
    user_stress_var_dict = {}
    user_stress_count_dict = {}
    user_done_ratio_dict = {}
    user_done_count_dict = {}
    user_undone_count_dict = {}
    for user in range(len(user_name_id_mapping)):
        if user in user_rate_list_dict:
            user_avg_rate_dict[user] = np.round(np.mean(user_rate_list_dict[user]), 1)
            user_rate_count_dict[user] = len(user_rate_list_dict[user])
            user_avg_difficulty_dict[user] = np.round(np.mean(user_difficulty_list_dict[user]), 1)
        else:
            user_avg_rate_dict[user] = '?'
            user_rate_count_dict[user] = 0
            user_avg_difficulty_dict[user] = 0
        if user in user_stress_list_dict:
            user_avg_stress_dict[user] = np.round(np.mean(user_stress_list_dict[user]), 1)
            user_stress_var_dict[user] = np.round(np.std(user_stress_list_dict[user]), 1)
            user_stress_count_dict[user] = len(user_stress_list_dict[user])
        else:
            user_avg_stress_dict[user] = '?'
            user_stress_var_dict[user] = '?'
            user_stress_count_dict[user] = 0
        if user in user_done_list_dict:
            user_done_ratio_dict[user] = np.round(np.mean(user_done_list_dict[user]), 1)
            user_done_count_dict[user] = np.sum(user_done_list_dict[user])
            user_undone_count_dict[user] = len(user_done_list_dict[user]) - np.sum(
                user_done_list_dict[user])
        else:
            user_done_ratio_dict[user] = "?"
            user_done_count_dict[user] = "?"
            user_undone_count_dict[user] = "?"
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_avg_rate_dict,
    #              title="Clusters with Tagged User Avg. Rate",
    #              x_label="User Factor",
    #              y_label="User Bias")
    heatmap_plot(n_clusters, cluster_avg_user_rate_list, equal_var=False,
                 title="User Avg. Rate Comparison",
                 fig_name="user_heatmap_avg_rate")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_rate_count_dict,
    #              title="Clusters with Tagged User Rate Count",
    #              x_label="User Factor",
    #              y_label="User Bias")
    heatmap_plot(n_clusters, cluster_user_rate_size_list, equal_var=False,
                 title="User Rate Count Comparison",
                 fig_name="user_heatmap_rate_count")

    # prBlue("Cluster Difficulty Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_avg_difficulty_dict,
    #              title="Clusters with Tagged User Avg. Item Difficulty",
    #              x_label="User Factor",
    #              y_label="User Bias")
    # heatmap_plot(n_clusters, cluster_avg_user_difficulty_list, equal_var=False,
    #              title="User Avg. Item Difficulty Comparison")

    # prBlue("Cluster Stress Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_avg_stress_dict,
    #              title="Clusters with Tagged User Avg. Stress Level",
    #              x_label="User Factor",
    #              y_label="User Bias")
    heatmap_plot(n_clusters, cluster_avg_user_stress_list, equal_var=False,
                 title="User Avg. Stress Comparison",
                 fig_name="user_heatmap_avg_stress")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_stress_var_dict,
    #              title="Clusters with Tagged User Stress Variance",
    #              x_label="User Factor",
    #              y_label="User Bias")
    heatmap_plot(n_clusters, cluster_user_stress_var_list, equal_var=False,
                 title="User Stress Variance Comparison",
                 fig_name="user_heatmap_stress_var")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_stress_count_dict,
    #              title="Clusters with Tagged User Stress Measure Count",
    #              x_label="User Factor",
    #              y_label="User Bias")
    # heatmap_plot(n_clusters, cluster_user_stress_size_list, equal_var=False,
    #              title="User Stress Measure Count Comparison")

    prBlue("Cluster Done Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_done_ratio_dict,
    #              title="Clusters with Tagged User Done Ratio",
    #              x_label="User Factor",
    #              y_label="User Bias")
    heatmap_plot(n_clusters, cluster_user_done_ratio_list, equal_var=False,
                 title="User Engagement Ratio Comparison",
                 fig_name="user_heatmap_engagement_ratio")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_done_count_dict,
    #              title="Clusters with Tagged User Done Count",
    #              x_label="User Factor",
    #              y_label="User Bias")
    heatmap_plot(n_clusters, cluster_user_done_count_list, equal_var=False,
                 title="User Engagement Count Comparison",
                 fig_name="user_heatmap_engagement_count")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_undone_count_dict,
    #              title="Clusters with Tagged User UnDone Count",
    #              x_label="User Factor",
    #              y_label="User Bias")
    # heatmap_plot(n_clusters, cluster_user_undone_count_list, equal_var=False,
    #              title="User UnDone Count Comparison")

    prBlue("Cluster PS Score Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_pss_score_dict,
    #              title="Clusters with Tagged User PSS Score",
    #              x_label="User Factor", y_label="User Bias")
    heatmap_plot(n_clusters, cluster_user_pss_score_list, equal_var=False,
                 title="User PS Score Comparison",
                 fig_name="user_heatmap_ps_score")
    #
    prBlue("Cluster CS Score Analysis")
    # for user_id in range(len(user_name_id_mapping)):
    #     if user_id not in user_coping_score_dict:
    #         user_coping_score_dict[user_id] = "?"
    # scatter_plot(data_frame=model_df, colors=colors, tags=user_coping_score_dict,
    #              title="Clusters with Tagged User Coping Score",
    #              x_label="User Factor", y_label="User Bias")
    heatmap_plot(n_clusters, cluster_user_coping_score_list, equal_var=False,
                 title="User Coping Strategy Score Comparison",
                 fig_name="user_heatmap_cs_score")
    return cluster_score1, cluster_score2, cluster_score3


def user_cluster_analysis(fold, clustering, n_clusters):
    score1_list = []
    score2_list = []
    score3_list = []
    for n in n_clusters:
        score1, score2, score3 = user_analysis(clustering, n_clusters=n, fold=fold)
        score1_list.append(score1)
        score2_list.append(score2)
        score3_list.append(score3)
    for score in score1_list[:-1]:
        print("{} ".format(score), end="")
    print(score1_list[-1])
    for score in score2_list[:-1]:
        print("{} ".format(score), end="")
    print(score2_list[-1])
    for score in score3_list[:-1]:
        print("{} ".format(score), end="")
    print(score3_list[-1])


def user_record_analysis():
    data = pickle.load(open("data/data.pkl", "rb"))
    for user_id in data['activity'].keys():
        records = data['activity'][user_id]
        rate = []
        stress = []
        rate_records = []
        stress_records = []
        done = []
        done_records = []
        for (user_id, time_id, resource, item_id, value) in records:
            if resource == "rate":
                rate.append(value)
                rate_records.append([user_id, time_id, resource, item_id, value])
            elif resource == "stress":
                stress.append(value)
                stress_records.append([user_id, time_id, resource, item_id, value])
            elif resource == "done":
                done.append(value)
                done_records.append([user_id, time_id, resource, item_id, value])
            else:
                raise ValueError
        # if len(rate) != 0:
        #     print("{};{};{};{}".format(user_id, len(rate), np.mean(rate), rate_records))
        # if len(stress) != 0:
        #     print("{};{};{};{}".format(user_id, len(stress), np.mean(stress), stress_records))
        if len(done) != 0:
            print("{};{};{};{}".format(user_id, len(done), np.mean(done), done_records))


def check(fold):
    output_data = torch.load("experiments/MVTFAgent/out/stress_output_fold_{}.pth.tar".format(fold))
    stress_item_factor = output_data["stress_item_factor"]
    stress_item_biases = output_data["stress_item_biases"]
    stress_user_biases = output_data["stress_user_biases"]
    time_biases = output_data["time_biases"]
    user_factors = output_data["user_factors"]
    time_factors = output_data["time_factors"]
    item_factors = output_data["item_factors"]
    user = 6
    print(item_factors)
    n_factors = user_factors.shape[1]
    for attempt in range(len(time_factors)):
        u_factor = user_factors[user].squeeze(dim=0)
        t_factor = time_factors[attempt].squeeze(dim=0)
        t_matrix = t_factor.reshape(n_factors, n_factors)
        stress = stress_user_biases[user].squeeze() + time_biases[attempt].squeeze() \
                 + stress_item_biases[torch.tensor(0)].squeeze()
        stress += torch.dot(torch.matmul(u_factor, t_matrix), stress_item_factor[torch.tensor(0)])
        stress = torch.sigmoid(stress) * 5.
        print(user, attempt, stress.item())


if __name__ == '__main__':
    # step 2. choose the same fold data as item cluster analysis for user cluster analysis
    user_cluster_analysis(fold=5, clustering="kmeans", n_clusters=[4])
    # check(5)
    # user_cluster_analysis(fold=1, clustering="spectral", n_clusters=[4])
    # user_cluster_analysis(fold=4, clustering="kmeans", n_clusters=[4])
    # user_record_analysis()
