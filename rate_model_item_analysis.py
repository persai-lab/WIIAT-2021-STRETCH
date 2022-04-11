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
from sklearn.manifold import TSNE


def item_analysis(clustering, n_clusters, fold, random_state=1024):
    prBlue("\n\nLoading Data...fold: {}, n_clusters: {}".format(fold, n_clusters))
    data = pickle.load(open("data/data.pkl", "rb"))
    item_name_id_mapping = data["item_name_id_mapping"]
    output_data = torch.load("experiments/MVTFAgent/out/rate_output_fold_{}.pth.tar".format(fold))
    print(output_data.keys())
    item_df = pd.read_csv("data/rec_item.csv")
    item_df = item_df[["rec_index", "coping_strategy", "difficulty", "tag", "type"]]

    # with open("data/rec_item.csv", "r") as f:
    #     for line in f:
    #         fields = line.strip().split(",")
    #         rec_index = fields[0]
    #         if rec_index in item_name_id_mapping:
    #             item_id = item_name_id_mapping[rec_index]
    #             print("{},{}".format(item_id, line), end="")
    #     # rec_index = str(rec_index)
    #     # if rec_index in item_name_id_mapping:
    #     #     item_id = item_name_id_mapping[rec_index]
    #     #     print("{};{};{};{};{};{};{}".format(
    #     #         item_id, rec_index, coping_strategy, difficulty, tag, type, short_dec
    #     #     ))

    # step 1. build tag name id mapping
    item_engagement_dict = {}
    tag_id_mapping = {}
    tag_index = 0
    for item_name, coping_strategy, difficulty, tag, type in item_df.to_numpy():
        item_name = str(item_name)
        if item_name in item_name_id_mapping:
            tags = tag.strip().split(", ")
            for t in tags:
                if t not in tag_id_mapping:
                    tag_id_mapping[t] = tag_index
                    tag_index += 1
            item_id = item_name_id_mapping[item_name]
            if coping_strategy in ["distancing", "escape avoidance"]:
                item_engagement_dict[item_id] = 1
            else:
                item_engagement_dict[item_id] = 0

    # step 2. create mappings
    item_type_dict = {}  # map item id to type name
    type_name_type_id_mapping = {}  # map type name to type id
    item_difficulty_dict = {}  # map item id to difficulty
    type_item_list_dict = {}  # map type name to item list
    type_index = 0  # used to build type_name_type_id_mapping
    strategy_index = 0
    item_strategy_dict = {}
    strategy_name_strategy_id_mapping = {}
    # item_id_y = {}
    for item_name, coping_strategy, difficulty, tag, type in item_df.to_numpy():
        item_name = str(item_name)
        if item_name in item_name_id_mapping:
            tags = tag.strip().split(", ")
            tags_vector = np.zeros(len(tag_id_mapping))
            for t in tags:
                tid = tag_id_mapping[t]
                tags_vector[tid] = 1
            item_id = item_name_id_mapping[item_name]
            item_type_dict[item_id] = type
            item_strategy_dict[item_id] = coping_strategy
            if type not in type_item_list_dict:
                type_item_list_dict[type] = []
            if type not in type_name_type_id_mapping:
                type_name_type_id_mapping[type] = type_index
                type_index += 1
            type_item_list_dict[type].append(item_id)
            if coping_strategy not in strategy_name_strategy_id_mapping:
                strategy_name_strategy_id_mapping[coping_strategy] = strategy_index
                strategy_index += 1

            if difficulty == "very easy":
                item_difficulty_dict[item_id] = 1
                type_id = type_name_type_id_mapping[type]
                # item_id_y[item_id] = [type_id, 5] + list(tags_vector)
            elif difficulty == "easy":
                item_difficulty_dict[item_id] = 2
                type_id = type_name_type_id_mapping[type]
                # item_id_y[item_id] = [type_id, 4] + list(tags_vector)
            elif difficulty == "medium":
                item_difficulty_dict[item_id] = 3
                type_id = type_name_type_id_mapping[type]
                # item_id_y[item_id] = [type_id, 3] + list(tags_vector)
            elif difficulty == "hard":
                item_difficulty_dict[item_id] = 4
                type_id = type_name_type_id_mapping[type]
                # item_id_y[item_id] = [type_id, 2] + list(tags_vector)
            elif difficulty == "very hard":
                item_difficulty_dict[item_id] = 5
                type_id = type_name_type_id_mapping[type]
                # item_id_y[item_id] = [type_id, 1] + list(tags_vector)
            else:
                raise ValueError
    pickle.dump(item_difficulty_dict, open("data/item_difficulty.pkl", "wb"))
    strategy_data = {
        "item_strategy_dict": item_strategy_dict,
        "strategy_name_id_mapping": strategy_name_strategy_id_mapping
    }
    pickle.dump(strategy_data, open("data/coping_strategy_data.pkl", "wb"))
    type_data = {
        "item_type_dict": item_type_dict,
        "type_name_id_mapping": type_name_type_id_mapping
    }
    pickle.dump(type_data, open("data/type_data.pkl", "wb"))

    for type in type_item_list_dict:
        print("type: {}, type-id: {}. size: {}, items: {}".format(
            type, type_name_type_id_mapping[type], len(type_item_list_dict[type]),
            type_item_list_dict[type]))

    # step 3. build item-id and rating-list mapping from user data
    item_rate_list_dict = {}
    item_done_list_dict = {}
    for user_id in data['activity'].keys():
        records = data['activity'][user_id]
        for (user_id, time_id, resource, item_id, value) in records:
            if resource == "rate":
                if item_id not in item_rate_list_dict:
                    item_rate_list_dict[item_id] = []
                item_rate_list_dict[item_id].append(value)
            elif resource == "done":
                if item_id not in item_done_list_dict:
                    item_done_list_dict[item_id] = []
                item_done_list_dict[item_id].append(value)

    item_avg_rate_list = []
    for i in item_done_list_dict:
        v = np.sum(item_done_list_dict[i])
        if v != 0 :
            item_avg_rate_list.append(v)

    print("rate: size: {}, mean: {}, std: {}, min: {}, med.: {}, max: {}".format(
        len(item_avg_rate_list), np.mean(item_avg_rate_list), np.std(item_avg_rate_list),
        np.min(item_avg_rate_list), np.median(item_avg_rate_list), np.max(item_avg_rate_list)
    ))
    # step 4. clustering based on item factor and item bias
    # concatenate item_factors and item_biases, and fit into clustering algo.
    prRed("******************************** clustering results ***********************************")
    item_factors = output_data["item_factors"].detach().numpy()
    rate_item_biases = output_data["rate_item_biases"].detach().numpy()
    done_item_biases = output_data["done_item_biases"].detach().numpy()
    X = []
    for item_name in item_name_id_mapping.keys():
        item_id = item_name_id_mapping[item_name]
        item_factor = item_factors[item_id]
        item_rate_bias = rate_item_biases[item_id]
        item_done_bias = done_item_biases[item_id]
        x = np.append(np.append(item_factor, item_rate_bias), item_done_bias)
        # x = np.append(item_factor, item_rate_bias)
        X.append(x)

    if clustering == "kmeans":
        clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    elif clustering == "spectral":
        clusters = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize",
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
    # print("silhouette score (larger is better): {}".format(cluster_score))
    # print("calinski harabasz score (larger is better): {}".format(cluster_score))
    # print("davies bouldin score (smaller is better): {}".format(cluster_score))

    # use type_id as cluster label and compute the clustering performance
    true_labels = []
    item_difficulty_list = []
    item_id_type_id_dict = {}
    item_id_strategy_id_dict = {}
    for item_id in sorted(item_type_dict.keys()):
        type_id = type_name_type_id_mapping[item_type_dict[item_id]]
        strategy_id = strategy_name_strategy_id_mapping[item_strategy_dict[item_id]]
        true_labels.append(type_id)
        item_difficulty_list.append(item_difficulty_dict[item_id])
        if item_id not in item_id_type_id_dict:
            item_id_type_id_dict[item_id] = type_id
        if item_id not in item_id_strategy_id_dict:
            item_id_strategy_id_dict[item_id] = strategy_id
    adj_rand_score = metrics.adjusted_rand_score(true_labels, clusters.labels_)
    adj_mutual_info_score = metrics.adjusted_mutual_info_score(true_labels, clusters.labels_)
    prRed("---------------- clustering performance with item type as true label ------------------")
    print("adj_rand_score: {}".format(adj_rand_score))
    print("adj_mutual_info_score: {}".format(adj_mutual_info_score))
    print("true (type    id): {}".format(true_labels))
    print("pred (cluster id): {}".format(list(clusters.labels_)))

    # compute the correlation between item bias, factor and item difficulty
    prRed("---------- correlation between item difficulty with item factor or bias ---------------")
    print("item factors: {}".format(np.squeeze(item_factors)))
    print("item biases: {}".format(np.squeeze(rate_item_biases)))
    print("item difficulty list: {}".format(item_difficulty_list))
    # print("item factor and difficulty correlation: {}".format(
    #     stats.pearsonr(np.squeeze(item_factors), np.array(item_difficulty_list, dtype=float))))
    print("item biases and difficulty correlation: {}".format(
        stats.pearsonr(np.squeeze(rate_item_biases), np.array(item_difficulty_list, dtype=float))))

    # step 5. clusters analysis, clusters comparison, and outlier analysis
    prRed("********************************* clusters analysis ***********************************")
    cluster_item_list = {}
    cluster_difficulty_list = {}
    cluster_type_list = {}
    cluster_strategy_list = {}
    cluster_engagement_list = {}
    cluster_rate_list = {}
    cluster_avg_item_rate_list = {}
    cluster_sum_item_rate_list = {}
    cluster_item_rate_size_list = {}
    cluster_item_done_ratio_list = {}
    cluster_item_done_size_list = {}
    cluster_item_undone_size_list = {}
    for item, label in enumerate(clusters.labels_):
        if label not in cluster_item_list:
            cluster_item_list[label] = []
            cluster_difficulty_list[label] = []
            cluster_type_list[label] = []
            cluster_strategy_list[label] = []
            cluster_engagement_list[label] = []
            cluster_rate_list[label] = []
            cluster_avg_item_rate_list[label] = []
            cluster_sum_item_rate_list[label] = []
            cluster_item_rate_size_list[label] = []
            cluster_item_done_ratio_list[label] = []
            cluster_item_done_size_list[label] = []
            cluster_item_undone_size_list[label] = []
        cluster_item_list[label].append(item)
        cluster_difficulty_list[label].append(item_difficulty_dict[item])
        cluster_type_list[label].append(type_name_type_id_mapping[item_type_dict[item]])
        cluster_strategy_list[label].append(
            strategy_name_strategy_id_mapping[item_strategy_dict[item]])
        cluster_engagement_list[label].append(item_engagement_dict[item])
        if item in item_rate_list_dict:  # if item is ever rated by users
            cluster_rate_list[label] += item_rate_list_dict[item]
            cluster_avg_item_rate_list[label].append(np.mean(item_rate_list_dict[item]))
            cluster_sum_item_rate_list[label].append(np.sum(item_rate_list_dict[item]))
            cluster_item_rate_size_list[label].append(len(item_rate_list_dict[item]))
        if item in item_done_list_dict:
            cluster_item_done_ratio_list[label].append(np.mean(item_done_list_dict[item]))
            cluster_item_done_size_list[label].append(np.sum(item_done_list_dict[item]))
            cluster_item_undone_size_list[label].append(
                len(item_done_list_dict[item]) - np.sum(item_done_list_dict[item])
            )
            # else:
            # cluster_rate_list[label] += item_rate_list_dict[item]
            # cluster_avg_item_rate_list[label].append(np.mean(item_rate_list_dict[item]))
            # cluster_item_rate_size_list[label].append(len(item_rate_list_dict[item]))

    print("cluster -> item list: {}".format(cluster_item_list))
    print("cluster -> difficulty list: {}".format(cluster_difficulty_list))
    print("cluster -> rate list: {}".format(cluster_rate_list))
    print("cluster -> avg item rate list: {}".format(cluster_avg_item_rate_list))
    print("cluster -> sum item rate list: {}".format(cluster_sum_item_rate_list))
    print("cluster -> item rate size list: {}".format(cluster_item_rate_size_list))
    print("cluster -> item done ratio list: {}".format(cluster_item_done_ratio_list))
    print("cluster -> item done count list: {}".format(cluster_item_done_size_list))
    print("cluster -> item undone count list: {}".format(cluster_item_undone_size_list))

    for cluster in sorted(cluster_avg_item_rate_list.keys()):
        value_list = cluster_item_rate_size_list[cluster]
        print("{} & {:.2f} & {:.2f} ".format(cluster, np.mean(value_list), np.std(value_list)),
              end="")
        value_list = cluster_avg_item_rate_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)), end="")
        value_list = cluster_item_done_ratio_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)), end="")
        value_list = cluster_item_done_size_list[cluster]
        print("& {:.2f} & {:.2f} ".format(np.mean(value_list), np.std(value_list)))

    # chi square test among cluster and item-type
    prBlue("Cluster Type Analysis")
    print(type_name_type_id_mapping)
    observed = np.zeros((n_clusters, len(type_name_type_id_mapping)))
    for cluster in sorted(cluster_type_list):
        type_list = cluster_type_list[cluster]
        for type_id in type_list:
            observed[cluster, type_id] += 1
    print(observed)
    expected = np.zeros((n_clusters, len(type_name_type_id_mapping)))
    for cluster in sorted(cluster_type_list):
        for type_id in range(len(type_name_type_id_mapping)):
            total = np.sum(observed)
            exp = np.sum(observed[:, type_id]) * np.sum(observed[cluster, :]) / total
            expected[cluster, type_id] = exp
    print(expected)
    m = np.zeros((n_clusters, len(type_name_type_id_mapping)))
    all_labels = []
    for cluster in sorted(cluster_type_list):
        obs = observed[cluster, :]
        exp = expected[cluster, :]
        chisq, p_val = stats.chisquare(obs, exp)
        print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        labels = []
        obs = observed[cluster, :]
        exp = expected[cluster, :]
        chisq, p_val = stats.chisquare(obs, exp)
        # print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        print("{} & ".format(cluster), end="")
        for type_id in range(len(type_name_type_id_mapping)):
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
    plt.ylabel("Item Cluster", fontsize=20)
    plt.tight_layout(pad=0.)
    plt.savefig("figures/item_chisquare_item_type.pdf")
    plt.show()
    plt.clf()




    prBlue("Cluster Strategy Type Analysis")
    print(strategy_name_strategy_id_mapping)
    m = np.zeros((n_clusters, len(strategy_name_strategy_id_mapping)))
    all_labels = []
    observed = np.zeros((n_clusters, len(strategy_name_strategy_id_mapping)))
    for cluster in sorted(cluster_strategy_list.keys()):
        strategy_list = cluster_strategy_list[cluster]
        for strategy_id in strategy_list:
            observed[cluster, strategy_id] += 1
    print(observed)
    expected = np.zeros((n_clusters, len(strategy_name_strategy_id_mapping)))
    for cluster in sorted(cluster_strategy_list.keys()):
        for strategy_id in range(len(strategy_name_strategy_id_mapping)):
            total = np.sum(observed)
            exp = np.sum(observed[:, strategy_id]) * np.sum(observed[cluster, :]) / total
            expected[cluster, strategy_id] = exp
    print(expected)
    for cluster in sorted(cluster_strategy_list.keys()):
        # obs = observed[cluster, :]
        # exp = expected[cluster, :]
        # chisq, p_val = stats.chisquare(obs, exp)
        # print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        # sta, p_val = stats.power_divergence(obs, exp)
        # print("power div test, cluster: {}, stat val: {}, p-val: {}".format(cluster, sta, p_val))

        labels = []
        obs = observed[cluster, :]
        exp = expected[cluster, :]
        chisq, p_val = stats.chisquare(obs, exp)
        # print("chisquare test, cluster: {}, stat val: {}, p-val: {}".format(cluster, chisq, p_val))
        print("{} & ".format(cluster), end="")
        for type_id in range(len(strategy_name_strategy_id_mapping)):
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
                cbar_kws={"shrink": 0.95, 'label': 'Normalized Difference: (Obs-Exp) / Exp'},
                xticklabels=xlabels)
    ax.tick_params(axis='x', rotation=15)
    plt.xlabel("Coping Strategy Type", fontsize=20)
    plt.ylabel("Item Cluster", fontsize=20)
    plt.tight_layout(pad=0.)
    plt.savefig("figures/item_chisquare_cs_type.pdf")
    plt.show()
    plt.clf()

    # use item factor and item bias to plot 2-dimensional scatter plots
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=11)
    tsne_obj = tsne.fit_transform(X)
    model_df = pd.DataFrame({"X": tsne_obj[:, 0], "Y": tsne_obj[:, 1],
                             "Cluster": clusters.labels_})
    # model_df = pd.DataFrame({"X": np.squeeze(item_factors), "Y": np.squeeze(rate_item_biases),
    #                          "Cluster": clusters.labels_})
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'slategray', 'darkcyan',
              'brown', 'dodgerblue', 'gold']
    colors = colors[:n_clusters]
    # plot scatter plots, colored with cluster label, tagged with item id
    scatter_plot(data_frame=model_df, colors=colors, tags=list(range(model_df.shape[0])),
                 y_label="Item Rate Bias",
                 title="Clusters with Tagged Item ID",
                 fig_name="item_cluster_with_item_id")

    prBlue("Cluster Difficulty Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_difficulty_dict,
    #              title="Clusters with Tagged Item Difficulty")
    # heatmap_plot(n_clusters, cluster_difficulty_list, equal_var=False,
    #              title="Item Difficulty Comparison")

    # prBlue("Cluster Type Analysis")
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_id_type_id_dict,
    #              title="Clusters with Tagged Item Type")
    # heatmap_plot(n_clusters, cluster_type_list, equal_var=False,
    #              title="Item Type Comparison")
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_engagement_dict,
    #              title="Clusters with Tagged Item Strategy Type")
    # heatmap_plot(n_clusters, cluster_engagement_list, equal_var=False,
    #              title="Item Strategy Type Comparison")

    prBlue("Cluster Rate Analysis")
    item_avg_rate_dict = {}
    item_rate_count_dict = {}
    for item in range(len(item_name_id_mapping)):
        if item in item_rate_list_dict:
            item_avg_rate_dict[item] = np.round(np.mean(item_rate_list_dict[item]), 1)
            item_rate_count_dict[item] = len(item_rate_list_dict[item])
        else:
            item_avg_rate_dict[item] = '?'
            item_rate_count_dict[item] = 0
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_avg_rate_dict,
    #              title="Clusters with Tagged Item Avg. Rate")
    heatmap_plot(n_clusters, cluster_avg_item_rate_list, equal_var=False,
                 title="Item Avg. Rate Comparison",
                 fig_name="item_heatmap_avg_rate")
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_rate_count_dict,
    #              title="Clusters with Tagged Item Rate Count")
    heatmap_plot(n_clusters, cluster_item_rate_size_list, equal_var=False,
                 title="Item Rate Count Comparison",
                 fig_name="item_heatmap_rate_count")

    prBlue("Cluster Done Analysis")
    item_done_ratio_dict = {}
    item_done_count_dict = {}
    item_undone_count_dict = {}
    for item in range(len(item_name_id_mapping)):
        if item in item_done_list_dict:
            item_done_ratio_dict[item] = np.round(np.mean(item_done_list_dict[item]), 1)
            item_done_count_dict[item] = np.sum(item_done_list_dict[item])
            item_undone_count_dict[item] = len(item_done_list_dict[item]) - np.sum(
                item_done_list_dict[item])
        else:
            item_done_ratio_dict[item] = "?"
            item_done_count_dict[item] = "?"
            item_undone_count_dict[item] = "?"
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_done_ratio_dict,
    #              title="Clusters with Tagged Item Done Ratio")
    heatmap_plot(n_clusters, cluster_item_done_ratio_list, equal_var=False,
                 title="Item Engagement Ratio Comparison",
                 fig_name="item_heatmap_engagement_ratio")
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_done_count_dict,
    #              title="Clusters with Tagged Item Done Count")
    heatmap_plot(n_clusters, cluster_item_done_size_list, equal_var=False,
                 title="Item Engagement Count Comparison",
                 fig_name="item_heatmap_engagement_count")
    # scatter_plot(data_frame=model_df, colors=colors, tags=item_undone_count_dict,
    #              title="Clusters with Tagged Item UnDone Count")
    # heatmap_plot(n_clusters, cluster_item_undone_size_list, equal_var=False,
    #              title="Item UnDone Count Comparison")

    return cluster_score1, cluster_score2, cluster_score3


def scatter_plot(data_frame, colors, tags, title=None, x_label="Item Factor",
                 y_label="Item Rate Bias", fig_name=None):
    plt.figure()
    fig, ax = plt.subplots()
    sns.set(font_scale=1.0)
    sns.scatterplot(x="X", y="Y", hue="Cluster", legend=True, data=data_frame, palette=colors)
    for item_id in range(data_frame.shape[0]):
        tag = tags[item_id]
        plt.text(x=data_frame.X[item_id] - 0.1, y=data_frame.Y[item_id] + 1.,
                 s=tag,
                 fontdict=dict(color=colors[data_frame.Cluster[item_id]], size=12))
    # plt.title(title, fontsize=18)
    # plt.xlabel(x_label, fontsize=18)
    # plt.ylabel(y_label, fontsize=18)
    ax.axis('off')
    ax.legend(loc="lower right", title="Cluster ID")
    plt.tight_layout(pad=0.5)
    plt.savefig("figures/{}.pdf".format(fig_name))
    plt.show()
    plt.clf()


def heatmap_plot(size, cluster_features_dict, equal_var=True, title=None, fig_name=None):
    mask = np.triu(np.ones((size, size), dtype=bool))
    m = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            value_i = cluster_features_dict[i]
            value_j = cluster_features_dict[j]
            test, p_val = stats.ttest_ind(value_i, value_j, equal_var=equal_var)
            m[i, j] = p_val
            # if j > i:
            #     prRed("cluster {} vs {}, Mean: {} vs {}, p-val {}".format(
            #         i, j, np.mean(value_i), np.mean(value_j), p_val))
            #     print("cluster {}: {}".format(i, value_i))
            #     print("cluster {}: {}".format(j, value_j))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.3)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(m, mask=mask, linewidths=1., square=True, cmap="YlGnBu_r",
                ax=ax, cbar=True, vmin=0, vmax=0.20, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.65, 'label': 'p-value', 'format': '%.2f'})
    plt.xlabel("Cluster", fontsize=15)
    plt.ylabel("Cluster", fontsize=15)
    plt.title(title, fontsize=15)
    plt.tight_layout(pad=0.)
    plt.savefig("figures/{}.pdf".format(fig_name))
    plt.show()
    plt.clf()


def item_cluster_analysis(fold, clustering, n_clusters=None):
    score1_list = []
    score2_list = []
    score3_list = []
    for n in n_clusters:
        score1, score2, score3 = item_analysis(clustering, n_clusters=n, fold=fold)
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


if __name__ == '__main__':
    # step 1. since we don't have all users' information, we should do the item analysis first
    # best choice fold 1 and fold 4
    item_cluster_analysis(fold=1, clustering="spectral", n_clusters=[5])  # 6 clusters is also fine
    # item_cluster_analysis(fold=4, clustering="kmeans", n_clusters=[4])

    # item_cluster_analysis(fold=2, clustering="kmeans", n_clusters=[5])
    # item_cluster_analysis(fold=5, clustering="kmeans", n_clusters=[5, 6])

    # step 2. choose the same fold data as item cluster analysis for user cluster analysis
    # user_cluster_analysis(fold=2, clustering="kmeans")
    # user_outlier_analysis(2)
