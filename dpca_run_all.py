import os
import sys
from scipy import stats
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pickle
import glob
import pandas as pd
from warnings import warn
import seaborn as sns
from dPCA import dPCA
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages


def main(fname):
    path = "/jukebox/witten/yousuf/rotation/" + fname.split("/")[-1][:-7] + "/"
    os.mkdir(path[:-1])

    # region plotting params
    mpl.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "xtick.bottom": True,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.linewidth": 1.5,
            "axes.labelsize": 16,
            "legend.fontsize": 16,
        }
    )


    def trial_xticks(ax, bin_sizes=[5, 25, 15, 7, 15]):
        bin_locs = np.cumsum(bin_sizes)
        ax.set_xticks([0] + bin_locs.tolist())
        ax.set_xticklabels([])
        ax.tick_params("x", length=17, width=1, which="major")
        ax.set_xlabel("Time", labelpad=10)

        periods = ["S", "Cue", "Delay", "Arm", "Reward"]
        for i in range(5):
            xloc = ([0] + bin_locs.tolist())[i] + bin_sizes[i] / 2
            ax.text(
                xloc,
                -0.03,
                periods[i],
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="top",
                transform=ax.get_xaxis_transform(),
                rotation=0,
            )


    cmap = sns.color_palette("terrain", 10, desat=0.45)
    colors = [cmap[-4], cmap[3]]

    # endregion

    # region initializing ouput pdfs and dicts
    choice_laser_pc1_pdf = PdfPages(path + "choice_laser_pc1.pdf")
    choice_laser_pc1_dist_pdf = PdfPages(path + "choice_laser_pc1_dist.pdf")
    choice_pc1_pdf = PdfPages(path + "choice_pc1.pdf")
    choice_pc1_dist_pdf = PdfPages(path + "choice_pc1_dist.pdf")
    evidence_pc1_pdf = PdfPages(path + "evidence_pc1.pdf")
    evidence_pc1_dist_pdf = PdfPages(path + "evidence_pc1_dist.pdf")

    choice_laser_pc1_dist = {
        "roff_minus_loff_mean": [],
        "ron_minus_loff_mean": [],
        "loff_minus_roff_mean": [],
        "lon_minus_roff_mean": [],
        "mouse_date": [],
        "pcorrect": [],
        "pengaged": [],
        "ds": [],
        "t": [],
    }

    choice_pc1_dist = {
        "roff_minus_loff_mean": [],
        "ron_minus_loff_mean": [],
        "loff_minus_roff_mean": [],
        "lon_minus_roff_mean": [],
        "mouse_date": [],
        "pcorrect": [],
        "pengaged": [],
        "d": [],
        "t": [],
    }
    evidence_pc1_dist = {
        "roff_minus_loff_mean": [],
        "ron_minus_loff_mean": [],
        "loff_minus_roff_mean": [],
        "lon_minus_roff_mean": [],
        "mouse_date": [],
        "pcorrect": [],
        "pengaged": [],
        "e": [],
        "t": [],
    }

    # endregion

    # region loading data
    with open(fname, "rb") as handle:
        data = pickle.load(handle)
    keys = data.keys()
    # selecting the key that includes single and multi-unit recordings:
    key = [key for i, key in enumerate(keys) if key.split("_")[1] == "all"]
    data = data[key[0]]

    # endregion

    # region initializing a few variables
    bin_sizes = [5, 25, 15, 7, 15]

    mouseID = np.array(data["mouseID"])
    mice = np.unique(mouseID)
    print(f"Mice in file: {mice}")

    ephys_loc = fname.split("_")[2][-3:]
    opto_loc = fname.split("_")[3][:-3]
    task = fname.split("_")[4][:3]

    # endregion

    for i_m, mouse in enumerate(mice):
        print(f"mouse {i_m+1} of {mice.size}")

        all_dates = np.asarray(data["date"])

        # dates associated with that mouse
        dates = np.unique(all_dates[mouseID == mouse])

        print(f"Dates with data from mouse {mouse}: {dates}")

        for i_d, date in enumerate(dates):
            # index of mouse/dates of interest in data:
            idx = np.nonzero((all_dates == date) * (mouseID == mouse))[0]
            print(f"Total Neurons: {idx.size}")

            if data["laserON"][idx[0]].sum() > 4:
                print(f"date {i_d+1} of {dates.size}")

                alldata_lsr = [None, None]
                trial_idx_lsr = [None, None]

                # loading in all data for session:
                for i_l, laser in enumerate([0, 1]):
                    # extracting index where the maze type is >7 for laser on vs off trials
                    trial_idx = np.nonzero(
                        (data["currMaze"][idx[0]] > 7)
                        * (data["laserON"][idx[0]] == laser)
                    )[0]
                    # add the following to the trial index boolean operation to include only state 3 trials:
                    # *(data['stateID'][idx[0]] == 3)

                    trial_idx_lsr[laser] = trial_idx

                    n_neurons = idx.size

                    [n_trials, n_bins] = data["timeSqueezedFR"][idx[0]][
                        trial_idx, :
                    ].shape

                    alldata = np.zeros((n_trials, n_neurons, n_bins))

                    correcttrials = np.where(
                        data["choice"][idx[0]][trial_idx]
                        == data["trialType"][idx[0]][trial_idx],
                        1,
                        0,
                    )
                    print(
                        f"fraction correct for laser {['off', 'on'][laser]}: {correcttrials.sum()}/{correcttrials.size}"
                        f" = {correcttrials.sum()/correcttrials.size:.2f}"
                    )

                    # for each neuron, get time squeezed firing rate for all trials and time points:
                    for i, id in enumerate(idx):
                        alldata[:, i, :] = data["timeSqueezedFR"][id][trial_idx, :]

                    alldata[np.isnan(alldata)] = 0  # replace nan values with 0

                    alldata_lsr[laser] = alldata
                    correcttrials = np.where(
                        data["choice"][idx[0]][trial_idx_lsr[0]]
                        == data["trialType"][idx[0]][trial_idx_lsr[0]],
                        1,
                        0,
                    )
                    pcorrect = correcttrials.sum() / correcttrials.size

                    pengaged = data["stateID"][idx[0]][trial_idx_lsr[0]] == 3
                    pengaged = pengaged.sum() / pengaged.size

                # region dPCA w laser and decision conditions

                try:
                    # region creating data matrices and fitting dPCA

                    choice1 = data["choice"][idx[0]][
                        trial_idx_lsr[0]
                    ]  # laser off choices
                    choice2 = data["choice"][idx[0]][
                        trial_idx_lsr[1]
                    ]  # laser on choices

                    # number of states for each experimental condition
                    n_time = n_bins
                    n_decision = 2
                    n_stim = 2

                    # finding the experimental condition with the least number of trials
                    # each condition should have the same number of trials for dPCA to work
                    n_trials = np.min(
                        (
                            alldata_lsr[0][choice1 == 0, :, :].shape[0],
                            alldata_lsr[0][choice1 == 1, :, :].shape[0],
                            alldata_lsr[1][choice2 == 0, :, :].shape[0],
                            alldata_lsr[1][choice2 == 1, :, :].shape[0],
                        )
                    )
                    print(f"number of trials included: {n_trials}")

                    # choosing random trials to include:
                    # NOTE: result seems to change significantly according to selected trials if n_trials is low
                    left_off_idx = np.random.randint(0, (choice1 == 0).sum(), n_trials)
                    right_off_idx = np.random.randint(0, (choice1 == 1).sum(), n_trials)
                    left_on_idx = np.random.randint(0, (choice2 == 0).sum(), n_trials)
                    right_on_idx = np.random.randint(0, (choice2 == 1).sum(), n_trials)

                    # data matrix which includes the average of all trials in each experimental condition:
                    X = np.empty((n_neurons, n_time, n_decision, n_stim))
                    X[:, :, 0, 0] = (alldata_lsr[0][choice1 == 0, :, :])[
                        left_off_idx
                    ].mean(axis=0)
                    X[:, :, 1, 0] = (alldata_lsr[0][choice1 == 1, :, :])[
                        right_off_idx
                    ].mean(axis=0)
                    X[:, :, 0, 1] = (alldata_lsr[1][choice2 == 0, :, :])[
                        left_on_idx
                    ].mean(axis=0)
                    X[:, :, 1, 1] = (alldata_lsr[1][choice2 == 1, :, :])[
                        right_on_idx
                    ].mean(axis=0)
                    X = X.transpose(0, 2, 3, 1)  # time should be last axis

                    # data matrix which includes each individual trial in each experimental condition,
                    # this is used for cross validation:
                    X_all = np.empty((n_trials, n_neurons, n_time, n_decision, n_stim))
                    X_all[:, :, :, 0, 0] = (alldata_lsr[0][choice1 == 0, :, :])[
                        left_off_idx
                    ]
                    X_all[:, :, :, 1, 0] = (alldata_lsr[0][choice1 == 1, :, :])[
                        right_off_idx
                    ]
                    X_all[:, :, :, 0, 1] = (alldata_lsr[1][choice2 == 0, :, :])[
                        left_on_idx
                    ]
                    X_all[:, :, :, 1, 1] = (alldata_lsr[1][choice2 == 1, :, :])[
                        right_on_idx
                    ]
                    X_all = X_all.transpose(0, 1, 3, 4, 2)

                    # mean centering each neuron:
                    X = X - X.mean(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
                    X_all = X_all - X_all.mean(axis=(0, 2, 3, 4)).reshape(
                        1, -1, 1, 1, 1
                    )  # mean centering each neuron

                    label = "dst"  # labeling the axes. d=decision/choice, s=laser/stim/inhib, t = time
                    dpca = dPCA.dPCA(labels=label, n_components=3, regularizer="auto")
                    dpca.opt_regularizer_flag = True
                    dpca.n_trials = 5  # number of cross validation folds
                    dpca.protect = [
                        "t"
                    ]  # prevents shuffling through time for cross validation
                    Xt = dpca.fit_transform(X, X_all)
                    # endregion

                    # region transforming individual trials into dPCA space and plotting pc vs time

                    # choose marginalization here (e.g. 'd' for decision space, 'ds' for decision-laser space):
                    marg = "ds"
                    pc = dpca.D[marg][:, :1]
                    choices = np.concatenate(
                        (
                            data["choice"][idx[0]][trial_idx_lsr[0]],
                            data["choice"][idx[0]][trial_idx_lsr[1]],
                        )
                    )

                    # transforming laser off trials:
                    data_T_off = np.squeeze(
                        alldata_lsr[0].transpose(0, 2, 1) @ pc, axis=2
                    )
                    # transforming laser on trials:
                    data_T_on = np.squeeze(
                        alldata_lsr[1].transpose(0, 2, 1) @ pc, axis=2
                    )

                    data_T_all = np.concatenate((data_T_off, data_T_on), axis=0)
                    # vector indicating which trials are laser on vs off
                    lsr = np.where(
                        np.arange(data_T_all.shape[0]) < data_T_off.shape[0],
                        "laser off",
                        "laser on",
                    )

                    # vector indicating which trials are left vs right choice
                    left_right = np.where(choices == 0, "left", "right")
                    if (
                        left_right[0] == "right"
                    ):  # to keep left/right color scheme consistent
                        colors = colors[::-1]

                    plot_df = pd.DataFrame(
                        data_T_all, columns=np.arange(data_T_all.shape[1])
                    )
                    plot_df["laser"] = lsr
                    plot_df["choice"] = left_right
                    plot_df = plot_df.melt(  # changing to long-form data frame to get error bars
                        id_vars=["laser", "choice"],
                        var_name="time",
                        value_name=marg + "-PC1",
                    )

                    # plotting
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    sns.lineplot(
                        data=plot_df,
                        x="time",
                        y=marg + "-PC1",
                        hue="choice",
                        style="laser",
                        palette=colors,
                        ax=ax,
                    )

                    sns.despine()
                    plt.legend(
                        loc="upper right",
                        frameon=False,
                        ncol=2,
                        bbox_to_anchor=(1, 1.1),
                    )
                    plt.title(
                        f"marg: {marg} | {ephys_loc} | {opto_loc} | {task} | Mouse {mouse:.0f} | Date {date:.0f}",
                        pad=40,
                    )
                    trial_xticks(ax)

                    plt.tight_layout()

                    choice_laser_pc1_pdf.savefig(fig)

                    plt.close(fig)
                    # endregion

                    # region computing distance between left/right trajectories and plotting

                    # more specfically, we compare (notation is choice_laser) L_off - R_off and L_on - R_off
                    # and vice versa: R_off - L_off and R_on - L_off

                    # all trajectories:
                    left_off = data_T_all[(lsr == "laser off") * (left_right == "left")]
                    right_off = data_T_all[
                        (lsr == "laser off") * (left_right == "right")
                    ]
                    left_on = data_T_all[(lsr == "laser on") * (left_right == "left")]
                    right_on = data_T_all[(lsr == "laser on") * (left_right == "right")]

                    # distances:
                    roff_minus_loff_mean = right_off - left_off.mean(axis=0)
                    loff_minus_roff_mean = left_off - right_off.mean(axis=0)
                    lon_minus_roff_mean = left_on - right_off.mean(axis=0)
                    ron_minus_loff_mean = right_on - left_off.mean(axis=0)

                    # number of trials in each choice category
                    r_trials = (
                        roff_minus_loff_mean.shape[0] + ron_minus_loff_mean.shape[0]
                    )
                    l_trials = (
                        loff_minus_roff_mean.shape[0] + lon_minus_roff_mean.shape[0]
                    )
                    lr_on_off_trials = (
                        roff_minus_loff_mean.shape[0],
                        ron_minus_loff_mean.shape[0],
                        loff_minus_roff_mean.shape[0],
                        lon_minus_roff_mean.shape[0],
                    )

                    dist_all = np.concatenate(
                        (
                            roff_minus_loff_mean,
                            ron_minus_loff_mean,
                            loff_minus_roff_mean,
                            lon_minus_roff_mean,
                        ),
                        axis=0,
                    )

                    # labels for data frame:
                    left_right_dist = np.repeat(["right", "left"], [r_trials, l_trials])
                    on_off = np.repeat(
                        ["laser off", "laser on", "laser off", "laser on"],
                        lr_on_off_trials,
                    )

                    plot_df = pd.DataFrame(
                        dist_all, columns=np.arange(dist_all.shape[1])
                    )
                    plot_df["choice"] = left_right_dist
                    plot_df["laser"] = on_off
                    plot_df = plot_df.melt(
                        id_vars=["choice", "laser"],
                        var_name="time",
                        value_name=marg + "-PC Distance",
                    )

                    # plotting:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    titles = [
                        r"$L_{laser} - \overline{R_{off}}$",
                        r"$R_{laser} - \overline{L_{off}}$",
                    ]
                    for i, c in enumerate(["left", "right"]):
                        sns.lineplot(
                            data=plot_df[plot_df.choice == c],
                            x="time",
                            y=marg + "-PC Distance",
                            hue="choice",
                            style="laser",
                            palette=colors[i : i + 1],
                            ax=axs[i],
                        )
                        trial_xticks(axs[i])
                        sns.despine()
                        axs[i].get_legend().remove()
                        axs[i].set_title(titles[i], pad=25)

                    handles, labels = axs[1].get_legend_handles_labels()
                    axs[1].legend(
                        handles[3:],
                        labels[3:],
                        loc="upper left",
                        frameon=False,
                        ncol=1,
                        bbox_to_anchor=(0, 1),
                    )
                    plt.suptitle(
                        f"{ephys_loc} | {opto_loc} | {task} | Mouse {mouse:.0f} | Date {date:.0f}"
                    )
                    plt.tight_layout()

                    choice_laser_pc1_dist_pdf.savefig(fig)
                    plt.close(fig)

                    dist_dict = {
                        "roff_minus_loff_mean": roff_minus_loff_mean,
                        "ron_minus_loff_mean": ron_minus_loff_mean,
                        "loff_minus_roff_mean": loff_minus_roff_mean,
                        "lon_minus_roff_mean": lon_minus_roff_mean,
                    }
                    for key, val in dist_dict.items():
                        choice_laser_pc1_dist[key].append(val)
                    choice_laser_pc1_dist["mouse_date"].append(f"{mouse}_{date}")
                    choice_laser_pc1_dist["pcorrect"].append(pcorrect)
                    choice_laser_pc1_dist["pengaged"].append(pengaged)
                    choice_laser_pc1_dist["t"].append(
                        dpca.explained_variance_ratio_["t"]
                    )
                    choice_laser_pc1_dist["ds"].append(
                        dpca.explained_variance_ratio_["ds"]
                    )

                    # endregion

                except Exception as e:
                    print("DPCA 1 ERROR")
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # endregion

                # region dPCA w decision conditions only

                try:
                    # region creating data matrices and fitting dPCA

                    choice = data["choice"][idx[0]][trial_idx_lsr[0]]

                    # number of states for each experimental condition
                    n_time = n_bins
                    n_decision = 2
                    n_stim = 2

                    # finding the experimental condition with the least number of trials
                    # each condition should have the same number of trials for dPCA to work
                    n_trials = np.min(((choice == 0).sum(), (choice == 1).sum()))
                    print(f"number of trials included: {n_trials}")

                    # choosing random trials to include:
                    # NOTE: result seems to change significantly according to selected trials if n_trials is low
                    left_idx = np.random.randint(0, (choice == 0).sum(), n_trials)
                    right_idx = np.random.randint(0, (choice == 1).sum(), n_trials)

                    # data matrix which includes the average of all trials in each experimental condition:
                    X = np.empty((n_neurons, n_time, n_decision))
                    X[:, :, 0] = (alldata_lsr[0][choice == 0, :, :])[left_idx].mean(
                        axis=0
                    )
                    X[:, :, 1] = (alldata_lsr[0][choice == 1, :, :])[right_idx].mean(
                        axis=0
                    )
                    X = X.transpose(0, 2, 1)  # time should be last axis

                    # data matrix which includes each individual trial in each experimental condition,
                    # this is used for cross validation:
                    X_all = np.empty((n_trials, n_neurons, n_time, n_decision))
                    X_all[:, :, :, 0] = (alldata_lsr[0][choice == 0, :, :])[left_idx]
                    X_all[:, :, :, 1] = (alldata_lsr[0][choice == 1, :, :])[right_idx]
                    X_all = X_all.transpose(0, 1, 3, 2)

                    # mean centering each neuron:
                    X = X - X.mean(axis=(1, 2)).reshape(-1, 1, 1)
                    X_all = X_all - X_all.mean(axis=(0, 2, 3)).reshape(
                        1, -1, 1, 1
                    )  # mean centering each neuron
                    label = "dt"  # labeling the axes. d=decision/choice, s=laser/stim/inhib, t = time
                    dpca = dPCA.dPCA(labels=label, n_components=3, regularizer="auto")
                    dpca.n_trials = 5  # number of cross validation folds
                    dpca.protect = ["t"]  # prevents shuffling through time
                    Xt = dpca.fit_transform(X, X_all)

                    # endregion

                    # region transforming individual trials into dPCA space and plotting pc vs time

                    # choose marginalization here (e.g. 'd' for decision space, 'ds' for decision-laser space):

                    marg = "d"
                    pc = dpca.D[marg][:, :1]
                    choices = np.concatenate(
                        (
                            data["choice"][idx[0]][trial_idx_lsr[0]],
                            data["choice"][idx[0]][trial_idx_lsr[1]],
                        )
                    )

                    # transforming laser off trials:
                    data_T_off = np.squeeze(
                        alldata_lsr[0].transpose(0, 2, 1) @ pc, axis=2
                    )
                    # transforming laser on trials:
                    data_T_on = np.squeeze(
                        alldata_lsr[1].transpose(0, 2, 1) @ pc, axis=2
                    )

                    data_T_all = np.concatenate((data_T_off, data_T_on), axis=0)
                    # vector indicating which trials are laser on vs off
                    lsr = np.where(
                        np.arange(data_T_all.shape[0]) < data_T_off.shape[0],
                        "laser off",
                        "laser on",
                    )

                    # vector indicating which trials are left vs right choice
                    left_right = np.where(choices == 0, "left", "right")
                    if (
                        left_right[0] == "right"
                    ):  # to keep left/right color scheme consistent
                        colors = colors[::-1]

                    plot_df = pd.DataFrame(
                        data_T_all, columns=np.arange(data_T_all.shape[1])
                    )
                    plot_df["laser"] = lsr
                    plot_df["choice"] = left_right
                    plot_df = plot_df.melt(  # changing to long-form data frame to get error bars
                        id_vars=["laser", "choice"],
                        var_name="time",
                        value_name=marg + "-PC1",
                    )

                    # plotting
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    sns.lineplot(
                        data=plot_df,
                        x="time",
                        y=marg + "-PC1",
                        hue="choice",
                        style="laser",
                        palette=colors,
                        ax=ax,
                    )

                    sns.despine()
                    plt.legend(
                        loc="upper right",
                        frameon=False,
                        ncol=2,
                        bbox_to_anchor=(1, 1.1),
                    )
                    plt.title(
                        f"marg: {marg} | {ephys_loc} | {opto_loc} | {task} | Mouse {mouse:.0f} | Date {date:.0f}",
                        pad=40,
                    )
                    trial_xticks(ax)

                    plt.tight_layout()
                    choice_pc1_pdf.savefig(fig)
                    plt.close(fig)

                    # endregion

                    # region computing distance between left/right trajectories

                    # more specfically, we compare (notation is choice_laser) L_off - R_off and L_on - R_off
                    # and vice versa: R_off - L_off and R_on - L_off

                    # all trajectories:
                    left_off = data_T_all[(lsr == "laser off") * (left_right == "left")]
                    right_off = data_T_all[
                        (lsr == "laser off") * (left_right == "right")
                    ]
                    left_on = data_T_all[(lsr == "laser on") * (left_right == "left")]
                    right_on = data_T_all[(lsr == "laser on") * (left_right == "right")]

                    # distances:
                    roff_minus_loff_mean = right_off - left_off.mean(axis=0)
                    loff_minus_roff_mean = left_off - right_off.mean(axis=0)
                    lon_minus_roff_mean = left_on - right_off.mean(axis=0)
                    ron_minus_loff_mean = right_on - left_off.mean(axis=0)

                    # number of trials in each choice category
                    r_trials = (
                        roff_minus_loff_mean.shape[0] + ron_minus_loff_mean.shape[0]
                    )
                    l_trials = (
                        loff_minus_roff_mean.shape[0] + lon_minus_roff_mean.shape[0]
                    )
                    lr_on_off_trials = (
                        roff_minus_loff_mean.shape[0],
                        ron_minus_loff_mean.shape[0],
                        loff_minus_roff_mean.shape[0],
                        lon_minus_roff_mean.shape[0],
                    )

                    dist_all = np.concatenate(
                        (
                            roff_minus_loff_mean,
                            ron_minus_loff_mean,
                            loff_minus_roff_mean,
                            lon_minus_roff_mean,
                        ),
                        axis=0,
                    )

                    # labels for data frame:
                    left_right_dist = np.repeat(["right", "left"], [r_trials, l_trials])
                    on_off = np.repeat(
                        ["laser off", "laser on", "laser off", "laser on"],
                        lr_on_off_trials,
                    )

                    plot_df = pd.DataFrame(
                        dist_all, columns=np.arange(dist_all.shape[1])
                    )
                    plot_df["choice"] = left_right_dist
                    plot_df["laser"] = on_off
                    plot_df = plot_df.melt(
                        id_vars=["choice", "laser"],
                        var_name="time",
                        value_name=marg + "-PC Distance",
                    )

                    # plotting:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    titles = [
                        r"$L_{laser} - \overline{R_{off}}$",
                        r"$R_{laser} - \overline{L_{off}}$",
                    ]
                    for i, c in enumerate(["left", "right"]):
                        sns.lineplot(
                            data=plot_df[plot_df.choice == c],
                            x="time",
                            y=marg + "-PC Distance",
                            hue="choice",
                            style="laser",
                            palette=colors[i : i + 1],
                            ax=axs[i],
                        )
                        trial_xticks(axs[i])
                        sns.despine()
                        axs[i].get_legend().remove()
                        axs[i].set_title(titles[i], pad=25)

                    handles, labels = axs[1].get_legend_handles_labels()
                    axs[1].legend(
                        handles[3:],
                        labels[3:],
                        loc="upper left",
                        frameon=False,
                        ncol=1,
                        bbox_to_anchor=(0, 1),
                    )
                    plt.suptitle(
                        f"{ephys_loc} | {opto_loc} | {task} | Mouse {mouse:.0f} | Date {date:.0f}"
                    )
                    plt.tight_layout()

                    choice_pc1_dist_pdf.savefig(fig)
                    plt.close(fig)

                    dist_dict = {
                        "roff_minus_loff_mean": roff_minus_loff_mean,
                        "ron_minus_loff_mean": ron_minus_loff_mean,
                        "loff_minus_roff_mean": loff_minus_roff_mean,
                        "lon_minus_roff_mean": lon_minus_roff_mean,
                    }
                    for key, val in dist_dict.items():
                        choice_pc1_dist[key].append(val)
                    choice_pc1_dist["mouse_date"].append(f"{mouse}_{date}")
                    choice_pc1_dist["pcorrect"].append(pcorrect)
                    choice_pc1_dist["pengaged"].append(pengaged)
                    choice_pc1_dist["t"].append(dpca.explained_variance_ratio_["t"])
                    choice_pc1_dist["d"].append(dpca.explained_variance_ratio_["d"])

                    # endregion

                except Exception as e:
                    print("DPCA 2 ERROR")
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # endregion

                # region dPCA w evidence conditions

                try:
                    # region creating data matrices and fitting dPCA

                    # getting evidence data which is in the form of #R_cues - #L_cues
                    evidence = data["nCues_RminusL"][idx[0]][trial_idx_lsr[0]]
                    ev_bins = 2
                    _, bins = np.histogram(evidence, ev_bins)
                    bins[-1] += 0.1

                    # assigning 0 for left evidence and 1 for right evidence:
                    evidence_level = np.digitize(evidence, bins) - 1
                    evidence_categories = np.sort(np.unique(evidence_level))

                    n_time = n_bins
                    n_decision = 2
                    n_stim = 2
                    n_evidence = bins.size - 1
                    n_trials_all = []

                    for i_ev, ev in enumerate(evidence_categories):
                        n = (evidence_level == ev).sum()
                        n_trials_all.append(n)

                    n_trials = np.min(n_trials_all)
                    print(f"number of trials included: {n_trials}")

                    # choosing random trials to include:
                    # NOTE: result seems to change significantly according to selected trials if n_trials is low
                    left_idx = np.random.randint(
                        0, (evidence_level == 0).sum(), n_trials
                    )
                    right_idx = np.random.randint(
                        0, (evidence_level == 1).sum(), n_trials
                    )
                    lr_idx = [left_idx, right_idx]

                    # data matrix which includes the average of all trials in each experimental condition:
                    X = np.empty((n_neurons, n_time, ev_bins))
                    for i_ev, ev in enumerate(evidence_categories):
                        X[:, :, i_ev] = (alldata_lsr[0][evidence_level == ev, :, :])[
                            lr_idx[i_ev]
                        ].mean(axis=0)
                    X = X.transpose(0, 2, 1)  # time should be last axis

                    # data matrix which includes each individual trial in each experimental condition,
                    # this is used for cross validation:
                    X_all = np.empty((n_trials, n_neurons, n_time, ev_bins))
                    for i_ev, ev in enumerate(evidence_categories):
                        X_all[:, :, :, i_ev] = (
                            alldata_lsr[0][evidence_level == ev, :, :]
                        )[lr_idx[i_ev]]
                    X_all = X_all.transpose(0, 1, 3, 2)
                    X = X - X.mean(axis=(1, 2)).reshape(
                        -1, 1, 1
                    )  # mean centering each neuron

                    X_all = X_all - X_all.mean(axis=(0, 2, 3)).reshape(
                        1, -1, 1, 1
                    )  # mean centering each neuron

                    dpca = dPCA.dPCA(labels="et", n_components=3, regularizer="auto")
                    dpca.n_trials = 5  # number of cross validation folds
                    dpca.protect = ["t"]  # prevents shuffling through time
                    Xt = dpca.fit_transform(X, X_all)

                    # endregion

                    # region transforming individual trials into dPCA space and plotting pc vs time

                    # choose marginalization here (e.g. 'e' for evidence space, 'es' for evidence-laser space):
                    marg = "e"
                    pc = dpca.D[marg][:, :1]
                    evidence1 = (
                        np.digitize(
                            data["nCues_RminusL"][idx[0]][trial_idx_lsr[0]], bins
                        )
                        - 1
                    )
                    evidence2 = (
                        np.digitize(
                            data["nCues_RminusL"][idx[0]][trial_idx_lsr[1]], bins
                        )
                        - 1
                    )
                    evidence = np.concatenate((evidence1, evidence2))

                    # transforming laser off trials:
                    data_T_off = np.squeeze(
                        alldata_lsr[0].transpose(0, 2, 1) @ pc, axis=2
                    )
                    # transforming laser on trials:
                    data_T_on = np.squeeze(
                        alldata_lsr[1].transpose(0, 2, 1) @ pc, axis=2
                    )

                    data_T_all = np.concatenate((data_T_off, data_T_on), axis=0)
                    # vector indicating which trials are laser on vs off
                    lsr = np.where(
                        np.arange(data_T_all.shape[0]) < data_T_off.shape[0],
                        "laser off",
                        "laser on",
                    )

                    # vector indicating which trials are left vs right evidence
                    left_right = np.where(evidence == 0, "left", "right")
                    if (
                        left_right[0] == "right"
                    ):  # to keep left/right color scheme consistent
                        colors = colors[::-1]

                    plot_df = pd.DataFrame(
                        data_T_all, columns=np.arange(data_T_all.shape[1])
                    )
                    plot_df["laser"] = lsr
                    plot_df["evidence"] = left_right
                    plot_df = plot_df.melt(  # changing to long-form data frame to get error bars
                        id_vars=["laser", "evidence"],
                        var_name="time",
                        value_name=marg + "-PC1",
                    )

                    # plotting
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    sns.lineplot(
                        data=plot_df,
                        x="time",
                        y=marg + "-PC1",
                        hue="evidence",
                        style="laser",
                        palette=colors,
                        ax=ax,
                    )

                    sns.despine()
                    plt.legend(
                        loc="upper right",
                        frameon=False,
                        ncol=2,
                        bbox_to_anchor=(1, 1.1),
                    )
                    plt.title(
                        f"marg: {marg} | {ephys_loc} | {opto_loc} | {task} | Mouse {mouse:.0f} | Date {date:.0f}",
                        pad=40,
                    )
                    trial_xticks(ax)

                    plt.tight_layout()

                    evidence_pc1_pdf.savefig(fig)
                    plt.close(fig)

                    # endregion

                    # region computing distance between left/right trajectories

                    # more specfically, we compare (notation is choice_laser) L_off - R_off and L_on - R_off
                    # and vice versa: R_off - L_off and R_on - L_off

                    # all trajectories:
                    left_off = data_T_all[(lsr == "laser off") * (left_right == "left")]
                    right_off = data_T_all[
                        (lsr == "laser off") * (left_right == "right")
                    ]
                    left_on = data_T_all[(lsr == "laser on") * (left_right == "left")]
                    right_on = data_T_all[(lsr == "laser on") * (left_right == "right")]

                    # distances:
                    roff_minus_loff_mean = right_off - left_off.mean(axis=0)
                    loff_minus_roff_mean = left_off - right_off.mean(axis=0)
                    lon_minus_roff_mean = left_on - right_off.mean(axis=0)
                    ron_minus_loff_mean = right_on - left_off.mean(axis=0)

                    # number of trials in each choice category
                    r_trials = (
                        roff_minus_loff_mean.shape[0] + ron_minus_loff_mean.shape[0]
                    )
                    l_trials = (
                        loff_minus_roff_mean.shape[0] + lon_minus_roff_mean.shape[0]
                    )
                    lr_on_off_trials = (
                        roff_minus_loff_mean.shape[0],
                        ron_minus_loff_mean.shape[0],
                        loff_minus_roff_mean.shape[0],
                        lon_minus_roff_mean.shape[0],
                    )

                    dist_all = np.concatenate(
                        (
                            roff_minus_loff_mean,
                            ron_minus_loff_mean,
                            loff_minus_roff_mean,
                            lon_minus_roff_mean,
                        ),
                        axis=0,
                    )

                    # labels for data frame:
                    left_right_dist = np.repeat(["right", "left"], [r_trials, l_trials])
                    on_off = np.repeat(
                        ["laser off", "laser on", "laser off", "laser on"],
                        lr_on_off_trials,
                    )

                    plot_df = pd.DataFrame(
                        dist_all, columns=np.arange(dist_all.shape[1])
                    )
                    plot_df["choice"] = left_right_dist
                    plot_df["laser"] = on_off
                    plot_df = plot_df.melt(
                        id_vars=["choice", "laser"],
                        var_name="time",
                        value_name=marg + "-PC Distance",
                    )

                    # plotting:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    titles = [
                        r"$L_{laser} - \overline{R_{off}}$",
                        r"$R_{laser} - \overline{L_{off}}$",
                    ]
                    for i, c in enumerate(["left", "right"]):
                        sns.lineplot(
                            data=plot_df[plot_df.choice == c],
                            x="time",
                            y=marg + "-PC Distance",
                            hue="choice",
                            style="laser",
                            palette=colors[i : i + 1],
                            ax=axs[i],
                        )
                        trial_xticks(axs[i])
                        sns.despine()
                        axs[i].get_legend().remove()
                        axs[i].set_title(titles[i], pad=25)

                    handles, labels = axs[1].get_legend_handles_labels()
                    axs[1].legend(
                        handles[3:],
                        labels[3:],
                        loc="upper left",
                        frameon=False,
                        ncol=1,
                        bbox_to_anchor=(0, 1),
                    )
                    plt.suptitle(
                        f"{ephys_loc} | {opto_loc} | {task} | Mouse {mouse:.0f} | Date {date:.0f}"
                    )
                    plt.tight_layout()

                    evidence_pc1_dist_pdf.savefig(fig)
                    plt.close(fig)

                    dist_dict = {
                        "roff_minus_loff_mean": roff_minus_loff_mean,
                        "ron_minus_loff_mean": ron_minus_loff_mean,
                        "loff_minus_roff_mean": loff_minus_roff_mean,
                        "lon_minus_roff_mean": lon_minus_roff_mean,
                    }
                    for key, val in dist_dict.items():
                        evidence_pc1_dist[key].append(val)
                    evidence_pc1_dist["mouse_date"].append(f"{mouse}_{date}")
                    evidence_pc1_dist["pcorrect"].append(pcorrect)
                    evidence_pc1_dist["pengaged"].append(pengaged)
                    evidence_pc1_dist["t"].append(dpca.explained_variance_ratio_["t"])
                    evidence_pc1_dist["e"].append(dpca.explained_variance_ratio_["e"])

                    # endregion

                except Exception as e:
                    print("DPCA 3 ERROR")
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # endregion

    # region saving pdf and pickle files
    with open(path + "evidence_pc1_dist.pickle", "wb") as handle:
        pickle.dump(evidence_pc1_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + "choice_laser_pc1_dist.pickle", "wb") as handle:
        pickle.dump(choice_laser_pc1_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path + "choice_pc1_dist.pickle", "wb") as handle:
        pickle.dump(choice_pc1_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    choice_laser_pc1_pdf.close()
    choice_laser_pc1_dist_pdf.close()
    choice_pc1_pdf.close()
    choice_pc1_dist_pdf.close()
    evidence_pc1_pdf.close()
    evidence_pc1_dist_pdf.close()

    # endregion


if __name__ == "__main__":
    fname = sys.argv[1]
    main(fname)
    # main('/jukebox/witten/yousuf/rotation/pickles2/loop_files/allSpikeData_DMS_direct_AoE.pickle')
