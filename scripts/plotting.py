import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy import interp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from helper_functions import *

def plotTrainingHistory(inDir, infile):
    """Quick function to translate a printout of the training of a neurla network into
    plots of the loss, accuracy, etc.

    Parameters
    ----------
    inDir : str
        Path to the file.
    infile : str
        File with output of keras training session.

    Returns
    -------
    numpy array
        the loss per epoch on the training set.
    numpy array
        the loss per epoch on the validation set.

    """
    stylePlots()
    fn = infile.split("/")[-1]
    with open(inDir + "/text/" + fn, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if "loss:" in line]
    N = np.arange(len(lines))

    loss = []
    val_loss = []
    acc = []
    val_acc = []
    fig, axs = plt.subplots(4, figsize=(10,20), sharex=True)

    for line in lines:
        loss.append(float(line.split()[5]))
        acc.append(float(line.split()[8]))
        val_loss.append(float(line.split()[11]))
        val_acc.append(float(line.split()[14]))

    axs[0].plot(N, loss)
    axs[0].set_title("Training Loss")
    axs[0].set_ylabel("Loss")

    axs[1].plot(N, val_loss)
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Validation Loss")

    axs[2].plot(N, acc)
    axs[2].set_title("Training Accuracy")
    axs[2].set_ylabel("Accuracy")

    axs[3].plot(N, val_acc)
    axs[3].set_title("Validation Accuracy")
    axs[3].set_ylabel("Accuracy")
    axs[3].set_xlabel("Epoch Number")

    plt.savefig(inDir + "/plots/Loss_%s.png"%fn[:-4],dpi=200, bbox_inches='tight')
    return loss, val_loss


def stylePlots():
    """A quick function to make subsequent plots look nice (requires seaborn).
    """
    sns.set_context("talk",font_scale=1.25)

    sns.set_style('white', {'axes.linewidth': 0.5})
    plt.rcParams['xtick.major.size'] = 15
    plt.rcParams['ytick.major.size'] = 15

    plt.rcParams['xtick.minor.size'] = 10
    plt.rcParams['ytick.minor.size'] = 10
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.minor.width'] = 2

    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.right'] = True

    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rcParams.update({
        #"text.usetex": True,turn off for now, not sure where my latex went
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

def plot_ROC_wCV(model, params, X_full, y_full, bounds, encoding_dict, fnstr='', plotpath='./', save=True):
    """A function to generate the Receiver Operator Characteristic curves for a classifier. The function
    splits up the data set input into a 5-fold cross-validation set and plots the mean and standard deviation of the
    performance across the 5 folds.

    Parameters
    ----------
    model : keras Model
        The fully trained keras model.
    X : numpy matrix
        The features of the dataset.
    y : numpy array
        The target encoded classifications of the dataset.
    bounds : tuple
        The temporal bounds (in days) to consider for plotting.
    encoding_dict : dictionary
        A mapping between the encoded values in y and the class names.
    fnstr : str
        A string for the filename of the plot.
    plotpath : str
        A string for the path where the plot will be saved.
    save : bool
        If true, the plot is saved.

    Returns
    -------
    float
        The mean accuracy across all splits.
    numpy array
        The accuracy of each split.

    """
    X = X_full[(np.nanmax(X_full[:, :, 0], axis=1) > bounds[0]) & (np.nanmax(X_full[:, :, 0], axis=1) < bounds[1]), :, :]
    y = y_full[(np.nanmax(X_full[:, :, 0], axis=1) > bounds[0]) & (np.nanmax(X_full[:, :, 0], axis=1) < bounds[1])]

    fig, c_ax = plt.subplots(1,1, figsize = (8, 8))
    ax = fig.gca()

    nsplit = 2
    cv = StratifiedKFold(n_splits=nsplit)
    classes = np.unique(y)
    colors = sns.color_palette('Dark2', params['Nclass'])
    mean_fpr = np.linspace(0, 1, 100)
    accuracy_tot = 0
    nclass = len(classes)
    for j in range(nclass):
        wrong = []
        allRight = []
        tprs = []
        allAcc = []
        aucs = []
        all_confMatrices = []
        for train, test in cv.split(X, y):
            Xtrain_resampled = X[train]
            ytrain_resampled = y[train]

            probas_ = model.predict(X[test])#[0]
            predictions = model.predict(X[test])
            predictDF = pd.DataFrame(data=predictions, columns=classes)
            predictions = predictDF.idxmax(axis=1)
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, j], pos_label=classes[j])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tempAccuracy =  np.sum(predictions == y[test])/len(y[test])*100
            allAcc.append(tempAccuracy)
            matr = confusion_matrix(y[test], predictions, normalize='true')
            all_confMatrices.append(matr)
            accuracy_tot += tempAccuracy
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        accuracy = accuracy_tot / (nsplit*len(classes))
        if std_auc < 0.01:
            ax.plot(mean_fpr, mean_tpr, color=colors[j],
                     label='%s (%0.2f $\pm$ <0.01)' % (encoding_dict[j].replace("SN", ""), mean_auc),
                     lw=2, alpha=.8)
        else:
            ax.plot(mean_fpr, mean_tpr, color=colors[j],
                     label='%s (%0.2f $\pm$ %0.2f)' % (encoding_dict[j].replace("SN", ""), mean_auc, std_auc),
                     lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[j], alpha=.3)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',alpha=.8)
    ax.set_xlabel("False Positive Rate");
    ax.set_ylabel("True Positive Rate");
    ax.legend()
    #ax.legend(loc=4)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    if save:
        plt.savefig(plotpath + "/Combined_MeanROC_Curve_%s.png"% fnstr,dpi=150)
    return accuracy, allAcc

def plot_PR_wCV(model, params, X_full, y_full, bounds, encoding_dict, fnstr='', plotpath='./', save=True):
    """A function to generate the Precision-Recall Curves for a classifier. The function
    splits up the data set input into a 5-fold cross-validation set and plots the mean and standard deviation of the
    performance across the 5 folds.

    Parameters
    ----------
    model : keras Model
        The fully trained keras model.
    X : numpy matrix
        The features of the dataset.
    y : numpy array
        The target encoded classifications of the dataset.
    bounds : tuple
        The temporal bounds (in days) to consider for plotting.
    encoding_dict : dictionary
        A mapping between the encoded values in y and the class names.
    fnstr : str
        A string for the filename of the plot.
    plotpath : str
        A string for the path where the plot will be saved.
    save : bool
        If true, the plot is saved.

    Returns
    -------
    float
        The mean accuracy across all splits.
    numpy array
        The accuracy of each split.

    """
    X = X_full[(np.nanmax(X_full[:, :, 0], axis=1) > leftBound) & (np.nanmax(X_full[:, :, 0], axis=1) < rightBound), :, :]
    y = y_full[(np.nanmax(X_full[:, :, 0], axis=1) > leftBound) & (np.nanmax(X_full[:, :, 0], axis=1) < rightBound)]

    fig, c_ax = plt.subplots(1,1, figsize = (8, 8))
    ax = fig.gca()

    nsplit = 2
    cv = StratifiedKFold(n_splits=nsplit)
    classes = np.unique(y)
    colors = sns.color_palette('Dark2', params['Nclass'])
    mean_r = np.linspace(0, 1, 100)
    accuracy_tot = 0
    nclass = len(classes)
    for j in range(nclass):
        ps = []
        allAcc = []
        aucs = []
        all_confMatrices = []
        for train, test in cv.split(X, y):
            Xtrain_resampled = X[train]
            ytrain_resampled = y[train]

            probas_ = model.predict(X[test])#[0]
            predictions = model.predict(X[test])#[0]
            predictDF = pd.DataFrame(data=predictions, columns=classes)
            predictions = predictDF.idxmax(axis=1)
            precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, j], pos_label=classes[j])
            ps.append(interp(mean_r, recall[::-1], precision[::-1]))
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)
            tempAccuracy =  np.sum(predictions == y[test])/len(y[test])*100
            allAcc.append(tempAccuracy)
            matr = confusion_matrix(y[test], predictions, normalize='true')
            all_confMatrices.append(matr)
            accuracy_tot += tempAccuracy
        mean_p = np.mean(ps, axis=0)
        mean_auc = auc(mean_r, mean_p)
        std_auc = np.std(aucs)
        accuracy = accuracy_tot / (nsplit*len(classes))
        if std_auc < 0.01:
            ax.plot(mean_r, mean_p, color=colors[j],
                     label='%s (%0.2f $\pm$ <0.01)' % (encoding_dict[j].replace("SN", ""), mean_auc),
                     lw=2, alpha=.8)
        else:
            ax.plot(mean_r, mean_p, color=colors[j],
                     label='%s (%0.2f $\pm$ %0.2f)' % (encoding_dict[j].replace("SN", ""), mean_auc, std_auc),
                     lw=2, alpha=.8)
        std_p = np.std(ps, axis=0)
        ps_upper = np.minimum(mean_p + std_p, 1)
        ps_lower = np.maximum(mean_p - std_p, 0)
        ax.fill_between(mean_r, ps_lower, ps_upper, color=colors[j], alpha=.3)

    ax.set_xlabel("Precision");
    ax.set_ylabel("Recall");
    ax.legend()
    #ax.legend(loc=4)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    if save:
        plt.savefig(plotpath + "/Combined_MeanPR_Curve_%s.png"% fnstr, dpi=150)
    return accuracy, allAcc

def plot_ROC_timeSeries(model, X, y, encoding_dict, fnstr='', plotpath='./', save=True):
    """A function to generate the mean AUROC as a function of the phase of the light curve segment. The function
    splits up the data set  into a 5-fold cross-validation set and plots the mean and standard deviation of the
    performance across the 5 folds.

    Parameters
    ----------
    model : keras Model
        The fully trained keras model.
    X : numpy matrix
        The features of the dataset.
    y : numpy array
        The target encoded classifications of the dataset.
    encoding_dict : dictionary
        A mapping between the encoded values in y and the class names.
    fnstr : str
        A string for the filename of the plot.
    plotpath : str
        A string for the path where the plot will be saved.
    save : bool
        If true, the plot is saved.

    """

    fig, c_ax = plt.subplots(1,1, figsize = (8, 8))
    ax = fig.gca()

    classes = np.unique(y)
    colors = sns.color_palette('Dark2', params['Nclass'])
    mean_fpr = np.linspace(0, 1, 100)
    accuracy_tot = 0
    nclass = len(classes)
    nsplit = 2
    cv = StratifiedKFold(n_splits=nsplit)
    temp_auc_arr = []
    auc_matrix ={}
    for i in np.arange(nclass):
        auc_matrix[i] = []
    for j in range(nclass):
        spacing = 3
        phases = np.arange(1, 30, spacing)
        empty = 1
        for train,test in cv.split(X, y):
            temp_auc_arr = []
            tprs = []
            aucs = []
            for phase in phases:
                #cut X and y
                Xtemp = X[test]
                ytemp = y[test]

                trimBool = [len(x[x>0]) > 0 for x in Xtemp[:, :, 0]]

                Xtemp = Xtemp[trimBool]
                ytemp = ytemp[trimBool]

                times = Xtemp[:, :, 0]
                trimmed = np.array([x[x>0] for x in times])

                under = np.array([x[-1] > (phase-spacing/2.) for x in trimmed])
                over = np.array([x[-1] < (phase+spacing/2.) for x in trimmed])
                phaseCut = under*over

                X_phase = Xtemp[phaseCut]
                y_phase = ytemp[phaseCut]

                probas_ = model.predict(X_phase)
                predictDF = pd.DataFrame(data=probas_, columns=classes)
                predictions = predictDF.idxmax(axis=1)
                fpr, tpr, thresholds = roc_curve(y_phase, probas_[:, j], pos_label=classes[j])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                temp_auc_arr.append(roc_auc)
            if empty:
                auc_matrix[j] = temp_auc_arr
                empty = 0
            else:
                auc_matrix[j] = np.vstack([auc_matrix[j], temp_auc_arr])
        meanauc = np.nanmean(auc_matrix[j], axis=0)
        stdauc = np.nanstd(auc_matrix[j], axis=0)
        ax.plot(phases, meanauc, color=colors[j],
                 label=r'%s (%0.2f $\pm$ %0.2f)' % (encoding_dict[j].replace("SN", ""), np.nanmean(meanauc), np.nanmean(stdauc)),
                 lw=2, alpha=.8)
        ax.fill_between(phases, meanauc-stdauc, meanauc+stdauc, color=colors[j],
                 lw=2, alpha=.5)
    ax.set_ylim((0, 1))
    ax.set_ylabel("AUROC");
    ax.set_xlabel(r"$t_n$ (Days)");
    ax.legend()
    if save:
        plt.savefig(plotpath + "/ROC_TimeSeries_%s.png"% fnstr,dpi=150)
    return

def plot_PR_timeSeries(model, params, X, y, encoding_dict, fnstr='', plotpath='./', save=True):
    """A function to generate the mean AUPRC as a function of the phase of the light curve segment. The function
    splits up the data set  into a 5-fold cross-validation set and plots the mean and standard deviation of the
    performance across the 5 folds.

    Parameters
    ----------
    model : keras Model
        The fully trained keras model.
    X : numpy matrix
        The features of the dataset.
    y : numpy array
        The target encoded classifications of the dataset.
    encoding_dict : dictionary
        A mapping between the encoded values in y and the class names.
    fnstr : str
        A string for the filename of the plot.
    plotpath : str
        A string for the path where the plot will be saved.
    save : bool
        If true, the plot is saved.

    """
    fig, c_ax = plt.subplots(1,1, figsize = (8, 8))
    ax = fig.gca()

    classes = np.unique(y)
    colors = sns.color_palette('Dark2', params['Nclass'])
    mean_r = np.linspace(0, 1, 100)
    accuracy_tot = 0
    nclass = len(classes)
    nsplit = 2
    cv = StratifiedKFold(n_splits=nsplit)
    temp_auc_arr = []
    pr_matrix ={}
    for i in np.arange(nclass):
        pr_matrix[i] = []
    for j in range(nclass):
        spacing = 3
        phases = np.arange(1, 30, spacing)
        empty = 1
        for train,test in cv.split(X, y):
            temp_auc_arr = []
            ps = []
            aucs = []
            for phase in phases:

                #cut X and y
                Xtemp = X[test]
                ytemp = y[test]

                trimBool = [len(x[x>0]) > 0 for x in Xtemp[:, :, 0]]

                Xtemp = Xtemp[trimBool]
                ytemp = ytemp[trimBool]

                times = Xtemp[:, :, 0]
                trimmed = np.array([x[x>0] for x in times])

                under = np.array([x[-1] > (phase-spacing/2.) for x in trimmed])
                over = np.array([x[-1] < (phase+spacing/2.) for x in trimmed])
                phaseCut = under*over

                X_phase = Xtemp[phaseCut]
                y_phase = ytemp[phaseCut]

                probas_ = model.predict(X_phase)
                predictDF = pd.DataFrame(data=probas_, columns=classes)
                predictions = predictDF.idxmax(axis=1)
                precision, recall, thresholds = precision_recall_curve(y_phase, probas_[:, j], pos_label=classes[j])
                ps.append(interp(mean_r, recall[::-1], precision[::-1]))

                # Use AUC function to calculate the area under the curve of precision recall curve
                pr_auc = auc(recall, precision)
                temp_auc_arr.append(pr_auc)
            if empty:
                pr_matrix[j] = temp_auc_arr
                empty = 0
            else:
                pr_matrix[j] = np.vstack([pr_matrix[j], temp_auc_arr])
        meanpr = np.nanmean(pr_matrix[j], axis=0)
        stdpr = np.nanstd(pr_matrix[j], axis=0)
        ax.plot(phases, meanpr, color=colors[j],
                 label=r'%s (%0.2f $\pm$ %0.2f)' % (encoding_dict[j].replace("SN", ""), np.nanmean(meanpr), np.nanmean(stdpr)),
                 lw=2, alpha=.8)
        ax.fill_between(phases, meanpr-stdpr, meanpr+stdpr, color=colors[j],
                 lw=2, alpha=.5)
    ax.set_ylim((0, 1))
    ax.set_ylabel("AUPRC");
    ax.set_xlabel(r"$t_n$ (Days)");
    ax.legend()
    if save:
        plt.savefig(plotpath + "/PRC_TimeSeries_%s.png"% fnstr,dpi=150)
    return
