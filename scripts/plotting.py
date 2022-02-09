from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def stylePlots():
    """A quick function to make subsequent plots look nice (requires seaborn).
    """
    sns.set_context("talk",font_scale=1.5)

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
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

def makeCM(model, X_train, X_test, y_train, y_test, encoding_dict, fn='gp_rnn_LSSTexp', ts=000000, c='Reds', plotpath='./plot/'):
    """A custom code to generate a confusion matrix for a classification model (allows for customization past what the built-in
    scikit-learn implementation allows).

    Parameters
    ----------
    model : keras model object
        The classification model to evaluate.
    Xtrain : 2d array-like
        Features of training set.
    Xtest : 2d array-like
        Features of test set.
    ytrain : 1d array-like
        Classes of objects in training set.
    ytest : 1d array-like
        Classes of objects in test set.
    fn : str
        Prefix to add to output filename
    ts : int
        The timestamp for the run (used to link plots to verbose output files)

    Returns
    -------
    None


    """
    # make predictions
    predictions = model.predict(X_test)
    predictDF = pd.DataFrame(data=predictions, columns=np.unique(list(encoding_dict.values())))
    predictDF['PredClass'] = predictDF.idxmax(axis=1)
    predictDF['TrueClass'] = [encoding_dict[x] for x in y_test]
    accTest = np.sum(predictDF['PredClass'] == predictDF['TrueClass'])/len(predictDF)*100

    #create confusion matrix
    CM = confusion_matrix(predictDF['TrueClass'], predictDF['PredClass'], normalize='true')
    fig = plt.figure(figsize=(10.0, 8.0), dpi=300) #frameon=false
    df_cm = pd.DataFrame(CM, columns=np.unique(predictDF['PredClass'].values), index = np.unique(predictDF['PredClass'].values))
    df_cm.index.name = 'True Label'
    df_cm.columns.name = 'Predicted Label'

    #plot it here:
    stylePlots()
    plt.figure(figsize = (10,7))
    sns.set(font_scale=2)
    g = sns.heatmap(df_cm, cmap=c, annot=True, fmt=".2f", annot_kws={"size": 30}, linewidths=1, linecolor='black', cbar_kws={"ticks": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, vmin=0.29, vmax=0.91)# font size
    g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
    g.set_yticklabels(g.get_yticklabels(), fontsize = 20)
    g.set_title("Test Set, Accuracy = %.2f%%"%accTest)
    plt.savefig(plotpath + "/%s_%i.png"%(fn, ts), dpi=200, bbox_inches='tight')
