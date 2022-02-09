from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def makeCM(model=None, X_train=None, X_test=None, y_train=None, y_test=None, encoding_dict=None, fn='gp_rnn_LSSTexp', ts=000000, c='Reds', plotpath='./plot/'):
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    Xtrain : type
        Description of parameter `Xtrain`.
    Xtest : type
        Description of parameter `Xtest`.
    ytrain : type
        Description of parameter `ytrain`.
    ytest : type
        Description of parameter `ytest`.
    fn : type
        Description of parameter `fn`.
    ts : type
        Description of parameter `ts`.

    Returns
    -------
    type
        Description of returned object.

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
    plt.figure(figsize = (10,7))
    sns.set(font_scale=2)
    g = sns.heatmap(df_cm, cmap=c, annot=True, fmt=".2f", annot_kws={"size": 30}, linewidths=1, linecolor='black', cbar_kws={"ticks": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, vmin=0.29, vmax=0.91)# font size
    g.set_xticklabels(g.get_xticklabels(), fontsize = 20)
    g.set_yticklabels(g.get_yticklabels(), fontsize = 20)
    g.set_title("Test Set, Accuracy = %.2f%%"%accTest)
    plt.savefig(plotpath + "/%s_%i.png"%(fn, ts), dpi=200, bbox_inches='tight')
