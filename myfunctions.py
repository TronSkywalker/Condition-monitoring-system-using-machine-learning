############################################################################
# Collection of functions
############################################################################

def find_true_keys(x):
    '''
    List all the keys of dictionary x, whose value is True
    Input arguments:
        x - dictionary
    '''
    output = []
    key_list = list(x.keys())
    value_list = list(x.values())
    for i in range(len(x)):
        if value_list[i] == True:
            output.append(key_list[i])
    return output

def listallwith(s, p):
    '''
    List all the strings in s,
    containing one substring of p
    '''
    # Convert into list, if variable p is string
    if isinstance(p, str):
        p = [p]
    new_list = []
    for i in range(len(s)):
        if any(ele in s[i] for ele in p):
            new_list.append(s[i])
    return new_list

import math
def roundup(x):
    ''' Round up value to next...
     - whole number (0 < x < 10)
     - tenth (10 < x < 100)
     - hundred (100 < x < 1000)
    '''
    if 0 < x < 10: # round up to next tenth
        return int(math.ceil(x / 1.0)) * 1
    if 10 < x < 100: # round up to next tenth
        return int(math.ceil(x / 10.0)) * 10
    if 100 < x < 1000: # round up to next hundred
        return int(math.ceil(x / 100.0)) * 100
    if 1000 < x < 10000: # round up to next hundred
        return int(math.ceil(x / 1000.0)) * 1000
    
# ROC curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)
    
