# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:58:49 2021

@author: BassitA
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np







def readfeatMatIDs(featMatFile, idsFile, n_samples, test = False):
    """Reads feature vectors and their corresponding subject IDs from csv files
            Parameters
            ----------
            
            featMatFile : path to the feature vectors file
            idsFile : path to the subjects IDs file
            n_samples : number of samples
            
            Returns
            -------
            
            featMat :   array-like of shape=(n_features, n_samples)
                        feature vectors corresponding to subjectIDs
            subjectIDs: array-like of shape=(n_samples,)
                        id of a subject is repeated n_occurrence of that subject 
                        it looks like [1,...,1,2,...,2,...,n,...,n] 
    """
    featMat = pd.read_csv(featMatFile, names = list(range(0,n_samples)))
    featMat = featMat.to_numpy()
    ids = pd.read_csv(idsFile, names = list(range(0,n_samples)))
    subjectIDs = ids.to_numpy() 
    subjectIDs = subjectIDs[0]
    # set subject IDs to start from 1 
    subjectIDs = subjectIDs - np.min(subjectIDs)+1  
    print('Feature vectors and IDs are loaded')
    (n_features, n_samples) = featMat.shape
    n_subjects = subjectIDs[-1]    
    print('-- Number of subjects : {}'.format(n_subjects))
    print('-- Number of samples : {}'.format(n_samples))    
    if test == True:
        # r is the number of samples per subject
        r = n_samples / n_subjects
        genComparison = n_subjects * (r * (r-1))/2
        impComparison = r**2 * (n_subjects * (n_subjects-1))/2
        print('-- Number of samples per subject : {:.2E}'.format(r))
        print('-- Number of genuine comparisons : {:.2E}'.format(genComparison))
        print('-- Number of impostor comparisons : {:.2E}'.format(impComparison))  
    else:
        print('-- Number of initial features : {}'.format(n_features))
    
    return featMat, subjectIDs








def subtractMeanAndTransformMatrix(dataMat, transform, globalMean):
    """Subtracts the global mean and reduces the features' dimensionality by applying transform 
            Parameters
            ----------
            dataMat :   array-like of shape=(n_features, n_samples)
            transform : array-like of shape=(n_reducedfeats, n_features)
                        transformation matrix to reduce the dimensionality from n_features to n_reducedfeats
            globalMean :    array-like of shape=(n_features,)
                            global mean per feature determined from pre-processing PCA
                
            Returns
            -------
            dataMat0reduced :   array-like of shape=(n_reducedfeats, n_samples)
                                zero-mean and features' dimension reduced  
    """
    row, col = dataMat.shape
    dataMat_ct = np.repeat(globalMean, col)
    dataMat_ct = dataMat - dataMat_ct.reshape(row, col)
    dataMat0reduced = np.matmul(transform, dataMat_ct)
    return dataMat0reduced



def plot_MultipleDETsSaved(plotFile, plotTitle, fmrList, tmrList, labelList, linewidthList):
    """Plots and saves the DET curves of fmrList and 1-tmrList
            Parameters
            ----------
            plotFile : path to where the plot should be saved
            plotTitle : path to  
            fmrList : list of FMRs 
            tmrList : list of TMRs  
            labelList : list of lables corresponding to the DET curve of the i-th fmrList and 1-tmrList
            linewidthList : list of integers specifying the linewidth of the i-th DET curve
            
            Returns
            -------
            
    """
    eer_line = np.logspace(-4,0,100) 
    fig, ax = plt.subplots()
    for fmr, tmr, lab, lw in zip(fmrList, tmrList, labelList, linewidthList):
        # fnmr is equal to 1-tmr
        ax.loglog(fmr, 1-tmr, label = lab, linewidth=lw)
    ax.loglog(eer_line, eer_line, label = 'EER')
    ax.set_aspect('equal')
    ax.set_xlim([1E-4, 1])
    ax.set_ylim([1E-4, 1])
    ax.set(xlabel='FMR', ylabel='FNMR')
    ax.legend()
    fig.suptitle('DET curves '+ plotTitle)
    plt.savefig(plotFile)






















