# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:14:38 2021

@author: BassitA
"""






import numpy as np








def get_fmr_tmr_Prec(matedScores, unmatedScores, precision = 0.1):
    """Calculates the FMR and TMR from mated and unmated scores
            Parameters
            ----------
            matedScores :   array-like of shape = (n_genuine,)
                            mated scores resulting from genuine comparison of two same-subject-samples
            unmatedScores : array-like of shape = (n_impostor,)  
                            unmated scores resulting from impostor comparison of two different-subject-samples
            precision :     step between two adjacent threshold scores
                            equals to 1 when the scores are integers and 1E-n when they are real-values 
            
            
            Returns
            -------
            fmr :   array-like representing the False Match Rate
                    the rate of impostor scores as a function of the threshold
            tmr :   array-like representing the True Match Rate
                    the rate of genuine scores as a function of the threshold
            scores : array-like representing all possible thresholds
    """
    # determine all possible thresholds
    minS = min(min(matedScores), min(unmatedScores))
    maxS = max(max(matedScores), max(unmatedScores))
    scores = np.arange(minS, maxS + precision, precision)
    lenMated = len(matedScores)
    lenUnMated = len(unmatedScores)
    # calculate the occurrence of mated scores 
    pgen = np.histogram(matedScores, bins=scores, density=False)[0]
    # calculate the genuine probability 
    pgen = pgen/lenMated
    # calculate the TMR as a function of the thresholds (that is scores)
    tmr = np.maximum(0, 1 - np.cumsum(pgen))
    # calculate the occurrence of unmated scores 
    pimp = np.histogram(unmatedScores, bins=scores, density=False)[0]
    # calculate the impostor probability
    pimp = pimp/lenUnMated
    # calculate the FMR as a function of the thresholds (that is scores)    
    fmr = np.maximum(0, 1 - np.cumsum(pimp))  
    # for a threshold equals to scores[i], FMR equals to fmr[i] and TMR equals to tmr[i] 
    return scores, fmr, tmr




" Important points "


def calc_eerPT(fmr, tmr):
    """Calculates the Equal Error Rate point from fmr and 1-tmr
            Parameters
            ----------
            fmr :   array-like representing the False Match Rate
                    the rate of impostor scores as a function of the threshold
            tmr :   array-like representing the True Match Rate
                    the rate of genuine scores as a function of the threshold
            
            Returns
            -------
            eer : point where fmr and fnmr (1-tmr) are equal
    """
    fnmr = 1-tmr
    x = fnmr - fmr
    if ((fmr.size == 0) or (fnmr.size == 0)):
        return np.inf
    
    index = np.argmin(np.abs(x))   
    
    if (index == 0):
        return (fnmr[index] + fmr[index])/2 

    if (index == len(fmr)):
        return (fnmr[index] + fmr[index])/2 

    if (fmr[index] <= fnmr[index]):
        l_index = index - 1
        r_index = index
    else:
        l_index = index
        r_index = index + 1

    d_1 = fmr[l_index] - fnmr[l_index]
    d_2 = fmr[r_index] - fnmr[r_index]

    if (d_1 - d_2) == 0:
        s_val = 0.5
    else:
        s_val = d_1 / (d_1 - d_2)

    eer = fnmr[l_index] + s_val*(fnmr[r_index] - fnmr[l_index])
    return eer

















