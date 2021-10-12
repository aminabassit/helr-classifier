# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:03:40 2021

@author: BassitA
"""


import numpy as np
from scipy.stats import norm
from scipy.integrate import dblquad




" HELR parameter functions "




def preprocess_PCA(trData, tr_ids, dPCA):
    # Assumes that IDs of users in training are consecutive
    nUsersTrain = np.max(tr_ids) - np.min(tr_ids) +1    
    
    dFeat,nFeatsTrain = trData.shape
    # dPCA must be less than or equal to number of training set -1 and feature length.
    if (dPCA == 0):
        dPCA = min(dFeat, nFeatsTrain - 1)
    else:
        dPCA = min(dFeat, nFeatsTrain - 1, dPCA)
        
    # (1) Compute global mean and subtract. Global mean is the mean per feature 
    featMean = np.mean(trData, axis = 1)
    global_featMean = np.repeat(featMean,nFeatsTrain)
    global_featMean = global_featMean.reshape(dFeat, nFeatsTrain)    
    # subtract 
    trData0 = trData - global_featMean  
    
    # (2) Compute local means    
    userMeans = np.zeros((dFeat, nUsersTrain))
    for i in range(nUsersTrain):
        ix = np.where(tr_ids == i+1)[0]
        wi = np.sqrt(len(ix)/nFeatsTrain) 
        userMeans[:,i] = wi * np.mean(trData0[:, ix[0]:ix[-1]+1], axis = 1)
        
    # (3) PCA Whiten
    u1, s1, vh1 = np.linalg.svd(trData0, full_matrices=True)  
   
    # (4) Compute PCA-whitening part of transform  
    u1 = u1[:, :dPCA]
    u1_tran = u1.transpose()
    s1 = s1[:dPCA]/np.sqrt(nFeatsTrain) # s1 should be an array containing the diag's values
        
    W = np.repeat(1/s1,dFeat)
    W = W.reshape(len(s1),dFeat)
    W = W * u1_tran    
    userMeansW = np.matmul(W,userMeans)     
    
    return nUsersTrain, W, userMeansW, featMean, dPCA
    


def process_LDA_after_PCA(trData, tr_ids, dPCA, dLDA): 
    # run PCA first
    nUsersTrain, W, userMeansW, globalMean, dPCA = preprocess_PCA(trData, tr_ids, dPCA)
    
    # dLDA  Must be less than or equal to number of individuals in training set -1 and feature length
    # Important !  feature length after PCA    
    if (dLDA == 0):
        dLDA = min(nUsersTrain - 1, dPCA)
    else:
        dLDA = min(nUsersTrain - 1, dPCA, dLDA)
        
    # LDA part
    u2, s2, vh2 = np.linalg.svd(userMeansW, full_matrices=True) 
    u2 = u2[:, :dLDA]
    u2_tran = u2.transpose()
    s2 = s2[:dLDA]    
    # Final Transform 
    final_transform = np.matmul(u2_tran,W)
    
    return final_transform, s2, globalMean, dLDA




def classifier_param(s2):
    #  Compute baseline classifier parameters
    Nu = np.power(s2, 2)    # variance per feature
    LogDetTerm = np.log(1-np.power(Nu, 2))
    LogDetTerm = - sum(LogDetTerm)/2  
    DiffFacs = (Nu/(1-Nu))/4
    SumFacs = (Nu/(1+Nu))/4
    return Nu, LogDetTerm, DiffFacs, SumFacs





def jointFeatureSameUserPDF(x, y, bsv):
    """Computes the joint 2D Gaussian PDF of two genuine feature values of the i-th feature
            Parameters
            ----------
            x : feature value on the x-axis of the i-th feature
            y : feature value on the y-axis of the i-th feature        
            bsv : between subject variance of the i-th feature
                
            Returns
            -------
            p : value of the joint 2D Gaussian PDF of two genuine feature values of the i-th feature
    """
    d = 1-np.power(bsv, 2)
    p = (np.power(x,2) - 2*x*y*bsv + np.power(y,2))/(2*d)
    p = np.exp(-p)/(2*np.pi*np.sqrt(d)) 
    return p


def findEquiProbThresholds(n):
    """Computes the bin's borders of a single feature for n feature levels
            Parameters
            ----------
            n : feature levels, n is usually of the form 2^b where b is the number bits spent to encode a feature
                
            Returns
            -------
            binBorders : array-like of shape=(n-1,) 
                         bin's borders for n feature levels 
    """
    # equiprobable division 
    p = np.arange(1,n)/n   
    # determine the borders of equiprobable bins assuming a Normal PDF N(0,1)
    binBorders = norm.ppf(p, loc=0, scale=1)
    return binBorders



def createThresholds(nB):
    """Computes the bin's borders for all features
            Parameters
            ----------
            nB : array-like of shape=(n_features,)
                 number of bits determining the number of feature levels; 
                 example: i-th feature is quantized on 2^nB[i] feature levels
            
            Returns
            -------
            binBorders : list of n_features arrays of shape=(2^nB[i] -1 ,) 
                         bin's borders for n_features features  
    """
    dFeat = len(nB)
    num_levels = np.power(2,nB)
    binBorders = [findEquiProbThresholds(num_levels[i]) for i in range(dFeat)] 
    return binBorders



def getIndexQF(x, t):
    """Computes index of feature x in quantization boundaries t array
            Parameters
            ----------
            x : feature value to be quantized according to t
            t : array-like of shape=(n-1,) 
                bin's borders for n feature levels 
            
            Returns
            -------
            x_index : quantized feature value; it also corresponds to the index in the HELR table w.r.t x-axis or y-axis
    """
    x_index = len(np.where(t<x)[0])
    return x_index





def jointFeatureSameUserProbability(nb, bsv):
    """Computes the 2D Gaussian genuine distribution of a single feature
            Parameters
            ----------
            nb :  number of bits determining the feature levels    
            bsv : between subject variance of that feature
                
            Returns
            -------
            pSame : array-like of shape=(2^nb,2^nb)
                    genuine distribution of a single feature
                    
    """
    inf = np.inf
    nL = int(2**nb)
    nL2 = int(nL/2)
    t = findEquiProbThresholds(nL)
    pSame = np.zeros((nL, nL))    
    # compute the bins that are on the diag and diag_inver of pSame with mirroring 
    for i in range(nL2):        
        if (i == 0):
            xmin = -inf 
            xmax = t[0]
            ymin1 = xmin
            ymax1 = xmax
            ymin2 = t[-1] 
            ymax2 = inf
        else:
            xmin = t[i-1] 
            xmax = t[i]
            ymin1 = xmin
            ymax1 = xmax
            ymin2 = t[-i-1]
            ymax2 = t[-i]            
        pSame[i,i] = dblquad(lambda x, y: jointFeatureSameUserPDF(x, y, bsv), xmin, xmax, lambda y: ymin1, lambda y: ymax1)[0]
        pSame[nL-1-i,nL-1-i] = pSame[i,i]
        pSame[i,nL-1-i] = dblquad(lambda x, y: jointFeatureSameUserPDF(x, y, bsv), xmin, xmax, lambda y: ymin2, lambda y: ymax2)[0]
        pSame[nL-1-i,i] = pSame[i,nL-1-i]
    # compute the bin out of both diags of pSame with mirroring
    for i in range(1,nL-1):
        xmin = t[i-1]
        xmax = t[i]
        for j in range(min(nL-i-1, i)):
            if (j == 0):
                ymin = -inf
                ymax = t[0]
            else:
                ymin = t[j-1]
                ymax = t[j]
            pSame[i,j] = dblquad(lambda x, y: jointFeatureSameUserPDF(x, y, bsv), xmin, xmax, lambda y: ymin, lambda y: ymax)[0]
            pSame[j,i] = pSame[i,j]
            pSame[nL-1-i, nL-1-j] = pSame[i,j]
            pSame[nL-1-j,nL-1-i] = pSame[i,j]
            
    return pSame




def generate_HELRtables(bsv, nB, dQ):
    """Generates HELR tables of all features 
            Parameters
            ----------
            nB :    array-like of shape=(n_features,)
                    number of bits determining the number of feature levels;  
            bsv :   array-like of shape=(n_features,)
                    between subject variance of n_features features
            dQ :  score quantization step; example: 2, 1.5, 1, 0.5, 0.2, 0.01, etc.  
                
            Returns
            -------
            helrTabs: list of 2D arrays storing the HELR tables
                    
    """
    nFeat = len(nB)
    helrTabs = []
    for i in range(nFeat):
        # impostor distribution equals to 1/(nL**2)
        nL = np.power(2,nB[i])
        # genuine distribution 
        p  = jointFeatureSameUserProbability(nB[i], bsv[i])
        # HELR table of the i-th feature 
        helrTab = np.round((np.log(p) + 2*np.log(nL))/dQ).astype(int)
        helrTabs.append(helrTab)
    return helrTabs





def computeLRQ(x, y, t, helrTabs):
    """Computes quantized LLR from two feature vectors using the HELR tables
            Parameters
            ----------
            x, y : feature vectors of shape=(n_features,)
            t : list of bin borders of n_features features          
            helrTabs: list of the HELR tables corresponding to n_features features
                
            Returns
            -------
            score : final score that is the sum of all individual scores 
                    
    """
    nFeat = len(x)
    score = 0
    for i in range(nFeat):
        # quantize the i-th feature value of x
        ix = getIndexQF(x[i], t[i])
        # quantize the i-th feature value of y
        iy = getIndexQF(y[i], t[i])
        # fetch the pre-computed HELR value of HELR(ix,iy)
        # add it to the final score
        score  += helrTabs[i][ix,iy]
    return score



def loglr_after_PCA_LDA(x, y, logDetTerm, diffFacs, sumFacs):
    """Computes unquantized LLR from two feature vectors using the estimated parameters
            Parameters
            ----------
            x, y : feature vectors of shape=(n_features,)
            logDetTerm, diffFacs, sumFacs : estimated parameters from the training phase
                
            Returns
            -------
            score : final score LLR score
                    
    """
    dsq = np.power(x-y, 2)
    ssq = np.power(x+y, 2)
    loglr = logDetTerm - np.sum(diffFacs * dsq) + np.sum(sumFacs *ssq )
    return loglr


def matedUnmatedScores_LLR_HELR(testIDs, featsTestTrans, logDetTerm, diffFacs, sumFacs, binBorders, helrTabs):
    """Computes mated and unmated scores for both LLR and HELR
            Parameters
            ----------
            testIDs : 
            featsTestTrans :  
            logDetTerm, diffFacs, sumFacs :  
            binBorders : list of n_features arrays of shape=(2^nB[i] -1 ,) 
                         bin's borders for n_features features      
            helrTabs: list of the HELR tables corresponding to n_features features
                
            Returns
            -------
            matedSc, unMatedSc : mated and unmated scores for LLR
            matedScQ, unMatedScQ : mated and unmated scores for HELR
                    
    """
    nTest = featsTestTrans.shape[1]    
    matedSc = []
    unMatedSc = []
    matedScQ = []
    unMatedScQ = []
    for i in range(nTest):
        x = featsTestTrans[:,i]
        for j in range(i+1,nTest):
            y = featsTestTrans[:,j]
            score = loglr_after_PCA_LDA(x,y, logDetTerm, diffFacs, sumFacs)
            scoreQ =  computeLRQ(x, y, binBorders, helrTabs)
            # Mated or Un-mated comparison?
            if (testIDs[i] == testIDs[j]): # Genuine comparison
                matedSc.append(score)
                matedScQ.append(scoreQ)
            else: # Impostor comparison
                unMatedSc.append(score)
                unMatedScQ.append(scoreQ)
    return np.asarray(matedSc), np.asarray(unMatedSc), np.asarray(matedScQ), np.asarray(unMatedScQ)







###############################################################################


def cosineSimilarity(x, y):
    """Computes Cosine Similarity between two feature vectors
            Parameters
            ----------
            x, y : feature vectors of shape=(n_features,)
                
            Returns
            -------
            cosineXY : Cosine(x,y)
                    
    """
    normX = np.sqrt(np.sum(x**2))
    normY = np.sqrt(np.sum(y**2))
    innerXY = np.sum(x*y)
    cosineXY = innerXY/(normX * normY)
    return cosineXY



def matedUnmatedScores_LLR_HELR_Cosine(testIDs, featsTest, featsTestTrans, logDetTerm, diffFacs, sumFacs, binBorders, helrTabs):
    """Computes mated and unmated scores for LLR, HELR and Cosine
            Parameters
            ----------
            testIDs : 
            featsTest :
            featsTestTrans :  
            logDetTerm, diffFacs, sumFacs :  
            binBorders : list of n_features arrays of shape=(2^nB[i] -1 ,) 
                         bin's borders for n_features features      
            helrTabs: list of the HELR tables corresponding to n_features features
                
            Returns
            -------
            matedSc, unMatedSc : mated and unmated scores for LLR
            matedScQ, unMatedScQ : mated and unmated scores for HELR
            matedScCosine, unMatedScCosine : mated and unmated scores for Cosine
                    
    """
    nTest = featsTestTrans.shape[1]    
    matedSc = []
    unMatedSc = []
    matedScQ = []
    unMatedScQ = []
    matedScCosine = []
    unMatedScCosine = []
    for i in range(nTest):
        x = featsTestTrans[:,i]
        xCos = featsTest[:,i]
        for j in range(i+1,nTest):
            y = featsTestTrans[:,j]
            yCos = featsTest[:,j]
            score = loglr_after_PCA_LDA(x,y, logDetTerm, diffFacs, sumFacs)
            scoreQ =  computeLRQ(x, y, binBorders, helrTabs)
            scoreCosine = cosineSimilarity(xCos, yCos)
            # Mated or Un-mated comparison?
            if (testIDs[i] == testIDs[j]): # Genuine comparison
                matedSc.append(score)
                matedScQ.append(scoreQ)
                matedScCosine.append(scoreCosine)
            else: # Impostor comparison
                unMatedSc.append(score)
                unMatedScQ.append(scoreQ)
                unMatedScCosine.append(scoreCosine)
    return np.asarray(matedSc), np.asarray(unMatedSc), np.asarray(matedScQ), np.asarray(unMatedScQ), np.asarray(matedScCosine), np.asarray(unMatedScCosine)








def scores_LLR_HELR_CosinePool(indexIJ, featsTest, featsTestTrans, logDetTerm, diffFacs, sumFacs, binBorders, helrTabs):
    """Computes LLR, HELR and Cosine scores for two feature vectors
            Parameters
            ----------
            indexIJ : index of two testIDs 
            featsTest : feature vectors 
            featsTestTrans : feature vectors of lower dimension 
            logDetTerm, diffFacs, sumFacs :  
            binBorders : list of n_features arrays of shape=(2^nB[i] -1 ,) 
                         bin's borders for n_features features      
            helrTabs: list of the HELR tables corresponding to n_features features
                
            Returns
            -------
            score : LLR score  
            scoreQ : HELR score
            scoreCosine : Cosine score
    """
    
    (i,j) = indexIJ
    x = featsTestTrans[:,i]
    xCos = featsTest[:,i] 
    y = featsTestTrans[:,j]
    yCos = featsTest[:,j]

    score = loglr_after_PCA_LDA(x,y, logDetTerm, diffFacs, sumFacs)
    scoreQ =  computeLRQ(x, y, binBorders, helrTabs)
    scoreCosine = cosineSimilarity(xCos, yCos)

    return score, scoreQ, scoreCosine

def generate_HELRPool(bsvi, nBi, dQ):
    """Generates a single HELR lookup table
            Parameters
            ----------
            bsvi : BSV of the i-th feature 
            nBi : number of bits determining the number of feature levels of the -th feature 
            dQ : score quantization step; example: 2, 1.5, 1, 0.5, 0.2, 0.01, etc. 
                
            Returns
            -------
            helrTab : HELR lookup table
    
    """
    nL = np.power(2,nBi) 
    p  = jointFeatureSameUserProbability(nBi, bsvi)
    helrTab = np.round((np.log(p) + 2*np.log(nL))/dQ).astype(int)
    return helrTab