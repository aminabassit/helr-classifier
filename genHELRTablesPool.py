# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:22:23 2021

@author: BassitA
"""


from generalFunctions import *
from helrClassifier import *
from bioPerformance import *
import pickle

import multiprocessing as mp
import gc
from itertools import repeat





def main():

    dataset = 'CelebA'
    trainFeatFile = './data/CelebA_trainset.csv'
    trainIDsFile = './data/CelebA_trainset_IDs.csv'
    trainSamples = 31351 
    nPCA = 0
    nLDA = 94
    dQ = 1.5 
    nFQ = 6 




    print('-- Estimation of LLR\'s parameters')

    # Load the training dataset
    trainFeat, trainIDs = readfeatMatIDs(trainFeatFile, trainIDsFile, trainSamples)

    gc.collect()
    final_transform, s2, globalMean, dimF = process_LDA_after_PCA(trainFeat, trainIDs, nPCA, nLDA)
    nu, logDetTerm, diffFacs, sumFacs = classifier_param(s2)
    print('---- Number of features after PCA/LDA : '+str(dimF))

    llrParamFile = './results/llrParam_'+dataset+'.pkl'
    pickle.dump((final_transform, s2, globalMean, dimF, nu, logDetTerm, diffFacs, sumFacs), open(llrParamFile, 'wb'))
    print('---- LLR\'s parameters estimated and saved')

    gc.collect()

    print('-- Generation of HELR tables')

    # Compute Between Subject Variation
    bsv = 4*diffFacs/(1+4*diffFacs)

    print('BSV = \n', bsv)
    
    # Specify number of bits per feature
    nB = nFQ * np.ones(dimF)
    # Compute bin's borders for all features
    binBorders = createThresholds(nB)
    
    # generate HELR tables
    

    args_list = zip( bsv, nB, repeat(dQ))
    pool = mp.Pool(64)    
    helrTabs = pool.starmap( generate_HELRPool, args_list)
    gc.collect()

    helrTabsFile = './results/helrTabs_dQ_{}_nFQ_{}_{}.pkl'.format(dQ, nFQ, dataset)
    pickle.dump((helrTabs, binBorders, dQ, nB), open(helrTabsFile, 'wb'))
    print('---- HELR tables generated and saved')
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
