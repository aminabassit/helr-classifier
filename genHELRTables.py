# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:22:23 2021

@author: BassitA
"""


from generalFunctions import *
from helrClassifier import *
from bioPerformance import *
import pickle




dataset = 'PUT'
trainFeatFile = './data/PUT_train.csv'
trainIDsFile = './data/PUT_train_IDs.csv'
trainSamples = 1100 
nPCA = 0
nLDA = 0
dQ = 1 # score quantization step Delta
nFQ = 6 # 2^nFQ is the number of feature levels 


# dataset = 'BMDB'
# trainFeatFile = './data/BMDB_train.csv'
# trainIDsFile = './data/BMDB_train_IDs.csv'
# trainSamples = 1350
# nPCA = 0
# nLDA = 36
# dQ = 0.5 
# nFQ = 4 


# dataset = 'FRGC'
# trainFeatFile = './data/FRGC_train.csv'
# trainIDsFile = './data/FRGC_train_IDs.csv'
# trainSamples = 12776
# nPCA = 100
# nLDA = 94
# dQ = 1.5 
# nFQ = 6  


# nPCA if 0 then dimension will be equal to min( n_individuals-1, n_features ) otherwise specify the desired dimension that must be less than min( n_individuals-1, n_features ) 
# nLDA if 0 then dimension will be equal to min( n_individuals-1, feature_dim_after_PCA ) otherwise specify the desired dimension that must be less than min( n_individuals-1, feature_dim_after_PCA )  




print('-- Estimation of LLR\'s parameters')

# Load the training dataset
trainFeat, trainIDs = readfeatMatIDs(trainFeatFile, trainIDsFile, trainSamples)


final_transform, s2, globalMean, dimF = process_LDA_after_PCA(trainFeat, trainIDs, nPCA, nLDA)
nu, logDetTerm, diffFacs, sumFacs = classifier_param(s2)
print('---- Number of features after PCA/LDA : '+str(dimF))

llrParamFile = './results/llrParam_'+dataset+'.pkl'
pickle.dump((final_transform, s2, globalMean, dimF, nu, logDetTerm, diffFacs, sumFacs), open(llrParamFile, 'wb'))
print('---- LLR\'s parameters estimated and saved')



print('-- Generation of HELR tables')

# Compute Between Subject Variation
bsv = 4*diffFacs/(1+4*diffFacs)
# Specify number of bits per feature
nB = nFQ * np.ones(dimF)
# Compute bin's borders for all features
binBorders = createThresholds(nB)
# generate HELR tables 
helrTabs = generate_HELRtables(bsv, nB, dQ)

helrTabsFile = './results/helrTabs_dQ_{}_nFQ_{}_{}.pkl'.format(dQ, nFQ, dataset)
pickle.dump((helrTabs, binBorders, dQ, nB), open(helrTabsFile, 'wb'))
print('---- HELR tables generated and saved')



