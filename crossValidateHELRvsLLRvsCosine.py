from generalFunctions import *
from helrClassifier import *
from bioPerformance import *
import multiprocessing as mp

import pickle
from datetime import datetime

import gc
from itertools import repeat






def main():
    now = datetime.now()
    timeTag = now.strftime("%d%m%Y_%H%M%S") 

    datasetTables = 'CelebA' 

    dataset = 'FRGC'
    testFeatFile = './data/FRGC_test.csv'
    testIDsFile = './data/FRGC_test_IDs.csv'
    testSamples = 16028 

    precisionD = 1E-7
    nFQ = 6
    dQ = 1.5

    print('Cross-validation experiment:')

    print('The HELR lookup tables are generated from the {} dataset then tested on the {} dataset'.format(datasetTables, dataset))


    print('-- Importing {} LLR parameters'.format(datasetTables))
    llrParamFile = './results/llrParam_'+datasetTables+'.pkl'
    (final_transform, s2, globalMean, dimF, nu, logDetTerm, diffFacs, sumFacs) = pickle.load(open(llrParamFile, 'rb'))
    print('-- Importing {} HELR tables'.format(datasetTables))   
            
    helrTabsFile = './results/helrTabs_dQ_{}_nFQ_{}_{}.pkl'.format(dQ, nFQ, datasetTables)
    (helrTabs, binBorders, dQ, nB) = pickle.load(open(helrTabsFile, 'rb'))



    print('-- Importing {} LLR parameters'.format(dataset))
    llrParamFileD = './results/llrParam_'+dataset+'.pkl'
    (final_transformD, _, globalMeanD, _, _, _, _, _) = pickle.load(open(llrParamFileD, 'rb'))
    print('-- Importing {} HELR tables'.format(dataset))

    

    
    print('-- Testing {} HELR tables on {} '.format(datasetTables, dataset))

    # Load the testing dataset
    testFeat, testIDs = readfeatMatIDs(testFeatFile, testIDsFile, testSamples, test = True)
    print('-- Subtract mean and reduce the features\' dimension')
    

    featsTestTrans = subtractMeanAndTransformMatrix(testFeat, final_transformD, globalMeanD)



    
    print('-- Run mated and unmated comparison for LLR, HELR and Cosine')
    
    pool = mp.Pool(64)

    indexesGenImpFile = './results/indexesGenImp_{}'.format(dataset)
    (genIndexes, impIndexes) = pickle.load(open(indexesGenImpFile, 'rb'))

    print('Indexes are loaded')




    gen_args = zip(genIndexes, repeat(testFeat), repeat(featsTestTrans), 
    repeat(logDetTerm), repeat(diffFacs), repeat(sumFacs), repeat(binBorders), repeat(helrTabs))

    imp_args = zip(impIndexes, repeat(testFeat), repeat(featsTestTrans), 
    repeat(logDetTerm), repeat(diffFacs), repeat(sumFacs), repeat(binBorders), repeat(helrTabs))

    

    print('Start pool.starmap')
    print('LLR HELR and Cosine mated')

    resultsMD = pool.starmap(scores_LLR_HELR_CosinePool, gen_args)
    scoreMD, scoreQMD, scoreCosineMD = zip(*resultsMD)
    gc.collect()

    print('LLR HELR and Cosine unmated')

    resultsUNMD = pool.starmap(scores_LLR_HELR_CosinePool, imp_args)
    scoreUNMD, scoreQUNMD, scoreCosineUNMD = zip(*resultsUNMD)
    gc.collect()

    print('End pool.starmap')

   

    scoreMD = np.array(scoreMD, dtype = float)
    scoreUNMD = np.array(scoreUNMD, dtype = float)
    scoreQMD = np.array(scoreQMD, dtype = int)
    scoreQUNMD = np.array(scoreQUNMD, dtype = int)
    scoreCosineMD = np.array(scoreCosineMD, dtype = float)
    scoreCosineUNMD = np.array(scoreCosineUNMD, dtype = float)



    print('-- Measure the performance of LLR')
    _, fmrLLR, tmrLLR = get_fmr_tmr_Prec(scoreMD, scoreUNMD, precision = 0.1)
    eerLLR = calc_eerPT(fmrLLR, tmrLLR)
    gc.collect()  


    print('-- Measure the performance of HELR')
    _, fmrHELR, tmrHELR = get_fmr_tmr_Prec(scoreQMD, scoreQUNMD, precision = 1)
    eerHELR = calc_eerPT(fmrHELR, tmrHELR)
    gc.collect()

    print('-- Measure the performance of Cosine')
    _, fmrCosine, tmrCosine = get_fmr_tmr_Prec(scoreCosineMD, scoreCosineUNMD, precision = precisionD)
    eerCosine = calc_eerPT(fmrCosine, tmrCosine)
    gc.collect()


    plotFile = './results/DETs_Cosine_LLR_HELR_dQ_{}_nFQ_{}_{}_cross_{}_{}.png'.format(dQ, nFQ, datasetTables, dataset,timeTag)
    plotTitle = 'LLR, HELR_{} (dQ = {}; nFQ = {}), Cosine on {} test set'.format(datasetTables, dQ, nFQ, dataset)  
    fmrList = [fmrLLR, fmrHELR, fmrCosine]
    tmrList = [tmrLLR, tmrHELR, tmrCosine] 
    labelList = ['LLR (EER = {:.2E})'.format(eerLLR), 'HELR (EER = {:.2E})'.format(eerHELR), 'Cosine (EER = {:.2E})'.format(eerCosine)] 
    linewidthList = [2, 2, 2]
    
    fmrTmrFile = './results/FMR_TMR_Cosine_LLR_HELR_dQ_{}_nFQ_{}_{}_cross_{}_{}.pkl'.format(dQ, nFQ, datasetTables, dataset,timeTag)
    pickle.dump((fmrList, tmrList, labelList), open(fmrTmrFile, 'wb'))

    
    plot_MultipleDETsSaved(plotFile, plotTitle, fmrList, tmrList, labelList, linewidthList)
    
    
    plotFile = './results/DETs_LLR_HELR_dQ_{}_nFQ_{}_{}_cross_{}_{}.png'.format(dQ, nFQ, datasetTables, dataset,timeTag)
    plotTitle = 'LLR, HELR_{} (dQ = {}; nFQ = {}) on {} test set'.format(datasetTables, dQ, nFQ, dataset) 
    plot_MultipleDETsSaved(plotFile, plotTitle, fmrList[:-1], tmrList[:-1], labelList[:-1], linewidthList[:-1])

    pool.close()
    pool.join()









if __name__ == '__main__':
    main()



