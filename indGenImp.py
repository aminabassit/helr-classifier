from generalFunctions import *
from helrClassifier import *
from bioPerformance import *

import pickle

import gc




def main():  


    dataset = 'FRGC'
    testFeatFile = './data/FRGC_test.csv'
    testIDsFile = './data/FRGC_test_IDs.csv'
    testSamples = 16028 
    print(dataset)

    # Load the testing dataset
    _, testIDs = readfeatMatIDs(testFeatFile, testIDsFile, testSamples, test = True)

    test_n = len(testIDs)
    
    print('test_n = ', test_n, ' testSamples = ', testSamples)

 
    genIndexes = []
    impIndexes = []
    for i in range(test_n):
        for j in range(i+1, test_n):
            if ( testIDs[i] == testIDs[j] ):
                genIndexes.append((i,j))
            else:
                impIndexes.append((i,j))


    indexesGenImpFile = './results/indexesGenImp_{}'.format(dataset)
    pickle.dump((genIndexes, impIndexes), open(indexesGenImpFile, 'wb'))

    print('len(genIndexes) = ', len(genIndexes), ' len(impIndexes) = ', len(impIndexes))


if __name__ == '__main__':
    main()


