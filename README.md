# HELR: Homomorphically Encrypted log Likelihood-Ratio Classifier


## Description 
HELR classifier [1] is a pre-computed LLR (Log Likelihood-Ratio) classifier [2].
It generates for each feature an HELR lookup table and hence speeds up and simplify the biometric recognition (1:1 comparison) reducing it to three elementary operations (i.e. selection, addition and comparison).
Thus, HELR classifier facilitates the use of biometric recognition in the encrypted domain without degrading the biometric accuracy.

HELR classifier supports any biometric modality that can be encoded as a fixed-length real-valued feature vector (e.g. face, fingerprint, dynamic signature, etc.) and does not support the one encoded as a binary-valued
feature vector such as irises.


## Experiments
For a dataset DATASET from repository data, run the following experiments:

- To generate HELR lookup tables run genHELRTables.py
- To test LLR and HELR run experimentHELRvsLLR.py
- To test LLR, HELR and Cosine run experimentHELRvsLLRvsCosine.py
- To launch the cross-validation experiment run:

      1) genHELRTablesPool.py
      2) indGenImp.py
      3) crossValidateHELRvsLLRvsCosine.py 

## Required Python Packages:

This is a Python 3 implementation that requires the following packages:
- NumPy  
- SciPy 

## References

[[ 1 ]](https://arxiv.org/pdf/2101.10631.pdf)
[[ 2 ]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1262034)

## Bibtex Citation

```
@misc{bassit2021biometric,
      title={Biometric Verification Secure Against Malicious Adversaries}, 
      author={Amina Bassit and Florian Hahn and Joep Peeters and Tom Kevenaar and Raymond N. J. Veldhuis and Andreas Peter},
      year={2021},
      eprint={2101.10631},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
