# Datasets

## Structure

This repository contains training and testing datasets that are organized as follows:

- Training phase files:
    -  'DATASET_train.csv' contains feature vectors of shape=(n_features, n_samples)
    - 'DATASET_train_IDs.csv' contains the corresponding subjects IDs that are in a consecutive order
    
- Testing phase files:
    - 'DATASET_test.csv' contains feature vectors of shape=(n_features, n_samples)
    - 'DATASET_test_IDs.csv' contains the corresponding subjects IDs that are in a consecutive order

## Public Datasets

The public datasets used for testing the HELR classifier are:

- **BMDB dataset**[1] (DS2 of BMDB  only genuine signatures are considered and skilled forgeries are not) for Dynamic Signatures. The features were extracted by the
algorithm described in [2].

- **PUT dataset**[3] for faces. The initial features were extracted by the VGGFace [4], the first layers’ weights were taken from [5] and the last layer was retrained for PUT to learn a projection from the 4096-dimensional last layer's output of the pre-trained model to a 64-dimensional latent space. Further details can be found in [6].

- **FRGC2.0 dataset**[7] (Experiment 1, mask II) for faces. The features were extracted by VGGFace[4].

- **CelebA dataset**[8] for faces. The features were extracted by VGGFace[4].

## References

[[ 1 ]](https://ieeexplore.ieee.org/document/4815263)
[[ 2 ]](https://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w4/papers/Gomez-Barrero_Implementation_of_Fixed-Length_CVPR_2016_paper.pdf)
[[ 3 ]](https://www.researchgate.net/profile/Adam-Schmidt-6/publication/232085001_The_PUT_face_database/links/09e4150d0bf1e5080f000000/The-PUT-face-database.pdf)
[[ 4 ]](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
[[ 5 ]](https://github.com/rcmalli/keras-vggface)
[[ 6 ]](https://aircconline.com/csit/papers/vol10/csit101901.pdf)
[[ 7 ]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1467368)
[[ 8 ]](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html?fbclid=IwAR1VeZyPhkTzsoD_Fq8ItPwvyA0W1MD7fHO0v7MVaps1oX1fSt95q5i8Wfo)

