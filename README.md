# rotation-invariant-random-features

**Rotation-Invariant Random Features Provide a Strong Baseline for Machine Learning on 3D Point Clouds**

Published in Transactions on Machine Learning Research

Reviewed on OpenReview [https://openreview.net/forum?id=nYzhlFyjjd](https://openreview.net/forum?id=nYzhlFyjjd)
## Abstract 
Rotational invariance is a popular inductive bias used by many fields in machine learning, such as computer vision and machine learning for quantum chemistry. 
Rotation-invariant machine learning methods set the state of the art for many tasks, including molecular property prediction and 3D shape classification. 
These methods generally either rely on task-specific rotation-invariant features, or they use general-purpose deep neural networks which are complicated to design and train.
However, it is unclear whether the success of these methods is primarily due to the rotation invariance or the deep neural networks. 
To address this question, we suggest a simple and general-purpose method for learning rotation-invariant functions of three-dimensional point cloud data 
using a random features approach. Specifically, we extend the random features method of [Rahimi and Recht, 2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) by deriving a version that is invariant to three-dimensional rotations and showing that it is fast to evaluate on point cloud data. 
We show through experiments that our method matches or outperforms the performance of general-purpose rotation-invariant neural networks on standard molecular property prediction benchmark datasets QM7 and QM9. 
We also show that our method is general-purpose and provides a rotation-invariant baseline on the ModelNet40 shape classification task. 
Finally, we show that our method has an order of magnitude smaller prediction latency than competing kernel methods.

![An image of a random feature matrix. Molecular point clouds line the rows to the left of the random feature matrix and random sums of spherical harmonic functions line the columns above. To the right of the feature matrix is a yellow column vector labeled Beta, an equals sign, and a blue column vector labeled Y.](assets/method_overview.png)

## How to Use This Repository

Sample commands are in the `sample_scripts/` folder. Note: these scripts do not include tuned hyperparameters for the methods. 

### Install Dependencies

The python dependencies are listed in `environment.yml`. They can be installed with the following command:
```
conda env create --name RIRF --file environment.yml
```

### Testing

Testing the functions implemented in `src/` can be done by running 
```
python -m pytest test/
```

## Citation

```
@article{
melia2023rotationinvariant,
title={Rotation-Invariant Random Features Provide a Strong Baseline for Machine Learning on 3D Point Clouds},
author={Owen Melia and Eric M Jonas and Rebecca Willett},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=nYzhlFyjjd},
note={}
}
```
