# rotation-invariant-random-features

**Rotation-Invariant Random Features Provide a Strong Baseline for Machine Learning on 3D Point Clouds**

## Abstract 

TODO

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
TODO
```