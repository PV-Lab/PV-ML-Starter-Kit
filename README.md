# PV-ML-Starter-Kit

#### Authors: Aleks Siemenn & Tonio Buonassisi

Starter kit for photovoltaics optimization using machine learning.

_________________________________________________________________________

This kit includes useful data visualization tools as well as a starter package for Bayesian optimization of photovoltaics data. The code to run this starter kit is found in the file [Run_Starter_Kit.ipynb](./Run_Starter_Kit.ipynb)

## Input Data Format

The input data consists of a `.csv file` with each experiment formatted into rows with each column being a real, continuous variable. The last column of the file should be the target variable that will be optimized. An example dataset on how to format the file properly can be found [here](./data/Example_PV_Dataset.csv).

## Running the Code

All required packages and the corresponding versions can be found in the [requirements.txt](./requirements.txt) file. Once installed in the Python environment of choice, the code can be run.

## Diagnostics Visualization

Several forms of data visualization are used to help the user better understand the biases, correlations, and variable importance within the dataset.

* **1D and 2D Histograms** plot the skew of data and their distributions against another variable. 
* **Correlation Coefficient Matrices** show how closely any two variables are coupled to each other.
* **SHAP Feature Importance** ranks each variable as how important it is in determining the magnitude and direction of the target output variable values.

## Bayesian Optimization

Bayesian optimization (BO) stems from the principles of [Bayes' Theorem](https://faculty.washington.edu/tamre/BayesTheorem.pdf). BO is a probabilistic tool used to help guide the selection of new data points within a dataset based on the computed means and variances of a surrogate model. BO is useful for driving experimental optimization when data are scarce or expensive to obtain.

By running the code found in [Run_Starter_Kit.ipynb](./Run_Starter_Kit.ipynb), a set of suggested conditions will be presented by the algorithm. These conditions either exploit the predicted means to obtain a highly optimum sample or explore the high variance regions of the dataset to minimize uncertainity for the next round of sampling.
