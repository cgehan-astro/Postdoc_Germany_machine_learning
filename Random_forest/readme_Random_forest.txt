### Project: Random_forest.py

I completed this project during my postdoc in Göttingen, Germany (2021 - 2024).


### Project overview

This project aims at predicting the values of 2 physical parameters with machine learning through a regression task using the random forest algorithm, from Lomb-Scargle periodograms of simulated synthetic light curves, i.e. flux time-series, where the flux comes from stars, which describe the effects of magnetic field lines at the stellar surface; those light curves are generated in the project Postdoc_Germany_light_curves by the program Lightcurve_with_activity_one_spot.py (description in readme_Lightcurve_with_activity_one_spot.txt). The 2 physical parameters to predict are the inclination of the rotation axis of the star with respect to the line-of-sight, and the latitude of the active regions on the surface of stars.


### Dataset

The dataset is composed of 8 parameters extracted from 1446 synthetic Lomb-Scargle periodograms that include only one dark spot with a fixed lifetime, computed for 90 different inclination values for the rotation axis (from 1 degree to 90 degrees with a 1 degree step) and for 35 different latitudes for the dark spot (from -85 degrees in the Southern hemisphere to 85 degrees in the Northern hemisphere with a 5 degrees step); a periodic signal is detectable in 1446 periodograms out of the total of 3150 periodograms, which are those used to train the random forest algorithm. The following 8 parameters extracted from the synthetic periodograms are used to train the random forest algorithm: the real and imaginary parts of the amplitude of the peak associated with the rotation period, the real and imaginary parts of the amplitude of the peak associated with the second harmonic of the rotation period of the spot, the phase of the peaks associated with the rotation period of the spot and with the second harmonic of the rotation period of the spot, and the power of the peaks associated with the rotation period of the spot and with the second harmonic of the rotation period of the spot.


### Methodology

## Data preprocessing

Data preprocessing steps involve turning the 8 raw parameters into 6 suitable, informative parameters for the machine learning algorithm: the power ratio between the peak associated with the rotation period and the peak associated with the second harmonic, the power ratio between the peak associated with the second harmonic and the peak associated with the rotation period, the ratio between the real part of the amplitude of the peak associated with the rotation period and the square root of its power, the ratio between the real part of the amplitude of the peak associated with the second harmonic and the square root of the peak associated with the rotation period, the ratio between the real part of the amplitude of the peak associated with the second harmonic and the square root of its power, and the weighted phase difference between the peak associated with the second harmonic and the peak associated with the rotation period.


## Model

The model is composed of 


## Training the model




### Results

The following results are printed in the terminal: the root mean squared error for the inclination and for the latitude, for both the train and test sets, the impurity-based feature importance (i.e. the relative importance of each feature in computing the predictions), and the median score of the estimator by cross-validation for both the inclination and the latitude (obtained by randomly splitting the training set into 10 non-overlapping subsets, i.e folds, then training and evaluating the model 10 times while picking a different fold for evaluation every time and using the other 9 folds for training). Six plots are also saved. The plot Correlation_between_input_parameters.pdf shows. 


### Installation: with anaconda

git clone https://github.com/cgehan-astro/Postdoc_Germany_machine_learning.git
cd Postdoc_Germany_machine_learning
conda env create -f environment.yml
