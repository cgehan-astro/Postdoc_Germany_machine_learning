### Project: Random_forest.py

I completed this project during my postdoc in Göttingen, Germany (2021 - 2024).


### Project overview

This project aims at predicting the values of 2 physical parameters with machine learning through a regression task using the random forest algorithm, from Lomb-Scargle periodograms of simulated synthetic light curves, i.e. flux time-series, where the flux comes from stars, which describe the effects of magnetic field lines at the stellar surface; those light curves are generated in the project Postdoc_Germany_light_curves by the program Lightcurve_with_activity_one_spot.py (description in readme_Lightcurve_with_activity_one_spot.txt). The 2 physical parameters to predict are the inclination of the rotation axis of the star with respect to the line-of-sight, and the latitude of the active regions on the surface of stars.


### Dataset

The dataset is composed of 8 parameters extracted from 1446 synthetic Lomb-Scargle periodograms that include only one dark spot with a fixed lifetime, computed for 90 different inclination values for the rotation axis (from 1 degree to 90 degrees with a 1 degree step) and for 35 different latitudes for the dark spot (from -85 degrees in the Southern hemisphere to 85 degrees in the Northern hemisphere with a 5 degrees step); a periodic signal is detectable in 1446 periodograms out of the total of 3150 periodograms, which are those used to train the random forest algorithm. The following 8 parameters, extracted from the synthetic periodograms and stored in the file input_random_forest_parameters.txt, are used to train the random forest algorithm: the real and imaginary parts of the amplitude of the peak associated with the rotation period, the real and imaginary parts of the amplitude of the peak associated with the second harmonic of the rotation period of the spot, the phase of the peaks associated with the rotation period of the spot and with the second harmonic of the rotation period of the spot, and the power of the peaks associated with the rotation period of the spot and with the second harmonic of the rotation period of the spot.


### Methodology

## Data preprocessing

Data preprocessing steps involve turning the 8 raw parameters into 6 suitable, informative parameters for the machine learning algorithm: the power ratio between the peak associated with the rotation period and the peak associated with the second harmonic, the power ratio between the peak associated with the second harmonic and the peak associated with the rotation period, the ratio between the real part of the amplitude of the peak associated with the rotation period and the square root of its power, the ratio between the real part of the amplitude of the peak associated with the second harmonic and the square root of the peak associated with the rotation period, the ratio between the real part of the amplitude of the peak associated with the second harmonic and the square root of its power, and the weighted phase difference between the peak associated with the second harmonic and the peak associated with the rotation period.


## Model

The model is composed of a random forest algorithm, which is a predictive modelling tool to extract information from existing data sets with the potential of uncovering new correlations in order to predict the value of one or several variables; to that end, random forest fits a number of decision tree regressors on various sub-samples of the data set and uses averaging to improve the predictive accuracy and to control over-fitting. In this project, the RandomForestRegressor random forest algorithm from the Scikit-Learn Python package was used with default parameters to perform a regression task and predict the inclination of the rotation axis of the star as well as the latitude dark spot based on other observables. Random forest modelling offers the possibility of assessing the importance of the different input parameters in predicting the inclination and spots latitudes, but does not allow us to derive a parametric relation between the predicted and the input parameters.


## Training the model

The data set is split in two parts: a traning set that is used to build the random forest algorithm and contains 80% of the total data set selected randomly (i.e. parameters from 1156 periodograms), and a test set consisting of remaining 20% of the total data set (i.e. parameters from 290 periodograms) that is used to assess the performance of the algorithm by comparing the predicted values with the actual input values.


### Results

The following results are printed in the terminal: the root mean squared error for the inclination and for the latitude, for both the train and test sets, the impurity-based feature importance (i.e. the relative importance of each feature in computing the predictions), and the median score of the estimator by cross-validation for both the inclination and the latitude (obtained by randomly splitting the training set into 10 non-overlapping subsets, i.e folds, then training and evaluating the model 10 times while picking a different fold for evaluation every time and using the other 9 folds for training). Six plots are also saved. The plot Correlation_between_input_parameters.pdf shows the range of latitudes for the spot as a function of the range of stellar inclinations in the entire dataset where the color code represents the value of the ratio between the power ratio between the peak associated with the rotation period and the peak associated with the second harmonic, which is one the parameters used as an input to train the random forest algorithm; this plot highlights the existing correlation between this input parameters and the parameters we want to infer (the latitudes of the spot and the inclination of the rotation axis of the star). The plot Train_test_sets_random_forest.pdf shows a similar plot, where this time the training set (i.e. parameters from 1156 periodograms) is represented in blue and the test set (i.e. parameters from 290 periodograms) is represented in red. The plot Random_forest_train_set_inclination_difference.pdf shows the difference between the predicted and input inclination values for the training set; the horizontal black line represents a difference of 0 degrees while the horizontal black dashed lines represent a difference of +- 10 degrees; some results are printed: the median value of the difference is 0 degrees, the median absolute value of the difference is 0.6 degrees, and for 98.3 % of the sample the maximum value of the difference is 10 degrees. The plot Random_forest_test_set_inclination_difference.pdf shows the same plot, this time for the test set; the median value of the difference is -0.2 degrees, the median absolute value of the difference is 1.4 degrees, and for 91 % of the sample the maximum value of the difference is 10 degrees. The plot Random_forest_train_set_latitude_difference.pdf shows a similar plot than Random_forest_train_set_inclination_difference.pdf, this time for the latitude of the spot; the median value of the difference is 0 degrees, the median absolute value of the difference is 0.2 degrees, and for 98.2 % of the sample the maximum value of the difference is 10 degrees. The plot Random_forest_test_set_latitude_difference.pdf shows a similar plot than Random_forest_test_set_inclination_difference.pdf, this time for the latitude of the spot; the median value of the difference is 0 degrees, the median absolute value of the difference is 0.6 degrees, and for 89.7 % of the sample the maximum value of the difference is 10 degrees. 


### Installation: with anaconda

git clone https://github.com/cgehan-astro/Postdoc_Germany_machine_learning.git
cd Postdoc_Germany_machine_learning
conda env create -f environment.yml
