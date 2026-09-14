### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15


### This program aims at predicting values with machine learning through a regression task using the random forest algorithm

import numpy as np
import pylab as plt
import matplotlib as mpl
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score



#### Preparing the dataset: meaningful parameters computed beforehand from periodograms of simulated synthetic light curves, i.e. a variation of flux over time, caused by one active region
### co-rotating with the surface of stars under the form of one dark spot with a lifetime of 10 times the stellar rotation period


## Retrieving the raw parameters

FFT_parameters_spot_all = np.loadtxt('./input_random_forest_parameters.txt')
FFT_parameters_spot_all = FFT_parameters_spot_all[np.where(FFT_parameters_spot_all[:,2] != -999.)[0]]		# selecting only valid parameters
FFT_parameters_spot_all = FFT_parameters_spot_all[np.where(FFT_parameters_spot_all[:,3] != -999.)[0]]
FFT_parameters_spot_all = FFT_parameters_spot_all[np.where(FFT_parameters_spot_all[:,5] != -999.)[0]]
FFT_parameters_spot_all = FFT_parameters_spot_all[np.where(FFT_parameters_spot_all[:,6] != -999.)[0]]
FFT_parameters_spot_all = FFT_parameters_spot_all[np.where(FFT_parameters_spot_all[:,8] != -999.)[0]]
FFT_parameters_spot_all = FFT_parameters_spot_all[np.where(FFT_parameters_spot_all[:,9] != -999.)[0]]
inclinations_spot_all = FFT_parameters_spot_all[:,0]	# inclination values of the rotation axis of the star with respect to the line-of-sight
latitudes_spot_all = FFT_parameters_spot_all[:,1]	# latitude values at which the dark spot is present
real_a1_spot = FFT_parameters_spot_all[:,2]		# real part of the amplitude of the peak associated with the rotation period of the spot in the Fourier transform of the light curve
real_a2_spot = FFT_parameters_spot_all[:,3]		# real part of the amplitude of the peak associated with the second harmonic of the rotation period of the spot in the Fourier transform
							# of the light curve
imag_a1_spot = FFT_parameters_spot_all[:,5]		# imaginary part of the amplitude of the peak associated with the rotation period of the spot in the Fourier transform of the light curve
imag_a2_spot = FFT_parameters_spot_all[:,6]		# imaginary part of the amplitude of the peak associated with the second harmonic of the rotation period of the spot in the Fourier transform
							# of the light curve
phase_1_spot = FFT_parameters_spot_all[:,8]		# phase of the peak associated with the rotation period of the spot in the Fourier transform of the light curve
phase_2_spot = FFT_parameters_spot_all[:,9]		# phase of the peak associated with the second harmonic of the rotation period of the spot in the Fourier transform of the light curve
P_1_spot = FFT_parameters_spot_all[:,14]		# power of the peak associated with the rotation period of the spot in the Fourier transform of the light curve
P_2_spot = FFT_parameters_spot_all[:,15]		# power of the peak associated with the second harmonic of the rotation period of the spot in the Fourier transform of the light curve


## Turning the raw parameters into suitable, informative parameters for the machine learning algorithm

P_12_spot = P_1_spot / P_2_spot		# power ratios between the peak associated with the rotation period and the peak associated with the second harmonic
P_21_spot = P_2_spot / P_1_spot		# power ratios between the peak associated with the second harmonic and the peak associated with the rotation period
real_A_11_spot = real_a1_spot / np.sqrt(P_1_spot)	# ratios between the real part of the amplitude of the peak associated with the rotation period and the square root of its power
real_A_21_spot = real_a2_spot / np.sqrt(P_1_spot)	# ratios between the real part of the amplitude of the peak associated with the second harmonic and the square root of the peak
							# associated with the rotation period
real_A_22_spot = real_a2_spot / np.sqrt(P_2_spot)	# ratios between the real part of the amplitude of the peak associated with the the second harmonic and the square root of its power
Delta_phi_21_spot = phase_2_spot - 2. * phase_1_spot	# weighted phase difference between the peak associated with the second harmonic and the peak associated with the rotation period


## Building an array contaning all the input parameters for the machine learning algorithm

number_features_ML_input = 7
ML_input = np.zeros((np.size(FFT_parameters_spot_all, axis=0), number_features_ML_input))
ML_input[:,0] = FFT_parameters_spot_all[:,0]		# first label, i.e. parameter we want to predict with a machine learning algorithm: inclination of the stellar rotation axis
ML_input[:,1] = FFT_parameters_spot_all[:,1] 		# second label, i.e. parameter we want to predict with a machine learning algorithm: latitude of the dark spot
ML_input[:,2] = P_21_spot[:]				# pre-processed informative parameters
ML_input[:,3] = real_A_11_spot[:]
ML_input[:,4] = real_A_21_spot[:]
ML_input[:,5] = real_A_22_spot[:]
ML_input[:,6] = Delta_phi_21_spot[:]



### Defining the training and test sets: respectively 80% and 20% of the whole data set, selected randomly

train_set, test_set = train_test_split(ML_input, test_size=0.2, random_state=42)	# we fix here the random state for reproducibility of the results at each run


## Plotting the training and test sets

plt.figure()
plt.scatter(train_set[:,0], train_set[:,1],s=5, c='b', label='Train set: ' + str(train_set[:,0].size) + ' periodograms')
plt.scatter(test_set[:,0], test_set[:,1], s=20, c='r', marker='+', label='Test set: ' + str(test_set[:,0].size) + ' periodograms')
plt.xticks(np.linspace(0,90,10), ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
plt.xlabel('Inclination i (' + r'$^\circ$' + ')', fontsize='x-large')
plt.ylabel('Latitude L (' + r'$^\circ$' + ')', fontsize='x-large')
plt.title('Input sample: ' + str(ML_input[:,0].size) + ' periodograms', fontsize='x-large')
plt.legend()
plt.savefig('Train_test_sets_random_forest.pdf')
plt.close()


## Plotting the labels, i.e. the latitude versus the inclination, color-coded with one of the preictors, i.e. the power ratio between the peak associated with the second harmonic and the peak associated with the rotation period, for the training and test sets combined

cm = mpl.cm.jet
fig, ax = plt.subplots()
cax = fig.add_axes([0.20, 0.80, 0.3, 0.04])
sc = ax.scatter(inclinations_spot_all, latitudes_spot_all, c=P_12_spot, vmin=min(P_12_spot), vmax=15., cmap=cm, s=50)
fig.colorbar(sc, cax=cax, ticks=[0.5, 4, 7, 11, 15], orientation='horizontal', label= 'Power ratio between the fundamental\nand the second harmonic ' + r'$P_{12}$')
ax.set_xlabel('Inclination i (' + r'$^\circ$' + ')', fontsize='x-large')
ax.set_ylabel('Latitude L (' + r'$^\circ$' + ')', fontsize='x-large')
ax.set_xlim(-1,92)
ax.set_ylim(-90,90)
ax.set_title('Correlation between the labels and one of the predictors', fontsize='x-large')
plt.savefig('./Correlation_between_input_parameters.pdf')
plt.close()


## Separating the predictors and the labels

predictor_features_ini = np.delete(train_set, 0, axis=1)			# train set
predictor_features = np.delete(predictor_features_ini, 0, axis=1)
label_features = np.zeros((np.size(predictor_features, axis=0),2))
label_features[:,0] = train_set[:,0]
label_features[:,1] = train_set[:,1]
predictor_features_test_ini = np.delete(test_set, 0, axis=1)			# test set
predictor_features_test = np.delete(predictor_features_test_ini, 0, axis=1)
label_features_test = np.zeros((np.size(predictor_features_test, axis=0),2))
label_features_test[:,0] = test_set[:,0]
label_features_test[:,1] = test_set[:,1]


### Predicting the desired parameters through machine learning using a random forest algorithm

random_forest_reg = RandomForestRegressor(random_state=42)					# building the random forest regressor
												# we fix here the random state for reproducibility of the results at each run
model_inclination_latitude = random_forest_reg.fit(predictor_features, label_features)		# building a forest of trees from the training set
inclination_latitude_predictions_random_forest = model_inclination_latitude.predict(predictor_features)		# predicting the values of the desired parameters for the train set: inclination and latitude
test_inclination_latitude_predictions = model_inclination_latitude.predict(predictor_features_test)		# predicting the values of the desired parameters for the test set: inclination and latitude


### Computing the root mean squared error between the predictions and the input values, i.e. the labels

RMSE_train_inclination = mean_squared_error(label_features[:,0], inclination_latitude_predictions_random_forest[:,0])		# root mean squared error for the inclination for the train set; similar to the reduced chi-square
RMSE_train_latitude = mean_squared_error(label_features[:,1], inclination_latitude_predictions_random_forest[:,1])		# root mean squared error for the latitude for the train set; similar to the reduced chi-square
RMSE_test_inclination = mean_squared_error(label_features_test[:,0], test_inclination_latitude_predictions[:,0])		# root mean squared error for the inclination for the test set; similar to the reduced chi-square
RMSE_test_latitude = mean_squared_error(label_features_test[:,1], test_inclination_latitude_predictions[:,1])			# root mean squared error for the latitude for the test set; similar to the reduced chi-square
print('\nRoot mean squared error for the inclination: ')
print(str(np.round(RMSE_train_inclination,1)) + ' degs for the train set')
print(str(np.round(RMSE_test_inclination,1)) + ' degs for the test set\n')
print('Root mean squared error for the latitude: ')
print(str(np.round(RMSE_train_latitude,1)) + ' degs for the train set ')
print(str(np.round(RMSE_test_latitude,1)) + ' degs for the test set\n')



### Computing the relative importance of each feature in computing the predictions: impurity-based feature importances

feature_importances_inclination_latitude = model_inclination_latitude.feature_importances_ * 100   		# in %
print('Impurity-based feature importances:')
print('Power ratio P_12: ' + str(np.round(feature_importances_inclination_latitude[0],1)) + ' %')
print('Power ratio P_21: ' + str(np.round(feature_importances_inclination_latitude[1],1)) + ' %')
print('Real amplitude ratio A_11: ' + str(np.round(feature_importances_inclination_latitude[2],1)) + ' %')
print('Real amplitude ratio A_21: ' + str(np.round(feature_importances_inclination_latitude[3],1)) + ' %')
print('Real amplitude ratio A_22: ' + str(np.round(feature_importances_inclination_latitude[4],1)) + ' %\n')



### Evaluating scores of the estimator by cross-validation: randomly splitting the training set into 10 non-overlapping subsets, i.e folds, then training and evaluating the model 10 times
### while picking a different fold for evaluation every time and using the other 9 folds for training

cross_val_rmses_inclination = -cross_val_score(model_inclination_latitude, predictor_features, label_features[:,0], scoring="neg_root_mean_squared_error", cv=10)   	# inclination
print('Median score of the estimator by cross-validation for the inclination over 10 evaluations of the model: ' + str(np.round(np.median(cross_val_rmses_inclination),1)) + ' degs' + '\n')
mean_cross_val_rmses_inclination = np.mean(cross_val_rmses_inclination)
cross_val_rmses_latitude = -cross_val_score(model_inclination_latitude, predictor_features, label_features[:,1], scoring="neg_root_mean_squared_error", cv=10)   	# latitude
mean_cross_val_rmses_latitude = np.mean(cross_val_rmses_latitude)
print('Median score of the estimator by cross-validation for the latitude over 10 evaluations of the model: ' + str(np.round(np.median(cross_val_rmses_latitude),1)) + ' degs')



### Plotting the difference between the predicted and input values of the desired parameters for the train and test sets: inclination and latitude

## Plotting the difference between the predicted and input inclination values for the train set

crit_inclination_difference_below_10 = np.where(inclination_latitude_predictions_random_forest[:,0] - label_features[:,0] <= 10)[0]
plt.figure()
plt.scatter(label_features[:,0], inclination_latitude_predictions_random_forest[:,0] - label_features[:,0], s=10, label='Median difference: ' + str('%.1f' % np.median(inclination_latitude_predictions_random_forest[:,0] - label_features[:,0])) + str(r'$^\circ$') + '\n' + 'Median absolute difference: ' + str('%.1f' % np.median(np.abs(inclination_latitude_predictions_random_forest[:,0] - label_features[:,0]))) + str(r'$^\circ$') + '\n' +  'Maximum difference of 10 ' + str(r'$^\circ$') + ': occurence rate of ' + str('%.1f' % (crit_inclination_difference_below_10.size/label_features[:,0].size*100)) + ' %')
plt.hlines(0,0,91, color='k')
plt.hlines(-10,0,91, color='k', linestyle='--')
plt.hlines(10,0,91, color='k', linestyle='--')
plt.xlim(0, 91)
plt.legend()
plt.xlabel('Input inclination value (°)')
plt.ylabel('Difference between the predicted\nand input inclination value (°)')
plt.savefig('./Random_forest_train_set_inclination_difference.pdf')
plt.close()


## Plotting the difference between the predicted and input latitude values for the train set

crit_latitude_difference_below_10 = np.where(inclination_latitude_predictions_random_forest[:,1] - label_features[:,1] <= 10)[0]
plt.figure()
plt.scatter(label_features[:,1], inclination_latitude_predictions_random_forest[:,1] - label_features[:,1], s=10, label='Median difference: ' + str('%.1f' % np.median(inclination_latitude_predictions_random_forest[:,1] - label_features[:,1])) + str(r'$^\circ$') + '\n' + 'Median absolute difference: ' + str('%.1f' % np.median(np.abs(inclination_latitude_predictions_random_forest[:,1] - label_features[:,1]))) + str(r'$^\circ$') + '\n' +  'Maximum difference of 10 ' + str(r'$^\circ$') + ': occurence rate of ' + str('%.1f' % (crit_latitude_difference_below_10.size/label_features[:,1].size*100)) + ' %')
plt.hlines(0,-90,90, color='k')
plt.hlines(-10,-90,90, color='k', linestyle='--')
plt.hlines(10,-90,90, color='k', linestyle='--')
plt.xlim(-90, 90)
plt.legend()
plt.xlabel('Input latitude value (°)')
plt.ylabel('Difference between the predicted\nand input latitude value (°)')
plt.savefig('./Random_forest_train_set_latitude_difference.pdf')
plt.close()


## Plotting the difference between the predicted and input inclination values for the test set

crit_inclination_difference_below_10_test = np.where(np.abs(test_inclination_latitude_predictions[:,0] - label_features_test[:,0]) <= 10)[0]
plt.figure()
plt.scatter(label_features_test[:,0], test_inclination_latitude_predictions[:,0] - label_features_test[:,0], s=10, label='Median difference: ' + str('%.1f' % np.median(test_inclination_latitude_predictions[:,0] - label_features_test[:,0])) + str(r'$^\circ$') + '\n' + 'Median absolute difference: ' + str('%.1f' % np.median(np.abs(test_inclination_latitude_predictions[:,0] - label_features_test[:,0]))) + str(r'$^\circ$') + '\n' +  'Maximum difference of 10 ' + str(r'$^\circ$') + ': occurence rate of ' + str('%.1f' % (crit_inclination_difference_below_10_test.size/label_features_test[:,0].size*100)) + ' %')
plt.hlines(0,0,91, color='k')
plt.hlines(-10,0,91, color='k', linestyle='--')
plt.hlines(10,0,91, color='k', linestyle='--')
plt.xlim(0, 91)
plt.legend()
plt.xlabel('Input inclination value (°)')
plt.ylabel('Difference between the predicted\nand input inclination value (°)')
plt.savefig('./Random_forest_test_set_inclination_difference.pdf')
plt.close()


## Plotting the difference between the predicted and input latitude values for the test set

crit_latitude_difference_below_10_test = np.where(np.abs(test_inclination_latitude_predictions[:,1] - label_features_test[:,1]) <= 10)[0]
plt.figure()
plt.scatter(label_features_test[:,1], test_inclination_latitude_predictions[:,1] - label_features_test[:,1], s=10, label='Median difference: ' + str('%.1f' % np.median(test_inclination_latitude_predictions[:,1] - label_features_test[:,1])) + str(r'$^\circ$') + '\n' + 'Median absolute difference: ' + str('%.1f' % np.median(np.abs(test_inclination_latitude_predictions[:,1] - label_features_test[:,1]))) + str(r'$^\circ$') + '\n' +  'Maximum difference of 10 ' + str(r'$^\circ$') + ': occurence rate of ' + str('%.1f' % (crit_latitude_difference_below_10_test.size/label_features_test[:,1].size*100)) + ' %')
plt.hlines(0,-90,90, color='k')
plt.hlines(-10,-90,90, color='k', linestyle='--')
plt.hlines(10,-90,90, color='k', linestyle='--')
plt.xlim(-90, 90)
plt.legend()
plt.xlabel('Input latitude value (°)')
plt.ylabel('Difference between the predicted\nand input latitude value (°)')
plt.savefig('./Random_forest_test_set_latitude_difference.pdf')
plt.close()


