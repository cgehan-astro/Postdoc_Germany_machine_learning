### Create the appropriate environment with anaconda: conda env create -f environment.yml
### python=3.9.25, ipython=8.15


### This program aims at predicting values with machine learning through a regression task using a 1D
### convolutional neural network

import numpy as np
import pylab as plt
import glob
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model                   # Sequential is a class allowing to create a linear stack of layers in our model; Model is a class allowing to define a more complex model architecture than the simple linear stack of layers provided by Sequential.
from keras import Input                                     # Input is a class that allowing to define the shape of the input data to our model.
from keras.layers import Dense, Dropout, Flatten            # Dense is a class representing a fully connected layer in our model; Dropout is a class applying dropout regularization to our model, which helps prevent overfitting;
							    # Flatten is a class flattening the output of a previous layer into a 1D array, which can then be passed to a fully connected layer.
from keras.layers import Conv1D, MaxPooling1D               # Conv1D is a class representing a 1D convolutional layer in our model ; MaxPooling1D is a class applying max pooling to the output of a previous layer, which helps reducing the spatial dimensions of the data.
from keras.layers import BatchNormalization                 # BatchNormalization is a class applying batch normalization to the output of a previous layer, which helps improve the stability and speed of training.
from keras.layers import LeakyReLU                          # LeakyReLU is a class applying the leaky rectified linear activation function to the output of a previous layer, which helps prevent the "dying ReLU" problem and can improve the performance of the model.
from tensorflow.keras.layers import Activation		    # Activation is a class applying an activation function to the output of a previous layer



### Defining required parameters

batch_size = 64         		# specifies the number of samples that will be propagated through the neural network at once during training; contributes massively to determining the learning parameters and affects the prediction accuracy
epochs = 20           			# number of times the entire dataset will be passed through the neural network during training
number_predicted_outputs = 2.   	# number of values to predict: inclination and latitude



### Preparing the dataset: meaningful parameters from the periodogram of simulated synthetic light curves, i.e. a variation of flux over time, caused by one active region co-rotating with the surface of stars under the form of one dark spot with a lifetime 
### of 10 times the stellar rotation period


## Retrieving the raw parameters

real_data_FFT_raw = np.loadtxt('./Real_FFT_spot.txt')
imag_FFT_raw = np.loadtxt('./Imag_FFT_spot.txt')[:,2:]		# real part of the amplitude of the Fourier transform
period_raw = np.loadtxt('./Period_FFT_spot.txt')[:,2:]		# period axis in the periodogram
real_FFT_raw = real_data_FFT_raw[:,2:]				# imaginary part of the amplitude of the Fourier transform
inclination_raw = real_data_FFT_raw[:,0]			# inclination values of the rotation axis of the star with respect to the line-of-sight
latitude_raw = real_data_FFT_raw[:,1]				# latitude values at which the dark spot is present


## Selecting only valid parameters: where there is exploitable signal in the periodogram

ML_boundaries_2_harmonics = np.loadtxt('./ML_boundaries_parameter_space.txt')		# file containing the inclinations and latitudes values corresponding to valid configurations giving an exploitable signal in the periodogram
L_min_boundaries_exploitable_signal = ML_boundaries_2_harmonics[:,0]
L_max_boundaries_exploitable_signal = ML_boundaries_2_harmonics[:,1]
i_boundaries_exploitable_signal = ML_boundaries_2_harmonics[:,2]
list_crit_2_harmonics = []
for i in range(np.size(inclination_raw, axis=0)):
    crit_i = np.where(i_boundaries_exploitable_signal == inclination_raw[i])[0]
    if latitude_raw[i] >= L_min_boundaries_exploitable_signal[crit_i] and latitude_raw[i] <= L_max_boundaries_exploitable_signal[crit_i]:
        list_crit_2_harmonics.append(i)
crit_2_harmonics = np.array(list_crit_2_harmonics)
real_FFT_raw = real_FFT_raw[crit_2_harmonics,:]
imag_FFT_raw = imag_FFT_raw[crit_2_harmonics,:]
period_raw = period_raw[crit_2_harmonics,:]
inclination_raw = inclination_raw[crit_2_harmonics]
latitude_raw = latitude_raw[crit_2_harmonics]


## Turning the raw parameters into suitable, informative parameters for the machine learning algorithm: normalizing the amplitudes of the Fourier transform

list_real_FFT = []
list_imag_FFT = []
list_abs_complex_FFT = []
list_period = []
list_inclination_latitude = []

list_different_inclinations = np.array(list(set(inclination_raw)))
for i in range(list_different_inclinations.size):
    crit = np.where(inclination_raw == list_different_inclinations[i])[0]
    real_FFT_crit = real_FFT_raw[crit]
    imag_FFT_crit = imag_FFT_raw[crit]
    period_crit = period_raw[crit]
    latitude_crit = latitude_raw[crit]
    Prot = 25.
    for j in range(crit.size):
        crit_A1 = np.where(abs(period_crit[j] - Prot) == min(abs(period_crit[j] - Prot)))[0]    	# finding the peak associated with the rotation period of the spot, i.e. the fundamental
        A1 = np.abs(complex(real_FFT_crit[j][crit_A1], imag_FFT_crit[j][crit_A1]))     			# absolute value of the amplitude of the fundamental
        norm_real_FFT_crit = real_FFT_crit[j]/A1        						# normalizing the real amplitude to the fundamental
        norm_real_FFT_crit = np.nan_to_num(norm_real_FFT_crit)      					# converting nan into zeros
        norm_imag_FFT_crit = imag_FFT_crit[j]/A1        						# normalizing the imaginary amplitude to the fundamental
        norm_imag_FFT_crit = np.nan_to_num(norm_imag_FFT_crit)      					# converting nan into zeros
        A = np.abs(real_FFT_crit[j] + 1j * imag_FFT_crit[j])      					# absolute value of the total, complex amplitude
        norm_A_crit = A/A1        									# normalizing the total, complex amplitude to the fundamental
        norm_A_crit = np.nan_to_num(norm_A_crit)      							# converting nan into zeros
        mat_inclination_latitude = np.zeros((1, 2))
        mat_inclination_latitude[0, 0] = list_different_inclinations[i]					# first label, i.e. parameter we want to predict with a machine learning algorithm: inclination of the stellar rotation axis 
        mat_inclination_latitude[0, 1] = latitude_crit[j]						# second label, i.e. parameter we want to predict with a machine learning algorithm: latitude of the dark spot
        list_inclination_latitude.append(mat_inclination_latitude[0, :])
        list_real_FFT.append(norm_real_FFT_crit)
        list_imag_FFT.append(norm_imag_FFT_crit)
        list_abs_complex_FFT.append(norm_A_crit)
        list_period.append(period_crit[j])
real_FFT = np.array(list_real_FFT))									# array containing the first predictor: the real amplitude
imag_FFT = np.array(list_imag_FFT))									# array containing the second predictor: the imaginary amplitude
abs_complex_FFT = np.array(list_abs_complex_FFT)							# array containing the second predictor: the total, complex amplitude
period = np.array(list_period)										# array containing the period axis
inclination_latitude = np.array(list_inclination_latitude)						# array containing the labels, i.e. inclination and latitude


## Building an array contaning all the input parameters for the machine learning algorithm

number_samples = np.size(real_FFT,axis=0)
number_timesteps = np.size(real_FFT,axis=1)
number_features = 3
combined_FFT_signal = np.zeros(((number_samples, number_timesteps, number_features))) 
for i in range(np.size(real_FFT,axis=0)):
    combined_FFT_signal[i, 0:, 0] = real_FFT[i,0:]
    combined_FFT_signal[i, 0:, 1] = imag_FFT[i,0:]
    combined_FFT_signal[i, 0:, 2] = abs_complex_FFT[i,0:]



### Defining the training and test sets: respectively 80% and 20% of the whole data set, selected randomly

train_set, test_set, train_label, test_label = train_test_split(combined_FFT_signal, inclination_latitude, test_size=0.2, random_state=42)		# we fix here the random state for reproducibility of the results at each run
train_set_core, valid_set, train_label_core, valid_label = train_test_split(train_set, train_label, test_size=0.2, random_state=13)			# core train set: 80% of the train set; validation set: 20% of the train set


### Configuring, training and saving the convolutional neural network model

inputShape = (number_timesteps, number_features)
fashion_model = Sequential()    				# stacking up layers one by one


## First convolutional layer

fashion_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=inputShape, padding='same'))				# 32 filters, kernel of size 3


## Second convolutional layer

fashion_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))							# 64 filters, kernel of size 3
fashion_model.add(Activation("relu"))
fashion_model.add(BatchNormalization(axis=-1))
fashion_model.add(Dropout(0.25))    													# adding a dropout layer with a rate of 0.25 to prevent overfitting by randomly dropping out some of the neurons during training
fashion_model.add(MaxPooling1D(2, padding='same'))  											# max-pooling layers of size 2
fashion_model.add(Activation("relu"))
fashion_model.add(BatchNormalization(axis=-1))


## Third convolutional layer

fashion_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))						# 128 filters, kernel of size 3
fashion_model.add(Activation("relu"))
fashion_model.add(BatchNormalization(axis=-1))
fashion_model.add(Dropout(0.25))    													# adding a dropout layer with a rate of 0.25 to prevent overfitting by randomly dropping out some of the neurons during training
fashion_model.add(MaxPooling1D(2, padding='same'))  											# max-pooling layers of size 2
fashion_model.add(Activation("relu"))
fashion_model.add(BatchNormalization(axis=-1))


## Adding a fully connected layer

fashion_model.add(Flatten()) 														# flattening into a 1D array
fashion_model.add(Dense(128, activation='relu'))											# 128 neurons
fashion_model.add(Activation("relu"))
fashion_model.add(BatchNormalization(axis=-1))
fashion_model.add(Dropout(0.5))   													# adding a dropout layer with a rate of 0.5 to prevent overfitting by randomly dropping out some of the neurons during training


## Adding an output layer to the model

fashion_model.add(Dense(number_predicted_outputs, activation='relu'))									# the number of neurons is equal to the number of predicted outputs to allow the model to output a value for each output
fashion_model.add(Activation("relu"))


## Building the model

fashion_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error']) 					# loss is the objective function that the model will try to minimize during training; optimizer is the algorithm used to update the weights
																	# of the model during training; metrics is a list of metrics used to evaluate the performance of the model during training and testing

## Training and saving the model

fashion_train = fashion_model.fit(train_set_core, train_label_core, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_set, valid_label))		# verbose controls the amount of information printed during training; validation_data specifies the
																					# data on which to evaluate the loss and any model metrics at the end of each epoch
fashion_model.save("fashion_model_CNN_inclination_latitude.h5py")													# saving the model

train_MSE_epochs = fashion_train.history['mean_squared_error']
val_MSE_epochs = fashion_train.history['val_mean_squared_error']
train_loss_epochs = fashion_train.history['loss']
val_loss_epochs = fashion_train.history['val_loss']
epochs = range(len(train_MSE_epochs))



### Plotting the mean square error versus the epoch for the train and validation sets

plt.figure()
plt.scatter(epochs, train_MSE_epochs, s=10, c='b')
plt.plot(epochs, train_MSE_epochs, 'b', label='Training MSE')
plt.scatter(epochs, val_MSE_epochs, s=10, c='r')
plt.plot(epochs, val_MSE_epochs, 'r', label='Validation MSE')
plt.title('Training and validation mean squared error')
plt.legend()
plt.savefig('Training_validation_MSE.pdf', format='pdf')
plt.close()


### Predicting the desired parameters through machine learning using a convolutional neural network

fashion_model = keras.models.load_model("fashion_model_dropout.h5py")		# loading the saved model
fashion_model.summary()


## Evaluating the model

train_eval = fashion_model.evaluate(train_set, train_label, verbose=0)		# evaluating the model for the train set
valid_eval = fashion_model.evaluate(valid_set, valid_label, verbose=0)		# evaluating the model for the validation set
test_eval = fashion_model.evaluate(test_set, test_label, verbose=0)		# evaluating the model for the test set
train_loss = train_eval[0]		# loss of the train set
train_MSE = train_eval[1]		# mean squared error of the train set
valid_loss = valid_eval[0]		# loss of the validation set
valid_MSE = valid_eval[1]		# mean squared error of the validation set
test_loss = test_eval[0]		# loss of the test set
test_MSE = test_eval[1]			# mean squared error of the test set


## Predicting the output parameters

predicted_train_labels = fashion_model.predict(train_set, verbose=0)			# train set
predicted_test_labels = fashion_model.predict(test_set, verbose=0)			# test set

predicted_train_labels_core = fashion_model.predict(train_set_core, verbose=0)		# core train set
predicted_valid_labels = fashion_model.predict(valid_set, verbose=0)			# validation set



### Plotting the predicted versus input inclination values for the train set

range_inclinations_plot = np.arange(-5, 95)
plt.figure()
plt.scatter(train_label[:,0], predicted_train_labels[:,0], s=10)
plt.plot(range_inclinations_plot, range_inclinations_plot, color='k')
plt.xlim(0, 91)
plt.xlabel('Input inclination value')
plt.ylabel('Predicted inclination value')
plt.savefig('./CNN_train_set_spot_real_imag_amplitude_inclination.pdf')
plt.close()



### Plotting the predicted versus input latitude values for the train set

range_latitudes_plot = np.arange(-90, 90)
plt.figure()
plt.scatter(train_label[:,1], predicted_train_labels[:,1], s=10)
plt.plot(range_latitudes_plot, range_latitudes_plot, color='k')
plt.xlim(-90, 90)
plt.xlabel('Input latitude value')
plt.ylabel('Predicted latitude value')
plt.savefig('./CNN_train_set_spot_real_imag_amplitude_latitude.pdf')
plt.close()



### Plotting the predicted versus input inclination values for the test set

plt.figure()
plt.scatter(test_label[:,0], predicted_test_labels[:,0], s=10)
plt.plot(range_inclinations_plot, range_inclinations_plot, color='k')
plt.xlim(0, 91)
plt.xlabel('Input inclination value')
plt.ylabel('Predicted inclination value')
plt.savefig('./CNN_test_set_spot_real_imag_amplitude_inclination.pdf')
plt.close()



### Plotting the predicted versus input latitude values for the test set

plt.figure()
plt.scatter(test_label[:,1], predicted_test_labels[:,1], s=10)
plt.plot(range_latitudes_plot, range_latitudes_plot, color='k')
plt.xlim(-90, 90)
plt.xlabel('Input latitude value')
plt.ylabel('Predicted latitude value')
plt.savefig('./CNN_test_set_spot_real_imag_amplitude_latitude.pdf')
plt.close()



### Plotting the predicted versus input inclination values for the core train set

plt.figure()
plt.scatter(train_label_core[:,0], predicted_train_labels_core[:,0], s=10)
plt.plot(range_inclinations_plot, range_inclinations_plot, color='k')
plt.xlim(0, 91)
plt.xlabel('Input inclination value')
plt.ylabel('Predicted inclination value')
plt.savefig('./CNN_train_core_set_spot_real_imag_amplitude_inclination.pdf')
plt.close()



### Plotting the predicted versus input latitude values for the core train set

plt.figure()
plt.scatter(train_label_core[:,1], predicted_train_labels_core[:,1], s=10)
plt.plot(range_latitudes_plot, range_latitudes_plot, color='k')
plt.xlim(-90, 90)
plt.xlabel('Input latitude value')
plt.ylabel('Predicted latitude value')
plt.savefig('./CNN_train_core_set_spot_real_imag_amplitude_latitude.pdf')
plt.close()



### Plotting the predicted versus input inclination values for the validation set

plt.figure()
plt.scatter(valid_label[:,0], predicted_valid_labels[:,0], s=10)
plt.plot(range_inclinations_plot, range_inclinations_plot, color='k')
plt.xlim(0, 91)
plt.xlabel('Input inclination value')
plt.ylabel('Predicted inclination value')
plt.savefig('./CNN_valid_set_spot_real_imag_amplitude_inclination.pdf')
plt.close()



### Plotting the predicted versus input latitude values for the validation set

plt.figure()
plt.scatter(valid_label[:,1], predicted_valid_labels[:,1], s=10)
plt.plot(range_latitudes_plot, range_latitudes_plot, color='k')
plt.xlim(-90, 90)
plt.xlabel('Input latitude value')
plt.ylabel('Predicted latitude value')
plt.savefig('./CNN_valid_set_spot_real_imag_amplitude_latitude.pdf')
plt.close()


