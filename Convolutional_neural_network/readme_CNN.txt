### Project: CNN.py

I completed this project during my postdoc in Göttingen, Germany (2021 - 2024).


### Project overview

This project aims at predicting the values of 2 physical parameters with machine learning through a regression task using a convolutional neural network, from simulated synthetic light curves, i.e. flux time-series, where the flux comes from stars, which describe the effects of magnetic field lines at the stellar surface; those light curves are generated in the project Postdoc_Germany_light_curves by the program Lightcurve_with_activity_one_spot.py (description in readme_Lightcurve_with_activity_one_spot.txt). The 2 physical parameters to predict are the inclination of the rotation axis of the star with respect to the line-of-sight, and the latitude of the active regions on the surface of stars.


### Dataset

The dataset is composed of 3 parameters extracted from 1446 synthetic light curves that include only one dark spot with a fixed lifetime, computed for 90 different inclination values for the rotation axis (from 1 degree to 90 degrees with a 1 degree step) and for 35 different latitudes for the dark spot (from -85 degrees in the Southern hemisphere to 85 degrees in the Northern hemisphere with a 5 degrees step); a periodic signal is detectable in 1446 light curves out of the total of 3150 light curves, which are those used to train the convolutional neural network. The following 3 parameters extracted from each synthetic light curve are used to train the convolutional neural network: the full real and imaginary parts of the amplitude of the Fourier transform of the light curve, stored in the files Real_FFT_spot.txt and Imag_FFT_spot.txt, respectively; as well as the absolute value of the full, complex amplitude of the Fourier transform of the light curve, computed as the absolute value of the sum of the real and imaginary parts of the amplitude of the Fourier transform of the light curve. The associated period axis in the periodogram comes from the file Period_FFT_spot.txt. The files Real_FFT_spot.txt, Imag_FFT_spot.txt and Period_FFT_spot.txt are absent because they are too big to be uploaded on GitHub. The file ML_boundaries_parameter_space is used to select the configurations of stellar inclinations and latitudes for the spot that result in a detectable periodic signal in the light curves.


### Methodology

## Data preprocessing

Data preprocessing steps involve turning the 3 raw parameters into 6 suitable, informative parameters for the machine learning algorithm: the full real and imaginary parts of the amplitude of the Fourier transform of the light curve normalized to the amplitude of the peak associated to the rotation period of the spot, and the absolute value of the full, complex amplitude of the Fourier transform of the light curve normalized to the amplitude of the peak associated to the rotation period of the spot.


## Model

The model consists in a 1D convolutional neural network, which is composed of an input layer containing the data used to train the convolutional neural network, some convolutional layers, a fully connected layer that converts the filtered input into a flat (1D) vector, and an output layer that produces the final regression predictions. The convolutional layers apply convolution learnable filters to the data in order to extract the relevant features; they produce feature maps that are then stacked together to form a new data set that captures the essential features of the original data set. The convolutional neural network learn the weights of the filters to find patterns in the images by minimizing a loss function during training; here the loss function is the mean squared error, and minimization is performed by the Adam optimizer algorithm.

In more details, 3 convolutional layers are used because layers deeper in the network, i.e. closer to the output predictions, learn more filters; therefore, the number of filters is increased for each layer (32, 64 and 128, respectively). 6 steps are performed after the second convolutional layer (composed of 64 neurons). The first step consists in applying an activation function in order to use stochastic gradient descent with back-propagation of errors to train deep neural networks by updating the filters weights; a rectified linear activation function is used, i.e. that outputs the input when positive, otherwise outputs 0, allowing to overcome the vanishing gradient problem encountered with traditional non-linear functions (such as sigmoid and hyperbolic tangent, which saturate and are only really sensitive to changes around their mid-point, resulting in deeper layers failing to receive useful gradient information to compute errors). The second step consists in applying batch normalization to the output of the convolution layer, which fixes the means and variances the inputs by re-centering and re-scaling them, in order to improve the stability and speed of the training by mitigating internal covariate shift. The third step consists in applying a dropout layer by randomly dropping out 25 % of the neurons during training, in order to prevent over-fitting by training a large number of neural networks with different architectures in parallel. The fourth step consists in applying a pooling layer that enables to reduce the spatial dimensions of the input in order to decrease the number of parameters, the computation time, and to avoid over-fitting. The fifth step consists in applying again a rectified linear activation function. The sixth step consists in applying again a batch normalization to the output. After the third convolutional layer (composed of 128 neurons), the same 6 steps are performed again. Then, a fully connected layer composed of 128 neurons is added, followed by the application of a rectified linear activation function, batch normalization, and a dropout layer with a dropping rate of 50 %. The output layer is finally added, which is composed of 2 neurons as there are 2 predicted outputs, i.e. the inclination of the stellar rotation axis and the latitude of the spot; a rectified linear activation function is finally applied.


## Training the model

The data set is split in two parts: a traning set that is used to build the convolutional neural network and contains 80% of the total data set selected randomly (i.e. parameters from 1156 light curves), and a test set consisting of remaining 20% of the total data set (i.e. parameters from 290 light curves) that is used to assess the performance of the algorithm by comparing the predicted values with the actual input values. 20% of the train set (i.e. parameters from 237 light curves) is used as a validation set, which serves to evaluate the loss and any model metrics at the end of each epoch, i.e. each time the entire dataset is passed through the neural network during training. The convolutional neural network is trained over 20 epochs with 64 samples propagated through the neural network at once during training; this parameter, i.e. the batch size, contributes massively to determining the learning parameters and affects the prediction accuracy.


### Results

The trained model is saved as fashion_model_CNN_inclination_latitude.h5py. Nine plots are also saved. The plots CNN_train_core_set_spot_real_imag_amplitude_inclination.pdf, CNN_valid_set_spot_real_imag_amplitude_inclination.pdf and CNN_test_set_spot_real_imag_amplitude_inclination.pdf show the predicted versus input values of the inclination of the stellar rotation axis for the train, validation and test sets, respectively; the black line represents a 1:1 relation. The plots CNN_train_core_set_spot_real_imag_amplitude_latitude.pdf, CNN_valid_set_spot_real_imag_amplitude_latitude.pdf and CNN_test_set_spot_real_imag_amplitude_latitude.pdf show similar plots, this time for the latitude of the spot. The plot Training_validation_MSE.pdf shows the mean square error versus the epoch for the training set in blue and for the validation set in red; the continuous decrease and the convergence of the mean square error for both data sets indicates that the data is not over-fitted, i.e. the trained model does not correspond too closely to the particular set of data used here and should therefore predict future observations reliably.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/Postdoc_Germany_machine_learning.git
cd Postdoc_Germany_machine_learning
conda env create -f environment.yml
