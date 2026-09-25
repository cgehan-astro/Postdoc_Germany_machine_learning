### Organisation of the repository

There are 2 programs in this repository.


### 1. ./Random_forest/Random_forest.py: description in readme_Random_forest.txt

This program takes as an input the file input_random_forest_parameters.txt and produces as outputs the plots  Correlation_between_input_parameters.pdf, Train_test_sets_random_forest.pdf, Random_forest_train_set_inclination_difference.pdf, Random_forest_test_set_inclination_difference.pdf, Random_forest_train_set_latitude_difference.pdf and Random_forest_test_set_latitude_difference.pdf.


### 2. ./Convolutional_neural_network/CNN.py: description in readme_CNN.txt

This program takes as inputs the files ML_boundaries_parameter_space.txt, Period_FFT_spot.txt, Real_FFT_spot.txt and Imag_FFT_spot.txt; it produces as outputs the model fashion_model_CNN_inclination_latitude.h5py as well as the plots Training_validation_MSE.pdf, CNN_train_set_spot_real_imag_amplitude_inclination.pdf, CNN_test_set_spot_real_imag_amplitude_inclination.pdf, CNN_valid_set_spot_real_imag_amplitude_inclination.pdf, CNN_train_set_spot_real_imag_amplitude_latitude.pdf, CNN_test_set_spot_real_imag_amplitude_latitude.pdf and CNN_valid_set_spot_real_imag_amplitude_latitude.pdf. The input files Period_FFT_spot.txt, Real_FFT_spot.txt and Imag_FFT_spot.txt are absent because they are too big to be uploaded on GitHub.


### Installation: with anaconda

git clone https://github.com/cgehan-astro/Postdoc_Germany_machine_learning.git
cd Postdoc_Germany_machine_learning
conda env create -f environment.yml
