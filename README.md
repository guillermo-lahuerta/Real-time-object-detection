# Object detection

## About this notebook

This notebook contains a simple implementation of a Deep Learning model for **object detection**.

The model presented in this repository has been totally implemented using open source tools (i.e., Python and TensorFlow).

## LeNet-5

Convolutional Neural Networks is the standard architecture for solving tasks associated with images (e.g., image classification). Some of the well-known deep learning architectures for CNN are LeNet-5 (7 layers), GoogLeNet (22 layers), AlexNet (8 layers), VGG (16–19 layers), and ResNet (152 layers).

For this project, we use LeNet-5, which has been successfully used on the MNIST dataset to identify handwritten-digit patterns. The LeNet-5 architecture is presented in the following schema.

## Data

The dataset used in this model, corresponds to the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
It is composed by a training set of 60,000 examples, and a test set of 10,000 examples.
The digits have been pre-processed to be size-normalized and centred in a fixed-size image of 28x28 pixels.


## Requirements

* Python 3
* Tensorflow 2

## How to run this app

Clone this repository and navigate to the main folder. To do so, open your Terminal (for MacOS/Linux) or your Command Prompt (for Windows) and run the following commands:
```
git clone https://github.com/guillermo-lahuerta/Object_detection.git
cd ./Object_detection/
```

I strongly suggest to create a virtual environment with Conda to manage dependencies and isolate projects. After installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html), run the following commands to update the base conda packages:
```
conda update conda
conda update python
conda update --all
```

Then, create the new conda environment called *object_detection* to store all the dependencies related to this repo:
```
conda create --name object_detection python=3.8.5
conda activate object_detection
```

Now, create a Jupyter Notebook kernel for the new environment:
```
conda install ipykernel jupyter
conda install -c conda-forge jupyter_nbextensions_configurator
python -m ipykernel install --user --name object_detection --display-name "object_detection"
```

Now, install all required packages:
```
pip install -r requirements.txt
```

### Install TensorFlow
It's time to install *TensorFlow*:

* If you have a NVIDIA graphics card with CUDA architectures, you should consider installing *tensorflow-gpu* (instead of the regular *tensorflow*), to speed up your deep learning models. Rather than using *pip* or *conda* to try to figure out which version of TensorFlow you need, I recommend finding the exact "*.whl*" file from [TensorFlow](https://www.tensorflow.org/install/pip#package-location)’s site. Once you have the *url* of the corresponding TensorFlow version that you need, run the following command (substitute *<whl_url>* with the exact url):
```
pip install <whl_url>
```

* If you don't have a NVIDIA graphics card, you should install the regular *TensorFlow* with the following command:
```
pip install tensorflow==2.4.0
```

### Run the notebook
To run the app jupyter notebook, use these commands:
```
conda activate object_detection
python ./app/app.py
```

## Resources

* [TensorFlow](https://www.tensorflow.org/)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
