# DeepLearning-Engine (2018)
A mini framework for assembling Machine Learning/Deep Learning models. 

For Machine learing/Deep learning tasks, the common process generally consists of steps: 

* Prepare/query dataset

    downlaoding dataset from the internet or servers, splitting dataset, or some other pre-processing.
  
* Setup a model

    building a new one or from a pre-trained one
  
* Config model's parameters

    Things like epoches, learning rate, callbacks, etc.
  
* Execute a task

    training a model, doing prediction using trained model, etc.

This engine is designed to split those tasks into steps, users can easily design their own steps and assemble them together to setup their specific tasks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This is a python projects, the engine framework itself is compitable with both python 2 and python 3. However, this engine introduces some steps classes for typical tasks liike training model and prediction, thes steps are based on python 3. 

Moreover, the steps and tasks utilize Keras, if you want to use others, just remove those steps and build your own ones.

For only the engine, following packages are required:

  * json
  * importlib

Following are necesary for the sample steps:

  * keras
  * sklearn
  * pickle
  * numpy
  * cv2
  * imutils

### Installing
step 1. Donwload the code

step 2. Assemble a task and run it.

## System architecture
The steps for a task is defined in a json file, which consists of two parts: step definition and step options. The engine firstly query the step definition and create an step instance for all the steps, then it config the steps with the step options.

After creating and assembling all the steps, it execute the steps one by one. 

An sample config file for training a model:

![step config](https://github.com/Borrk/DeepLearning-Engine/raw/master/doc/step-config-file.png)

The engine architecture:
![Engine](https://github.com/Borrk/DeepLearning-Engine/raw/master/doc/System-Architecture.png)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Brook Huang** - *All the work*

See also the list of [contributors](https://github.com/Borrk/Enzyme-labeled-instrument.git/contributors) who participated in this project.

## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**

