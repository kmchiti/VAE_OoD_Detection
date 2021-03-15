# VAE_OoD_Detection

## Requirements
Python: 3.6+

To install required packages:

```setup
pip install -r requirements.txt
```

## Model Details
* Developed by Kamran and Alireza for the IFT6390 first project.
* Model has been developed on 2021-03-10
* Using Variational Autoencoder to identify if a sample is coming from a similar distribution as the training dataset or not

# Factors
* The model is trained on the MNIST dataset and it is mostly accurate for images with similar resolution, colour and content.
* The model is evaluated on images from MNIST and Fashion MNIST datasets to test the accuracy of identifying in distribution and out of distribution samples

## Metrics
* By calculating the log-likelihood of a new sample and comparing it to a threshold, we classify it as Out of Distribution or In distribution. The choice of a threshold depends on the particular application. We have tuned the threshold, as same as other model hyper parameters, by validation set.


## Intended Use
* Intended to be used for detecting if a sample is coming from a similar distribution as the training dataset or not
* Particularly intended for investigating if a picture is of a number or not
* Not suitable for classifying other types of images such as animals or objects
* Should not be used for essential topics such as navigation in smart cars

## Reproducibility
We have provide a python script named run.py taking the following arguments:
* Download the dataset :
```setup
python run.py fetch
```
* Train the model on the dataset:
```setup
python run.py train
```
* Compute the evaluation metrics and figures needed in the model card and project report which you can find in latex folder:
```setup
python run.py evaluate
```
* Compile model card and report PDF documents based on the evaluation metrics obtained:
```setup
python run.py build_paper
```
