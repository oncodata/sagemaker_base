# Steps to train a DSMIL model

(This repository  contains examples for training a stomach cancer detection model)

## Downloading a dataset from TCGA

First, you need to download two files from GDC, the first one is the manifest file of the images that you want to use. The second one is the biospeciment json file with information regarding the patients for the dataset

With the manifest file, use the *download_from_gdc* tool at [https://github.com/oncodata/download_from_gdc](https://github.com/oncodata/download_from_gdc) To download the svs images

## Generate the base dataset json

All our training code is expected to be run using a simplified json that contains the relevant information about each one of the dataset's images and 
also the location of such images. To generate this json, use the *json_creation* tool at [https://github.com/oncodata/json_creation](https://github.com/oncodata/json_creation)

## Generating the features

Once the svs files are downloaded and the base json was created, it is necessary to generate the features. To do so, use the *feature_generation* tool at [https://github.com/oncodata/feature_making](https://github.com/oncodata/feature_making)

## Training the model

Once we have the base json and the features, it is necessary to create the train and test jsons (which are separated by patients). To do so, use the *separate_jsons_by_patient.ipynb* notebook.

Once the train and test jsons are created, use the *train_dsmil.ipynb* notebook to train and test the model. It will generate a .pt file with the models checkpoint.

## Generating heatmaps for the model

If you wish to take a look at how the model is identifying cancer, you may use the *create_heatmaps* notebook.

IMPORTANT: The creation of heatmaps does feature generation in inference time. So, if you are using Mocov3 (use at the example), you need to follow the same steps in [https://github.com/oncodata/feature_making](https://github.com/oncodata/feature_making) to download the trained model.