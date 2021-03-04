# Introduction

[PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) is a Kaggle challenge where the goal is to predict the adoptability of pets using their online profiles and metadata provided by PetFinder.my, Malaysia’s leading animal welfare platform.

Supervisor: Erwan Scornet (École Polytechnique)

# Data

 - Training data: 14993 pets
 - Testing data: 3972 pets

## Data Fields

### Structured Data

 - PetID - Unique hash ID of pet profile
 - AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict.
 - Type - Type of animal (1 = Dog, 2 = Cat)
 - Name - Name of pet (Empty if not named)
 - Age - Age of pet when listed, in months
 - Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
 - Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
 - Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
 - Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
 - Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
 - Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
 - MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
 - FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
 - Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
 - Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
 - Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
 - Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
 - Quantity - Number of pets represented in profile
 - Fee - Adoption fee (0 = Free)
 - State - State location in Malaysia (Refer to StateLabels dictionary)
 - RescuerID - Unique hash ID of rescuer
 - VideoAmt - Total uploaded videos for this pet
 - PhotoAmt - Total uploaded photos for this pet
 - Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

## Unstructured Data

### Image Related

 - Raw Images
 - Metadata: Annotation, Dominant Colors, Crop Hints

### Text Related

 - Sentiment: Sentences, Documents
 - Entities
 - Language


## Data Exploration

 - Categorical Features: [
    'Type', 'Name', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
    'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State',
    'RescuerID', 'PetID'
]

 - Continuous Features: [
    'Age', 'MaturitySize', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt',
]

 - Textual Features: [
    'Description'
]

- Image Features: []

Target Feature: 'AdoptionSpeed'

### Histograms

![alt text](https://github.com/hayashimasa/Pet_Adoption/blob/master/visualizations/rawdata_hist.png?raw=true)

# Preprocessing

## Normalization

 - Power law distribution : [
    'Age', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt'
]
 - Gaussian Distribution = ['MaturitySize']

## Categorical Embedding

Categorical Features:
 - 'Type': 2 -> 1
 - 'Name': 9060 -> 100
 - 'Breed1': 176 -> 50
 - 'Breed2': 135 -> 50
 - 'Gender': 3 -> 2
 - 'Color1': 7 -> 3
 - 'Color2': 7 -> 3
 - 'Color3': 6 -> 3
 - 'Vaccinated': 3 -> 2
 - 'Dewormed': 3 -> 2
 - 'Sterilized': 3 -> 2
 - 'Health': 3 -> 2
 - 'State': 14 -> 5
 - 'RescuerID': 5595 -> 100
 - 'PetID': 14993 -> 100


Embedding Architecture: categorical features -> embedding -> concat -> fc(100) -> fc(100)

### Visualization with t-SNE

![alt text](https://github.com/hayashimasa/Pet_Adoption/blob/master/visualizations/catemb_3dtsne.png?raw=true)

## Textual Features

doc2vec, word2vec

## Image Features

CNN feature encoding (ResNet)

# Model

## Gradient Boosting Tree (LightGBM)

## Neural Network (PyTorch)

Multilayer Perceptron

# Training

## Stratified K-Fold

Used stratfied 5-fold to obtain indicies for training (11994/80%) and validation (2999/20%) sets.

To generate a smaller dataset, stratified 5-fold was apply to one of the validations sets (2999/20%) to obtain smaller training (2399/16%) and validation (600/4%) sets

![alt text](https://github.com/hayashimasa/Pet_Adoption/blob/master/visualizations/mse_loss.png?raw=true)

# Result

## Metric

Quadratic Weighted Kappa: 0.259

![alt text](https://github.com/hayashimasa/Pet_Adoption/blob/master/visualizations/qwk.png?raw=true)

# References
1. Airbnb: GDBT -> Deep Learning
2. Entity Embeddding of Categorical Variables
3. ResNet
4. DenseNet
5. doc2vec
