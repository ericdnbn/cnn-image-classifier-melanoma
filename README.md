# Identifying Skin Lesions In Dermoscopic Images With Neural Networks

### Author: Eric Denbin

<p align="center">
  <img src=images/derma.jpeg />
</p>

## Business Understanding

Skin cancer is the most common form of cancer in the United States and worldwide. More people are diagnosed with skin cancer each year in the U.S. than all other cancers combined. 

Specific types of skin lesions are usually diagnosed through biopsies, serial imaging, or single image expert consensus. 

It is estimated that in 2018, biopsies of benign tumors cost between $624 million and $1.7 billion(https://ascopubs.org/doi/abs/10.1200/JCO.2018.36.15_suppl.e18903).



## Data Understanding

My dataset consists of 7,179 dermoscopic images from the Internation Skin Imaging Collaboration(ISIC) archive (https://www.isic-archive.com/). All patients were 10-90 years old and the images were taken in the course of clinical care.

The ISIC archive contains over 150,000 images, 70,000 of which have been made public. I downloaded only dermoscopic images to ensure a certain standard of quality in regard to the data. The archive contains 23,704 dermoscopic images of benign lesions, 2,240 dermoscopic images of malignant lesions, and 2,212 dermoscopic images of unknown lesions. I downloaded 2,401 images of benign lesions for training and validation, and 600 images of benign lesions for testing. I downloaded 1500 images of malignant lesions for training and validation, and 600 for testing. For unkown lesions, I downloaded 1500 images for training and validation, and 600 for testing.

The following file structure provides the ground truth labeling needed to train the models. If you wish to run my code, you will need to download images from the ISIC archive into the same directory format:
```
└── dermoscopic_images
    ├── train
    │    ├──benign
    |    ├──malignant
    │    └──unknown
    └── test
         ├──benign
         ├──malignant
         └──unknown
```



## Modeling with neural networks

![fsm](images/confusion_matrix_fsm)

My first simple model consists of a basic fully connected dense neural network with two hidden layers, plus an output layer. 
This model serves mainly as a proof of concept and provides baseline accuracy and recall scores.

To improve on my first simple model, I began iterating on convoluational neural networks. The following represents various different improvements made over these iterations:
 - Using the full dataset
 - Adding more dense layers to improve the power of the network
 - Adding convolutional layers to improve pattern recognition
 - Adding dropout layers to reduce computational workload and overfitting
 - Adding batch normalization layers to reduce computational workload and overfitting
 - Using L2 regularization to avoid overfitting
 - Trying different kernel sizes

![confusion matrix best cnn](images/confusion_matrix_best_cnn)

*How iterations improved model*

I continued to iterate with the pre-trained 'imagenet' model, adding more layers, including:
 - Introducing `BatchNormalization` layers, which reduce the range of input values to speed training and improve performance
 - Adjusting number of layers to change the complexity of the model

 
Collectively, I iterated through ten models, adjusting these features parameters among others. My final model has the following architecture:

![final model summary]()



## Final Evaluation

The final model...

![final confusion matrix]()

On unseen testing data...



## Conclusions

### Recommendations

- I recommend that this model be used by medical professionals to screen patients who are having skin lesions examined.

- I recommend that this model be used to reduce the number of biopsies taken of benign lesions.

- Finally I recommend that this model be used to expedite the process of serial imaging, and replace single image expert consensus


### Possible Next Steps

- Round out the ISIC archive with more dermoscopic images of malignant skin lesions(basal cell carcinoma, squamous cell carcinmoa)

- Create a classifier that can idneitfy specific lesions, not just the risk they pose(benign, malignant, unknown)



## For More Information

See the full analysis in the [Jupyter Notebook](./.ipynb) or review this [presentation](./.pdf)



### Structure of Repository:

```
├── callback_checkpoints
├── dermoscopic_images (dataset)
├── functions.py
├── images (for readme, presentation)
├── models (saved .h5 files of trained models)
├── skin_lesion_image_classifier_notebook.ipynb
└── README.md
```
