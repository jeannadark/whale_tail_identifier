# Whale-Tail-Identifier

## Project Description

This project is my solution to the Kaggle Humpback Whale Identification Challenge. To aid whale conservation efforts, scientists rely on captured images to identify whale species by the shape of their tails and unique markings. The challenge is to build an algorithm that can help identify individual whales on images. 

## Dataset

The original training dataset was submitted to Kaggle by HappyWhale. The data contained approximately 25K images of whales (comprising 5,005 different whale species, including a "new_whale" category for yet unidentified whale types). 

Examples of different images pertaining to one whale type:

![GitHub Logo](/whale1.png)
![GitHub Logo](/whale2.png)

## Analysis

### Data Exploration & Synthetic Data Generation

The training data is heavily imbalanced, with 9K images pertaining to the "new_whale" category and less than 100 images per the rest of the categories (with some categories having just 1 image in the whole set). 

To create more balance in the dataset and deal with the "new_whale" type, the following approaches were considered:

    - regroup the "new_whale" category into meaningful clusters based on image similarity (if there are any);
    - if no meaningful clusters, downsize the "new_whale" category for the purposes of better prediction accuracy;
    - generate synthetic data for other underrepresented whale types. 

The images in the "new_whale" category were proven to be too dissimilar for them to form any meaningful clusters based on nearest-neighbor similarity. The algorithm incorrectly grouped whales with different markings and tail shapes into one cluster (possibly grouping them based on other similar image attributes). For the purposes of a better prediction accuracy, the "new_whale" category was dropped from the dataset.

Synthetic data was generated for the rest of the underrepresented whale types by combining random backgrounds & foregrounds to generated approximately 40 extra images for each whale category (thereby resolving the issue of whale types having a representation of just a few images). 

Examples of synthetically generated data:

![GitHub Logo](/synth1.png)
![GitHub Logo](/synth2.png)
![GitHub Logo](/synth3.png)

### Image Preprocessing

To facilitate better prediction accuracy and capture all possible ways a whale's tail might appear on the image, the following transformations were applied - horizontal/vertical flips, image normalization, rotation, etc. 

### Dataset Split

Since the generated data was too large handle, the training was performed on a randomly sample of the data, representing 2,572 different whale categories, with an 80-20 split into train (35K images) and validation sets (9K images).

### Training

A 9-layered convolutional neural network was used to train on the preprocessed data. After training for 20 epochs, the accuracy achieved on the validation set approximated 63%, which is a lot higher than the "naive" training on the original data without synthetic images and with the "new_whale" category present. 

The curve below represents an almost exponential increase in model accuracy. 

![GitHub Logo](/acc.png)

## Conclusion







