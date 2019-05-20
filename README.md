# Medical Imaging - Camelyon 16 challenge
Instructor: Prof. Josh Gordon

Team: Beichun Qi (bq2139), Boxi Xia (bx2150)

#### [Video demo and code walkthrough](https://youtu.be/zNceQO2JufE)
#### [Presentation slide](https://docs.google.com/presentation/d/1cr5YLsgH_RgjFtjSHn1h_C9uAYCYLqixkGY_dGliKzg/edit?usp=sharing)
#### [Github repository: /bq2139/medical-imaging-project](https://github.com/bq2139/medical-imaging-project)

## Project Overview

Visual inspection of whole-slide images created from tissue samples is the most common way of detecting breast cancer metastases. However, it can be a tedious job because tissue samples are volumes. One tissue sample can be sliced to produce many tissue images for pathologists to inspect, while tumors are usually very small compared to a whole slide. This can cause fatigue and mistakes. Also, visual inspection requires expertise, which is rare.

Our goal is to develop a tool that can radically reduce the workload of pathologists and the misdiagnosis rate for common diagnostic tasks using mostly off the shelf software and well-known algorithms. Considering the recent achievements of deep neural networks in various image classification tasks, we decided to build a deep neural network to classify the slide images. The prediction of the neural network can form a second opinion for a pathologist - the opinion is to assist, not to replace pathologists.

To approach our goal, we first investigated the training data. We examined a few slide examples and found that tumor cells are always inside tissue cells, and often a large portion of the slide has no tissue (looks clean). We thus created the tissue masks for the slides images to exclude the non-tissue area. We generated tissue masks for level 1 slides, which is our target resolution level, based on level 8 slide images, since a 128 * 128 patch in level 1 corresponds to one pixel in level 8. We also slices the whole images down to small image patches (256 * 256) for our furture network to process. Then, our target is down to the tissue patches.

However, the number of tumor patches to normal (non-tumor) patches are still extremely imbalanced, so we took some steps to deal with the imbalanced data. First, we downloaded all the camelyon dataset and used ASAP to extract slide images (code in [preprocess_mask](./preprocess_mask.ipynb)). Doing so, we gathered more tumor patches, but training data is still imbalanced. So we used all the tumor patches and randomly sampled normal patches. Finally, we used data augmentation on the tumor patches. This includes random rotation, flip, hue, saturation, contrast and brightness. After those steps, we ensured that the number of tumor patches are comparative to normal patches.

Because the size of training data is huge, loading the image patches while training is expensive and would be an inefficient use of training time. To combat this problem, we utilized TFRecords to generate TFRecord dataset files (243GB training/2.5GB testing) from preprocessed image patches. Data augmentation was performed during training, which include random rotation / flip / hue / saturation / contrast / brightness.

We used the existing Xception model with a dense layer as our baseline model. Trained from fresh start, the baseline achieved 96% training accuracy and 97% validation accuracy after 150 epochs. Using trained weights achieve slightly better result with only 50 epochs. We also tried replacing all relu activations with selu activation, but the result showed no favor for the latter.

Another experiment we did was octave convolution. We compared MobilenetV2 (using lightweight depthwise convolution by default) and Resnet50 with its convolution layer replaced by octave convolution The experiment achieved training accuracy of 96% and 97% respectively, and validation accuracy of 97% both. We ploted the confusion matrix for every model.

We assembled the above steps to an end-to-end pipeline by performing the final step: ploting the probability heatmap of the slide. The heatmap shows clearly the tumor regions while also being conservative about the areas where it is unsure about.



### Reference
+ [CAMELYON16](https://camelyon17.grand-challenge.org/Data/)
+ [OpenSlide](https://openslide.org/)
+ [DeepZoom viewer](https://github.com/openslide/openslide-python/tree/master/examples/deepzoom)
+ [Detecting Cancer Metastases on Gigapixel Pathology Images](https://arxiv.org/abs/1703.02442)
+ [Octave convolution paper](https://arxiv.org/abs/1904.05049)
+ [Octave convolution keras code](https://github.com/titu1994/keras-octconv)
