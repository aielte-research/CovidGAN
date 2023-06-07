# CovidGAN

## Overview

After Ian Goodfellow introduced GANs in 2014 and the first models started showing promising results, their popularity skyrocketed. A GANs task is to learn the distribution of the training data, then they are used for data generation. 
They can be used for sensiteve data protection and data generation for augmentation purposes. Because a GAN's job is to generate data (and not to classify them), it is challanging to measure their efficiency simply. 
With this project the goal is to measure GANs' efficiency from a data augmentation perspective. 
The question one might ask themselves: Is it worth training a GAN to generate data for augmentation, or other methods are more succesful and cheaper to implement?

The idea is to simulate a situation during classification when the dataset isn't balanced, and one class in particular is under-respresented (like fradulent transactions in a bank system or 
covid chest x-ray images in the beginning of the pandemic).
Take the under-respresnted class and train a GAN on it and balance the dataset with GAN-generated data
	1. Then train a classificator on the 'gan-balanced' dataset.
	2. Train a classificator on the 'unbalanced' dataset. (For benchmark purposes)
	3. And finally, train a third classificator with some other kind of augmentation (for eg. simple 'oversampling')
	4. Compare the results and see, wether 
		- the 'gan-balanced' case improved on the classification accuracy, 
		- and could it compete with the 'simple' augmentation method, for eg. oversampling

The dataset used by this project is: COVID_QU_EX dataset, which is a chest x-ray dataset with 3 classes: [covid, normal, viral].
As of 2022 September, this dataset is a balanced.

To have a test dataset which is balanced, 20% of the images were put aside for testing the classificator.

From the remaining images, came 5 simulated dataset, where the covid class is under-represented. The 5 simulated dataset kept 5 different portions of the covid images. 
Namely they kept: 1 or 100%, 0.8 or 80%, 0.6, 0.4, 0.2 ratio of the covid images.
The simulated datset with 100% of the covid images, was made to be a benchmark.
With the other four ratios, each simulated dataset was used for training a GAN. 

To achieve this, this project uses Lippizzaner, a GAN training program developed by the ALFA research group from MIT. Lipizzaner combines evolutionary learning algorithms and neural networks to manage the pitfalls of training a GAN.
This was used because it solved many of the problems of training a GAN (mode-collapse, vanishing gradients) 

We used a 2\*2 grid in the Lipizzaner framework, with a convolutional grayscale network. Each GAN training roughly took 8 hours. 

Onto the classification:
We tested the classification for the 5 datasets, with the 3 picture augmentation method (None, oversampling, gan). Some test cases were excluded because of redundancy. (oversampling and gan doesn't affect the dataset when we have 100% of the Covid pictures) 
All test cases so far: 5 + 4 + 4 = 13

Each case was repeated with using classic affin augmentation, such as affin rotation and five crop, and without it. 
All test cases so far: 13\*2 = 26

The classification was repeated with 3 different networks.
Namely: [ResNet18, VGG16, EfficientNet_b0]
All test cases so far: 26\*3 = 78

This was repeated 5 times, with 5 different train-test split of the COVID_QU_EX dataset.

All test cases: 78\*5 = 390 
We completed the hyperparameter optimization for only one split, and used this one set of hyperaprameters for the other splits. (78 case optimization)

Each test case took betwween 30 minutes and 2 hours of training.

The results were evaluated with simple classification metrics. 

#Future research possibilites: 
It might be possible that a GAN with given parameters could be fine-tuned for one kind of classificator network, and the same GAN wouldn't be good for a different kind of classificator network
This would mean that a GAN only learns to generate image for a specific kind of network, and it doesn't learn the pictures' distribution generally 
