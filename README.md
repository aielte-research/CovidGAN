# CovidGAN

## Overview

After Ian Goodfellow introduced GANs in 2014 and the first models started showing promising results, their popularity skyrocketed. A GANs task is to learn the distribution of the training data, then they are used for data generation. 
They can be used for sensiteve data protection and data generation for augmentation porpuses. Because a GAN's job is to generate data (and not to classify them), it is challanging to measure their efficiency simply. 
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
		- and could it compete with the 'simple' augmentation method

The dataset used by this project is: COVID_QU_EX dataset, which is a chest x-ray dataset with 3 classes: [covid, normal, viral].
As of 2022 September, this dataset is a balanced.

To have a test dataset which is balanced, 20% of the images were put aside for testing the classificator.

From the remaining images, came 5 simulated dataset, where the covid class is under-represented. The 5 simulated dataset kept 5 different portions of the covid images. 
Namely they kept: 1 or 100%, 0.8 or 80%, 0.6, 0.4, 0.2 ratio of the covid images.
The simulated datset with 100% of the covid images, was made to be a benchmark.
With the other four ratios, each simulated dataset was used for training a GAN. 

This was repeated 5 times, with 5 different train-test split of the COVID_QU_EX dataset.

The classification was repeated with 3 different networks used for classification. 
Namely: [Resnet18, Vgg16, EfficientNet_b0]

So: 
5  different splits, for each split 5 different ratios , of wich 4 was used for training GAN
3 different networks, and 3 different augmentation mode
Roughly 200 classificator 

To achieve this, this project uses Lippizzaner, a GAN training program developed by the ALFA research group from MIT. Lipizzaner combines evolutionary learning algorithms and neural networks to manage the pitfalls of training a GAN.
This was used because it solved many of the problems of training a GAN (mode-collapse, vanishing gradients) 

Problems during the project:
Lipizzaner requires a lot of resource because effectively it is training several different networks in the same time, 
and if the network is a little bit more complex then a simple Perceptron network, the memory demand is huge.

Because of the roughly 200 test cases, training the classificators couldn't take up much time so low epoch numbers were used to finish with the trainings in time

#Future research possibilites: 

In this case no fine-tuning were made, all test cases trained on the same parameters, but it is posssible that some cases could be fine-tuned to achieve a few more percent all-in-all in accuracy

It might be possible that a GAN with given parameters could be fine-tuned for one kind of classificator network, and the same GAN wouldn't be good for a different kind of classificator network
This would mean that a GAN only learns to generate image for a specific kind of network, and it doesn't learn the pictures' distribution generally 
