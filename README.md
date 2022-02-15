Do not expect any updates to this project.

# ActiveLearningMadeEasy
A small library to compare various active-learning querying strategies. It has all the common querying strategies and some advanced strategies involving adversarial attacks.

Feel free to reach out with any questions about the code.

Reference Material:
- [Guide to Active Learning](https://www.datacamp.com/community/tutorials/active-learning)
- [Ways to Measure Uncertainty](https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b)
- [DeepFool Active Learning Paper (DFAL)](https://arxiv.org/pdf/1802.09841.pdf)


# Table of Contents
1. [Dependencies](#dependencies)
2. [API](#api)
3. [How to Use](#how-to-use)
4. [Known Issues](#known-issues)
5. [Citation](#citation)

# Dependencies
This was written and tested with:
- Python 3.6
- Tensorflow 2.6
- Numpy
- Scipy
- Matplotlib

# API

Dataset Class:
- Holds training and test data.
```python
import datasets

dataset               = datasets.Dataset(name, train_data, train_labels, test_data, test_labels)
mnist_dataset         = datasets.MNIST(normalize=False)            ## Child Class of Dataset
fashion_mnist_dataset = datasets.Fashion_MNIST(normalize=False)    ## Child Class of Dataset
```
```python
## Attributes
Dataset.name
Dataset.train_data
Dataset.train_label
Dataset.test_data
Dataset.test_label

## Methods
Dataset.shuffle()    ## Shuffles the training data and labels
```

Model Class:
- Holds a TensorFlow model.
```python
import models

model        = models.Model(name, model)
lenet5_model = models.LeNet5(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)    ## Child Class of Model
vgg8_model   = models.VGG8(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)      ## Child Class of Model
```
```python
## Attributes
Model.name
Model.model
Model.history        ## Populated by Trainer

## Methods
Model.evaluate(dataset, verbose='auto')    ## Evalutate model's accuracy on dataset's test data.
Model.plot_training_history()              ## Plot model's accuracy and validation accuracy during training.
```

Trainer Class:
- Trains Models.
```python
import trainers

supervised_trainer = trainers.Supervised(epoch=15, batch_size=32)                       ## Child of _Trainer Class
active_trainer     = trainers.ActiveLearning(epoch=15, batch_size=32,                   ## Child of _Trainer Class
                                             sampling_method=None, subsample_size=0,              
                                             active_learning_rounds=20, num_labels_to_learn=128,
                                             adversary=None)
```
```python
## Attributes
_Trainer.name
_Trainer.epoch
_Trainer.batch_size
ActiveLearning.sampling_method           ## Function that sorts unlabeled data based on some criteria (querying strategy)
ActiveLearning.subsample_size            ## Size of subsample to apply pick next queries from
ActiveLearning.active_learning_rounds    ## Number of rounds of training & querying
ActiveLearning.num_labels_to_learn       ## Number of unlabeled samples to query labels for
ActiveLearning.adversary                 ## Function that generates adversarial data (misclassification attack)
                                         ## If an adversary is provided, the trainer will use the adversary to 
                                         ## generate adversarial versions of the queried data. The adversarial
                                         ## inputs will be added to the 'learned' pool with the original labels.

## Methods
_Trainer.train_model(model, dataset, verbose='auto', validation=True)
```

attacks Module:
- Contains misclassification attacks. (Actually only contains one attack right now.)
```python
import attacks

adversary_input, noise, original_labels, adversary_labels = attacks.deepfool(model, original_input, 
                                                                     overshoot=0.02, max_iterations=50, 
                                                                     max_classes=10, epsilon=1e-4)
```
sampling_methods Module:
- Contains various functions for query selection.
```python
import sampling_methods

pidx = sampling_methods.random(model, not_learned_data)
pidx = sampling_methods.max_entropy(model, not_learned_data)
pidx = sampling_methods.max_ratio(model, not_learned_data)
pidx = sampling_methods.min_confidence(model, not_learned_data)
pidx = sampling_methods.min_margin(model, not_learned_data)
pidx = sampling_methods.min_perturbation(model, not_learned_data, attack)
```

# How To Use
1. Open main.py
2. Uncomment one of each:
   - Dataset
   - Model
   - Sampler
   - Adversary
   - Trainer
4. Run

# Known Issues
These aren't really issues... just things that would be nice to have.
- A parameter to specify initial number of training samples. Right now, the first active learning round is always queried randomly and every round after is queried using the specified strategy. (Easy Fix)
- The Model.plot_training_history() graph could be improved, but I think most people will make their own anyway. (Pointless Fix)
- Should probably rename sampling_methods to querying_strategy. (Easy Fix)

# Citation
- [LeNet-5 Architecture](https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/)
- [VGG Architecture](https://www.kaggle.com/blurredmachine/vggnet-16-architecture-a-complete-guide)
- [DeepFool Active Learning Paper (DFAL)](https://arxiv.org/pdf/1802.09841.pdf)
- [DeepFool Tensorflow Implementation](https://github.com/gongzhitaao/tensorflow-adversarial)
  - I used this as a reference, but mine is significantly better :)
