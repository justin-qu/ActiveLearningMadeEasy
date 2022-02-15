from scipy.stats import entropy
from scipy.special import softmax
import numpy as np



## Shuffle
def random(model, not_learned_data):
    return np.random.permutation(len(not_learned_data))

## Sort by Prediction Entropy (decreasing)
## https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
def max_entropy(model, not_learned_data):
    predictions = model.predict(not_learned_data)
    normalized_predictions = softmax(predictions, axis=1)
    
    pred_entropy = entropy(normalized_predictions, axis=1)
    return np.argsort(pred_entropy)[::-1]

## Sort by Ratio of Confidence (decreasing)
## https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
def max_ratio(model, not_learned_data):
    predictions = model.predict(not_learned_data)
    normalized_predictions = softmax(predictions, axis=1)
    
    normalized_predictions.sort(axis=1)
    pred_ratio = normalized_predictions[:, -2] / normalized_predictions[:, -1]
    return np.argsort(pred_ratio)[::-1]

## Sort by Prediction Confidence
## https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
def min_confidence(model, not_learned_data):
    predictions = model.predict(not_learned_data)
    
    pred_confidence = np.amax(predictions, 1)
    return np.argsort(pred_confidence)

## Sort by Margin of Confidence
## https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
def min_margin(model, not_learned_data):
    predictions = model.predict(not_learned_data)
    normalized_predictions = softmax(predictions, axis=1)

    normalized_predictions.sort(axis=1)
    pred_margin = normalized_predictions[:, -1] - normalized_predictions[:, -2]
    return np.argsort(pred_margin)

## Sort by Minimum Perturbation (L2-norm) Required to Misclassify Image.
## https://arxiv.org/pdf/1802.09841.pdf
def min_perturbation(model, not_learned_data, attack):
    _ , noise, _, _ = attack(model, not_learned_data)
    perturbation = np.linalg.norm(noise, axis=(1,2)).flatten()

    return np.argsort(perturbation)
    
