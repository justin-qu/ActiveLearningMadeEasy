import tensorflow as tf
import numpy as np
import copy

__all__ = ['deepfool']

## More information about DeepFool can be found in the link below.
## https://arxiv.org/pdf/1511.04599.pdf
def deepfool(model, original_input, overshoot=0.02, max_iterations=50, max_classes=10, epsilon=1e-4):
    predictions     = model(original_input)
    classes         = np.argsort(predictions, axis=1)[:, ::-1]
    original_labels = classes[:, 0]

    if max_classes > len(classes[0]):
        max_classes = len(classes[0])
    else:
        classes = classes[:, :max_classes]

    adversary_input     = copy.deepcopy(original_input)
    adversary_labels    = copy.deepcopy(original_labels)
    
    noise = np.zeros(original_input.shape)

    iteration = 0
    tensor_input = tf.Variable(adversary_input, dtype=tf.float32)
    unfinished_indicies = original_labels == adversary_labels

    while sum(unfinished_indicies) > 0 and iteration < max_iterations:
        len_i               = tensor_input.shape[0]
        min_perturbations   = np.array([np.inf] * len_i)
        best_gradients      = np.zeros(tensor_input.shape)
        loss_values         = [0] * max_classes
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tensor_input)
            predictions = model(tensor_input)
            
            for k in range(max_classes):
                loss_values[k] = tf.gather_nd(predictions, np.stack((np.arange(len_i), classes[unfinished_indicies, k]), axis=1))

        original_losses     = loss_values[0]
        original_gradients  = tape.gradient(original_losses, tensor_input)
        
        for k in range(1, max_classes):                
            current_gradients = tape.gradient(loss_values[k], tensor_input)

            gradients_k = current_gradients - original_gradients
            loss_k      = loss_values[k] - original_losses

            perturbations_k = np.abs(loss_k) / np.linalg.norm(gradients_k, axis=(1,2)).flatten()

            update_indices  = (perturbations_k < min_perturbations)
            min_perturbations[update_indices]   = perturbations_k[update_indices]
            best_gradients[update_indices]      = gradients_k[update_indices]

        noise_i = best_gradients * ((min_perturbations + epsilon) / np.linalg.norm(best_gradients, axis=(1,2)).flatten())[:, None, None, None]
        noise[unfinished_indicies] += noise_i

        adversary_input[unfinished_indicies] = original_input[unfinished_indicies] + ((1 + overshoot) * noise[unfinished_indicies])
        
        predictions = model(adversary_input[unfinished_indicies])
        adversary_labels[unfinished_indicies] = np.argmax(predictions, axis=1)

        unfinished_indicies = original_labels == adversary_labels
        tensor_input = tf.Variable(adversary_input[unfinished_indicies], dtype=tf.float32)

        iteration += 1

    noise = (1 + overshoot) * noise

    return adversary_input, noise, original_labels, adversary_labels
    
