from tensorflow.keras import layers, models, losses, optimizers
import datasets
import models as my_models
import trainers
import sampling_methods
import attacks



if __name__  == "__main__":
    ## Create a Tensorflow Model
    def create_model(lr, beta1, beta2, epsilon):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D())
        
        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D())
        
        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(10))

        model.summary()

        adam = optimizers.Adam(lr, beta1, beta2, epsilon)
        model.compile(optimizer=adam,
                      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    ## Load a dataset (Uncomment One Line)
    dataset = datasets.MNIST(normalize=True)
##    dataset = datasets.Fashion_MNIST(normalize=True)
    
    ## Shuffle Training Data Order
##    dataset.shuffle()

    ## Generate a model (Uncomment One Line)
##    model = my_models.LeNet5(0.0008, 0.75, 0.85, 1e-8)
##    model = my_models.VGG8()
    model = my_models.Model("Custom", create_model(0.0008, 0.75, 0.85, 1e-8))

    ## Select a sampling method for active learning (Uncomment One Line)
##    sampler = None
##    sampler = sampling_methods.random
##    sampler = sampling_methods.max_entropy
##    sampler = sampling_methods.max_ratio
    sampler = sampling_methods.min_confidence
##    sampler = sampling_methods.min_margin
##    sampler = lambda model, data: sampling_methods.min_perturbation(model, data, attacks.deepfool)
    
    ## Select an attack to generate adversarial samples during learning. (Uncomment One Line)
    adversary = None
##    adversary = attacks.deepfool

    ## Select a training method (Uncomment One Line)
##    trainer = trainers.Supervised(epoch=5, batch_size=128)
    trainer = trainers.ActiveLearning(epoch=5, batch_size=256,
                                      active_learning_rounds=20, num_labels_to_learn=256, subsample_size=4096,
                                      sampling_method=sampler, adversary=adversary)

    trainer.train_model(model, dataset)

    print("Test Accuracy: {}".format(model.evaluate(dataset)))
    model.plot_training_history()
    
