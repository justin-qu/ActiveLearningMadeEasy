from tensorflow.keras import layers, models, losses, optimizers
import matplotlib.pyplot as plt

__all__ = ['Model', 'LeNet5', 'VGG8']
    
class Model:
    def __init__(self, name, model):
        self.name       = name
        self.model      = model
        self.history    = None

    def __str__(self):
        return "{} Model".format(self.name)

    def evaluate(self, dataset, verbose='auto'):
        test_loss, test_acc = self.model.evaluate(dataset.test_data, dataset.test_labels, verbose=verbose)
        return test_acc

    def plot_training_history(self):
        plt.title(self.name + " Model Training Accuracy")
        plt.plot(self.history['accuracy'], label='Accuracy')
        plt.plot(self.history['val_accuracy'], label = 'Val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    

## More information about the LeNet-5 architecture can be found in the link below.
## https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/
class LeNet5(Model):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        model = self._create_model(lr, beta1, beta2, epsilon)
        
        super().__init__("LeNet-5", model)

    def _create_model(self, lr, beta1, beta2, epsilon):
        model = models.Sequential()
        
        model.add(layers.Conv2D(6, (5, 5), kernel_initializer='he_uniform', activation='tanh', padding='same', input_shape=(28, 28, 1)))
        model.add(layers.AveragePooling2D())

        model.add(layers.Conv2D(16, (5, 5), kernel_initializer='he_uniform', activation='tanh'))
        model.add(layers.AveragePooling2D()) 

        model.add(layers.Conv2D(120, (5, 5), kernel_initializer='he_uniform', activation='tanh'))

        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(10))

        model.summary()

        adam = optimizers.Adam(lr, beta1, beta2, epsilon)
        model.compile(optimizer=adam,
                      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model



## More information about the VGG-N architecture can be found in the link below.
## https://www.kaggle.com/blurredmachine/vggnet-16-architecture-a-complete-guide
class VGG8(Model):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        model = self._create_model(lr, beta1, beta2, epsilon)
        
        super().__init__("VGG-8", model)

    def _create_model(self, lr, beta1, beta2, epsilon):
        model = models.Sequential()
        
        model.add(layers.UpSampling2D(size=(8,8), input_shape=(28, 28, 1)))
        
        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D()) 

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.MaxPooling2D())
        
        model.add(layers.Flatten())

        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(10))

        model.summary()

        adam = optimizers.Adam(lr, beta1, beta2, epsilon)
        model.compile(optimizer=adam,
                      loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model
        
