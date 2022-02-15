import sampling_methods
import numpy as np

__all__ = ['Supervised', 'ActiveLearning']

class _Trainer():
    def __init__(self, name, epoch, batch_size):
        self.name       = name
        self.epoch      = epoch
        self.batch_size = batch_size

        assert (type(epoch) is int and epoch > 0)
        assert (type(batch_size) is int and batch_size > 0)
    
    def train_model(self, model, dataset, verbose='auto', validation=True):
        pass



class Supervised(_Trainer):
    def __init__(self, epoch=15, batch_size=32):
        super().__init__("Supervised Learning", epoch, batch_size)
        
    def train_model(self, model, dataset, verbose='auto', validation=True):
        if verbose != 'auto':
            assert (type(verbose) is int and verbose in range(3))

        if validation:
            history = model.model.fit(dataset.train_data, dataset.train_labels,
                                      validation_data=(dataset.test_data, dataset.test_labels),
                                      epochs=self.epoch, batch_size=self.batch_size, verbose=verbose)
        else:
            history = model.model.fit(dataset.train_data, dataset.train_labels,
                                      epochs=self.epoch, batch_size=self.batch_size, verbose=verbose)
            history.history['val_loss']     = [0] * self.epoch
            history.history['val_accuracy'] = [0] * self.epoch

        model.history = history.history



## More information about the LeNet-5 architecture can be found in the link below.
## https://www.datacamp.com/community/tutorials/active-learning
class ActiveLearning(_Trainer):
    def __init__(self, epoch=10, batch_size=32,
                 sampling_method=None, subsample_size=0,
                 active_learning_rounds=20, num_labels_to_learn=128,
                 adversary=None):
        super().__init__("Active Learning", epoch, batch_size)

        ## If no sampling method is specified, just label next N unlabeled images
        if sampling_method is None:
            sampling_method = lambda model, data: np.arange(len(data))

        self.sampling_method        = sampling_method
        self.subsample_size         = subsample_size
        self.active_learning_rounds = active_learning_rounds
        self.num_labels_to_learn    = num_labels_to_learn
        self.adversary              = adversary

        assert (type(subsample_size) is int and subsample_size >= 0)
        assert (type(active_learning_rounds) is int and active_learning_rounds > 0)
        assert (type(num_labels_to_learn) is int and num_labels_to_learn > 0)
        
    def train_model(self, model, dataset, verbose='auto', validation=True):
        if verbose != 'auto':
            assert (type(verbose) is int and verbose in range(3))
        
        learned_data        = dataset.train_data[:0]
        learned_labels      = dataset.train_labels[:0]
        not_learned_data    = dataset.train_data[0:]
        not_learned_labels  = dataset.train_labels[0:]

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        ## Label the first N elements in the 'not-learned' list
        def label(n):
            nonlocal learned_data, learned_labels, not_learned_data, not_learned_labels

            if n > len(not_learned_data):
                n = len(not_learned_data)
            
            learned_data        = np.concatenate((learned_data, not_learned_data[:n]))
            learned_labels      = np.concatenate((learned_labels, not_learned_labels[:n]))
            not_learned_data    = not_learned_data[n:]
            not_learned_labels  = not_learned_labels[n:]

        ## Train the model, record the history.
        def train(i):
            nonlocal self, model, dataset, learned_data, learned_labels, history, verbose
            
            if verbose:
                print("\nRound {}\nLearned Samples: {}\n".format(i, len(learned_data)))

            if validation:
                history_i = model.model.fit(learned_data, learned_labels,
                                          validation_data=(dataset.test_data, dataset.test_labels),
                                          epochs=self.epoch, batch_size=self.batch_size, verbose=verbose)
            else:
                history_i = model.model.fit(learned_data, learned_labels,
                                          epochs=self.epoch, batch_size=self.batch_size, verbose=verbose)
                history_i.history['val_loss']     = [0] * self.epoch
                history_i.history['val_accuracy'] = [0] * self.epoch

            history['loss']         += history_i.history['loss']
            history['accuracy']     += history_i.history['accuracy']
            history['val_loss']     += history_i.history['val_loss']
            history['val_accuracy'] += history_i.history['val_accuracy']

        ## Sort the 'not-learned' list with a sampling method.
        def pick_samples(n):
            nonlocal self, model, not_learned_data, not_learned_labels
            
            if n and n > self.num_labels_to_learn:
                n = min(n, len(not_learned_data))
                
                pidx = np.random.permutation(len(not_learned_data))
                not_learned_data    = not_learned_data[pidx]
                not_learned_labels  = not_learned_labels[pidx]
                
                pidx = self.sampling_method(model.model, not_learned_data[:n])
                
                not_learned_data[:n]    = not_learned_data[pidx]
                not_learned_labels[:n]  = not_learned_labels[pidx]
            else:
                pidx = self.sampling_method(model.model, not_learned_data)
                not_learned_data    = not_learned_data[pidx]
                not_learned_labels  = not_learned_labels[pidx]

        ## If an attack is provided, generate artificial samples by adding adversary images with their original label to the 'learned' list
        def use_adversary(attack, n):
            nonlocal model, learned_data, learned_labels, not_learned_data, not_learned_labels
            
            if n > len(not_learned_data):
                n = len(not_learned_data)

            adversary_data, _, _, _ = attack(model.model, not_learned_data[:n])

            learned_data    = np.concatenate((learned_data, adversary_data))
            learned_labels  = np.concatenate((learned_labels, not_learned_labels[:n]))

        
        for i in range(self.active_learning_rounds - 1):
            label(self.num_labels_to_learn)
            
            if len(not_learned_data) == 0:
                break
            
            train(i+1)
            pick_samples(self.subsample_size)

            if self.adversary is not None:
                use_adversary(self.adversary, self.num_labels_to_learn)

        label(self.num_labels_to_learn)
        train(i+1)

        model.history = history
