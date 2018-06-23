from keras.callbacks import Callback


class ShowTestAccuracyEachEpoch(Callback):
    def __init__(self, test_data, test_labels):
        self.test_data = test_data
        self.test_labels = test_labels

    def on_epoch_end(self, epoch, logs={}):
        x = self.test_data
        y = self.test_labels
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))