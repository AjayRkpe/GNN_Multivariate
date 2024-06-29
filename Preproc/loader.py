#Here a custom data generator function is written to override the inbuilt data input process of tensorflow so that the input of this model can be used

import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data_array, batch_size=32, shuffle=True):
        'Initialization'
        self.data = data_array
        self.batch_size = batch_size
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size)) 

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        gr = self.data[0][0]
        X = np.empty((self.batch_size, gr.shape[0], gr.shape[1], gr.shape[2]))
        for i3, id in enumerate(list_IDs_temp):
            X[i3,] = self.data[0][id]

        
        y = np.empty((self.batch_size, self.data[0][1].shape[0]))
        for i4, id1 in enumerate(list_IDs_temp):
            y[i4,] = self.data[1][id]

        return X, y
