import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import pickle
import numpy as np

class Model():
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        print(self.input_shape)
 
    def built_model(self):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(16, (3,3), strides=(1, 1),padding='same',  activation='relu', input_shape= self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3,3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        self.model = model 

    def summary(self):
        return self.model.summary()

    def compile(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']) 

    def fit(self, x_train, y_train, epochs=1):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=True)
        
    def save(self, path):
        tf.keras.models.save_model(self.model, path, overwrite=True, save_format='.h5')

    def load_model(self, path):
        tf.keras.models.load_model(path)
# if __name__ == "__main__":
#     model = Model()
#     model.built_model()
#     model.summary()
#     model.compile()
#     model.fit(np.random.rand(100, 28, 28, 1), np.random.rand(100, 28, 28, 1, 1))




















# class Model():
#     def __init__(self):
#         pass 

#     def build_model(self):
#         print('Creating model.')
#         self.model = tf.keras.models.Sequential()
#         self.model.add(Conv2D(16, (3,3), strides=(1, 1), activation='relu', input_shape= (28, 28, 1)))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))

#         print('..')
#         self.model.add(Conv2D(16, (3,3), strides=(1, 1), activation='relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))

#         self.model.add(Flatten())
#         self.model.add(Dense(1024, activation='relu'))
#         self.model.add(Dense(10, activation='softmax'))
#         print('..')
#         return self.model 

#     def compile(self, optimizer, loss, learning_rate = 0.001):
#         print('Compiling model.')
#         self.optimizer = optimizer 
#         self.loss = loss 
#         self.learning_rate = learning_rate

#         if self.optimizer == 'adam':
#             self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#         if self.optimizer == 'rmsprop':
#             self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

#         if self.optimizer == 'sgd':
#             self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
#         if self.loss == 'sparescategorical':
#             self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
#         if self.loss == 'mse':
#             self.loss = 'mse'
#         print('..')
#         self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
#         print('..')
#         return self.model 
    
#     def fit(self, x_train, y_train, epochs=10):
#         print('Fitiing model..')
#         self.epochs = epochs 
#         self.xtrain = x_train
#         self.ytrain = y_train 

#         self.model = self.model.fit(self.xtrain,
#                                     self.ytrain, 
#                                     epochs=self.epochs,
#                                     verbose=True
#                                  )
#         print('..')

#         return self.model 

#     def save(self, filePath = None):
#         # self.model.save('model/mnist')
#         filename = 'finalized_model.sav'
#         pickle.dump(self.model, open(filename, 'wb'))
#         print('Model saved!')
    
#     def load_model(self, filePath):
#         pickle.load(filePath)
        

#     def predict(self, input):
#         prediction = self.model.predict(input)

#         return prediction 