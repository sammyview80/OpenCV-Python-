import tensorflow as tf 
import numpy as np
import cv2 as cv 
from loadData import load_data
from model import Model 
import matplotlib.pyplot as plt
import PIL.ImageOps


class Main():

    def run(self):
        x_train, y_train, x_test, y_test = load_data(show=True)
        
        model = Model(input_shape=x_train[0, :, :, :].shape)
        model.built_model()
        model.summary()
        model.compile()
        model.fit(x_train, y_train, epochs=3)
        model.save('model/mnist')

        self.y_train = y_train

    def load(self):
        # Loading the model 
        self.model = tf.keras.models.load_model('model/mnist')

    
    def predict(self, imageArray):
        _, self.y_train, _, _ = load_data()
        unique_outputs = np.unique(self.y_train)
        preds = self.model.predict(imageArray)
        prediction = unique_outputs[preds.argmax()]
        return prediction

    def preprocess(self, imageArray):
        # Inverting the image (I figured out the range was form 0-1)
        imageArray = (1 - imageArray)

        # Resizing the image into (28, 28)
        imageArray = cv.resize(imageArray, (28, 28))

        # plt.imshow(imageArray, plt.cm.binary)
        # plt.show()

        # Adding dimension to the imageArray 
        imageArray = imageArray[None,:,:, np.newaxis]

        return imageArray
    
        

if __name__ == "__main__":
    Main().run()