import tensorflow as tf 
import matplotlib.pyplot as plt 

def load_data(show=False):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizing the data 
    x_train = x_train/255
    x_test = x_test/255

    if show:
        # Showing the data 
        plt.figure(figsize=(10, 10))
        for i in range(10):
            plt.subplot(4, 3, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train[i])
            plt.xlabel(y_train[i])
        plt.show()

    # Covolution layer need's 3 dimension to procedure
    x_train = x_train[:,:,:,tf.newaxis]
    x_test = x_test[:,:,:,tf.newaxis]

    # print(x_test.shape)

    return x_train, y_train, x_test, y_test  


load_data()