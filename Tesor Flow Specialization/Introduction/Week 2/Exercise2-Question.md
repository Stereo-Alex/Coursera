
## Exercise 2
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:
1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class

I've started the code for you below -- how would you finish it? 


```python
import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"
```


```python
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
   

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    # We normalize 
    X_train, X_test = x_train / 255.0, x_test / 255.0 
    
    callbacks = myCallback()

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
        
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(X_train, y_train, epochs=10, callbacks=[callbacks])
    
    # model fitting
    return history.epoch, history.history['acc'][-1]
```


```python
train_mnist()
```

    WARNING: Logging before flag parsing goes to stderr.
    W0427 16:07:08.586483 140572864296768 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor


    Epoch 1/10
    60000/60000 [==============================] - 9s 155us/sample - loss: 0.2018 - acc: 0.9411
    Epoch 2/10
    60000/60000 [==============================] - 8s 140us/sample - loss: 0.0798 - acc: 0.9758
    Epoch 3/10
    60000/60000 [==============================] - 9s 145us/sample - loss: 0.0526 - acc: 0.9832
    Epoch 4/10
    60000/60000 [==============================] - 9s 143us/sample - loss: 0.0373 - acc: 0.9883
    Epoch 5/10
    59712/60000 [============================>.] - ETA: 0s - loss: 0.0277 - acc: 0.9907
    Reached 99% accuracy so cancelling training!
    60000/60000 [==============================] - 9s 152us/sample - loss: 0.0276 - acc: 0.9907





    ([0, 1, 2, 3, 4], 0.9906833)




```python
# Now click the 'Submit Assignment' button above.
# Once that is complete, please run the following two cells to save your work and close the notebook
```


```javascript
%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();
```


```javascript
%%javascript
IPython.notebook.session.delete();
window.onbeforeunload = null
setTimeout(function() { window.close(); }, 1000);
```
