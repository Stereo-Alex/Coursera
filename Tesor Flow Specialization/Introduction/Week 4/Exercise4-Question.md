
Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.


```python
import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()
```


```python
# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            # Change 'accuracy' to 'acc' for the Coursera autograder!
            if(logs.get('acc')>DESIRED_ACCURACY): 
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape= (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation= 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
        
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss = 'binary_crossentropy',
                 optimizer = 'Adam',
                 metrics = ['accuracy'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)
    
    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory('/tmp/h-or-s', target_size = (150, 150),
                                                      batch_size = 20, class_mode = 'binary'
    
    )

    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(train_generator, steps_per_epoch = 4,
                                 epochs = 20,
                                 callbacks = [callbacks])
    # model fitting
    return history.history['acc'][-1]
```


```python
# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()
```

    W0428 12:52:49.659329 140039431321408 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where


    Found 80 images belonging to 2 classes.
    Epoch 1/20
    4/4 [==============================] - 4s 1s/step - loss: 1.3429 - acc: 0.5125
    Epoch 2/20
    4/4 [==============================] - 0s 74ms/step - loss: 0.7310 - acc: 0.5000
    Epoch 3/20
    4/4 [==============================] - 0s 75ms/step - loss: 0.6817 - acc: 0.7375
    Epoch 4/20
    4/4 [==============================] - 0s 77ms/step - loss: 0.6354 - acc: 0.7000
    Epoch 5/20
    4/4 [==============================] - 0s 79ms/step - loss: 0.4927 - acc: 0.9000
    Epoch 6/20
    4/4 [==============================] - 0s 76ms/step - loss: 0.3303 - acc: 0.9375
    Epoch 7/20
    4/4 [==============================] - 0s 75ms/step - loss: 0.1917 - acc: 0.9375
    Epoch 8/20
    4/4 [==============================] - 0s 74ms/step - loss: 0.2009 - acc: 0.9125
    Epoch 9/20
    4/4 [==============================] - 0s 75ms/step - loss: 0.2893 - acc: 0.9125
    Epoch 10/20
    4/4 [==============================] - 0s 74ms/step - loss: 0.2042 - acc: 0.8625
    Epoch 11/20
    4/4 [==============================] - 0s 58ms/step - loss: 0.2122 - acc: 0.9125
    Epoch 12/20
    4/4 [==============================] - 0s 75ms/step - loss: 0.1730 - acc: 0.9125
    Epoch 13/20
    4/4 [==============================] - 0s 76ms/step - loss: 0.1663 - acc: 0.9625
    Epoch 14/20
    4/4 [==============================] - 0s 73ms/step - loss: 0.1890 - acc: 0.9125
    Epoch 15/20
    3/4 [=====================>........] - ETA: 0s - loss: 0.1591 - acc: 1.0000
    Reached 99.9% accuracy so cancelling training!
    4/4 [==============================] - 0s 58ms/step - loss: 0.1509 - acc: 1.0000





    1.0




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
