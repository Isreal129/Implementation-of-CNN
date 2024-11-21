# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
Develop a model that can classify images of handwritten digits (0-9) from the MNIST dataset with high accuracy. The model should use a convolutional 
neural network architecture and be optimized using early stopping to avoid overfitting.

## Neural Network Model

![Screenshot 2024-09-23 105356](https://github.com/user-attachments/assets/eaf9841d-2555-428e-a642-58b245b237c8)


## DESIGN STEPS

### STEP 1:
Import the necessary libraries and  load the dataset

### STEP 2:
Reshape and normalize the data 

### STEP 3:
Create the EarlyStoppingCallback function 

### STEP 4:
Create the convulational model and compile the model

### STEP 5:
Train the model

## PROGRAM

### Name: Isreal Moses B
### Register Number: 212221040060


```python
import os
import numpy as np
import tensorflow as tf
# Provide path to get the full path
data_path = os.path.join( "mnist.npz")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
print('Name:Isreal Moses           Register Number:212221040060   \n')
import numpy as np

# GRADED FUNCTION: reshape_and_normalize
def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """
    
    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)  # Add the channel dimension
    
    # Normalize pixel values
    images = images / 255.0  # Divide by the maximum pixel value (255 for images)

    ### END CODE HERE ###

    return images
# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

# GRADED CLASS: EarlyStoppingCallback
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.98
        if logs.get('accuracy') >= 0.995:
            # Stop training once the above condition is met
            self.model.stop_training = True
            print("\nReached 99.5% accuracy so cancelling training!") 
            
# GRADED FUNCTION: convolutional_model
def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    
    
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
        tf.keras.layers.Flatten(),  # Flatten the output
        tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with 64 neurons
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
    ]) 

   

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
          
    return model
model = convolutional_model()
# Train your model (this can take up to 5 minutes)
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])
```

## OUTPUT

### Reshape and Normalize output
![Screenshot 2024-09-16 111126](https://github.com/user-attachments/assets/f1096f05-1ce8-4051-ad8b-ea432f895ad5)



### Training the model output
![Screenshot 2024-09-16 111140](https://github.com/user-attachments/assets/9f71ea39-5ff0-4844-a330-cb25141deb55)




## RESULT
Hence a convolutional deep neural network for digit classification was successfully developed.
