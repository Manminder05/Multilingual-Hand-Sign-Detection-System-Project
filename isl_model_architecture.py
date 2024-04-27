import tensorflow
from keras.models import load_model
from keras.utils import plot_model

# Load the saved model
model = load_model('D:\A to Z sign detection language\model_asl.h5')

# Plot the model architecture
plot_model(model, to_file='asl_model_architecture.png', show_shapes=True, show_layer_names=True)
