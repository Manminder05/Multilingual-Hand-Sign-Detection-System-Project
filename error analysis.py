import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

# Load the trained model
model = load_model('D:\A to Z sign detection language\model_isl.h5')

# Load the test data
try:
    X_test = np.load('D:\A to Z sign detection language\X_test_isl.npy')
    y_test = np.load('D:\A to Z sign detection language\y_test_isl.npy')
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

def perform_error_analysis(model, X_test, y_test):
    # Predict probabilities for each class
    y_pred_prob = model.predict(X_test)

    # Convert probabilities to class labels
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Convert one-hot encoded labels to integers
    y_true = np.argmax(y_test, axis=1)

    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(26), yticklabels=range(26))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Save the classification report to a file
    report = classification_report(y_true, y_pred, output_dict=True)
    with open('classification_report.txt', 'w') as f:
        for key, value in report.items():
            f.write(f'{key} : {value}\n')

# Use the function to perform error analysis
perform_error_analysis(model, X_test, y_test)
