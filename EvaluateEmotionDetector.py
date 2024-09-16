import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the entire model from the saved .h5 file
model_path = 'emotion_model.h5'  # Path to your .h5 file
emotion_model = load_model(model_path)
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)  # Ensure that images are not shuffled for consistent evaluation

# Perform predictions on test data
predictions = emotion_model.predict(test_generator)

# Get true labels and predicted labels
true_labels = test_generator.classes
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
print("-----------------------------------------------------------------")
c_matrix = confusion_matrix(true_labels, predicted_labels)
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=list(emotion_dict.values()))
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(true_labels, predicted_labels, target_names=list(emotion_dict.values())))
