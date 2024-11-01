import os
from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage import measure
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# File paths
Train_img_path = "C:\\Users\\Miskat\\Desktop\\Project\\Python\\Bank_Note_GUI\\bangla_banknote_v2\\Training"
categories = ['1', '10', '100', '1000', '2', '20', '5', '50', '500']


# Predict function
def predict_banknote():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    original = Image.open(file_path)

    if file_path:
        preprocessed_input = preprocess_image(file_path)
        flattened_input = preprocessed_input.reshape(1, -1)
        pre_label = svm_classifier.predict(flattened_input)[0]
        pre_category = categories[pre_label]

        
        load_and_display_image(original)
        result_label.config(text=f"Predicted Banknote Category: {pre_category}")

# Load and display image function
def load_and_display_image(image):
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


# Preprocessing function
def preprocess_image(image_path):
        # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply median blur for denoising
    denoised_image = cv2.medianBlur(gray_image, 3)
    # Apply Sobel filter
    sobel_x = cv2.Sobel(denoised_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    histogram, _ = np.histogram(sobel_combined, bins=256, range=(0, 256))
    # Find a valley or peak in the histogram for thresholding
    # You can adjust this threshold selection method based on your images
    threshold_value = np.argmax(histogram[50:]) + 50
    # Apply binary thresholding
    _, binary_image = cv2.threshold(sobel_combined, threshold_value, 255, cv2.THRESH_BINARY)
    # Find contours using skimage
    contours = measure.find_contours(binary_image, 0.8)
    # Draw contours on the original image
    contour_image = np.copy(image)
    for contour in contours:
        contour = np.flip(contour, axis=1)  # skimage uses (row, column) coordinates
        cv2.polylines(contour_image, [contour.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    return contour_image

# Preprocess Training Images and store them in a numpy array
training_images = []
training_labels = []
for category in categories:
    category_path = os.path.join(Train_img_path, category)

    for image_filename in os.listdir(category_path):
        image_path = os.path.join(category_path, image_filename)

         # Check if the image format is supported (e.g., 'jpg', 'jpeg', 'png')
        supported_formats = ('.jpg', '.jpeg', '.png')
        if image_path.lower().endswith(supported_formats):

            # Apply preprocessing to the image
            preprocessed_image = preprocess_image(image_path)

            # Append the preprocessed image and its label to the arrays
            training_images.append(preprocessed_image)
            training_labels.append(categories.index(category))

# Convert lists to numpy arrays
training_images = np.array(training_images)
training_labels = np.array(training_labels)
# Feature vectors (flattened image features) and labels
X = training_images.reshape(len(training_images), -1)
y = training_labels
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Load SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
# Train the SVM classifier on the training data
svm_classifier.fit(X_train, y_train)
# Predict the classes on the test data
y_pred = svm_classifier.predict(X_test)








root = Tk()
root.title("Banknote Detection")

root.geometry("1350x690+0+0")
root.resizable(False, False)
img1 = Image.open("C:\\Users\\Miskat\\Desktop\\Project\\Python\\Bank_Note_GUI\\Image\\X edu.jpg")
img1 = img1.resize((1530, 790), Image.ANTIALIAS)
photoimg1 = ImageTk.PhotoImage(img1)

#Create a label to display the background image
bg_img = Label(root, image=photoimg1)
bg_img.place(x=0, y=0, relwidth=1, relheight=1)
# Create a custom button style using attributes
imgB = Image.open("C:\\Users\\Miskat\\Desktop\\Project\\Python\\Bank_Note_GUI\\Image\\Help.jpg")
imgB = imgB.resize((220, 220), Image.ANTIALIAS)
photoimgB = ImageTk.PhotoImage(imgB)

B = Button(bg_img, image=photoimgB, cursor="hand2", command=predict_banknote)
B.place(x=550,y=400, width=220, height=220)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Helvetica", 20, "bold"))
result_label.pack()

root.mainloop()