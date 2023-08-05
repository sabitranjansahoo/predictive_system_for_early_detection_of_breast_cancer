import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import streamlit as st

# Load the trained model
model = load_model("C:\\Users\\sabit\\Downloads\\breast_cancer_model (1).h5")

st.title('Breast Cancer Diagnosis System')
uploaded_file = st.file_uploader("Upload Mammogram", type=['jpg', 'jpeg', 'pgm', 'png'])
st.markdown("""---""")
col1,col2 = st.columns(2)
name = col1.text_input("*Name*")
age = col2.text_input("*Age*")
gender = col1.selectbox("*Gender*",("Male","Female"))
phone = col2.text_input("*Phone No.*")
date = col1.date_input("*Date*")
ref = col2.text_input("*Referred By*")

def predict_diagnosis(image_path,img2,segmented_image,contour_img):
    # Load the image to predict
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (64, 64))

    # Preprocess the image
    rows, cols = img.shape
    X = []
    for angle in range(360):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rotated = cv2.warpAffine(img, M, (cols, rows))
        X.append(img_rotated)
    X = np.array(X)
    a, b, c = X.shape
    X = np.reshape(X, (a, b, c, 1))

    # Predict the class of the image
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    col1.image([img2], caption=['[Input Image]'])
    if(col1.button("Submit")):
        col2.image([segmented_image], caption=['[Segmented Image]'])

        # Print the predicted class
        with st.empty():
                col2.write("      ")
                col2.write("      ")
                col2.write("      ")
                col2.write("      ")
                col2.write("      ")
        col1.write("**Report Section:**")
        col1.caption("Name:")
        col2.write(name)
        col1.caption("Age:")
        col2.write(age)
        col1.caption("Gender:")
        col2.write(gender)
        col1.caption("Phone Number:")
        col2.write(phone)
        col1.caption("Date:")
        col2.write(date)
        col1.caption("Referred By:")
        col2.write(ref)
        if np.all(y_pred == 0):
            col1.write('Remark: The image is :green[Normal].')
        elif np.any(y_pred == 2):
            col1.write('Remark: The image is :red[**Malignant**].')
        else:
            col1.markdown('Remark: The image is :red[Benign].')

def segment_and_diagnose(image_path):
    input_image = cv2.imread(image_path)
    img2 = cv2.resize(input_image, (256, 256))

    # Convert the input image to grayscale
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur with a kernel size of (3, 3) to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Apply adaptive thresholding using the "adaptive Gaussian" method
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 2)

    # Apply erosion followed by dilation to remove small noise and connect broken contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a black image
    contour_img = np.zeros(img2.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)

    # Apply segmentation to the input image
    segmented_image = cv2.bitwise_and(img2, img2, mask=contour_img)
    # Display the input image and segmented image
    # display_images(img2,segmented_image,contour_img)

    # Save the uploaded image to a temporary file
    temp_file = 'temp.jpg'
    image.save(temp_file)

    # Predict the diagnosis
    diagnosis = predict_diagnosis(temp_file,img2,segmented_image,contour_img)
    # st.write(diagnosis)

if uploaded_file is not None:
    # Save the uploaded image
    image = Image.open(uploaded_file)
    image_path = 'uploaded_image.jpg'
    image.save(image_path)
    # Segment and diagnose the uploaded image
    segment_and_diagnose(image_path)