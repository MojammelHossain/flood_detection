import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()

def resize_image(image):
    # Resize the image to (512, 512)
    resized_image = image.resize((512, 512))
    return resized_image

def normalize_image(image):
    # Normalize the image by dividing pixel values by 255
    normalized_image = np.array(image) / 255.0
    return normalized_image

def load_model():
    # Register the custom objects
    # with tf.keras.utils.custom_object_scope({'AdamW': tf.keras.optimizers.AdamW}):
        # Load the semantic segmentation model
    model = tf.keras.models.load_model("D:/MsCourse/AI/project/flood_detection/tensorflow/model/linknet/linknet_ex_patchify_epochs_2000_28-May-23.hdf5",compile=False)
    return model


def predict_segmentation(model, image):
    # Preprocess the image for model input
    input_image = np.expand_dims(image, axis=0)
    # Perform prediction
    predicted_mask = model.predict(input_image)
    # Convert predicted mask to binary
    pred_mask = np.argmax(predicted_mask,axis=3)
    return pred_mask

def main():
    st.title("Flood Detection")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an flood image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Resize the image
        resized_image = resize_image(original_image)

        # Normalize the image
        normalized_image = normalize_image(resized_image)

        # Load the segmentation model
        model = load_model()

        # Perform segmentation prediction
        predicted_mask = predict_segmentation(model, normalized_image)
        fig, ax = plt.subplots()
        plt.imshow(predicted_mask[0])
        plt.axis('off')  # Optional: Remove axis labels
        plt.show()

        # Display the Matplotlib figure in Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()


