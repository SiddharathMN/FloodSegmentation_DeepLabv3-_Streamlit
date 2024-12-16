import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, AveragePooling2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNetV2



# Define the ConvBlock custom layer
class ConvBlock(Layer):
    def __init__(self, filters: int = 256, kernel_size: int = 3, dilation_rate: int = 1, **kwargs) -> None:
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, padding='same', strides=1, dilation_rate=dilation_rate, activation=None, use_bias=False),
            BatchNormalization(),
            ReLU()
        ])

    def call(self, X):
        return self.net(X)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate
        }

# Register the custom layer
get_custom_objects().update({'ConvBlock': ConvBlock})


def AtrousSpatialPyramidPooling(X):
    _, H, W, C = X.shape
    image_pool = AveragePooling2D(pool_size=(H, W), name="ASPP-ImagePool-AP")(X)
    image_pool = ConvBlock(kernel_size=1, name="ASPP-ImagePool-CB")(image_pool)
    image_pool = UpSampling2D(size=(H // image_pool.shape[1], W // image_pool.shape[2]), name="ASPP-ImagePool-US")(image_pool)
    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-Conv1")(X)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-Conv6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-Conv12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-Conv18")(X)
    combined = Concatenate(name="ASPP-Concatenate")([image_pool, conv_1, conv_6, conv_12, conv_18])
    output = ConvBlock(kernel_size=1, name="ASPP-Out")(combined)
    return output


# Model definition (same as before) with adjustments for path lengths
IMAGE_SIZE = 256
MODEL_NAME = "DeepLab-FloodArea-MobileNetV2"
inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="InputLayer")
backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
DCNN = backbone.get_layer('block_13_expand_relu').output
ASPP = AtrousSpatialPyramidPooling(DCNN)
ASPP = UpSampling2D(size=(IMAGE_SIZE // 4 // ASPP.shape[1], IMAGE_SIZE // 4 // ASPP.shape[2]), name="ASPP-UpSample-Out")(ASPP)
LLF = backbone.get_layer('block_3_expand_relu').output
LLF = ConvBlock(filters=48, kernel_size=1, name="LLF-ConvBlock")(LLF)
concat = Concatenate(axis=-1, name="Concat-LLF-HLF")([LLF, ASPP])
y = ConvBlock(name="TopConvBlock1")(concat)
y = ConvBlock(name="TopConvBlock2")(y)
y = UpSampling2D(size=(IMAGE_SIZE // y.shape[1], IMAGE_SIZE // y.shape[2]), name="Feature-UpSample")(y)
output = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', use_bias=False, name="OutputLayer")(y)
DeepLabV3 = Model(inputs, output, name=MODEL_NAME)





@st.cache_resource  # Cache the model loading
def load_deeplab_model(model_path: str):
    model = tf.keras.models.load_model(model_path, custom_objects={'ConvBlock': ConvBlock})
    return model


def preprocess_image(image):
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    return image_input




def display_segmentation(image, prediction):

    mask = np.where(prediction > 0.5, 1, 0) # Adjust threshold as needed
    mask = np.squeeze(mask, axis=0)
    mask = np.squeeze(mask, axis=-1)  # Remove channel dimension if it exists

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray') # Use 'gray' colormap for binary mask
    ax[1].set_title('Segmented Output')
    ax[1].axis('off')

    st.pyplot(fig)



# Streamlit UI
st.title("✨ DeepLabV3 Segmentation 🏜")

image_path = st.file_uploader("Upload Image 🚀", type=["png", "jpg", "bmp", "jpeg"])
if image_path is not None:
    with st.spinner("Working.. 💫"):

        image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        model_path = "C:\\Users\\SIDDHARATH\\Segment-Anything-Streamlit\\DeepLabV3_model.h5" # Replace with your actual model path
        model = load_deeplab_model(model_path)

        image_input = preprocess_image(image)
        prediction = model.predict(image_input)

        display_segmentation(image, prediction)

else:
    st.warning('⚠ Please upload your Image! 😯')

st.markdown("<br><hr><center>Made with ❤️ by <strong>Your Name</strong></center><hr>", unsafe_allow_html=True)