import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import io

# 1. App Configuration
st.set_page_config(
    page_title="AI Super-Resolution Upscaler",
    page_icon="✨",
    layout="centered"
)

# 2. Optimized Pre-trained Model Caching
@st.cache_resource
def load_esrgan_model():
    # Utilizing the migrated, active Kaggle hosting link for the ESRGAN model
    model_url = "https://www.kaggle.com/models/google/esrgan/TensorFlow2/esrgan-tf2/1"
    return hub.load(model_url)

with st.spinner("Initializing ESRGAN Deep Model... Please wait."):
    esrgan_model = load_esrgan_model()

# Helper function to prepare images for the GAN architecture
def preprocess_image(uploaded_image):
    # If image has an alpha channel (PNG), convert it to RGB
    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")
        
    # Safeguard: Resize massive images slightly so the container does not run out of RAM
    if max(uploaded_image.size) > 800:
        uploaded_image.thumbnail((800, 800))
        
    img_array = np.array(uploaded_image)
    # Convert matrix array into a float32 Tensor and append batch dimension
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return tf.expand_dims(img_tensor, axis=0)

# Helper function to transform native tensors back to standard images
def postprocess_tensor(tensor):
    # Remove batch wrapper, clip output values safely between 0 and 255
    out_tensor = tf.squeeze(tensor, axis=0)
    out_tensor = tf.clip_by_value(out_tensor, 0, 255)
    out_tensor = tf.round(out_tensor)
    
    # Cast to integer array matrix
    out_array = out_tensor.numpy().astype(np.uint8)
    return Image.fromarray(out_array)

# 3. Frontend Layout Setup
st.title("✨ AI Super-Resolution Studio")
st.markdown("Upload any low-resolution, compressed, or blurry image. The **Generative Adversarial Network (ESRGAN)** will reconstruct missing pixels and output a crisp x4 upscaled photo.")

file_slot = st.file_uploader("Upload low-res source image...", type=["jpg", "jpeg", "png"])

if file_slot is not None:
    source_img = Image.open(file_slot)
    
    # Structural column grid representation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(source_img, use_container_width=True)
        st.caption(f"Dimensions: {source_img.size[0]}x{source_img.size[1]} px")
        
    # Trigger AI processing loop
    if st.button("🌟 Run Generative Upscaling"):
        with st.spinner("GAN is running feature pass estimations..."):
            try:
                # Format pipeline tensors
                input_tensor = preprocess_image(source_img)
                
                # Execute pre-trained model mapping pass
                raw_prediction = esrgan_model(input_tensor)
                
                # Format back into an image map
                hd_image = postprocess_tensor(raw_prediction)
                
                with col2:
                    st.subheader("AI Enhanced Image")
                    st.image(hd_image, use_container_width=True)
                    st.caption(f"Dimensions: {hd_image.size[0]}x{hd_image.size[1]} px (4x Spatial Increase)")
                
                # 4. In-Memory Download Buffer File generation
                img_buffer = io.BytesIO()
                hd_image.save(img_buffer, format="PNG")
                processed_bytes = img_buffer.getvalue()
                
                st.success("Upscaling successfully executed!")
                st.download_button(
                    label="📥 Download HD Image",
                    data=processed_bytes,
                    file_name="upscaled_hd_result.png",
                    mime="image/png"
                )
            except Exception as system_error:
                st.error(f"An execution interrupt occurred during process generation: {str(system_error)}")
