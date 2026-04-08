import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

st.title("AI Infrastructure Crack Detection System")
st.write("Upload a pipeline or concrete surface image to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for faster mobile processing
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    gray = image_resized.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray_enhanced = enhancer.enhance(2.0)

    # Convert to numpy array
    img_array = np.array(gray_enhanced, dtype=float)

    # Simple smoothing using 3x3 average filter (vectorized)
    kernel_size = 3
    pad = kernel_size // 2
    padded = np.pad(img_array, pad, mode='edge')
    smoothed = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

    # Subtract smoothed image to get local contrast
    edges = img_array - smoothed
    edges[edges < 0] = 0

    # Threshold based on percentile
    threshold = np.percentile(edges, 95)
    edge_binary = edges > threshold

    crack_pixels = np.sum(edge_binary)
    crack_pixel_threshold = 500  # smaller for subtle cracks

    draw = ImageDraw.Draw(image_resized)

    if crack_pixels > crack_pixel_threshold:
        height, width = edge_binary.shape
        for y in range(height):
            x_positions = np.where(edge_binary[y, :])[0]
            if len(x_positions) > 0:
                start = x_positions[0]
                prev = x_positions[0]
                for x in x_positions[1:]:
                    if x == prev + 1:
                        prev = x
                    else:
                        draw.line((start, y, prev, y), fill="red", width=2)
                        start = x
                        prev = x
                draw.line((start, y, prev, y), fill="red", width=2)

        severity = "Low"
        if crack_pixels > 3000:
            severity = "Moderate"
        if crack_pixels > 10000:
            severity = "Severe"

        st.subheader("Detected Crack Overlay")
        st.image(image_resized, use_container_width=True)
        st.subheader("Inspection Result")
        st.success("Surface Crack Detected")
        st.write("Severity Level:", severity)
        st.write("Crack Pixel Count:", crack_pixels)
    else:
        st.subheader("Inspection Result")
        st.info("No Crack Detected on Surface")
        st.write("Crack Pixel Count:", crack_pixels)

    # Download full-size processed image
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # full resolution
    buffer.seek(0)
    st.download_button(
        label="Download Full-Resolution Processed Image",
        data=buffer,
        file_name="crack_detection_result.png",
        mime="image/png"
    )