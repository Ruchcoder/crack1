import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import io

# ---------------- TITLE ---------------- #
st.title("Infrastructural Defect Detection System")
st.write("Detects structural cracks and corrosion (rust) on infrastructure surfaces.")

# ---------------- UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for speed
    max_width = 600
    w_percent = max_width / float(image.width)
    new_height = int(float(image.height) * w_percent)
    image_resized = image.resize((max_width, new_height))

    # ---------------- CRACK DETECTION ---------------- #
    gray = image_resized.convert("L")

    enhancer = ImageEnhance.Contrast(gray)
    gray_enhanced = enhancer.enhance(2.0)

    img_array = np.array(gray_enhanced, dtype=float)

    padded = np.pad(img_array, 1, mode='edge')
    smoothed = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

    edges = img_array - smoothed
    edges[edges < 0] = 0

    threshold = np.percentile(edges, 97)
    edge_binary = edges > threshold

    crack_pixels = np.sum(edge_binary)
    crack_threshold = 500

    # ---------------- RUST DETECTION ---------------- #
    img_np = np.array(image_resized)

    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    # Improved rust detection condition
    rust_mask = (R > 120) & (G > 60) & (B < 100)
    rust_pixels = np.sum(rust_mask)
    rust_threshold = 1000

    # ---------------- DRAW RESULTS ---------------- #
    draw = ImageDraw.Draw(image_resized)

    height, width = edge_binary.shape

    # Draw crack lines (RED)
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

    # Draw rust areas (YELLOW overlay)
    for y in range(height):
        x_positions = np.where(rust_mask[y, :])[0]
        for x in x_positions:
            draw.point((x, y), fill="yellow")

    # ---------------- SEVERITY ---------------- #
    # Crack severity
    crack_severity = "None"
    if crack_pixels > 500:
        crack_severity = "Low"
    if crack_pixels > 3000:
        crack_severity = "Moderate"
    if crack_pixels > 10000:
        crack_severity = "High"

    # Rust severity
    rust_severity = "None"
    if rust_pixels > 1000:
        rust_severity = "Low"
    if rust_pixels > 5000:
        rust_severity = "Moderate"
    if rust_pixels > 15000:
        rust_severity = "High"

    # ---------------- DISPLAY ---------------- #
    st.subheader("Detection Overlay")
    st.image(image_resized, use_container_width=True)

    # ---------------- REPORT ---------------- #
    st.subheader("Inspection Report")

    # Crack result
    if crack_pixels > crack_threshold:
        st.success("Crack/Openings Detected")
        st.write(f"Crack Severity Level: {crack_severity}")
    else:
        st.info("No Significant Crack/Openings Detected")

    st.write(f"Crack Pixel Count: {crack_pixels}")

    # Rust result
    if rust_pixels > rust_threshold:
        st.warning("Rust/Corrosion Detected")
        st.write(f"Rust Severity Level: {rust_severity}")
    else:
        st.info("No Significant Rust/Corrosion Detected")

    st.write(f"Rust Pixel Count: {rust_pixels}")

    # Recommendation
    st.markdown("### Recommended Action")
    if crack_pixels > crack_threshold or rust_pixels > rust_threshold:
        st.write("Immediate inspection and maintenance required.")
    else:
        st.write("Routine monitoring recommended.")

    # ---------------- DOWNLOAD ---------------- #
    buffer = io.BytesIO()
    image_resized.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="inspection_result.png",
        mime="image/png"
    )
