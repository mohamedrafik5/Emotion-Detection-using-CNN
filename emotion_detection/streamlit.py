import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageOps, ImageFont
import io

st.set_page_config(page_title="😃 Emotion Detection App", page_icon="👌", layout="centered")
st.title("😃 Emotion Detection from Images")
st.markdown("Upload an image and detect facial emotions using your FastAPI backend.")

API_URL = "http://localhost:8000/predict_emotion/"

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

def to_api_bytes(pil_img: Image.Image, format_hint: str = "JPEG") -> bytes:
    """Return bytes of the *EXIF-corrected* image for the API."""
    buf = io.BytesIO()
    fmt = "PNG" if format_hint.lower() == "png" else "JPEG"
    pil_img.save(buf, format=fmt, quality=95)
    return buf.getvalue()

if uploaded_file is not None:
    # Open and correct EXIF rotation
    raw = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(raw)

    st.image(image, caption="Uploaded Image (EXIF-corrected)", use_container_width=True)

    if st.button("🔍 Detect Emotions"):
        with st.spinner("Analyzing image..."):
            try:
                api_bytes = to_api_bytes(image, uploaded_file.type.split("/")[-1] if uploaded_file.type else "jpeg")
                response = requests.post(API_URL, files={"file": api_bytes})

                if response.status_code == 200:
                    result = response.json()
                    boxes = result.get("bounding_boxes", [])
                    labels = result.get("labels", [])

                    draw = ImageDraw.Draw(image)
                    W, H = image.size

                    # ✅ assume your API returns (x, y, w, h)
                    for (x, y, w, h), label in zip(boxes, labels):
                        try:
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)

                            # Clamp to image bounds
                            x1 = max(0, min(x1, W - 1))
                            y1 = max(0, min(y1, H - 1))
                            x2 = max(0, min(x2, W - 1))
                            y2 = max(0, min(y2, H - 1))

                            # Draw bounding box (thicker for visibility)
                            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=6)

                            # ✅ Adaptive font size (max 60px, min 24px)
                            font_size = max(24, min(60, image.width // 25))
                            try:
                                # Try common fonts across OS
                                font = ImageFont.truetype("arial.ttf", font_size)
                            except:
                                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)

                            # Draw label text (larger and blue)
                            draw.text((x1 + 8, max(y1 - font_size - 5, 0)), label, fill="blue", font=font)
                        except Exception as e:
                            st.warning(f"⚠️ Skipping invalid box {box}: {e}")

                    st.image(image, caption="Detected Emotions", use_container_width=True)
                    with st.expander("📊 Raw Prediction Data"):
                        st.json(result)
                    st.success("✅ Emotion detection complete!")
                else:
                    st.error(f"❌ API request failed (status {response.status_code})")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("⬆️ Please upload an image to start.")






# streamlit run emotion_detection\streamlit.py
