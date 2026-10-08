"""Run with: streamlit run web/app.py"""
import hashlib
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from core import analyse, csv_bytes, export_zip, segment

st.set_page_config(page_title="QGeo Web", layout="wide")
st.title("QGeo Web")
st.caption("Research prototype · Image segmentation and shape measurements")
st.write("Upload a rock photograph, select the region to analyse, and review colour classes before measuring features.")
uploaded = st.file_uploader("Rock image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded is None:
    st.info("Upload an image to begin. Images are processed in the application session.")
    st.stop()

data = uploaded.getvalue()
digest = hashlib.sha256(data).hexdigest()
if st.session_state.get("source_digest") != digest:
    for key in ("masks", "names", "crop_key"):
        st.session_state.pop(key, None)
    st.session_state.source_digest = digest

try:
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if pil.width * pil.height > 16_000_000:
        st.error("This prototype accepts images up to 16 million pixels. Reduce the image size and upload again.")
        st.stop()
    original = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
except (OSError, ValueError, Image.DecompressionBombError):
    st.error("The image could not be decoded.")
    st.stop()

h, w = original.shape[:2]
st.subheader("1. Select the analysis region")
left, right = st.columns([2, 1])
with right:
    st.caption(f"Original resolution: {w} × {h} pixels. Analysis uses original pixels.")
    x0, x1 = st.slider("Horizontal crop (pixels)", 0, w, (0, w))
    y0, y1 = st.slider("Vertical crop (pixels)", 0, h, (0, h))
    st.caption("Exclude rulers, background and marker lines. This version uses a rectangular crop.")
with left:
    preview = original.copy()
    cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 255), max(1, w // 500))
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width="stretch")
if x1 <= x0 or y1 <= y0:
    st.warning("Select a crop with positive width and height.")
    st.stop()
crop = original[y0:y1, x0:x1].copy()
crop_key = (digest, x0, x1, y0, y1)
if st.session_state.get("crop_key") != crop_key:
    st.session_state.crop_key = crop_key
    st.session_state.pop("masks", None)
    st.session_state.pop("names", None)

st.subheader("2. Segment and review colour classes")
count = st.number_input("Initial colour classes", 2, 12, 3)
if st.button("Segment crop", type="primary"):
    try:
        with st.spinner("Segmenting original crop pixels…"):
            st.session_state.masks = segment(crop, int(count))
            st.session_state.names = [f"Class {i+1}" for i in range(count)]
            st.session_state.initial_clusters = int(count)
    except ValueError as error:
        st.error(str(error))
if "masks" not in st.session_state:
    st.stop()
masks, names = st.session_state.masks, st.session_state.names
st.caption("Colour classes are not automatic mineral identifications. Names should reflect your interpretation.")
for i, (mask, name) in enumerate(zip(masks, names)):
    with st.expander(f"{name} · {100 * mask.mean():.2f}% of crop", expanded=len(masks) <= 4):
        a, b = st.columns([2, 1])
        a.image(cv2.cvtColor(np.where(mask[..., None], crop, 0), cv2.COLOR_BGR2RGB), width="stretch")
        names[i] = b.text_input("Class name", value=name, key=f"name_{crop_key}_{i}_{name}")
        b.caption("Black in this preview indicates excluded pixels; exported binary masks preserve membership.")

selected = st.multiselect("Classes to adjust", list(range(len(masks))), format_func=lambda i: f"{i+1}: {names[i]}")
a, b, c = st.columns(3)
if a.button("Merge selected", disabled=len(selected) < 2):
    merged = np.logical_or.reduce([masks[i] for i in selected])
    remaining = [i for i in range(len(masks)) if i not in selected]
    st.session_state.masks = [masks[i] for i in remaining] + [merged]
    st.session_state.names = [names[i] for i in remaining] + ["Merged class"]
    st.rerun()
if b.button("Remove selected", disabled=not selected or len(selected) == len(masks)):
    remaining = [i for i in range(len(masks)) if i not in selected]
    st.session_state.masks = [masks[i] for i in remaining]
    st.session_state.names = [names[i] for i in remaining]
    st.rerun()
split_count = c.number_input("Classes after splitting", 2, 12, 2)
if c.button("Split selected class", disabled=len(selected) != 1):
    index = selected[0]
    try:
        children = segment(crop, int(split_count), masks[index])
        st.session_state.masks = masks[:index] + children + masks[index+1:]
        st.session_state.names = names[:index] + [f"{names[index]} part {j+1}" for j in range(split_count)] + names[index+1:]
        st.rerun()
    except ValueError as error:
        st.error(str(error))

st.subheader("3. Calibrate and measure")
st.write("Enter the physical height of the selected crop, not the whole sample. Curvature and perspective can affect measurements; unwrapping is not included in this prototype.")
physical_height = st.number_input("Selected crop height (mm)", min_value=0.001, value=10.0, format="%.3f")
confirmed = st.checkbox("I have checked the physical height of this crop")
minimum_area = st.number_input("Minimum contour area (pixels²)", min_value=0.0, value=0.0)
st.caption("Object detection uses external contours from the desktop code. Contour areas fill internal holes. Aspect ratio uses an axis-aligned bounding box and can change with orientation. Area percentages include all class pixels, regardless of the contour filter.")
if not confirmed:
    st.info("Confirm calibration to enable measurements and exports.")
    st.stop()
scale = physical_height / crop.shape[0]
rows, summaries = analyse(masks, names, scale, minimum_area)
st.dataframe(summaries, width="stretch", hide_index=True)
st.caption(f"Scale: {scale:.6f} mm per pixel. Removed classes remain excluded; percentages refer to the entire crop.")
if rows:
    st.dataframe(rows[:1000], width="stretch", hide_index=True)
    if len(rows) > 1000:
        st.caption("Preview shows the first 1,000 objects. Downloads include all objects.")
else:
    st.info("No positive-area contours meet the selected filter.")
metadata = dict(source=uploaded.name, original_size=[w,h], crop=[x0,y0,x1,y1],
                physical_crop_height_mm=physical_height, mm_per_pixel=scale,
                minimum_contour_area_px=minimum_area, class_names=names,
                seed=0, initial_clusters=st.session_state.initial_clusters, version="QGeo Web prototype 1")
st.download_button("Download object measurements (CSV)", csv_bytes(rows), "qgeo_objects.csv", "text/csv", disabled=not rows)
st.download_button("Download results and masks (ZIP)", export_zip(crop,masks,names,rows,summaries,metadata), "qgeo_results.zip", "application/zip")
