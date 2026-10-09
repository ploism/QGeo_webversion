"""QGeo browser workspace. Run: streamlit run app.py"""
import hashlib
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from core import analyse, csv_bytes, export_zip, segment


def navigate(view):
    st.session_state.workspace = view


def preview(image, size=(420, 230)):
    """Fit a display copy without changing the analysis image."""
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    rgb.thumbnail(size)
    canvas = Image.new("RGB", size, (24, 28, 34))
    canvas.paste(rgb, ((size[0] - rgb.width) // 2, (size[1] - rgb.height) // 2))
    return canvas


def checkpoint():
    history = st.session_state.setdefault("history", [])
    history.append((list(st.session_state.masks), list(st.session_state.names)))
    st.session_state.history = history[-10:]


def changed():
    st.session_state.revision = st.session_state.get("revision", 0) + 1
    st.rerun()


st.set_page_config(page_title="QGeo Web", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.3rem; padding-bottom: 1rem;}
h1 {font-size: 1.8rem !important;}
</style>""", unsafe_allow_html=True)
st.title("QGeo Web")
with st.sidebar:
    st.subheader("Image")
    uploaded = st.file_uploader("Rock image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded is None:
    st.info("Upload a rock image using the left panel to open the workspace.")
    st.stop()
digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
if st.session_state.get("source_digest") != digest:
    for key in ("masks", "names", "crop_key", "bounds", "history"):
        st.session_state.pop(key, None)
    st.session_state.source_digest = digest
    st.session_state.workspace = "Crop"
    st.session_state.revision = st.session_state.get("revision", 0) + 1
try:
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded.getvalue())))
    if pil.width * pil.height > 16_000_000:
        st.error("This prototype accepts images up to 16 million pixels.")
        st.stop()
    original = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
except (OSError, ValueError, Image.DecompressionBombError):
    st.error("The image could not be decoded.")
    st.stop()
h, w = original.shape[:2]
with st.sidebar:
    st.caption(f"{uploaded.name} · {w} × {h} pixels")
view = st.radio("Workspace", ["Crop", "Analyse", "Results"], horizontal=True,
                key="workspace", label_visibility="collapsed")
st.session_state.setdefault("bounds", (0, w, 0, h))
x0, x1, y0, y1 = st.session_state.bounds
if view == "Crop":
    with st.sidebar:
        st.subheader("Crop region")
        x0, x1 = st.slider("Horizontal crop (pixels)", 0, w, (x0, x1), key=f"crop_x_{digest}")
        y0, y1 = st.slider("Vertical crop (pixels)", 0, h, (y0, y1), key=f"crop_y_{digest}")
        st.session_state.bounds = (x0, x1, y0, y1)
        st.caption("Exclude rulers, background and marker lines. Changing the crop resets its analysis.")
if x1 <= x0 or y1 <= y0:
    st.warning("Select a crop with positive width and height.")
    st.stop()
crop = original[y0:y1, x0:x1].copy()
crop_key = (digest, x0, x1, y0, y1)
if st.session_state.get("crop_key") != crop_key:
    st.session_state.crop_key = crop_key
    for key in ("masks", "names", "history"):
        st.session_state.pop(key, None)
    st.session_state.revision = st.session_state.get("revision", 0) + 1

if view == "Crop":
    a, b = st.columns([2, 1])
    marked = original.copy()
    cv2.rectangle(marked, (x0, y0), (x1-1, y1-1), (0, 200, 255), max(1, w // 500))
    a.image(preview(marked, (800, 580)), caption="Original image and selected region", width="stretch")
    b.image(preview(crop, (420, 460)), caption="Analysis crop", width="stretch")
    b.caption("Analysis retains the original pixel resolution. These images are display previews.")
    b.button("Continue to analysis", type="primary", on_click=navigate, args=("Analyse",))
    st.stop()

if view == "Analyse":
    with st.sidebar:
        st.subheader("Segmentation")
        count = st.number_input("Initial colour classes", 2, 12, 3, key="cluster_count")
        if st.button("Segment crop", type="primary"):
            try:
                with st.spinner("Segmenting…"):
                    new_masks = segment(crop, int(count))
                if "masks" in st.session_state:
                    checkpoint()
                st.session_state.masks = new_masks
                st.session_state.names = [f"Class {i+1}" for i in range(count)]
                st.session_state.initial_clusters = int(count)
                changed()
            except ValueError as error:
                st.error(str(error))
        grid_columns = st.slider("Images per row", 2, 6, 4)
    if "masks" not in st.session_state:
        st.image(preview(crop, (800, 530)), width="stretch")
        st.info("Choose the number of colour classes in the left panel, then select Segment crop.")
        st.stop()
    masks, names = st.session_state.masks, st.session_state.names
    revision = st.session_state.revision
    selected = [i for i in range(len(masks)) if st.session_state.get(f"select_{revision}_{i}", False)]
    focus, gallery = st.columns([1, 3])
    with focus:
        st.subheader("Reference")
        focus_image = np.where(masks[selected[0]][..., None], crop, 0) if len(selected) == 1 else crop
        st.image(preview(focus_image, (420, 480)), width="stretch")
        st.caption(names[selected[0]] if len(selected) == 1 else "Analysis crop")
        st.caption("Select one class to inspect it here, or several classes to merge them.")
    with gallery:
        st.subheader(f"Colour classes · {len(masks)}")
        with st.container(height=610, border=True):
            for start in range(0, len(masks), grid_columns):
                columns = st.columns(grid_columns)
                for offset, col in enumerate(columns):
                    index = start + offset
                    if index >= len(masks):
                        break
                    with col:
                        with st.container(border=True):
                            mask = masks[index]
                            st.image(preview(np.where(mask[..., None], crop, 0)), width="stretch")
                            st.checkbox(f"Select class {index+1}", key=f"select_{revision}_{index}")
                            names[index] = st.text_input("Class name", value=names[index],
                                                       key=f"name_{revision}_{index}", label_visibility="collapsed")
                            st.caption(f"{100 * mask.mean():.2f}% of crop")
    with st.sidebar:
        st.subheader("Selected classes")
        st.caption(", ".join(names[i] for i in selected) if selected else "Select images in the grid.")
        if st.button("Merge selected", disabled=len(selected) < 2, width="stretch"):
            checkpoint()
            merged = np.logical_or.reduce([masks[i] for i in selected])
            keep = [i for i in range(len(masks)) if i not in selected]
            st.session_state.masks = [masks[i] for i in keep] + [merged]
            st.session_state.names = [names[i] for i in keep] + ["Merged class"]
            changed()
        split_count = st.number_input("Classes after splitting", 2, 12, 2)
        if st.button("Split selected", disabled=len(selected) != 1, width="stretch"):
            try:
                index = selected[0]
                children = segment(crop, int(split_count), masks[index])
                checkpoint()
                st.session_state.masks = masks[:index] + children + masks[index+1:]
                st.session_state.names = names[:index] + [f"{names[index]} part {j+1}" for j in range(split_count)] + names[index+1:]
                changed()
            except ValueError as error:
                st.error(str(error))
        if st.button("Remove selected", disabled=not selected or len(selected) == len(masks), width="stretch"):
            checkpoint()
            keep = [i for i in range(len(masks)) if i not in selected]
            st.session_state.masks = [masks[i] for i in keep]
            st.session_state.names = [names[i] for i in keep]
            changed()
        if st.button("Undo last change", disabled=not st.session_state.get("history"), width="stretch"):
            st.session_state.masks, st.session_state.names = st.session_state.history.pop()
            changed()
        st.button("View measurements", on_click=navigate, args=("Results",), width="stretch")
    st.caption("Colour classes require user interpretation before assigning mineral names. Mask percentages use the whole crop.")
    st.stop()

if "masks" not in st.session_state:
    st.info("Segment the crop in the Analyse workspace before opening results.")
    st.button("Go to analysis", on_click=navigate, args=("Analyse",))
    st.stop()
masks, names = st.session_state.masks, st.session_state.names
with st.sidebar:
    st.subheader("Calibration")
    physical_height = st.number_input("Selected crop height (mm)", min_value=0.001, value=10.0,
                                      format="%.3f", key=f"height_{crop_key}")
    confirmed = st.checkbox("I have checked the physical crop height", key=f"confirm_{crop_key}")
    minimum_area = st.number_input("Minimum contour area (pixels²)", min_value=0.0, value=0.0)
    st.caption("Enter the height of the crop, not the whole sample. Curvature and perspective are not corrected here.")
    st.button("Return to images", on_click=navigate, args=("Analyse",), width="stretch")
st.subheader("Measurements")
if not confirmed:
    st.info("Check the physical crop height in the left panel to enable measurements and downloads.")
    st.stop()
scale = physical_height / crop.shape[0]
rows, summaries = analyse(masks, names, scale, minimum_area)
st.dataframe(summaries, width="stretch", hide_index=True)
st.caption(f"Scale: {scale:.6f} mm per pixel. Removed classes remain excluded from analysis.")
if rows:
    st.dataframe(rows[:1000], width="stretch", hide_index=True, height=360)
    if len(rows) > 1000:
        st.caption("Preview shows 1,000 objects. Downloads include all objects.")
else:
    st.info("No positive-area contours meet the selected filter.")
metadata = dict(source=uploaded.name, original_size=[w,h], crop=[x0,y0,x1,y1],
                physical_crop_height_mm=physical_height, mm_per_pixel=scale,
                minimum_contour_area_px=minimum_area, class_names=names,
                seed=0, initial_clusters=st.session_state.initial_clusters, version="QGeo Web workspace 2")
a, b = st.columns(2)
a.download_button("Download measurements (CSV)", csv_bytes(rows), "qgeo_objects.csv", "text/csv", disabled=not rows)
b.download_button("Download results and masks (ZIP)", export_zip(crop,masks,names,rows,summaries,metadata), "qgeo_results.zip", "application/zip")
with st.expander("Measurement method"):
    st.write("External contours and metrics reuse the desktop implementation. Contour areas fill internal holes. Aspect ratio uses an axis-aligned bounding box and depends on orientation. Area percentages include all class pixels, independently of the contour filter.")
