# QGeo Web prototype

The web app runs alongside the existing desktop application. No desktop files are replaced.

## Run locally

Use Python 3.11 or 3.12 in a separate virtual environment:

```bash
python -m venv .venv-web
source .venv-web/bin/activate
pip install -r web/requirements.txt
streamlit run web/app.py
```

On Windows, activate the environment with `.venv-web\Scripts\activate`.

## Deploy

Commit the web folder to a development branch of the existing GitHub repository. In Streamlit Community Cloud, select that repository and branch, choose `web/app.py` as the entrypoint, and select Python 3.11 or 3.12. The requirements file beside the entrypoint supplies the web dependencies. Do not use the desktop requirements for this app.

## Scope

Supports upload, rectangular cropping, original-resolution colour segmentation, class naming, merging, removal, splitting, calibrated contour metrics, CSV and ZIP downloads. Public access does not require making any additional source repository public.

This prototype does not include free perspective cropping, six-point cylindrical unwrapping, 3D viewing, persistent projects or batch processing. Uploaded images are held in session memory, not deliberately written to disk by the app. Hosting provider infrastructure and logging still apply.

## Scientific behaviour

Colour classes require user interpretation. Black source pixels remain valid data through explicit boolean masks. Segmentation uses the desktop OpenCV k-means settings with seed zero. Splitting excludes pixels outside the selected class. Original source resolution is preserved. A physical crop height must be confirmed before measurements become available.

Contour measurements reuse `src/shape_detection.py`. External contours do not account for internal holes, and bounding-box aspect ratio depends on orientation. Degenerate zero-area contours are excluded. Area percentages use mask membership over the entire crop, independently of the contour filter. Cylindrical projection and perspective are not corrected in this version.

Validation with desktop exports and representative mineral labels is still required before research use. The original Apache 2.0 attribution in the repository README applies.
