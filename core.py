"""Browser-independent processing for QGeo Web."""
import csv
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shape_detection import contour_segmentation, generate_results


def segment(image, count, active=None, seed=0):
    """Keep membership separate from RGB values so black pixels are valid data."""
    if active is None:
        active = np.ones(image.shape[:2], dtype=bool)
    pixels = image[active].astype(np.float32)
    if count < 2 or count > min(12, len(pixels)):
        raise ValueError("Choose 2 to 12 clusters, no more than the selected pixels.")
    if len(np.unique(pixels, axis=0)) < count:
        raise ValueError("The selected region has fewer distinct colours than clusters.")
    cv2.setRNGSeed(seed)
    _, labels, _ = cv2.kmeans(
        pixels, count, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_PP_CENTERS,
    )
    masks = []
    for cluster in range(count):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[active] = labels.ravel() == cluster
        masks.append(mask)
    return masks


def analyse(masks, names, mm_per_pixel, minimum_area=0):
    rows, summaries = [], []
    denominator = masks[0].size
    for mask, name in zip(masks, names):
        binary = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        contours = [c for c in contour_segmentation(binary)
                    if c.get_area() > 0 and c.get_area() >= minimum_area]
        values = generate_results(contours, mm_per_pixel)
        for index, value in enumerate(values, 1):
            rows.append(dict(class_name=name, object_id=index,
                             aspect_ratio=value[1], area_mm2=value[2],
                             perimeter_mm=value[3], equivalent_radius_mm=value[4],
                             equivalent_length_mm=value[5],
                             centre_x_px=value[6], centre_y_px=value[7]))
        summaries.append(dict(class_name=name, area_percent=round(100 * mask.sum() / denominator, 3),
                              measured_objects=len(contours)))
    return rows, summaries


def csv_bytes(rows):
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def export_zip(image, masks, names, rows, summaries, metadata):
    import json
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("objects.csv", csv_bytes(rows))
        archive.writestr("summary.csv", csv_bytes(summaries))
        archive.writestr("settings.json", json.dumps(metadata, indent=2))
        for index, (mask, name) in enumerate(zip(masks, names), 1):
            _, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
            archive.writestr(f"mask_{index}.png", encoded.tobytes())
            coloured = np.where(mask[..., None], image, 0)
            _, encoded = cv2.imencode(".png", coloured)
            archive.writestr(f"colour_{index}.png", encoded.tobytes())
    return output.getvalue()
