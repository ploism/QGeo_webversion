"""Record and replay QGeo operations with colour-based cluster correspondence."""
import json
from functools import lru_cache

import cv2
import numpy as np

from core import segment


def class_colours(image, masks):
    return [image[mask].mean(axis=0).tolist() for mask in masks]


def match_colours(image, masks, reference):
    """Optimal one-to-one matching of mean colours in CIELAB, within this split."""
    actual = np.asarray(class_colours(image, masks), dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    def lab(colours):
        return cv2.cvtColor((colours / 255).reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    distances = np.linalg.norm(lab(target)[:, None, :] - lab(actual)[None, :, :], axis=2)
    count = len(masks)

    @lru_cache(None)
    def assign(used):
        row = used.bit_count()
        if row == count:
            return 0.0, ()
        best = None
        for column in range(count):
            if used & (1 << column):
                continue
            cost, order = assign(used | (1 << column))
            candidate = (float(distances[row, column]) + cost, (column,) + order)
            if best is None or candidate < best:
                best = candidate
        return best

    _, order = assign(0)
    differences = [float(distances[row, column]) for row, column in enumerate(order)]
    return [masks[column] for column in order], differences


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("format") != "qgeo-protocol" or protocol.get("version") != 1:
        raise ValueError("This is not a supported QGeo protocol file.")
    if protocol.get("algorithm") != "opencv-kmeans-bgr-v1":
        raise ValueError("Unsupported segmentation algorithm.")
    steps = protocol.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 100:
        raise ValueError("A protocol must contain 1 to 100 steps.")
    count = 0
    for number, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise ValueError(f"Invalid step {number}.")
        op = step.get("op")
        if number == 1 and op != "segment":
            raise ValueError("The first protocol step must segment the crop.")
        if op in ("segment", "split"):
            clusters = step.get("count")
            if type(clusters) is not int or not 2 <= clusters <= 12:
                raise ValueError(f"Invalid cluster count in step {number}.")
            try:
                colours = np.asarray(step.get("colours"), dtype=float)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid colours in step {number}.") from None
            if colours.shape != (clusters, 3) or not np.isfinite(colours).all() or np.any((colours < 0) | (colours > 255)):
                raise ValueError(f"Invalid reference colours in step {number}.")
            if op == "segment":
                if number != 1:
                    raise ValueError("Only the first step may segment the whole crop.")
                count = clusters
            else:
                index = step.get("index")
                if type(index) is not int or not 0 <= index < count:
                    raise ValueError(f"Invalid selected class in step {number}.")
                count += clusters - 1
        elif op in ("merge", "remove"):
            indices = step.get("indices")
            if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < count for i in indices) or len(set(indices)) != len(indices):
                raise ValueError(f"Invalid class selections in step {number}.")
            if op == "merge" and len(indices) < 2:
                raise ValueError("A merge requires at least two classes.")
            if op == "remove" and len(indices) == count:
                raise ValueError("A protocol cannot remove every class.")
            count -= len(indices) - (1 if op == "merge" else 0)
        else:
            raise ValueError(f"Unknown operation in step {number}.")
        if count > 128:
            raise ValueError("A protocol may produce at most 128 classes.")
    names = protocol.get("class_names")
    if not isinstance(names, list) or len(names) != count or any(not isinstance(n, str) or len(n) > 250 for n in names):
        raise ValueError("The final class names do not match the protocol output.")
    return protocol


def load_protocol(data):
    if len(data) > 250_000:
        raise ValueError("Protocol files must be smaller than 250 KB.")
    try:
        return validate_protocol(json.loads(data))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("The protocol file must contain valid JSON.") from None


def make_protocol(name, steps, names):
    return validate_protocol(dict(format="qgeo-protocol", version=1,
                                  name=name, algorithm="opencv-kmeans-bgr-v1",
                                  scope="operations on the selected crop; crop and physical calibration are set per image",
                                  steps=steps, class_names=list(names)))


def replay(image, protocol):
    validate_protocol(protocol)
    masks, differences = [], []
    for number, step in enumerate(protocol["steps"], 1):
        try:
            op = step["op"]
            if op == "segment":
                masks, delta = match_colours(image, segment(image, step["count"]), step["colours"])
                differences.extend(delta)
            elif op == "split":
                index = step["index"]
                children = segment(image, step["count"], masks[index])
                children, delta = match_colours(image, children, step["colours"])
                differences.extend(delta)
                masks = masks[:index] + children + masks[index+1:]
            else:
                selected = step["indices"]
                retained = [m for i, m in enumerate(masks) if i not in selected]
                if op == "merge":
                    retained.append(np.logical_or.reduce([masks[i] for i in selected]))
                masks = retained
        except ValueError as error:
            raise ValueError(f"Protocol step {number} ({step['op']}) could not be applied: {error}") from error
    return masks, list(protocol["class_names"]), differences
