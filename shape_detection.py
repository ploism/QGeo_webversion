"""
Shape detection implementation for
the geology-cuantifier project.
"""

from typing import List, Tuple

import cv2
import numpy as np

# COLORS in BGR
DEF_COLOR = [(255, 255, 0)]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

STATISTICS_CM = [
    "Rel. de Aspecto",
    "Area (cm^2)",
    "Perimetro (cm)",
    "Radio Equiv. (cm)",
    "Largo Equiv. (cm)",
    "Punto Medio X (px)",
    "Punto Medio Y (px)",
]
STATISTICS_MM = [
    "Rel. de Aspecto",
    "Area (mm^2)",
    "Perimetro (mm)",
    "Radio Equiv. (mm)",
    "Largo Equiv. (mm)",
    "Punto Medio X (px)",
    "Punto Medio Y (px)",
]
STATISTICS_DESC = [
    "Indica la relación que existe entre el alto y ancho de la figura, no toma en cuenta la orientación que esta tenga",
    "Area promedio de las figuras",
    "Perimetro promedio de las figuras",
    "Se calcula mediante el area del figura suponiendo que esta es una circunferencia",
    "Se calcula mediante el area del figura suponiendo que esta es un rectangulo con la misma relación de aspecto",
    "Coordenada x del promedio de los puntos medios de todas las figuras del mineral",
    "Coordenada y del promedio de los puntos medios de todas las figuras del mineral",
]


class ContourData(object):
    """
    This class is in charge of holding
    relevant data of a contour, such as
    It's bounding rectangle and a mask
    with only the pixels in that region.
    """

    def __init__(self, contour) -> None:
        """
        Class constructor, instantiates class
        params. such as the bounding rect. of a contour
        and It's mask.
        """
        self.contour = contour
        self.r = cv2.boundingRect(contour)
        self.group = 0

    def aspect_ratio(self) -> float:
        """
        When called, this function returns
        the aspect ratio of the bounding rect
        of the contour.
        """
        _, _, w, h = self.r
        asp_rat = w / h
        if asp_rat < 1:
            asp_rat = 1 / asp_rat
        return np.round(asp_rat, 2)

    def get_area(self) -> float:
        """
        When called, this function returns
        the total area of the contour.
        """
        return cv2.contourArea(self.contour)

    def get_perimeter(self) -> float:
        """
        When called, this function returns
        the total perimeter of the contour.
        """
        return cv2.arcLength(self.contour, True)

    def get_equiv_radius(self) -> float:
        """
        When called, this function returns
        the equivalent radius associated to
        the area of the contour.
        """
        return np.sqrt(self.get_area() / np.pi)

    def get_equiv_lenght(self) -> float:
        """
        When called, this function returns
        the equivalent lenght associated to
        the area of the contour.
        """
        return np.sqrt(self.get_area() / self.aspect_ratio())

    def get_middle_point(self) -> Tuple[float, float]:
        """
        When called, this function returns
        the middle point of the bounding rect
        of the contour.
        """
        x, y, w, h = self.r
        return (x + w / 2, y + h / 2)

    def get_all_statistics(self, pixel2mm) -> List:
        """
        When called, this function returns
        the results of all statistics implemented
        in the class.
        """
        return [
            self.group,
            np.round(self.aspect_ratio(), 2),
            np.round(self.get_area() * pixel2mm * pixel2mm, 4),
            np.round(self.get_perimeter() * pixel2mm, 4),
            np.round(self.get_equiv_radius() * pixel2mm, 2),
            np.round(self.get_equiv_lenght() * pixel2mm, 2),
            np.round(self.get_middle_point()[0], 0),
            np.round(self.get_middle_point()[1], 0),
        ]


def contour_segmentation(img):
    """
    Finds the contours of the image, for each computes
    It's bounding rect and mask. Then draws the contours
    and returns a list with all the data.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, threshInv) = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    data = [ContourData(cnt) for cnt in contours]
    return data


def contour_agrupation(contours):
    """
    Runs a basic agrupation by comparing the aspect ratio
    of each contour's bounding rect.
    """
    for c in contours:
        asp_rat = c.aspect_ratio()
        if asp_rat <= 1.2:
            # Square like rectangles
            c.group = 0
        elif asp_rat <= 1.5:
            c.group = 1
        else:
            c.group = 2


def cluster_segmentation(cluster, contours, colors):
    """
    Once each CountourData instance is successfully
    agrupated, this function draws each bounding rect.

    The rectangle colour will be given by the group of the
    ContourData instance.
    """
    im = np.copy(cluster)
    for c in contours:
        color = colors[c.group]
        cv2.rectangle(im, c.r, color, 2)
    return im


def generate_results(contours, pixel2mm):
    """
    Calculate statistics for each group.
    """
    final_results = []
    for c in contours:
        final_results.append(c.get_all_statistics(pixel2mm))
    return final_results


def image_agrupation(img_org, contours, groups):
    """
    Agrupate all images of the contours in one image for each group
    """
    shape = img_org.shape
    masks = [np.zeros(shape[:2], dtype="uint8") for _ in range(groups)]
    conts = [[] for _ in range(groups)]
    for i in range(len(contours)):
        group = contours[i].group
        conts[group].append(contours[i].contour)
    for i in range(groups):
        cv2.drawContours(masks[i], conts[i], -1, (255, 255, 255), -1)
        masks[i] = cv2.bitwise_and(img_org, img_org, mask=masks[i])
    return masks
