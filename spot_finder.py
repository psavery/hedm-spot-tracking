import logging

import cv2
import numpy as np
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class Spot:
    i: int
    '''The row pixel coordinate of the spot: img[i, j]'''
    j: int
    '''The column pixel coordinate of the spot: img[i, j]'''
    w: int
    '''The full width of the bounding box of the spot'''
    max: float
    '''The maximum pixel value in the spot'''
    sum: float
    '''The sum of all pixel values in the spot'''

    @property
    def bounding_box(self):
        return (
            self.i - self.w / 2,
            self.j - self.w / 2,
            self.i + self.w / 2,
            self.j + self.w / 2,
        )


class SpotFinder:
    def __init__(
        self,
        threshold: float = 0.0,
        minimum_separation: float = 1,
        min_area: int = 1,
        max_area: int = 1000,
    ):
        self.threshold = threshold
        self.minimal_separation = minimum_separation
        self.min = min_area
        self.max = max_area

    def find_spots(self, img: np.ndarray) -> list[Spot]:
        '''
        Finds blobs in the image using the configured parameters via either blob_log or blob_dog

        Returns the blobs found in the image as coordinates and widths

        Returns: A list of the spots found in the image
        '''
        binary_image = img / np.max(img) > self.threshold

        params = cv2.SimpleBlobDetector_Params()  # type: ignore
        params.filterByArea = True
        params.minArea = self.min
        params.maxArea = self.max
        params.minDistBetweenBlobs = self.minimal_separation

        params.filterByCircularity = False
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)  # type: ignore
        LOGGER.debug('Detecting blobs')
        keypoints = detector.detect(binary_image.astype(np.uint8) * 255)

        spots: list[Spot] = []
        for kp in keypoints:
            j, i = kp.pt
            w = kp.size * np.sqrt(2) / 2

            pixels = img[
                max(int(np.round(i - w)), 0) : int(np.round(i + w)),
                max(int(np.round(j - w)), 0) : int(np.round(j + w)),
            ]
            spots.append(Spot(i, j, w, np.max(pixels), np.sum(pixels)))

        LOGGER.debug('Detected %d blobs', len(spots))
        return spots
