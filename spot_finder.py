import logging

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import label

LOGGER = logging.getLogger(__name__)


@dataclass
class Spot:
    i: int
    '''The row pixel coordinate of the spot: img[i, j]'''
    j: int
    '''The column pixel coordinate of the spot: img[i, j]'''
    w: int
    '''The full width of the bounding box of the spot'''
    bounding_box: tuple[int, int, int, int]
    '''The bounding box of the spot: (rmin, cmin, rmax, cmax)'''
    max: float
    '''The maximum pixel value in the spot'''
    sum: float
    '''The sum of all pixel values in the spot'''


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax


class SpotFinder:
    def __init__(
        self,
        threshold: float = 25.0,
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
        binary_image = img > self.threshold

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

            w = kp.size

            li = max(0, int(np.round(i - w)))
            hi = min(img.shape[0] - 1, int(np.round(i + w)))
            lj = max(0, int(np.round(j - w)))
            hj = min(img.shape[1] - 1, int(np.round(j + w)))

            pixels = img[li : hi + 1, lj : hj + 1]

            labels, n = label(pixels)  # type: ignore
            counts = [np.count_nonzero(labels == i) for i in range(1, n + 1)]
            l = np.argmax(counts) + 1
            mask = labels == l

            i, j = np.unravel_index(np.argmax(pixels), pixels.shape)
            i += li
            j += lj

            bbox = np.array(bbox2(mask)) + np.array([li, lj, li, lj])
            spots.append(
                Spot(
                    int(i),
                    int(j),
                    w * 2,
                    tuple(bbox),
                    np.max(pixels[mask]),
                    np.sum(pixels[mask]),
                )
            )

        LOGGER.debug('Detected %d blobs', len(spots))
        return spots
