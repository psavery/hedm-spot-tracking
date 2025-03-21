import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


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

    def find_spots(self, img: np.ndarray) -> np.ndarray:
        """
        Finds blobs in the image using the configured parameters via either blob_log or blob_dog

        Returns the blobs found in the image as coordinates and widths

        Returns: [i, j, w] where i and j are the coordinates of the center of each blob and w is the width
        """
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
        LOGGER.debug("Detecting blobs")
        keypoints = detector.detect(binary_image.astype(np.uint8) * 255)
        blobs = np.array([[kp.pt[0], kp.pt[1], kp.size] for kp in keypoints])
        LOGGER.debug("Detected %d blobs", len(blobs))
        if len(blobs) == 0:
            return blobs
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # Convert sigma to width

        return blobs
