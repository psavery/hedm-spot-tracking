from dataclasses import dataclass

import numpy as np
import rtree


@dataclass
class TrackedSpot:
    x: float
    y: float
    w: float
    missing_count: int

    @property
    def bounding_box(self):
        return (
            self.x - self.w / 2,
            self.y - self.w / 2,
            self.x + self.w / 2,
            self.y + self.w / 2,
        )


class SpotTracker:
    def __init__(
        self, overlap_threshold: float = 0.5, missing_frame_threshold: int = 1
    ):
        self.overlap_threshold = overlap_threshold
        self.missing_frame_threshold = missing_frame_threshold

        self.current_spots: dict[int, TrackedSpot] = {}

        self.next_id = 0

        self.spot_index = rtree.index.Index()

    def overlap(
        self, x1: float, y1: float, w1: float, x2: float, y2: float, w2: float
    ) -> float:
        """
        Returns the fraction of overlap between two rectangles given by their center and width
        """
        lx = max(x1 - w1 / 2, x2 - w2 / 2)
        ly = max(y1 - w1 / 2, y2 - w2 / 2)
        hx = min(x1 + w1 / 2, x2 + w2 / 2)
        hy = min(y1 + w1 / 2, y2 + w2 / 2)

        dx = max(0, hx - lx)
        dy = max(0, hy - ly)

        return dx * dy / (w1 * w1 + w2 * w2 - dx * dy)

    def track_spots(self, spots: np.ndarray) -> np.ndarray:
        """
        Compares the spots to the currently tracked spots and updates the current spots with the new ones.

        Returns the currently tracked spots as a numpy array structured by [x, y, w]
        """

        for x, y, w in spots:
            bounding_box = (x - w / 2, y - w / 2, x + w / 2, y + w / 2)
            hits = list(self.spot_index.intersection(bounding_box))

            for hit in hits:
                spot = self.current_spots[hit]

                if spot.missing_count == 0:
                    continue

                if (
                    self.overlap(x, y, w, spot.x, spot.y, spot.w)
                    > self.overlap_threshold
                ):
                    self.spot_index.delete(hit, spot.bounding_box)
                    spot.x = x
                    spot.y = y
                    spot.w = w
                    spot.missing_count = 0
                    self.spot_index.insert(hit, spot.bounding_box)
                    break
            else:
                self.current_spots[self.next_id] = TrackedSpot(x, y, w, 0)
                self.spot_index.insert(self.next_id, bounding_box)
                self.next_id += 1

        for i, spot in list(self.current_spots.items()):
            spot.missing_count += 1
            if spot.missing_count > self.missing_frame_threshold:
                del self.current_spots[i]
                self.spot_index.delete(i, spot.bounding_box)

        return np.array(
            [[i, spot.x, spot.y, spot.w] for i, spot in self.current_spots.items()]
        )
