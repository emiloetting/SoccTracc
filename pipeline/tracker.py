import numpy as np
from scipy.optimize import linear_sum_assignment


class Filter:
    # Class variable for unique IDs - starts at 0
    _id_counter = -1

    def __init__(self, z, cls):
        # Assign unique track ID
        Filter._id_counter += 1
        self.track_id = Filter._id_counter

        # Track information
        self.class_id = int(cls)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Transition matrix F (constant velocity)
        self.F = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix H (only position and size)
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # Process noise Q
        self.Q = np.diag([1, 1, 1, 1, 5, 5])

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float64) * 2.0

        # Control matrix B (control input is the optical flow having an effect only on position of the boxes)
        self.B = np.zeros((6, 2), dtype=np.float64)
        self.B[0, 0] = 1
        self.B[1, 1] = 1

        # Initialize state vector: [x, y, w, h, vx, vy]
        self.x = np.array([z[0], z[1], z[2], z[3], 0, 0], dtype=np.float32)

        # Initial covariance P
        if cls == 0:  # ball
            self.P = np.diag([0.01, 0.01, 0.01, 0.01, 100, 100])
        else:
            self.P = np.diag([0.01, 0.01, 0.01, 0.01, 10, 10])

        # Identity matrix for updates
        self.I = np.eye(6)

    def predict(self, opt_flow):
        """Predicts the next state with optical flow as control.
        Args:
            opt_flow (tuple): Optical flow vector (dx, dy).
        Returns:
            None
        """
        # Predict state with control input = optical flow
        dx, dy = opt_flow
        u = np.array([-dx, -dy], dtype=np.float64)
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # Update age and time
        self.age += 1
        self.time_since_update += 1

    def update(self, z):
        """Updates state with a new measurement.
        Args:
            z (np.ndarray): Measurement [x, y, w, h].
        Returns:
            None
        """
        # Kalman filter update equations
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)

        # Update hits and time
        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        """Gets current bounding box state.
        Returns:
            np.ndarray: Current state [x, y, w, h].
        """
        return self.x[:4].copy()

    def get_velocity(self):
        """Gets current velocity.
        Returns:
            np.ndarray: Current velocity [vx, vy].
        """
        return self.x[4:6].copy()

    def should_be_deleted(self, max_age):
        """Determines if track should be deleted due to age.
        Args:
            max_age (int): Max frames without updates.
        Returns:
            bool: True if track should be deleted.
        """
        return self.time_since_update > max_age

    def is_valid(self, min_hits=3):
        """Checks if track is confirmed based on hits.
        Args:
            min_hits (int): Minimum hits to confirm.
        Returns:
            bool: True if hits >= min_hits.
        """
        return self.hits >= min_hits

class Tracker:
    def __init__(self):
        self.name = "Tracker"  # Do not change the name of the module as otherwise recording replay would break!
        self.filters: list[Filter] = []

        self.max_disappeared = 10
        self.min_hits = 3  # minimum hits to confirm a track
        self.iou_threshold = 0.1

    def start(self, data):
        """Placeholder for start. No operation.
        Args:
            data (dict): Input data.
        Returns:
            None
        """
        pass

    def stop(self, data):
        """Clears all tracks.
        Args:
            data (dict): Input data.
        Returns:
            None
        """
        self.filters = []

    def step(self, data):
        """Processes detections and updates tracks.
        Args:
            data (dict): Contains:
                detections (list): Detection boxes [x,y,w,h].
                classes (list): Class indices.
                opticalFlow (tuple): Optical flow vector (dx, dy).
        Returns:
            dict: Contains:
                tracks (np.ndarray): Bounding boxes [x,y,w,h].
                trackVelocities (np.ndarray): Velocities [vx,vy].
                trackAge (list): Ages of tracks.
                trackClasses (list): Classes of tracks.
                trackIds (list): Track identifiers.
        """
        detections = data.get("detections", [])
        classes = data.get("classes", [])

        if len(detections) == 0:
            detections = np.empty((0, 4))
        else:
            detections = np.array(detections)

        # Predict tracks
        for filter in list(self.filters):
            filter.predict(data["opticalFlow"])
            # Delete old tracks
            if filter.should_be_deleted(self.max_disappeared):
                self.filters.remove(filter)

        # Associate detections to tracks
        matches = self.associate_detections_to_tracks(detections, self.filters, classes)

        # TODO check if necessary/ better option for unmatched detections than creating new tracks
        for detection, track in enumerate(matches):
            if track is None:  # sentinels that no match was found
                filter = Filter(detections[detection], classes[detection])
                self.filters.append(filter)
            else:
                self.filters[track].update(
                    detections[detection]
                )  # feed the filter the bounding box

        # Return results
        return self.format_output()

    def associate_detections_to_tracks(self, detections, filters, classes):
        """Associates detections to tracks via Hungarian algorithm.
        Args:
            detections (np.ndarray): Detection boxes.
            filters (list[Filter]): Active track filters.
            classes (list): Class indices for detections.
        Returns:
            np.ndarray: Match indices or None for unmatched.
        """
        matches = np.full(len(detections), None)
        # if no tracks: no matches, all detections unmatched
        if not filters or len(detections) == 0:
            return matches

        cost_matrix = self.calculate_cost_matrix(detections, filters, classes)
        detections, tracks = linear_sum_assignment(cost_matrix)

        valid = cost_matrix[detections, tracks] <= 1.0 - self.iou_threshold
        matches[detections[valid]] = tracks[valid]

        return matches

    def calculate_cost_matrix(self, detections, tracks, classes):
        """Computes cost matrix combining IOU and class mismatch.
        Args:
            detections (np.ndarray): Detection boxes.
            tracks (list[Filter]): Existing track filters.
            classes (list): Class indices.
        Returns:
            np.ndarray: Cost matrix (num_detections x num_tracks).
        """
        # Convert to arrays
        trk_states = np.array([t.get_state() for t in tracks])  # shape: (T, 4)

        # Detection box corners
        x1_min = detections[:, 0] - detections[:, 2] / 2
        y1_min = detections[:, 1] - detections[:, 3] / 2
        x1_max = detections[:, 0] + detections[:, 2] / 2
        y1_max = detections[:, 1] + detections[:, 3] / 2

        # Track box corners
        x2_min = trk_states[:, 0] - trk_states[:, 2] / 2
        y2_min = trk_states[:, 1] - trk_states[:, 3] / 2
        x2_max = trk_states[:, 0] + trk_states[:, 2] / 2
        y2_max = trk_states[:, 1] + trk_states[:, 3] / 2

        # Pairwise intersection
        inter_x_min = np.maximum(x1_min[:, None], x2_min[None, :])
        inter_y_min = np.maximum(y1_min[:, None], y2_min[None, :])
        inter_x_max = np.minimum(x1_max[:, None], x2_max[None, :])
        inter_y_max = np.minimum(y1_max[:, None], y2_max[None, :])
        inter_w = np.maximum(0.0, inter_x_max - inter_x_min)
        inter_h = np.maximum(0.0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h  # shape: (D, T)

        # Union areas
        area_det = detections[:, 2] * detections[:, 3]  # (D,)
        area_trk = trk_states[:, 2] * trk_states[:, 3]  # (T,)
        union = area_det[:, None] + area_trk[None, :] - inter_area
        iou = np.where(union > 0, inter_area / union, 0.0)  # (D, T)

        # IoU cost
        iou_cost = 1.0 - iou

        # Class-mismatch cost
        det_cls = np.array(classes)[:, None]  # (D, 1)
        trk_cls = np.array([t.class_id for t in tracks])[None, :]  # (1, T)
        class_cost = np.where(det_cls == trk_cls, 0.0, 1000.0)  # (D, T)

        # Combined cost matrix
        cost_matrix = iou_cost + class_cost
        return cost_matrix

    def format_output(self):
        """Formats track filters into output dictionary.
        Returns:
            dict: Keys: tracks, trackVelocities, trackAge, trackClasses, trackIds.
        """
        trackInfo = np.full((len(self.filters), 9), None)

        for idx, filter in enumerate(self.filters):
            if filter.is_valid(self.min_hits):
                trackInfo[idx, 0:6] = filter.x
                trackInfo[idx, 6] = filter.age
                trackInfo[idx, 7] = filter.class_id
                trackInfo[idx, 8] = filter.track_id

        validTracks = trackInfo[trackInfo[:, 8] != None]
        return {
            "tracks": validTracks[:, 0:4],
            "trackVelocities": validTracks[:, 4:6],
            "trackAge": validTracks[:, 6].tolist(),
            "trackClasses": validTracks[:, 7].tolist(),
            "trackIds": validTracks[:, 8].tolist(),
        }
