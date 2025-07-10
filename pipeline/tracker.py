import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track

class Filter:
    """
    Filter class that implements a Kalman filter for tracking
    State vector: [x, y, w, h, vx, vy] - Position, Size, Velocity
    """
    
    # Class variable for unique IDs - starts at 0
    _id_counter = -1
    def __init__(self, z, cls):
        """
        Initialize a new track
        z: Detection [x, y, w, h] 
        cls: Class (0=Ball, 1=Goalkeeper, 2=Player, 3=Referee)
        """
        # Assign unique track ID 
        Filter._id_counter += 1
        self.track_id = Filter._id_counter
        
        
        # Track information
        self.class_id = int(cls)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Transition matrix F (constant velocity)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],  
            [0, 1, 0, 0, 0, 1],  
            [0, 0, 1, 0, 0, 0],  
            [0, 0, 0, 1, 0, 0],  
            [0, 0, 0, 0, 1, 0],  
            [0, 0, 0, 0, 0, 1]   
        ], dtype=np.float64)
        
        # Measurement matrix H (only position and size)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float64)
        
        # Process noise Q
        self.Q = np.eye(6, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01 
        
        # Measurement noise R
        self.R = np.eye(4, dtype=np.float64) * 1.0
        
        # Control matrix B (no control input)
        self.B = np.zeros((6, 1), dtype=np.float64)
        
        # Initialize state vector: [x, y, w, h, vx, vy]
        self.x = np.array([z[0], z[1], z[2], z[3], 0, 0], dtype=np.float64)
        
        # Initial covariance P
        self.P = np.eye(6, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0  
        self.P *= 10.0
        
        # Identity matrix for updates
        self.I = np.eye(6)
    
    
    
    def predict(self):
        """
        Prediction step of Kalman filter 
        """
        # Predict state with no control input
        u = np.zeros(1)
        self.x = np.dot(self.F,self.x)+ np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F,self.P),self.F.T)+self.Q
        # Update age and time
        self.age += 1
        self.time_since_update += 1
       
        
    def update(self,z):
        """
        Update step of Kalman filter 
        z: new measurement [x, y, w, h]
        """
        # Kalman filter update equations
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H,self.P),self.H.T)+self.R
        K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(S))
        self.x = self.x+np.dot(K,y)
        self.P = np.dot((self.I - np.dot(K,self.H)),self.P)
        
        # Update hits and time
        self.hits += 1
        self.time_since_update = 0

    
    def get_state(self):
        """
        Current state: [x, y, w, h]
        """
        return self.x[:4].copy()
    
    def get_velocity(self):
        """
        Current velocity: [vx, vy]
        """
        return self.x[4:6].copy()
    
    def should_be_deleted(self, max_age=30):
        """
        Determine if track should be deleted
        max_age: maximum number of frames without detection
        """
        return self.time_since_update > max_age
    
    def is_valid(self, min_hits=3):
        """
        Is the track confirmed (enough hits)?
        """
        return self.hits >= min_hits and self.time_since_update == 0
    
class Tracker:
    def __init__(self):
        self.name = "Tracker"  # Do not change the name of the module as otherwise recording replay would break!
        self.filters: list[Filter] = []
        
        self.max_disappeared = 30  
        self.min_hits = 3         # minimum hits to confirm a track
        self.iou_threshold = 0.3   

    def start(self, data):
        """
        Initialize tracker
        """
        pass

    def stop(self, data):
        """
        Stop tracker
        """
        self.filters = []

    def step(self, data):
        """
        Main processing step for tracker
        """
        detections = data.get('detections', [])
        classes = data.get('classes', [])

        if len(detections) == 0:
            detections = np.empty((0, 4))
        else:
            detections = np.array(detections)
        
        # Predict tracks
        for filter in list(self.filters):
            filter.predict()
            # Delete old tracks
            if filter.should_be_deleted():
                self.filters.remove(filter)
            
        # Associate detections to tracks
        matches = self.associate_detections_to_tracks(detections, self.filters, classes)

        # TODO check if necessary/ better option for unmatched detections than creating new tracks
        for detection, track in matches:
            if track is None: #sentinels that no match was found
                filter = Filter(detections[detection], classes[detection])
                self.filters.append(filter)
            elif detection is None:
                self.filters[track].predict()
            else:
                self.filters[track].update(detections[detection]) # feed the filter the bounding box

        # Return results
        return self.format_output()
    
    def associate_detections_to_tracks(self, detections, filters, classes):
        """
        Associate detections to tracks based on Hungarian Algorithm
        """
        # matches = np.full(len(detections), None)
        # if no tracks: no matches, all detections unmatched
        
        if not filters or len(detections) == 0:
            matches = np.full((len(detections), 2), None)
            matches[:,0] = np.arange(len(detections))
            return matches 
        
        cost_matrix = self.calculate_cost_matrix(detections, filters, classes)
        
        _detections, _tracks = linear_sum_assignment(cost_matrix)
        
        valid = cost_matrix[_detections, _tracks] <= 1.0 - self.iou_threshold
        
        valid_detections = _detections[valid]
        valid_tracks = _tracks[valid]

        d_indx = np.arange(len(detections))
        f_indx = np.arange(len(filters))
        unmatched_detections = np.setdiff1d(d_indx, valid_detections)
        unmatched_tracks=np.setdiff1d(f_indx, valid_tracks)

        matches = np.full((len(valid_tracks)+len(unmatched_detections)+len(unmatched_tracks), 2), None)
        matches[d_indx, 0] = d_indx
        matches[_detections[valid], 1] = _tracks[valid]
        
        matches[len(valid_detections):len(valid_detections)+len(unmatched_detections), 0] = unmatched_detections
        matches[len(valid_detections)+len(unmatched_detections):, 1] = unmatched_tracks

        return matches
    
    def calculate_cost_matrix(self, detections, tracks, classes):
        """
        Calculate cost matrix for association between detections and tracks
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
        inter_area = inter_w * inter_h                           # shape: (D, T)

        # Union areas
        area_det = detections[:, 2] * detections[:, 3]                       # (D,)
        area_trk = trk_states[:, 2] * trk_states[:, 3]           # (T,)
        union = area_det[:, None] + area_trk[None, :] - inter_area
        iou = np.where(union > 0, inter_area / union, 0.0)        # (D, T)

        # IoU cost
        iou_cost = 1.0 - iou

        # Class-mismatch cost
        det_cls = np.array(classes)[:, None]                     # (D, 1)
        trk_cls = np.array([t.class_id for t in tracks])[None, :]# (1, T)
        class_cost = np.where(det_cls == trk_cls, 0.0, 1000.0)    # (D, T)

        # Combined cost matrix
        cost_matrix = iou_cost + class_cost
        return cost_matrix
    
    def format_output(self):
        """
        create output dictionary
        """ 
        trackInfo = np.full((len(self.filters), 9), None)
        
        for idx, filter in enumerate(self.filters):
            if filter.is_valid(self.min_hits): # bei mehr als 5 hits und wenn es genau in unserem step geupdated
                trackInfo[idx, 0:6] = filter.x
                trackInfo[idx, 6] = filter.age
                trackInfo[idx, 7] = filter.class_id
                trackInfo[idx, 8] = filter.track_id
                
        validTracks = trackInfo[trackInfo[:, 8] != None]
        return {
            "tracks":           validTracks[:, 0:4],
            "trackVelocities":  validTracks[:, 4:6],
            "trackAge":         validTracks[:, 6].tolist(),
            "trackClasses":     validTracks[:, 7].tolist(),
            "trackIds":         validTracks[:, 8].tolist()
        }
