import numpy as np
from scipy.optimize import linear_sum_assignment
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
    
    def is_confirmed(self, min_hits=3):
        """
        Is the track confirmed (enough hits)?
        """
        return self.hits >= min_hits
    
class Tracker:
    def __init__(self):
        self.name = "Tracker"  # Do not change the name of the module as otherwise recording replay would break!
        self.tracks: list = []
        self.detections = None
        self.classes = None
        
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
        self.tracks = []

    def step(self, data):
        """
        Main processing step for tracker
        """
        self.detections = data.get('detections', [])
        self.classes = data.get('classes', [])
        
        detections = self.detections
        classes = self.classes
        
        if len(detections) == 0:
            detections = np.empty((0, 4))
        else:
            detections = np.array(detections)
        
        # Predict tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matches, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(
                detections, self.tracks, classes
            )
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))
        
        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        # TODO check if necessary/ better option for unmatched detections than creating new tracks
        for det_idx in unmatched_dets:
            if det_idx < len(classes):
                cls = classes[det_idx]
            else:
                cls = 2  # Default: Player
            
            new_track = Filter(detections[det_idx], cls)
            self.tracks.append(new_track)
        
        # Delete old tracks
        self.tracks = [t for t in self.tracks if not t.should_be_deleted(self.max_disappeared)]

        # Return results
        return self.format_output()
    
    def associate_detections_to_tracks(self, detections, tracks, classes):
        """
        Associate detections to tracks based on Hungarian Algorithm
        """

        # if no tracks: no matches, all detections unmatched, no unmatched tracks
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        cost_matrix = self.calculate_cost_matrix(detections, tracks, classes)
        
        # Hungarian Algorithm 
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Only assign detections to tracks if the IoU is above the threshold
            matches = []
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= 1.0 - self.iou_threshold:
                    matches.append([row, col])
            
            # Find unmatched detections and tracks
            unmatched_detections = []
            for d in range(len(detections)):
                if d not in [m[0] for m in matches]:
                    unmatched_detections.append(d)
            
            unmatched_tracks = []
            for t in range(len(tracks)):
                if t not in [m[1] for m in matches]:
                    unmatched_tracks.append(t)
        else:
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(tracks)))
        
        return matches, unmatched_detections, unmatched_tracks
    
    def calculate_cost_matrix(self, detections, tracks, classes):
        """
        Calculate cost matrix for association between detections and tracks
        """
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, detection in enumerate(detections):
            for t, track in enumerate(tracks):
                # IoU costs
                iou = self.calculate_iou(detection, track.get_state())
                iou_cost = 1.0 - iou
                
                # Class costs: 1000 if classes don't match
                det_class = classes[d] 
                if det_class == track.class_id:
                    class_cost = 0.0
                else:
                    class_cost = 1000.0
                
                # Total cost
                cost_matrix[d, t] = iou_cost + class_cost
        
        return cost_matrix
    
    def calculate_iou(self, box1, box2):
        """
        Calculate intersection over union for two boxes
        """
        # Box1 coordinates
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        # Box2 coordinates
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def format_output(self):
        """
        create output dictionary
        """
        if len(self.tracks) == 0:
            return {
                "tracks": np.empty((0, 4)),
                "trackVelocities": np.empty((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": [],
            }
        
        
        confirmed_tracks = []

        for t in self.tracks:
            if t.is_confirmed(self.min_hits) and t.time_since_update == 0:
                confirmed_tracks.append(t)

        if len(confirmed_tracks) == 0:
            return {
                "tracks": np.empty((0, 4)),
                "trackVelocities": np.empty((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": [],
            }
        
        tracks_list = []
        velocities_list = []
        ages = []
        classes = []
        ids = []

        for t in confirmed_tracks:
            tracks_list.append(t.get_state())
            velocities_list.append(t.get_velocity())
            ages.append(int(t.age))
            classes.append(int(t.class_id))
            ids.append(t.track_id)

        tracks = np.array(tracks_list)
        velocities = np.array(velocities_list)

        return {
            "tracks": tracks,
            "trackVelocities": velocities,
            "trackAge": ages,
            "trackClasses": classes,
            "trackIds": ids
        }
