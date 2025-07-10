# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track
class Filter:
    def __init__(self, z, cls):
        # TODO: Implement filter initializstion
        pass

    # TODO: Implement remaining funtionality for an individual track


class Tracker:
    def __init__(self):
        self.name = "Tracker"  # Do not change the name of the module as otherwise recording replay would break!

    def start(self, data):
        # TODO: Implement start up procedure of the module
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        # TODO: Implement processing of a detection list
        # The task of the tracker module is to identify (temporal) consistent tracks out of the given list of detections
        # The tracker maintains a list of known tracks which is initially empty.
        # The tracker then tries to associate all given detections from the detector to existing tracks. A meaningful metric needs to be defined
        # to decide which detection should be associated with each track and which detections better stay unassigned.
        # After the association step, one must handle there different cases:
        #   1) Detections which have not beed associated with a track: For these, create a new filter class and initialize its state based on the detection
        #   2) Tracks which have a detection: The state of these can be updated based on the associated detection
        #   3) Tracks which have no detection: It makes sense to allow for a few missing frames, nonetheless it is still necessary to predict the
        #      current filter state (e.g. based on the optical flow measurement and the object velocity). If too many frames are missing, the track can be deleted

        # Note: You can access data["detections"] and data["classes"] to receive the current list of detections and their corresponding classes
        # You must return a dictionary with the given fields:
        #       "tracks":           A Nx4 NumPy Array containing a 4-dimensional state vector for each track. Similar to the detections,
        #                           the track state containts the center point (X,Y) as well as the bounding box width and height (W, H)
        #       "trackVelocities":  A Nx2 NumPy Array with an additional velocity estimate (in pixels per frame) for each track
        #       "trackAge":         A Nx1 List with the track age (number of total frames this track exists). The track age starts at
        #                           1 on track creation and increases monotonically by 1 per frame until the track is deleted.
        #       "trackClasses":     A Nx1 List of classes associated with each track. Similar to detections, the following mapping must be used
        #                               0: Ball
        #                               1: GoalKeeper
        #                               2: Player
        #                               3: Referee
        #       "trackIds":         A Nx1 List of unique IDs for each track. IDs must not be reused and be unique during the lifetime of the program.
        return {
            "tracks": None,
            "trackVelocities": None,
            "trackAge": None,
            "trackClasses": None,
            "trackIds": None,
        }
