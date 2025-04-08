import cv2 as cv
import numpy as np

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!
        self.last_frame = np.array([])
        self.frame = None
        self.features = None
        self.feature_params = dict(
                        maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
        self.lk_params = dict(
                            winSize  = (10, 10),
                            maxLevel = 2,
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
                            )

    def start(self, data):
        # TODO: Implement start up procedure of the module
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        if self.last_frame.size == 0:
            self.last_frame = cv.cvtColor(data['image'], cv.COLOR_BGR2GRAY)
            self.features = cv.goodFeaturesToTrack(self.last_frame, mask = None, **self.feature_params)
            return {
                "opticalFlow": np.asarray([0, 0])
            }
        
        self.frame = cv.cvtColor(data["image"], cv.COLOR_BGR2GRAY)
        _new_features, _status, _err = cv.calcOpticalFlowPyrLK(self.last_frame, self.frame, self.features, None, **self.lk_params)
        
        # sort out features that have not been found
        if _new_features is not None:
            tracked_new = _new_features[_status == 1]
            tracked_old = self.features[_status == 1]
        
        flow = -np.median(tracked_new - tracked_old, axis=0)
        
        self.last_frame = self.frame.copy()
        
        # TODO: Implement processing of a single frame
        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image. 
        # You 

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {
           "opticalFlow": flow
        }

