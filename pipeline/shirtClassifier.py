import numpy as np
import cv2 as cv
import os
import itertools as it
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans

cwd = os.getcwd()
player_paths = os.path.join(cwd,'.faafo', 'full_players')
torso_paths = os.path.join(cwd,'.faafo', 'torsos')
class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier" # Do not change the name of the module as otherwise recording replay would break!
        self.current_frame = 1
        self.currently_tracked_objs = []
        self.torsos_bgr = []
        self.torso_means = []
        self.clusterer = None
        self.classifier = None
        self.labels_pred = None
        self.team_a_colors = []
        self.team_b_colors = []
        self.team_a_color = None
        self.team_b_color = None
    
    def start(self, data):
        self.clusterer = KMeans(n_clusters=3)
        self.classifier = KNeighborsClassifier(n_jobs=-1)

    def stop(self, data):
        # remove faafo images, will be removed later anyways after 
        player_files = [os.path.join(player_paths, file) for file in os.listdir(player_paths)]
        torso_files = [os.path.join(torso_paths, file) for file in os.listdir(torso_paths)]
        for file_path in player_files:
            os.remove(file_path)
        for file_path in torso_files:
            os.remove(file_path)
        

    def step(self, data):
        # TODO: Implement processing of a current frame list
        # The task of the shirt classifier module is to identify the two teams based on their shirt color and to assign each player to one of the two teams

        # Note: You can access data["image"] and data["tracks"] to receive the current image as well as the current track list
        # You must return a dictionary with the given fields:
        #       "teamAColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team A
        #       "teamBColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team B
        #       "teamClasses"       A list with an integer class for each track according to the following mapping:
        #           0: Team not decided or not a player (e.g. ball, goal keeper, referee)
        #           1: Player belongs to team A
        #           2: Player belongs to team B

        # Get images of detected objetcs
        self.get_players_boxes(data)    # WORKS

        # Get top half (torsos) of detected objects
        self.torsos_bgr.extend([self.torso(player=player, idx=index) for index, player in enumerate(self.currently_tracked_objs)])    # WORKS (Debuggeg by displaying)
        
        # print('Currently_tracked_objects: ', self.currently_tracked_objs)
        # Reduce Image features by calculating mean color (BGR) of image
        for torso in self.torsos_bgr:     # whole of For-Loop WORKS AS WELL
            mean_pxl = list(np.mean(torso, axis=(0,1)).astype(int))
            self.torso_means.append(mean_pxl)
        
        # Clear list of BGR torsos to avoid processing them twice in next step
        self.torsos_bgr = []

        if self.current_frame < 8:
            self.team_a_color = (255,255,255)   # Placeholder color until correct colors are calculated. Works due to only class 1 being pred until frame 8
            self.team_b_color = None
            self.labels_pred = [1]*len(data["tracks"])  # same variable but stores placeholder data until correct data is calculated (frame 8)
            

        # On 4th frame: cluster all LAB-quantized torsos
        if self.current_frame == 8:    
      
            # KMEANS 
            # fitting and predicting useful to be combine
            self.clusterer.fit_predict(self.torso_means)

            # Make sure labels fit required classes (1,2: teams, 0: rest): 
            # rest is expected to have least objects tracked -> least occuring label will be switched with most occuring
            # Check Label occurances
            labels = self.clusterer.labels_
            label_hist = np.bincount(labels)

            # Add Index to label_hist
            index_array = np.arange(0,len(label_hist))

            # Add index_array to label hist
            hist_indexed = np.vstack((index_array,label_hist))

            # To correctly switch labels: 
            # Add 5 to each index to make switch up easier and not risk messing up other labels
            # If min(index_line) == 5: class with least objects (rest class) already labeled right -> 
            # -> If first class (origin. 0) has least amnt label: just undo prior changes, labels in correct order, nothing to change

            # IMPROVEMENT FOR LATER: USE NP.ARGMIN()
            # Currently mentally unable to...

            # If bin 0 (label 0) has most occurances and other labels have equally many occurances
            if np.max(hist_indexed[1,:]) == hist_indexed[1,0] and hist_indexed[1,1] == hist_indexed[1,2]:
                label += 5      # Change current labels
                hist_indexed[0] += 5    # Change label names in hist

                # For there are no clear teams: label mith most appearances muts NOT be 0, further specification impossible
                # Label 1 (with most occurances) will be label 0 and vice versa
                labels[labels == 6] = 0      
                labels[labels == 5] = 1
                labels[labels == 7] = 2

            # If label 1 hast least occurances -> make label 1 label 0 and vice versa
            elif np.min(hist_indexed[1,:]) == hist_indexed[1,1]:     
                labels += 5     
                hist_indexed[0] += 5    

                # We know: former bin 1 of (0,1,2) has least opccurances -> should be label 0, is currently 6 (due to 5-Addition)
                # Same process as before, seperated for chain of thought clarification / understandability
                labels[labels == 6] = 0
                labels[labels == 5] = 1
                labels[labels == 7] = 2
                hist_indexed[0] -= 5    # Change label names in hist
            
            # If label 2 has least occurances -> make label 2 label 0 and vice versa
            elif np.min(hist_indexed[1,:]) == hist_indexed[1,2]:
                labels += 5    
                hist_indexed[0] += 5    

                # We know: former bin 2 of (0,1,2) has least opccurances -> should be label 0, is currently 7 (due to 5-Addition)
                labels[labels == 7] = 0
                labels[labels == 6] = 1
                labels[labels == 5] = 2
                hist_indexed[0] -= 5
            
            # Train classifier to predict labels of players in all coming frames
            # We choose: KNN Classification

            self.classifier.fit(X=self.torso_means, y=labels)
            self.current_frame +=1 

            # Last step: Determine Team color
            # Goal: make them mean Torso color of mean torso colors of each team -> but lower green channel
            a_indices = np.where(labels == 1)
            self.team_a_colors = self.torso_means[a_indices]
            self.team_a_color = np.clip(
                a=(np.mean(self.team_a_colors, axis=0).astype(int)) - [0,100,0],    # Mean over mean color, make sure it's int, subtract green noise
                a_min=0,                                                            # Make sure after subtraction no value exceed 8bit-ins to avoid display issues
                a_max=255)

            b_indices = np.where(labels == 2)
            self.team_b_colors = self.torso_means[b_indices]
            self.team_b_color = np.clip(
                a=(np.mean(self.team_b_colors, axis=0).astype(int)) - [0,100,0], 
                a_min=0, 
                a_max=255)
        
            # Cache colors to avoid having to calculate them over and over again
            self.team_a_color = tuple(self.team_a_color.tolist())
            self.team_b_color = tuple(self.team_b_color.tolist())
            

        elif self.current_frame >= 9:
            # After gathering the data and training the classifier:
            # images get cut and prepared (see top of method), down here: only classification happens
            self.labels_pred = self.classifier.predict(X=self.torso_means) 
            
        self.currently_tracked_objs = []
        self.current_frame += 1

        return { "teamAColor": self.team_a_color,
                 "teamBColor": self.team_b_color,
                 "teamClasses": self.labels_pred } 
    
    def get_players_boxes(self, data):
        """Extracts all players' bounding boxes from image in data, slices players from image into np.Array"""
        img = data['image']
        player_boxes = data['tracks'] # is numpy array
        
        for idx, player_box in enumerate(player_boxes):
            x,y,w,h = player_box
            half_width = .5 * w
            half_height = .5 * h
            top_left_corner = (int(y - half_height), int(x - half_width))
            bottom_right_corner = (int(y + half_height), int(x + half_width))
            player = img[top_left_corner[0]:bottom_right_corner[0], top_left_corner[1]:bottom_right_corner[1]]
            cv.imwrite(f'.faafo/full_players/player_{idx}.jpg', player)
            self.currently_tracked_objs.append(player)
        return self.currently_tracked_objs
    
    def torso(self, player: np.array, idx: int):
        rows = len(player)
        torso = player[:int(.5*rows),:,:]   # top half of player only interesting -> the part where shirt is
        cv.imwrite(f'.faafo/torsos/player_{idx}.jpg', player)
        return torso

