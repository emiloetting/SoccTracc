import numpy as np
import cv2 as cv
import os
import sys
import itertools as it
from scipy.spatial import cKDTree
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
    
    def start(self, data):
        self.clusterer = KMeans(n_clusters=3)

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
            print('histogram :', label_hist)
            print('Index Array: ', index_array)
            hist_indexed = np.vstack((index_array,label_hist))
            print(hist_indexed)

            self.current_frame +=1 

        elif self.current_frame == 9:
            pass
            
            


            

        self.currently_tracked_objs = []
        self.current_frame += 1

        return { "teamAColor": (1,1,1),
                 "teamBColor": (1,1,1),
                 "teamClasses": [1]*len(data["tracks"]) } # Replace with actual team classes
    
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
    
    def get_quantized_lab(self):
        """Qunatized LAB color space to 48 bins"""
        l_channel = np.linspace(20, 230, num=3)
        a_channel = np.linspace(0, 255, num=4)
        b_channel = np.linspace(0, 255, num=4)
        return np.array(list(it.product(l_channel, a_channel, b_channel))) # Create cartesian product
    
    def torso(self, player: np.array, idx: int):
        rows = len(player)
        torso = player[:int(.5*rows),:,:]   # top half of player only interesting -> the part where shirt is
        cv.imwrite(f'.faafo/torsos/player_{idx}.jpg', player)
        return torso
    
    def prepare_color_tree(self, quantized_palette):
        """Quantizes given color space (current implementation for LAB) and feeds colors into cKDTree"""
        color_quantizer_tree = cKDTree(quantized_palette.astype(np.float32))
        return color_quantizer_tree
    
    def quantize_colors(self, img: np.array, color_quant_tree, color_space_in_given_tree) -> np.array:
        height, width, depth = img.shape
        row_img = img.reshape(-1, 3).astype(np.float32)     # type conversion for tree 
        distances, indices = color_quant_tree.query(row_img, k=[1]) # k=[1] for first-nearest neighbor
        quantized_row_img = color_space_in_given_tree[indices]
        reshaped_quantized = quantized_row_img.reshape(height, width, depth)
        return reshaped_quantized.astype(np.uint8)  # convert back
    
    def get_relative_torso_hist(self, torso: np.array):
        pxl_row = torso.reshape(-1,3) 
        matches = (pxl_row[:, None, :] == self.quantized_color_space[None, :, :]).all(axis=2)    # -> bool-Array of shape (Amnt. Pxl, Amnt. Bins)
        bin_counts = matches.sum(axis=0)    # Sum over all pxl in each bin
        amnt_pxls = pxl_row.shape[0]
        rlt_amnt_pxls = bin_counts/amnt_pxls    # get relative occurance to make sure different img sizes don't matter
        return rlt_amnt_pxls