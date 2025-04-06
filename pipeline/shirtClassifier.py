import numpy as np
import cv2



class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier" # Do not change the name of the module as otherwise recording replay would break!
        self.currently_tracked_objs = []
        self.current_frame = 0
    
    def start(self, data):
        # TODO: Implement start up procedure of the module
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

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
        self.current_frame += 1
        self.mean_colors = []
        self.currently_tracked_objs = []
        self.get_players_boxes(data)
        for player in self.currently_tracked_objs:
            pass
        

            
        return { "teamAColor": (1,1,1),
                 "teamBColor": (1,1,1),
                 "teamClasses": [1]*len(data["tracks"]) } # TODO: Replace with actual team classes
    
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
            cv2.imwrite(f'.faafo/player_{idx}.jpg', player)
            self.currently_tracked_objs.append(player)
        return self.currently_tracked_objs