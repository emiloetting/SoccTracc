import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from uuid import uuid4

class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"  # Do not change the name of the module as otherwise recording replay would break!
        self.current_frame = 0

    def start(self, data):
        self.clusterer = KMeans(n_clusters=3, max_iter=200)

    def stop(self, data):
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
        # Get images of detected objetcs
        player_imgs = self.get_players_boxes(data)
        lst_torsos_test = []
        lst_torsos_train = []
        # Reduce Image features by calculating mean color (BGR) of image
        for img in player_imgs:
            # Mask out green and black pixels
            masked_img = self.mask_img(img)
            # Calculate mean color of torso image (LAB) and append to list
            mean_lab = self.get_mean_lab_color(masked_img)
            lst_torsos_test.append(mean_lab)
            if self.current_frame < 9:  # Collect data for training
                lst_torsos_train.append(mean_lab)

            # Add to list of all mean pxls of all torsos in current frame    

        arr_torsos_test = np.array(lst_torsos_test)
        
        # Set data to return before clustering is completed
        if self.current_frame < 8:
            return {
                "teamAColor": (255, 255, 255),
                "teamBColor": (0, 0, 0),
                "teamClasses": [1] * len(data["tracks"]), #fake labels
            }
        
        # Frame 8: clustering takes place
        if self.current_frame == 8:
            # TORSOS HERE ALREADY IN LAB
            # Build cluster 
            arr_torsos_train = np.array(lst_torsos_train)
            self.clusterer.fit(arr_torsos_train)

            # Remap labels to 0, 1, 2 (0 = Rest, 1 = Team A, 2 = Team B)
            predictions = self.clusterer.predict(arr_torsos_train)
            self.generate_translation_layer(predictions)
            labels = self.translate_predictions(predictions)  
    
            # Last step: Determine Team color
            # Will be cached, not calcuted every frame
            
            self.team_a_color = self.get_main_team_color(labels==1, arr_torsos_train)  # method returns color in BGR
            self.team_b_color = self.get_main_team_color(labels==2, arr_torsos_train)
        
        # valid_mask = np.all(arr_torsos_test != None, axis=1)
        # predictions: np.ndarray = np.zeros(len(data["tracks"]))
        predictions = self.clusterer.predict(arr_torsos_test) #predictions[valid_mask]
        labels = self.translate_predictions(predictions)

        return {
            "teamAColor": self.team_a_color,
            "teamBColor": self.team_b_color,
            "teamClasses": labels.tolist(),
        }

    def get_players_boxes(self, data): # boxes contain torsos
        """Extracts all players' bounding boxes from image in data, slices players from image into np.Array"""
        img = data["image"]
        player_boxes = data["tracks"] 
        player_imgs = []
        for player_box in player_boxes:
            x, y, w, h = player_box
            x1, y1, x2, y2 = int(x-w/2), int(y-h/4), int(x+w/2), int(y) #just take y as min height because torso
            cutout = img[y1:y2, x1:x2, ...]
            # cv.imwrite(f"box-{uuid4().hex[0:5]}.png", cutout)
            if cutout.size == 0: player_imgs.append(None)
            else: player_imgs.append(cutout)
        return player_imgs
    
    def generate_translation_layer(self, labels: np.ndarray):
        bins = np.bincount(labels)
        self.translation =  np.argsort(bins)
    
    def translate_predictions(self, labels: np.ndarray) -> np.ndarray:
        """Reorganizes class labels into a desired format: 0 (Rest), 1 (Team A), 2 (Team B).
        This method processes an array of labels by analyzing their frequency and remapping 
        them based on specific rules. The remapping ensures that the label with the least 
        occurrences is assigned to the "Rest" class (0), while the other labels are assigned 
        to "Team A" (1) and "Team B" (2).
        - Computes a histogram of label occurrences.
        - Adjusts label values by adding 5 to avoid conflicts during remapping.
        - Determines the label with the least occurrences and assigns it to class 0.
        - Reassigns the remaining labels to classes 1 and 2 based on their original values.
        Args:
            labels (np.array): Array of labels to be reorganized.
        Returns:
            labels_remapped (np.array): Reorganized labels with values 0, 1, and 2.
        """
        remap = labels.copy()
        for i in range(len(self.translation)):
            remap[labels == self.translation[i]] = i
        
        return remap

    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """Applies a green mask to the BGR input-image, blacking out all green pixels.
        Used to remove green noise (pitch-background) from images.
        Args:
            img (np.array): Input image in BGR format.
        Returns:
            np.array: Image with green pixels blacked out. !!HSV!!
        """
        if img.size == 0: return np.empty(0)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Create a mask for green pixels
        greens = cv.inRange(img, np.array([38, 40, 40]), np.array([70, 255, 255])).astype(bool)
        # blacks = cv.inRange(img, np.asarray([0, 0, 0]), np.asarray([180, 255, 12])).astype(bool)
        # Apply the mask to the original image
        cleansed = img[~greens].reshape(-1, 1, 3)
        if cleansed.size == 0: return np.empty(0)
        cleansed = cv.cvtColor(cleansed, cv.COLOR_HSV2BGR)
        return cleansed
    
    def get_mean_lab_color(self, img: np.ndarray) -> tuple:
        """Calculates the mean color of an image.
        Args:
            img (np.array): Input image. (BGR)
        Returns:
            tuple: Mean color in LAB format.
        """
        # If no 'colored' pixel to be be collected: return green in LAB
        if img.size == 0:
            return np.array([127, 0, 255])
        
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        
        # Calculate mean pixel color as float, ensure correct dtype
        mean_lab = np.mean(lab, axis=(0,1)).astype(np.uint8)
        
        # cv.imwrite(f"mask-{uuid4().hex[0:5]}.png", img)
        
        return mean_lab
    
    def get_main_team_color(self, players_indices: np.ndarray, torso_means: np.ndarray) -> tuple:
        """Calculates the average color of a team based on the given players' images.
        Args:
            players (np.array): Array of player images.
        Returns:
            tuple: Average color in BGR format.
        """
        team_colors = torso_means[players_indices]
        team_color = np.mean(team_colors, axis=0).astype(np.uint8)
        # Reshpae to 1x1 image with 3 channels to prepare for conversion with cv
        team_color = team_color.reshape(-1, 1, 3)

        # Convert from LAB to BGR (Torsos are LAB)
        team_color = cv.cvtColor(team_color, cv.COLOR_LAB2BGR).ravel()

        return tuple(int(i) for i in team_color)
    
