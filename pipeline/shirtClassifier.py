import numpy as np
import cv2 as cv
import os
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering

cwd = os.getcwd()
player_paths = os.path.join(cwd, ".faafo", "full_players")
torso_paths = os.path.join(cwd, ".faafo", "torsos")
masked_torso_paths = os.path.join(cwd, ".faafo", "masked_torsos")
class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"  # Do not change the name of the module as otherwise recording replay would break!
        self.current_frame = 1
        self.currently_tracked_objs = []
        self.torsos_bgr = []
        self.torso_means = []
        self.current_torsos_in_frame = []
        self.clusterer = None
        self.classifier = None
        self.labels_pred = None
        self.team_a_color = None
        self.team_b_color = None

    def start(self, data):
        self.clusterer = AgglomerativeClustering(n_clusters=3, linkage="ward", compute_full_tree=True, metric="euclidean")
        self.classifier = NearestCentroid()

    def stop(self, data):
        # remove faafo images, will be removed later anyways after
        player_files = [
            os.path.join(player_paths, file) for file in os.listdir(player_paths)
        ]
        torso_files = [
            os.path.join(torso_paths, file) for file in os.listdir(torso_paths)
        ]
        masked_torsos = [
            os.path.join(masked_torso_paths, file) for file in os.listdir(masked_torso_paths)
        ]
        for file_path in player_files:
            os.remove(file_path)
        for file_path in torso_files:
            os.remove(file_path)
        for file_path in masked_torsos:
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
        #TODO      -1: Player belongs to team B     <- was previously 2, problem lies in Display-module, expects -1 instead of 2

        # Get images of detected objetcs
        self.get_players_boxes(data)  # Internally updates self.currently_tracked_objs

        # Get top half (torsos) of detected objects
        current_torsos = []

        for index, player in enumerate(self.currently_tracked_objs):
            # Only use upper half of player image (torso)
            torso = self.torso(player=player, idx=index)
            current_torsos.append(torso)
        
        
        self.torsos_bgr.extend(current_torsos)  # dtype self.torsos_bgr: list[lists of np.arrays]

        # Reduce Image features by calculating mean color (BGR) of image
        for indx, torso in enumerate(self.torsos_bgr):

            # Mask green pixels (pitch background), black out green pixels
            masked_torso = self.green_masking(torso, indx)  

            # Transform masked image to LAB color space for better clustering
            masked_torso_lab = cv.cvtColor(masked_torso, cv.COLOR_BGR2LAB)

            # Make sure dtpye = np.uint8    
            masked_torso_lab = np.array(masked_torso_lab, dtype=np.uint8)

            # Calculate mean color of torso image (BGR) and append to list
            mean_pxl = self.calc_mean_img_color(masked_torso_lab)   # Ignores black pxl, if only black: returns [220,  43,  210] (green in LAB space)

            if self.current_frame < 9:  # Collect data for training
                self.torso_means.append(mean_pxl)  # leave as list to avoid flattening to 1D by np.append() (behaves differently for some reason)

            # Add to list of all mean pxls of all torsos in current frame    
            self.current_torsos_in_frame.append(mean_pxl)


        # Clear list of BGR torsos to avoid processing them twice in next step
        self.torsos_bgr = []

        # Set data to return before clustering is completed
        if self.current_frame < 8:
            # Set fake default team colors (already in BGR as later required)
            self.team_a_color = (255, 255, 255)
            self.team_b_color = (0, 0, 0)

            # Set fake labels (each player is of team 1)
            self.labels_pred = [1] * len(data["tracks"]) 

        # Frame 8: clustering takes place
        if self.current_frame == 8:
            # TORSOS HERE ALREADY IN LAB
            
            # Cast self.torso_means as np.Array for easy reshaping
            self.torso_means = np.array(self.torso_means)

            # Reshape each image to 1D array (3 channels) before clustering
            torso_means_reshaped = np.array([torso.reshape(-3, ) for torso in self.torso_means], dtype=np.uint8)
            
            # Build cluster 
            self.clusterer.fit_predict(torso_means_reshaped)    

            # Remap labels to 0, 1, 2 (0 = Rest, 1 = Team A, 2 = Team B)
            labels = self.clusterer.labels_
            labels_remapped = self.organize_classes(labels)  

            # Train KNN classifier with training data (torsos) and labels
            self.classifier.fit(X=torso_means_reshaped, y=labels_remapped)  # Fit classifier with training data (torsos) and labels (team colors)

            # Use clusterer as classifier -> Prev. KNN does about same and takes longer
            self.labels_pred = self.classifier.predict(self.current_torsos_in_frame) # Labels for players in this frame were calculated in prev. step, however 

            # Fit to changed requirement of team-class 2 to be -1
            self.labels_pred[self.labels_pred == 2] = -1
            self.labels_pred = self.labels_pred.tolist()
            
            # Last step: Determine Team color
            # Will be cached, not calculated every frame
            a_indices = np.where(labels_remapped == 1)[0]  # np.where() returns tuple, first value is needed
            self.team_a_color = self.avg_team_color(a_indices)  # method returns color in BGR

            b_indices = np.where(labels_remapped == -1)[0]
            self.team_b_color = self.avg_team_color(b_indices)

            # Reset torsos for next frame
            self.torso_means = [] 

        elif self.current_frame >= 9:
            # After gathering the data and training the classifier:
            # images get cut and prepared (see top of method), down here: only classification happens
            self.torso_means = []
            self.current_torsos_in_frame = np.array(self.current_torsos_in_frame)
        
            self.labels_pred = (self.classifier.predict(X=self.current_torsos_in_frame))

            # Fit to changed requirement of team-class 2 to be -1
            self.labels_pred[self.labels_pred == 2] = -1
            self.labels_pred = self.labels_pred.tolist()    # inplace operation not possible for to_list()

        self.currently_tracked_objs = []
        self.current_torsos_in_frame = []
        self.current_frame += 1

        return {
            "teamAColor": self.team_a_color,
            "teamBColor": self.team_b_color,
            "teamClasses": self.labels_pred,
        }

    def get_players_boxes(self, data):
        """Extracts all players' bounding boxes from image in data, slices players from image into np.Array"""
        img = data["image"]
        player_boxes = data["tracks"] 

        for idx, player_box in enumerate(player_boxes):
            x, y, w, h = player_box

            half_width = 0.5 * w
            half_height = 0.5 * h
            top_left_corner = (int(y - half_height), int(x - half_width))
            bottom_right_corner = (int(y + half_height), int(x + half_width))
            player = img[
                top_left_corner[0] : bottom_right_corner[0],
                top_left_corner[1] : bottom_right_corner[1],
            ]
            height, width, _ = player.shape
            if height == 0 or width == 0:
                self.currently_tracked_objs.append(np.zeros((3,3,3), dtype=np.uint8))  # Will be ignored in calc_mean_img_color()
                continue
            # cv.imwrite(f'.faafo/full_players/player_{idx}.jpg', player)
            self.currently_tracked_objs.append(player)

    def torso(self, player: np.array, idx: int):
        rows = len(player)
        # top half of player only interesting -> the part where shirt is
        torso = player[:int(np.ceil(0.5 *rows)), :, :]  
        # top half of player only interesting -> the part where shirt is
        # cv.imwrite(f'.faafo/torsos/player_{idx}.jpg', torso)
        if torso.size == 0:
            return np.zeros((3,3,3), dtype=np.uint8) # Will be ignored in calc_mean_img_color()
        return torso
        
    
    def organize_classes(self, labels: np.array) -> np.array:
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
            labels_remapped (np.array): Reorganized labels with values 0, 1, and -1.  # Attention: -1 was previously 2
        """
        label_hist = np.bincount(labels)

        # Add Index to label_hist
        index_array = np.arange(0, len(label_hist))

        # Add index_array to label hist
        hist_indexed = np.vstack((index_array, label_hist))


        # If bin 0 (label 0) has most occurances and other labels have equally many occurances
        if (
            np.max(hist_indexed[1, :]) == hist_indexed[1, 0]
            and hist_indexed[1, 1] == hist_indexed[1, 2]
        ):
            labels += 5  # Change current labels

            # For there are no clear teams: label mith most appearances muts NOT be 0, further specification impossible
            # Label 1 (with most occurances) will be label 0 and vice versa
            labels[labels == 6] = 0
            labels[labels == 5] = 1
            labels[labels == 7] = -1

        # If label 1 hast least occurances -> make label 1 label 0 and vice versa
        elif np.min(hist_indexed[1, :]) == hist_indexed[1, 1]:
            labels += 5

            # We know: former bin 1 of (0,1,2) has least opccurances -> should be label 0, is currently 6 (due to 5-Addition)
            # Same process as before, seperated for chain of thought clarification / understandability
            labels[labels == 6] = 0
            labels[labels == 5] = 1
            labels[labels == 7] = -1

        # If label 2 has least occurances -> make label 2 label 0 and vice versa
        elif np.min(hist_indexed[1, :]) == hist_indexed[1, 2]:
            labels += 5

            # We know: former bin 2 of (0,1,2) has least opccurances -> should be label 0, is currently 7 (due to 5-Addition)
            labels[labels == 7] = 0
            labels[labels == 6] = 1
            labels[labels == 5] = -1

        return labels

    def green_masking(self, bgr_img: np.array, idx: int) -> np.array:
        """Applies a green mask to the BGR input-image, blacking out all green pixels.
        Used to remove green noise (pitch-background) from images.
        Args:
            bgr_img (np.array): Input image in BGR format.
            idx (int): Index of the player for debugging purposes.
        Returns:
            np.array: Image with green pixels blacked out.
        """
        # Convert BGR to HSV color space
        hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the green color in HSV
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Create a mask for green pixels
        mask = cv.inRange(hsv_img, lower_green, upper_green)

        # Invert the mask to keep non-green pixels
        mask_inv = cv.bitwise_not(mask)

        # Apply the mask to the original image
        green_cleansed_img = cv.bitwise_and(bgr_img, bgr_img, mask=mask_inv)
        
        # Safe masked image for debugging purposes
        #cv.imwrite(f'.faafo/masked_torsos/player_{idx}_masked.jpg', green_cleansed_img)

        return green_cleansed_img
    
    def avg_team_color(self, players_indices: np.array) -> tuple:
        """Calculates the average color of a team based on the given players' images.
        Args:
            players (np.array): Array of player images.
        Returns:
            tuple: Average color in BGR format.
        """
        team_colors = self.torso_means[players_indices]
        team_color = np.clip(
            a=(np.mean(team_colors, axis=0).astype(np.uint8)),
            a_min=0,
            a_max=255,
            )
        # Reshpae to 1x1 image with 3 channels to prepare for conversion with cv
        team_color = team_color.reshape(1, 1, 3)

        # Convert from LAB to BGR (Torsos are LAB)
        team_color = cv.cvtColor(team_color, cv.COLOR_LAB2BGR).tolist()[0][0]

        return tuple(team_color)
    
    def calc_mean_img_color(self, img: np.array) -> tuple:
        """Calculates the mean color of an image. Ignores completely black pixels -> are caused by green masking and should not be considered for color calculation.
        Args:
            img (np.array): Input image.
        Returns:
            tuple: Mean color in BGR format.
        """
        # flatten image to 2D array (rows = pixels, columns = BGR values)
        pixels = img.reshape(-1, 3) 

        # mask to ignore black pixel in mean calculation later
        mask = np.all(pixels != [0, 128, 128], axis=1) 

        # Apply mask -> collect only 'coloured' pixel
        filtered_pixels = pixels[mask]

        # If no 'colored' pixel to be be collected: return green in LAB
        if not np.any(filtered_pixels != 0): 
            return [220,  43,  210] 
        
        # Calculate mean pixel color as float, ensure correct dtype
        mean_color = np.mean(filtered_pixels, axis=0)

        # Convert from float back to int
        mean_color = np.clip(mean_color, 0, 255).astype(np.uint8)

        # Return as list for compatibility with other methods
        return mean_color.tolist()