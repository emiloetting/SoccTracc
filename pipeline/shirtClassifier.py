import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"  # Do not change the name of the module as otherwise recording replay would break!
        self.current_frame = 0

    def start(self, data):
        """
        Initializes the KMeans clusterer.
        Args:
            data (dict): Unused input data dictionary.
        Returns:
            None
        """
        self.clusterer = KMeans(n_clusters=3, max_iter=200)

    def stop(self, data):
        pass

    def step(self, data):
        """
        Processes a single frame to classify player shirts into two teams.
        Args:
            data (dict): Dictionary containing:
                image (np.ndarray): Frame in BGR format.
                tracks (list of tuples): Bounding boxes (x, y, w, h) for detected players.
        Returns:
            dict: Contains:
                teamAColor (tuple of int): BGR color for Team A.
                teamBColor (tuple of int): BGR color for Team B.
                teamClasses (list of int): Class label per track (0=Rest, 1=Team A, 2=Team B).
        """
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
        arr_torsos_test = np.array(lst_torsos_test)

        # Set data to return before clustering is completed
        if self.current_frame < 8:
            return {
                "teamAColor": (255, 255, 255),
                "teamBColor": (0, 0, 0),
                "teamClasses": [1] * len(data["tracks"]),  # fake labels
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

            self.team_a_color = self.get_main_team_color(
                labels == 1, arr_torsos_train
            )  # method returns color in BGR
            self.team_b_color = self.get_main_team_color(labels == 2, arr_torsos_train)

        predictions = self.clusterer.predict(arr_torsos_test)  # predictions[valid_mask]
        labels = self.translate_predictions(predictions)

        return {
            "teamAColor": self.team_a_color,
            "teamBColor": self.team_b_color,
            "teamClasses": labels.tolist(),
        }

    def get_players_boxes(self, data):  # boxes contain torsos
        """
        Extracts upper body images for each tracked player.
        Args:
            data (dict): Dictionary with keys:
                image (np.ndarray): Full frame in BGR.
                tracks (list of tuples): Bounding boxes (x, y, w, h).
        Returns:
            list of np.ndarray: Cropped torso images for each player.
        """
        img = data["image"]
        player_boxes = data["tracks"]
        player_imgs = []
        for player_box in player_boxes:
            x, y, w, h = player_box
            x1, y1, x2, y2 = (
                int(x - w / 2),
                int(y - h / 4),
                int(x + w / 2),
                int(y),
            )  # just take y as min height because torso
            cutout = img[y1:y2, x1:x2, ...]
            if cutout.size == 0:
                player_imgs.append(np.empty(0))
            else:
                player_imgs.append(cutout)
        return player_imgs

    def generate_translation_layer(self, labels: np.ndarray):
        """
        Builds a translation mapping from raw cluster labels to standardized class order.
        Args:
            labels (np.ndarray): Raw cluster labels.
        Returns:
            None
        """
        bins = np.bincount(labels)
        self.translation = np.argsort(bins)

    def translate_predictions(self, labels: np.ndarray) -> np.ndarray:
        """
        Remaps cluster labels to standardized classes: 0=Rest, 1=Team A, 2=Team B.
        Args:
            labels (np.ndarray): Raw cluster labels.
        Returns:
            np.ndarray: Remapped labels with values 0, 1, or 2.
        """
        remap = labels.copy()
        for i in range(len(self.translation)):
            remap[labels == self.translation[i]] = i

        return remap

    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """
        Masks out green pixels representing the pitch background in the input BGR image.
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            np.ndarray: Cleansed image with green pixels removed.
        """
        if img.size == 0:
            return np.empty(0)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Create a mask for green pixels
        greens = (img[..., 0] >= 32) & (img[..., 0] <= 70)
        # Apply the mask to the original image
        cleansed = img[~greens].reshape(-1, 1, 3)
        if cleansed.size == 0:
            return np.empty(0)
        cleansed = cv.cvtColor(cleansed, cv.COLOR_HSV2BGR)
        return cleansed

    def get_mean_lab_color(self, img: np.ndarray) -> tuple:
        """
        Calculates the mean LAB color of the input image.
        Args:
            img (np.ndarray): Input image in BGR format.
        Returns:
            tuple of uint8: Mean LAB color channels.
        """
        # If no 'colored' pixel to be be collected: return green in LAB
        if img.size == 0:
            return np.array([127, 0, 255])

        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

        # Calculate mean pixel color as float, ensure correct dtype
        mean_lab = np.mean(lab, axis=(0, 1)).astype(np.uint8)

        return mean_lab

    def get_main_team_color(
        self, players_indices: np.ndarray, torso_means: np.ndarray
    ) -> tuple:
        """
        Determines the average BGR color of a team based on LAB torso colors.
        Args:
            players_indices (np.ndarray): Boolean mask or array of indices for players in the team.
            torso_means (np.ndarray): Array of shape (n_samples, 3) with LAB mean colors.
        Returns:
            tuple of int: Average BGR color for the team.
        """
        team_colors = torso_means[players_indices]
        team_color = np.mean(team_colors, axis=0).astype(np.uint8)
        # Reshape to 1x1 image with 3 channels to prepare for conversion with cv
        team_color = team_color.reshape(-1, 1, 3)

        # Convert from LAB to BGR (Torsos are LAB)
        team_color = cv.cvtColor(team_color, cv.COLOR_LAB2BGR).ravel()

        return tuple(int(i) for i in team_color)
