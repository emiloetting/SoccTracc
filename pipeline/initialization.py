import os
import gdown
import shutil
from tqdm import tqdm



class Initializer():
    def __init__(self, cwd: str, metas: dict):
        self.temp_dir = os.path.join(cwd, "temp")
        # separate temp dir for model downloads to avoid interfering with video temp
        self.cwd = cwd
        self.metas = metas
        os.makedirs(self.temp_dir, exist_ok=True)

    def grab_model_on_absence(self, model_type:str) -> None:
        """Downloads finetuned YOLO if model can not be found in project directory.
        
        Args:
            model_type (str): Which model to grab. Must be in ["yolov8m-football.pt", "yolov8n-football.ptn"]
            metas (dict): Dict containing url and hash for each file (20 vids + 2 YOLO-weights)
            
        Returns:
            None"""
        
        # Prepare
        detector_dir = os.path.join(self.cwd, "pipeline", "detector_models")
        os.makedirs(detector_dir, exist_ok=True)

        print("Downloading model...")

        # download into a dedicated model temp directory
        download_path = os.path.join(self.temp_dir, model_type)
        gdown.download(url=self.metas[model_type], 
                       output=download_path, 
                       quiet=True)

        print("Moving files...")

        # model_path is already the full path including filename
        model_path = os.path.join(detector_dir, model_type)
        shutil.copyfile(download_path, model_path)

        # Clean model temp dir
        print("Removing model temporary dir...")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("Done.\n")
        return


    def grab_vids(self, indices: int|list|str="all", ) -> None:
        """Downloads videos from external DGrive directory to demontrate project. Includes security check using SHA256-hashes.
        
        Args:
            indices (list|int|str): Index, "all" or list of indices of videos. Must be in range [1, 20]. If int: only specified video will be collected.

        Returns: 
            None"""
        
        # Make temp dir to store downloads before sorting
        os.makedirs(self.temp_dir, exist_ok=True)

        # Download specified vids
        if indices == "all":
            with tqdm(total=20, desc="Downloading videos") as pbar:
                for i in range(1, 21):
                    gdown.download(url=self.metas[str(i)+".mp4"], 
                                   output=os.path.join(self.temp_dir, f"{i}.mp4"), 
                                   quiet=True)
                    pbar.update(1)
        
        elif isinstance(indices, int):
            if not (1 <= indices <= 20):
                raise ValueError("Index out of range, must be in [1, 20]")
            print("Downloading video...")
            gdown.download(url=self.metas[str(indices)+".mp4"], 
                           output=os.path.join(self.temp_dir, f"{str(indices)+'.mp4'}"), 
                           quiet=True)

        elif isinstance(indices, list):
            if not all([isinstance(i, int) for i in indices]):
                raise ValueError("All entries in list must be integers.")
            if not all([1 <= i <= 20 for i in indices]):
                raise ValueError("All indices must be in range [1, 20].")
            
            print("Downloading videos...")
            with tqdm(total=len(indices), desc="Downloading videos") as pbar:
                for i in indices:
                    gdown.download(url=self.metas[str(i)+".mp4"], 
                                   output=os.path.join(self.temp_dir, f"{i}.mp4"), 
                                   quiet=True)
                    pbar.update(1)
    
        print("Moving files...")

        # Make vids dir if not existing
        os.makedirs("videos", exist_ok=True)

        # sort files numerically (so 2.mp4 comes before 10.mp4)
        tmp_list = os.listdir(self.temp_dir)
        sorted_files = sorted(tmp_list, key=lambda x: int(os.path.splitext(x)[0]))
        for file in sorted_files:
            shutil.copyfile(os.path.join(self.temp_dir, file), os.path.join(self.cwd, "videos", file))

        # Clean temp dir
        print("Removing temporary dir...")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("Done.\n")
        return


    def initialize_files(self, precheck_existence: bool) -> None:
        """Downloads all required files for project to run. Includes security check using SHA256-hashes."""
        
        if not precheck_existence:
            self.grab_model_on_absence("m")
            self.grab_model_on_absence("n")
            self.grab_vids("all")
            return
        
        # Weights
        all_models = ["yolov8m-football.pt", "yolov8n-football.pt"]
        model_dir = os.path.join(self.cwd, "pipeline", "detector_models")

        existing_models = [file for file in os.listdir(model_dir)
                           if os.path.isfile(os.path.join(model_dir, file)) and file.endswith(".pt")]
        
        required = [model for model in all_models if model not in existing_models]
        if not required:
            print("All models already present. No download needed.")

        else:   
            for model in required:
                self.grab_model_on_absence(model)
            print("Gathered all missing models.\n")
        
        # Vids
        video_dir = os.path.join(self.cwd, "videos")
        existing_videos = os.listdir(video_dir)
        existing_indices = [int(file.split(".")[0]) for file in existing_videos
                           if os.path.isfile(os.path.join(video_dir, file)) and file.endswith(".mp4")]
        required_videos = [i for i in range(1, 21) if i not in existing_indices]

        if required_videos:
            self.grab_vids(required_videos)
            print("Gathered all missing videos.")

        else: 
            print("All videos already present. No download needed.")
        print("Initialization completed.\n")
        return
