import os
import json
from engine import Engine, npTensor, rgbImage, lst
from modules import VideoReader, Display,  recordReplayMultiplex, RRPlexMode
from pipeline.detector import Detector
from pipeline.opticalflow import OpticalFlow
from pipeline.tracker import Tracker
from pipeline.shirtClassifier import ShirtClassifier
from pipeline.initialization import Initializer



SYNC_DATA = True    # Synchronize data with the GDrive cloud storage

# Choose model by setting string as DETECTOR_SIZE (n = faster performance for less precision, m = more precise, requires more computational power)
#   - faster:       "yolov8n-football.pt"
#   - more precise: "yolov8m-football.pt"

MODEL = "yolov8n-football.pt"
VIDEO = 3


if __name__ == "__main__":

  cwd = os.getcwd()
  print(cwd)
  
  if SYNC_DATA:
      with open(os.path.join(cwd, "meta.json")) as f:
          metas = json.load(f)
      init = Initializer(cwd=cwd, metas=metas)
      init.initialize_files(precheck_existence=True)

  recordMode = RRPlexMode.BYPASS
  shape = (960, 540)
  engine = Engine(
    modules=[
      VideoReader(targetSize=shape),
      recordReplayMultiplex(Detector(model=MODEL, cwd=cwd), RRPlexMode.BYPASS),
      recordReplayMultiplex(OpticalFlow(), RRPlexMode.BYPASS),
      recordReplayMultiplex(Tracker(), RRPlexMode.BYPASS),
      recordReplayMultiplex(ShirtClassifier(), RRPlexMode.BYPASS),
      Display(historyBufferSize=1000)
      ],
    signals={
      "image": rgbImage(shape[0], shape[1]),
      "opticalFlow": npTensor((2,)),
      "detections": npTensor((-1, 4)),
      "classes": npTensor((-1,)),
      "tracks": npTensor((-1, 4)),
      "trackVelocities": npTensor((-1, 2)),
      "trackAge": lst(),
      "trackClasses": lst(),
      "trackIds": lst(),
      "teamClasses": lst(),
      "terminate": bool,
      "stopped": bool,
      "testout": int
    })

  data = { "video": f'videos/{VIDEO}.mp4' }
  signals = engine.run(data)