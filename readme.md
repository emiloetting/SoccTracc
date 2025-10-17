# SoccTracc⚽✨🔍
<p align=center>

![MPT_demo (online-video-cutter com)](https://github.com/user-attachments/assets/a855e84f-8227-4406-9ade-c1e13b956246)


## Welcome to SoccTracc, a project on tracking soccer players 🏃

**Key Features**
- :link:[YOLO]([https://github.com/ultralytics])-based player detection 🎯
- calculation of optical flow 📹
- clustering-based player classification 🎭
- individual tracking for each detected player using Kalman-filter 🔭

<p align=center>
  
<img width="500" height="200" alt="grafik" src="https://github.com/user-attachments/assets/cc11ca33-07ce-47cd-8ee0-132ba24ac95f" />

</p>

## Installation Guide📦
1. Grab your clone by using:
```bash
git clone https://github.com/emiloetting/SoccTracc.git
```
2. After cloning, collect dependancies into your environment with:
```bash
pip install -r requirements.txt
```

## Quickstart with automatic download📫
In order to run the code, there are some external files required (weights of finetuned model & demo videos).  
All of these can be found in a dedicated Google Drive storage: https://drive.google.com/drive/folders/1G8HcAsniumJZfdiyTxgne5Y0p9k_DQo5 

But don't worry, per default, these files will automatically be collected on execution of `main.py` using :link:[Gdown]([https://github.com/wkentaro/gdown]) :)
<p align=center>
<img width="889" height="83" alt="image" src="https://github.com/user-attachments/assets/db0bf1b7-f7b4-4e13-894c-83a7e404b410" />

Note✏️: Ensure you set your working directory to the repo-clone  

Once you collected the files, change `SYNC_DATA` to `False` in order to avoid auto-download on each execution.  
By default, the smaller model will be used on video 3 as displayed in the demo-gif:  
<p aling=center>
<img width="1778" height="173" alt="image" src="https://github.com/user-attachments/assets/e2d6ac35-898c-464b-9769-fc96860d4888" />

</p>
Feel free to change and explore ;)  

Now, all that's left to do is change your working directory to the repo-clone, execute the `main.py` and have a blast🎉💫

## Troubleshooting: GDown not working?
Sometimes GDown faces troubles trying to grab the data from cloud storage.  
In this case, simply try again and it's likely to work. 
Should the download simply not work, you can access the files directly using the link mentioned above.  
❗Important: file locations❗:  
To make the code work, don't forget to put the files into the correct subdirectories:
- videos:        `SoccTracc/videos/`
- model weights: `SoccTracc/pipeline/detector_models/`

## Any questions left?📨
Should any questions remain, feel free to contact us, we'll be happy to help!😊

