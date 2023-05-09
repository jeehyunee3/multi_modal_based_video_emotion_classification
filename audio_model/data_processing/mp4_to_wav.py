# Load Packages
import pandas as pd
import shutil
from tqdm import tqdm
from pathlib import Path

# 1. Set filepath
filePath='./'

# 2. Load m4a video files
video_files = tqdm(list(Path(filePath).glob("**/*.mp4")))
print(len(video_files))

#3. Convert m4a format to wav
for video_file in video_files:
    name = video_file.stem
    print(name + '.wav') 
    shutil.copy(video_file, outPath + name + '.wav') 

print("finish save")