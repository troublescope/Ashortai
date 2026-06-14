# 🎬 OpenSource Clipping — Google Colab Guide

This guide provides a "copy-paste" solution to run the AI Auto-Clipper in Google Colab.

## 💡 Pre-requisites
1. **GPU Runtime**: In Colab, go to `Runtime` -> `Change runtime type` -> select `T4 GPU`.
2. **API Key**: Get a free Gemini API Key from [Google AI Studio](https://aistudio.google.com/apikey).

---

## 🛠️ Cell 1: Setup Environment
Copy and paste this into your first Colab cell to install everything.

```python
# @title 🛠️ Step 1: Setup Environment
import os

print("⏳ Cleaning workspace...")
!rm -rf ./* ./.*

print("⏳ Cloning OpenSource Clipping...")
!git clone https://github.com/troublescope/Ashortai.git .

print("⏳ Installing FFmpeg and dependencies (1-2 mins)...")
!apt-get -qq update && apt-get -qq install -y ffmpeg
!pip install -q -r requirements.txt

print("\n✅ Setup Complete! Move to the next cell.")
```

---

## 🚀 Cell 2: Run the AI Clipper
Copy and paste this into your second cell. It uses Colab Forms for easy input.

```python
# @title 🚀 Step 2: Run the AI Clipper
# @markdown Enter your credentials and video details below:

GOOGLE_API_KEY = "" # @param {type:"string"}
VIDEO_URL = "https://www.youtube.com/watch?v=Dc4_aBFAYWE" # @param {type:"string"}
NUMBER_OF_CLIPS = 3 # @param {type:"slider", min:1, max:10, step:1}
OUTPUT_RATIO = "9:16" # @param ["9:16", "1:1", "16:9"]
FONT_STYLE = "HORMOZI" # @param ["HORMOZI", "STORYTELLER", "CINEMATIC", "DEFAULT"]

import os
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    print("❌ ERROR: Please enter your GOOGLE_API_KEY from Google AI Studio!")
else:
    print(f"🎬 Starting processing for: {VIDEO_URL}")
    !python main.py \
        --url "{VIDEO_URL}" \
        --clips {NUMBER_OF_CLIPS} \
        --ratio "{OUTPUT_RATIO}" \
        --font-style "{FONT_STYLE}" \
        --whisper-compute-type "float32"

    print("\n✅ DONE! You can find your videos in the 'outputs/' folder on the left sidebar.")
```

---

## 📂 Downloading Results
1. Click the **Folder icon** 📁 on the left sidebar of Colab.
2. Navigate to the `outputs` directory.
3. Right-click your generated `.mp4` clips and select **Download**.
