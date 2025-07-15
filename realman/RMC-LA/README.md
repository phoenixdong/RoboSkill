# 🤖 Realman Robot Skill Server

This project provides skill services for a Realman robot, including object grasping with a wrist-mounted camera and target navigation. It integrates:

- 🦾 Realman Arm SDK
- 🦿 Realman Chassis Control (TCP)
- 🎯 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for grounding-based visual detection
- 📷 Intel RealSense Camera
- 🧠 FastMCP protocol server

---

## 📦 Prerequisites

- Python 3.10+
- RealSense D435i (or compatible)
- PyTorch 2.6.0
- Redis server
- Realman robot (arm + chassis)
- GroundingDINO (see below)

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/FlagOpen/RoboSkill

cd RoboSkill/realman/RMC-LA

```

### 2. Install dependencies

We recommend using a fresh virtual environment (e.g. with conda or venv):

```bash
pip install -r requirements.txt
```

### 3. Install GroundingDINO
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install -e .

```
Download the weights:
```bash
mkdir weights
wget https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -P weights/
```

## 🔧 Configuration
Before running, you must configure IP and port settings in `skill.py`:

```bash
# chassis
chassis_host = "127.0.0.1"   # 🔁 Set to actual robot chassis IP
chassis_port = 5000

# arm
arm_host = "127.0.0.1"       # 🔁 Set to actual robotic arm IP
arm_port = 5000

```

## 🚀 Running the Service
```bash
python skill.py
```
This will:

+ Start the MCP-compatible skill server on http://0.0.0.0:8000

+ Load the GroundingDINO model

+ Initialize the RealSense camera

+ Initialize the Realman robotic arm



## 📂 Output

+ `./output/wrist_obs.png`: Captured color image

+ `./output/annotated_image.png`: Visualized prediction with bounding boxes

+ `./output/wrist_obs_depth.npy`: Depth image



## 🛠️ Troubleshooting

+ ❌ No RealSense Device Found
Make sure the camera is plugged in and pyrealsense2 is installed.

+ ❌ Grasping Fails
Ensure good lighting and that the target object is in the field of view.

+ ❌ Cannot connect to robot
Double-check the IP/port values in skill.py.


## 📄 License



## ✨ Acknowledgments
+ [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
+ [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)