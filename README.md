🛰️ AI Swarm Rescue System – Disaster Response using Drones & AI

An intelligent multi-drone coordination system that uses YOLOv8 object detection, Reinforcement Learning (PPO), and A* path planning to autonomously detect, rescue, and map survivors in disaster-hit areas — powered by Streamlit Dashboard for real-time visualization.

live demo on render : https://ai-swarm-rescue-system-1.onrender.com

🚀 Overview

Natural disasters often leave humans trapped or stranded in hard-to-reach zones.
Our AI Swarm Rescue System enables autonomous drones to:

Detect humans or hazards using YOLOv8.

Coordinate movements using Reinforcement Learning + A*.

Display real-time mission maps and summaries through an interactive web dashboard.

This project demonstrates how AI, Robotics, and Computer Vision can save lives during disaster management.

🧠 Features

✅ YOLOv8 Real-Time Detection – Locates humans, vehicles, and obstacles.
✅ Multi-Drone Coordination – Swarm control with PPO agents.
✅ Dynamic Map Visualization – Grid-based map with heatmaps & icons.
✅ Path Planning (A*) – Efficient obstacle avoidance and target tracking.
✅ Mission Dashboard – Displays metrics, logs, and rescue progress.
✅ Disaster Scenario Simulation – Earthquake, Flood, and Wildfire modes.
✅ Hybrid Policy Control – Combines Reinforcement Learning with Planner decisions.

🏗️ System Architecture
                 ┌─────────────────────────────┐
                 │  Streamlit Command Center   │
                 │  - Mission Dashboard        │
                 │  - Video Feed + Map         │
                 └────────────┬────────────────┘
                              │
                   Real-Time Detection (YOLOv8)
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
 RL Controller (PPO Agents)              Path Planner (A*)
        │                                           │
        └───────────────┬───────────────────────────┘
                        │
                Swarm Environment (Simulation)
                        │
                 Drones ↔ Humans ↔ Obstacles


⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/AI_Swarm_Rescue_System.git
cd AI_Swarm_Rescue_System

2️⃣ Create a Virtual Environment
python -m venv venv310
venv310\Scripts\activate  # (Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Dashboard
streamlit run dashboard/app.py

🎮 Usage

1️⃣ Choose Webcam / Simulation mode.
2️⃣ Select Disaster Type (Flood / Fire / Earthquake).
3️⃣ Start the stream and observe:

YOLO detects humans in live feed.

Drones autonomously navigate & rescue.

Dashboard updates rescue metrics, mission logs, and heatmaps.

🧠 Model Training
🎯 YOLOv8 Fine-Tuning
yolo train data=data/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640

🤖 PPO Training (Multi-Agent)
python rl/train_rl.py

📊 Example Output
Module	Description
🧍 Human Detection	Real-time identification using YOLOv8
🚁 Drone Movement	Autonomous navigation on A* grid
🗺️ Map Visualization	Dynamic grid with drone & survivor positions
📈 Mission Dashboard	Live stats, logs, and rescue summary
🧩 Technologies Used

Python 3.10

YOLOv8 (Ultralytics)

Stable-Baselines3 (PPO)

OpenCV

NumPy / Pandas

Streamlit

Matplotlib

screen shots:

<img width="978" height="1004" alt="image" src="https://github.com/user-attachments/assets/52f99a48-18e0-43ce-924f-baceb227e445" />

<img width="1920" height="1021" alt="Screenshot (1005)" src="https://github.com/user-attachments/assets/7fb6bccc-2845-4418-8c05-50837431f8be" />

<img width="1920" height="1000" alt="Screenshot (1007)" src="https://github.com/user-attachments/assets/835cb459-495d-4788-b771-e8ffb05d5096" />

🤝 Contributions

Pragathi Vaishnavi  – Lead Developer, AI/ML Integration, RL Agent Design.
