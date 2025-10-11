# main.py (hybrid)
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from path_planner import AStarPlanner
from simulation.visualizer import draw_env
from rl.env import SwarmRescueEnv
from rl.train_rl import load_ppo, make_env_fn
from stable_baselines3.common.vec_env import DummyVecEnv

YOLO_MODEL = "yolov8m.pt"
CONF_THRESH = 0.5
NUM_DRONES = 3

# load yolo
model = YOLO(YOLO_MODEL)

# env + planner
env = SwarmRescueEnv(grid_size=20, num_drones=NUM_DRONES, num_humans=5)
planner = AStarPlanner(env.grid_size)
state = env.reset()

# Try to load PPO
ppo_model = None
try:
    ppo_model = load_ppo("rl/checkpoints/ppo_swarm.zip")
    print("Loaded PPO model.")
except Exception:
    print("No PPO model found; running planner-only + random actions fallback.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    human_positions = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            if label == "person" and conf >= CONF_THRESH:
                b = box.xyxy[0].cpu().numpy().astype(int)
                x = int(((b[0] + b[2]) / 2) / frame.shape[1] * env.grid_size)
                y = int(((b[1] + b[3]) / 2) / frame.shape[0] * env.grid_size)
                human_positions.append([x, y])
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (b[0], b[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # update env human positions
    env.place_humans(human_positions)

    # Hybrid action selection
    # 1) planner suggests per-drone next cell if human(s) visible
    planned_next = [None] * env.num_drones
    for di in range(env.num_drones):
        if len(human_positions) > 0:
            # pick nearest human for that drone
            dpos = tuple(env.drone_pos[di])
            dists = [np.linalg.norm(np.array(h)-np.array(dpos)) for h in human_positions]
            nearest = human_positions[int(np.argmin(dists))]
            path = planner.plan(dpos, tuple(nearest), env.obstacles)
            if len(path) > 0:
                planned_next[di] = path[0]  # next cell

    # 2) convert planned_next to actions (0,1,2,3,4). If planned_next None -> use PPO (if available) else stay/random.
    actions = []
    for di in range(env.num_drones):
        p = planned_next[di]
        if p is not None:
            # compute action to go from current pos to p
            cx, cy = env.drone_pos[di]
            nx, ny = p
            if nx == cx and ny == cy:
                action = 0
            elif nx == cx and ny == cy - 1:
                action = 1  # up
            elif nx == cx and ny == cy + 1:
                action = 2  # down
            elif nx == cx - 1 and ny == cy:
                action = 3  # left
            elif nx == cx + 1 and ny == cy:
                action = 4  # right
            else:
                action = 0
            actions.append(action)
        else:
            # planner not suggesting — fallback to PPO or random
            if ppo_model:
                # SB3 expects vectorized env — create a temporary env wrapper for prediction
                single_env = DummyVecEnv([make_env_fn(grid_size=env.grid_size, num_drones=env.num_drones, num_humans=env.num_humans)])
                # set env state to current env; this is a hack: we cannot easily set internal state; instead use model.predict on obs
                obs = env._get_obs().reshape(1, -1)
                action, _ = ppo_model.predict(obs, deterministic=True)
                # action will be array of size num_drones
                if hasattr(action, "__len__"):
                    actions.extend(list(action))
                else:
                    # single scalar fallback
                    actions.append(int(action))
            else:
                # random fallback
                actions.append(0)

    # if actions list too long (because PPO returned a vector), shorten/pad to num_drones
    if len(actions) > env.num_drones:
        actions = actions[:env.num_drones]
    while len(actions) < env.num_drones:
        actions.append(0)

    obs, reward, done, info = env.step(actions)

    # create map image
    map_img = draw_env(env.grid_size, env.drone_pos, env.human_pos, env.obstacles)

    # compose frame+map
    frame_resized = cv2.resize(frame, (640,480))
    map_resized = cv2.resize(map_img, (480,480))
    combined = np.hstack((frame_resized, map_resized))
    cv2.imshow("Hybrid Rescue - Live", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
