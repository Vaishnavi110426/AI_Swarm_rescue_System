# dashboard/app.py
import sys
import os
import time
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import heapq

# Add root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.env import SwarmRescueEnv

# ---------------- Streamlit Setup ----------------
st.set_page_config(layout="wide")
st.title("🛰️ AI Disaster Rescue Simulator v2.0")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ Settings")
use_camera = st.sidebar.checkbox("Use Webcam", value=True)
show_map = st.sidebar.checkbox("Show Map", value=True)
show_detection = st.sidebar.checkbox("Run YOLO Detection", value=True)
grid_size = st.sidebar.slider("Grid size", 10, 40, 20)
num_drones = st.sidebar.slider("Number of drones", 1, 5, 3)
detection_conf = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.3)
record_demo = st.sidebar.checkbox("Record Demo", value=False)

# ---------------- Create Environment ----------------
env = SwarmRescueEnv(grid_size=grid_size, num_drones=num_drones, num_humans=0)
base_pos = (0, 0)  # Base location for drones

# ---------------- Stream Slots ----------------
frame_slot = st.empty()
map_slot = st.empty()

# ---------------- Load YOLO Model ----------------
if show_detection:
    if "yolo_model" not in st.session_state:
        model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
        st.session_state.yolo_model = YOLO(model_path)
        st.sidebar.success("✅ YOLOv8 model loaded.")
    model = st.session_state.yolo_model
else:
    model = None

# ---------------- Camera Setup ----------------
cap = None
if use_camera:
    cap = cv2.VideoCapture(0)

# ---------------- Video Recording Setup ----------------
video_writer = None
if record_demo:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('rescue_demo.avi', fourcc, 20.0, (640, 480))

# ---------------- Control Buttons ----------------
col1, col2 = st.columns(2)
start_stream = col1.button("▶ Start Stream", key="start_btn")
reset_env = col2.button("🔄 Reset Environment", key="reset_btn")

if reset_env:
    env.reset()
    st.success("Environment reset successfully!")

# ---------------- Helper Functions ----------------
def map_to_grid(x, y, frame_w, frame_h, grid_size):
    gx = int((x / frame_w) * grid_size)
    gy = int((y / frame_h) * grid_size)
    return gx, gy

def astar(start, goal, grid):
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start)-np.array(goal))}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0]:
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                tentative_g = g_score[current]+1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current
    return [start]

def move_drones(drone_positions, targets, grid, rescued_flags):
    new_positions = []
    occupied = set()
    for idx, (dx, dy) in enumerate(drone_positions):
        if rescued_flags[idx]:
            # Drone returning to base
            path = astar((dx, dy), base_pos, grid)
            if len(path) > 0:
                nx, ny = path[0]
            else:
                nx, ny = dx, dy
        elif targets:
            # Assign drone to highest priority human (closest to hazard = top-left)
            nearest = min(targets, key=lambda t: (t[0]**2 + t[1]**2))
            path = astar((dx, dy), nearest, grid)
            if len(path) > 0:
                nx, ny = path[0]
                # Check if drone reached human
                if (nx, ny) == nearest:
                    rescued_flags[idx] = True
                    targets.remove(nearest)
            else:
                nx, ny = dx, dy
        else:
            nx, ny = dx, dy
        if (nx, ny) in occupied:
            nx, ny = dx, dy
        occupied.add((nx, ny))
        new_positions.append((nx, ny))
    return new_positions

def draw_dynamic_map(grid_size, drone_pos, object_positions, human_heatmap=None):
    cell = 24
    img_size = grid_size*cell
    map_img = np.full((img_size,img_size,3),30,dtype=np.uint8)
    for i in range(0, img_size, cell):
        cv2.line(map_img,(i,0),(i,img_size),(60,60,60),1)
        cv2.line(map_img,(0,i),(img_size,i),(60,60,60),1)
    # heatmap
    if human_heatmap is not None:
        max_heat = np.max(human_heatmap) if np.max(human_heatmap)>0 else 1
        for y in range(grid_size):
            for x in range(grid_size):
                intensity = int((human_heatmap[y,x]/max_heat)*255)
                if intensity>0:
                    x1,y1=x*cell,y*cell
                    x2,y2=x1+cell,y1+cell
                    overlay_color = (0,0,intensity)
                    map_img[y1:y2,x1:x2] = cv2.addWeighted(map_img[y1:y2,x1:x2],0.5,
                                                           np.full((cell,cell,3),overlay_color,dtype=np.uint8),0.5,0)
    colors={"person":(0,0,255),"car":(255,0,0),"bicycle":(0,255,255)}
    for obj_type, positions in object_positions.items():
        for gx,gy in positions:
            cx,cy=gx*cell+cell//2,gy*cell+cell//2
            color=colors.get(obj_type,(200,200,200))
            cv2.circle(map_img,(cx,cy),cell//4,color,-1)
    for dx,dy in drone_pos:
        cx,cy=dx*cell+cell//2,dy*cell+cell//2
        cv2.circle(map_img,(cx,cy),cell//3,(0,255,0),-1)
    cv2.rectangle(map_img,(0,0),(img_size-1,img_size-1),(150,150,150),2)
    return map_img

# ---------------- Main Stream Loop ----------------
if start_stream:
    st.info("Press Stop to end stream.")
    human_heatmap=np.zeros((grid_size,grid_size),dtype=int)
    obstacles = [(np.random.randint(0,grid_size), np.random.randint(0,grid_size)) for _ in range(grid_size)]
    rescued_flags = [False]*len(env.drone_pos)
    try:
        while True:
            if use_camera and cap:
                ret, frame = cap.read()
                if not ret:
                    st.warning("No camera frame.")
                    break
            else:
                frame = np.ones((480,640,3),dtype=np.uint8)*50
                cv2.putText(frame,"Camera disabled",(50,250),cv2.FONT_HERSHEY_SIMPLEX,1.2,(200,200,200),2)

            frame_h, frame_w, _ = frame.shape
            object_positions=defaultdict(list)

            # YOLO detection
            if show_detection and model:
                small_frame=cv2.resize(frame,(320,240))
                results=model(small_frame,conf=detection_conf)
                annotated_frame=cv2.resize(results[0].plot(),(frame_w,frame_h))
                for box in results[0].boxes:
                    cls=int(box.cls[0])
                    conf=float(box.conf[0])
                    label=model.names[cls]
                    if conf<detection_conf: continue
                    x1,y1,x2,y2=box.xyxy[0].tolist()
                    cx,cy=(x1+x2)/2,(y1+y2)/2
                    gx,gy=map_to_grid(cx*frame_w/320,cy*frame_h/240,frame_w,frame_h,grid_size)
                    object_positions[label].append((gx,gy))
                    if label=="person":
                        human_heatmap[gy,gx]+=1
            else:
                annotated_frame=frame

            # Update environment
            grid=np.zeros((grid_size,grid_size),dtype=int)
            for ox,oy in obstacles:
                grid[oy,ox]=1
            env.human_pos=object_positions.get("person",[])
            env.drone_pos=move_drones(env.drone_pos, env.human_pos, grid, rescued_flags)

            # Sidebar metrics
            for obj, positions in object_positions.items():
                st.sidebar.metric(f"{obj.capitalize()} Detected", len(positions))
            st.sidebar.metric("Drones Active", len(env.drone_pos))
            if env.human_pos:
                st.sidebar.warning(f"⚠️ Humans detected: {len(env.human_pos)}")

            # Display feed
            frame_rgb=cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, width=640,
                             caption="Live Feed (Detection)" if show_detection else "Live Feed")
            if show_map:
                map_img=draw_dynamic_map(grid_size, env.drone_pos, object_positions,human_heatmap)
                map_slot.image(map_img, width=480, caption="Dynamic Map")

            # Save demo video
            if record_demo and video_writer:
                video_writer.write(cv2.resize(frame_rgb,(640,480)))

            time.sleep(0.05)

    except Exception as e:
        st.error(f"Stream stopped: {e}")
    finally:
        if cap: cap.release()
        if video_writer: video_writer.release()
        st.success("Demo recording saved as rescue_demo.avi" if record_demo else "")
