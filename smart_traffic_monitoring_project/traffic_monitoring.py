# (Code truncated here for brevity in notebook environment)
# The full code is already provided in the canvas above.
# ------------- IMPORTS -------------
import cv2
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import webbrowser

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# Constants
FRAME_WIDTH = 1280
BOX_TOP_LEFT = (-50, 60)
BOX_BOTTOM_RIGHT = (FRAME_WIDTH + 50, 360)
DETECTION_ZONE_LENGTH_METERS = 10
LINE_X_Y = BOX_TOP_LEFT[1] + 100
LINE_Y_Y = LINE_X_Y + 160

vehicle_log = []

# ------------- MAIN VIDEO PROCESSING FUNCTION -------------
def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    vehicle_entry_time = {}
    vehicle_speed_record = {}
    vehicle_exit_recorded = set()
    vehicle_counted = set()
    vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.3)[0]

        if results is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, track_id, class_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_name = model.names[int(class_id)]
                rid = int(track_id)

                if class_name in vehicle_classes:
                    in_box = BOX_TOP_LEFT[0] <= cx <= BOX_BOTTOM_RIGHT[0] and BOX_TOP_LEFT[1] <= cy <= BOX_BOTTOM_RIGHT[1]

                    if in_box:
                        if rid not in vehicle_entry_time:
                            vehicle_entry_time[rid] = datetime.now()

                        if rid not in vehicle_counted:
                            vehicle_count += 1
                            vehicle_counted.add(rid)

                        time_spent = (datetime.now() - vehicle_entry_time[rid]).total_seconds()
                        speed_kmph = (DETECTION_ZONE_LENGTH_METERS / time_spent) * 3.6 if time_spent > 0 else 0
                        vehicle_speed_record[rid] = speed_kmph

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID:{rid} | {int(speed_kmph)} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        if rid in vehicle_entry_time and rid not in vehicle_exit_recorded:
                            vehicle_exit_recorded.add(rid)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # ROI Box Area
        cv2.rectangle(frame, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 0, 255), 3)
        cv2.line(frame, (BOX_TOP_LEFT[0], LINE_X_Y), (BOX_BOTTOM_RIGHT[0], LINE_X_Y), (0, 255, 255), 2)
        cv2.line(frame, (BOX_TOP_LEFT[0], LINE_Y_Y), (BOX_BOTTOM_RIGHT[0], LINE_Y_Y), (0, 255, 255), 2)

        cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Vehicle Detection & Speed Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    for rid in vehicle_entry_time:
        crossing_time = vehicle_entry_time[rid].strftime("%Y-%m-%d %H:%M:%S")
        speed = int(vehicle_speed_record.get(rid, 0))
        vehicle_log.append({'Vehicle ID': rid, 'Crossing Time': crossing_time, 'Speed (km/h)': speed})

    show_summary()


# ------------- EXPORT TO EXCEL -------------
def export_to_excel():
    df = pd.DataFrame(vehicle_log)
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Success", "Data exported to Excel successfully!")


# ------------- SUMMARY WINDOW -------------
def show_summary():
    summary_win = tk.Tk()
    summary_win.title("Vehicle Summary Report")
    summary_win.geometry("600x400")

    tk.Label(summary_win, text="Vehicle Crossing Summary",
             font=("Helvetica", 16, "bold")).pack(pady=10)

    tree = ttk.Treeview(summary_win, columns=("ID", "Time", "Speed"), show="headings")
    tree.heading("ID", text="Vehicle ID")
    tree.heading("Time", text="Crossing Time")
    tree.heading("Speed", text="Speed (km/h)")
    tree.pack(fill="both", expand=True)

    for entry in vehicle_log:
        tree.insert("", "end",
                    values=(entry['Vehicle ID'], entry['Crossing Time'], entry['Speed (km/h)']))

    tk.Button(summary_win, text="Export to Excel", font=("Helvetica", 12),
              bg="#4CAF50", fg="white", command=export_to_excel).pack(pady=10)

    summary_win.mainloop()


# ------------- APPLICATION GUI -------------
def create_gui():

    def browse_video():
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            root.destroy()
            run_video(file_path)

    def open_crri_website():
        webbrowser.open("https://crridom.gov.in/")

    root = tk.Tk()
    root.title("CRRI - Smart Traffic Monitoring")
    root.geometry("900x550")
    root.configure(bg="#f2f6f9")

    tk.Label(root, text="Central Road Research Institute (CRRI)",
             font=("Helvetica", 22, "bold"), fg="#003366", bg="#f2f6f9").pack(pady=5)

    tk.Button(root, text="Run Default Video", font=("Helvetica", 12, "bold"),
              bg="#4CAF50", fg="white",
              command=lambda: [root.destroy(), run_video("traffic.mp4")]).pack(pady=10)

    tk.Button(root, text="Select Video from Gallery", font=("Helvetica", 12, "bold"),
              bg="#2196F3", fg="white", command=browse_video).pack(pady=5)

    tk.Button(root, text="Visit CRRI Website", font=("Helvetica", 11),
              bg="#f44336", fg="white", command=open_crri_website).pack(pady=15)

    root.mainloop()


# ------------- START THE PROGRAM -------------
create_gui()
