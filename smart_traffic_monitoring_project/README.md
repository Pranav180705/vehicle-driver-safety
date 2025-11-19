# Smart Traffic Monitoring System

A computer vision based traffic monitoring system using YOLOv8 for vehicle detection, tracking, counting and speed estimation. Includes a Tkinter GUI and Excel export for reports.
This project demonstrates the successful development and implementation of a computer vision-based smart traffic monitoring system using the YOLOv8 object detection model. The system is capable of real-time vehicle detection, classification, counting, and speed estimation, all while offering a clean and intuitive user interface.
6.1 Key conclusions from the project:
•	The YOLOv8 model offers reliable and real-time vehicle tracking performance, even on CPU-only systems using the lightweight yolov8n.pt version.
•	Speed estimation using a fixed-distance method between two virtual lines proved to be simple yet effective.
•	The integration of GUI and data export enhances the usability of the system for field deployment or academic study.
•	The solution provides a scalable foundation for future enhancements, including integration with live CCTV feeds, license plate recognition, and rule violation detection.
In essence, the project fulfills the goal of developing a cost-effective, efficient, and real-time smart traffic monitoring system suitable for urban traffic analysis and road safety enforcement initiatives.
6.2 Future Improvements (Brief)
1.	Live Camera Integration
Connect the system to real-time CCTV or traffic cameras for continuous monitoring.
2.	License Plate Recognition (ANPR)
Add automatic number plate recognition to identify and record vehicle registration numbers.
3.	Violation Detection
Detect traffic rule violations like signal jumping, wrong-lane driving, and over-speeding.
4.	Weather & Night Adaptation
Improve detection in low-light or bad weather using infrared or thermal camera support.



## Quick start

1. Clone the repo:
```bash
git clone <repo-url>
cd smart-traffic-monitoring-repo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python traffic_monitoring.py
```

## License
MIT
