# Robot Head

Arduino servo control and OAK camera tracking. Run these commands from the repository root.

Set up Python dependencies and upload the Arduino Uno firmware (once):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt platformio
pio run -d robot_head_arduino -t upload --upload-port /dev/ttyUSB0
```

Connect the Arduino and OAK camera, then run one script at a time:

```bash
python scripts/arm.py                           # Type commands such as: 1 90
python scripts/arm2.py                          # Smooth control with arrows/WASD; q quits
python scripts/person_follow.py --device 10.12.222.68  # Face tracking
python scripts/focused_person_follow.py --device-ip 10.12.222.68  # Focused tracking + point cloud (RVC4)
python scripts/cloud.py --device-ip 10.12.222.68  # Stereo depth + point cloud
```

Replace the camera IP as needed. Servo scripts default to `/dev/ttyUSB0`; use `--port` for `arm2.py`, `--serial-port` for `focused_person_follow.py`, or edit the port constant in the other servo scripts. Point clouds require calibrated CAM_A/CAM_D cameras.

Open <http://localhost:8082> for camera views. Add `--no-servos` to focused tracking to run without the Arduino.

Stereo uses LENS Neural Depth on RVC4, with CAM_D (left), CAM_A (right), and
640×400 color camera inputs. The `NEURAL_DEPTH_576X360` model produces native
576×360 depth, reducing inference and point-cloud memory versus the largest model.
The point cloud uses the matching rectified left image directly, bypassing
same-camera depth alignment that crashes the tested RVC4 SDK. Coordinates are
in metres; invalid points and depths outside 0.2–8 m are removed. The focused
tracker maps the inverted sensor's cloud to the viewer's Y-up coordinates,
preserving Y and depth while correcting X. NeuralDepth emits grayscale
rectified images, so the cloud appearance is grayscale. Neural Depth requires
Luxonis OS 1.20.4 or newer. See the
[NeuralDepth documentation](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/neural_depth).

Neural camera outputs are capped at 8 FPS to avoid a growing inference backlog;
focused tracking keeps its separate 13 FPS default. PCL logs include frame age
at the host, before browser rendering. Keep `--no-servos` only for camera trials;
omit it to enable Arduino control, using `--serial-port` if needed.

Run a 15-second hardware check without servos or the web viewer, optionally
saving a PLY snapshot (use a textured scene with objects within 0.2–8 m):

```bash
python scripts/check_point_cloud.py --device-ip 10.12.204.183 --output /tmp/robot_head_cloud.ply
```

The check fails on stalled delivery, sparse output, implausible metric scale,
or a 90th-percentile frame age above 750 ms.
Use Ctrl+C to stop a viewer script. Ctrl+Z suspends it and leaves the viewer
ports occupied; resume that shell job with `fg` before stopping it with Ctrl+C.
