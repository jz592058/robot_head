"""Track the largest face using a head crop and display the stereo point cloud."""

import argparse
from datetime import timedelta
import time

import depthai as dai
from depthai_nodes.message import Collection, GatheredData
from depthai_nodes.node import (
    CoordinatesMapper,
    ExtendedNeuralNetwork,
    FrameCropper,
    GatherData,
    ParsingNeuralNetwork,
)
from dotenv import load_dotenv

from cloud import (
    DEPTH_FPS,
    OUTPUT_SIZE as STEREO_SIZE,
    configureNeuralDepth,
    createPointCloud,
    getStereoSockets,
)
from person_follow import (
    DEFAULT_PORT,
    connect_to_arm,
    get_first_face_detection,
    get_latest_message,
    initialize_servos,
    send_pending_servo_updates,
    update_servo_targets,
)

HIGH_RES_SIZE = (2000, 2000)
LOW_RES_SIZE = (640, 640)
FOCUSED_SERVO_STEP = 3  # 50% faster than person_follow.py's 2-degree step
FRAME_TYPE = dai.ImgFrame.Type.BGR888i
PERSON_MODEL = dai.NNModelDescription(
    "luxonis/scrfd-person-detection:25g-640x640", platform="RVC4"
)
FACE_MODEL = dai.NNModelDescription("luxonis/yunet:320x240", platform="RVC4")


class LargestDetection(dai.node.HostNode):
    """Pass on only the largest detection while keeping its frame metadata."""

    def build(self, detections):
        self.link_args(detections)
        return self

    def process(self, message):
        if message.detections:
            largest = max(
                message.detections,
                key=lambda det: (det.xmax - det.xmin) * (det.ymax - det.ymin),
            )
            for detection in message.detections.copy():
                if detection is not largest:
                    message.detections.remove(detection)
        self.out.send(message)


class HeadRegion(dai.node.HostNode):
    """Turn a person detection into an upper-body region for face search."""

    def build(self, detections):
        self.link_args(detections)
        return self

    def process(self, message):
        for detection in message.detections:
            original_ymin = detection.ymin
            original_height = detection.ymax - original_ymin
            detection.ymin = max(0.0, original_ymin - 0.03)
            detection.ymax = min(1.0, original_ymin + original_height * 0.15)
        self.out.send(message)


class FacesFromCollection(dai.node.HostNode):
    """Extract face detections after the coordinate mapper."""

    def build(self, mapped):
        self.link_args(mapped)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, message):
        if isinstance(message, GatheredData):
            reference = message.reference_data
            items = message.items
        elif isinstance(message, Collection):
            reference = message
            items = message.items
        else:
            raise TypeError(f"Unexpected mapped detections: {type(message)}")
        if items:
            return items[0]
        empty = dai.ImgDetections()
        empty.setTimestamp(reference.getTimestamp())
        empty.setTimestampDevice(reference.getTimestampDevice())
        empty.setSequenceNum(reference.getSequenceNum())
        return empty


class PointCloudForViewer(dai.node.HostNode):
    """Map the upside-down sensor's optical coordinates to the Y-up viewer."""

    def build(self, point_cloud):
        self.link_args(point_cloud)
        self.inputs["message"].setMaxSize(1)
        self.inputs["message"].setBlocking(False)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, message):
        points, colors = message.getPointsRGB()
        # The inverted sensor already gives physically upper points positive Y,
        # which is also up in the 3D viewer. Flipping Y would turn them upside down.
        # Keep the horizontal mounting correction; preserve Z and point colors.
        points[:, 0] *= -1
        message.setPointsRGB(points, colors)
        message.updateBoundingBox()
        return message


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python3 focused_person_follow.py --device-ip 10.12.222.68",
    )
    parser.add_argument(
        "-d",
        "--device",
        "--device-ip",
        dest="device",
        help="Camera device ID or IP address (for example, 10.12.222.68)",
    )
    parser.add_argument("--fps-limit", type=int, default=13)
    parser.add_argument("--serial-port", default=DEFAULT_PORT)
    parser.add_argument(
        "--no-servos", action="store_true", help="Run detection without moving the servos"
    )
    parser.add_argument(
        "--duration", type=float, help="Stop after this many seconds (useful for a trial run)"
    )
    args = parser.parse_args()
    if args.fps_limit <= 0:
        parser.error("--fps-limit must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def rotate_180(pipeline, camera_output, size):
    """Rotate each camera branch before either inference or web streaming."""
    rotate = pipeline.create(dai.node.ImageManip)
    rotate.initialConfig.addRotateDeg(180)
    rotate.setMaxOutputFrameSize(size[0] * size[1] * 3)
    camera_output.link(rotate.inputImage)
    return rotate.out


def build_face_zoom(pipeline, detections, frames, *, sparse=False):
    """Crop axis-aligned face boxes; avoid retaining high-res frames while idle."""
    script = pipeline.create(dai.node.Script)
    if sparse:
        # Focused detections can stop when no person is visible. Keep only the
        # newest high-res frame instead of waiting for a timestamp match forever.
        script.inputs["frame"].setBlocking(False)
        script.inputs["frame"].setMaxSize(1)
        script.inputs["detections"].setBlocking(False)
        script.inputs["detections"].setMaxSize(1)
        frames.link(script.inputs["frame"])
        detections.link(script.inputs["detections"])
        input_script = (
            "        detections = node.inputs['detections'].get()\n"
            "        if not detections.detections:\n"
            "            continue\n"
            "        frame = node.inputs['frame'].tryGet()\n"
            "        if frame is None:\n"
            "            continue\n"
        )
    else:
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=10))
        sync.setSyncAttempts(-1)
        frames.link(sync.inputs["frame"])
        detections.link(sync.inputs["detections"])
        sync.out.link(script.inputs["synced"])
        input_script = (
            "        group = node.inputs['synced'].get()\n"
            "        frame = group['frame']\n"
            "        detections = group['detections']\n"
        )

    script.setScript(
        "try:\n"
        "    while True:\n"
        + input_script
        + """        for det in detections.detections:
            rect = RotatedRect()
            rect.center.x = (det.xmin + det.xmax) / 2
            rect.center.y = (det.ymin + det.ymax) / 2
            rect.size.width = det.xmax - det.xmin + 0.04
            rect.size.height = det.ymax - det.ymin + 0.04
            rect.angle = 0
            if rect.size.width <= 0 or rect.size.height <= 0:
                continue
            cfg = ImageManipConfig()
            cfg.addCropRotatedRect(rect, True)
            cfg.setOutputSize(320, 320, ImageManipConfig.ResizeMode.LETTERBOX)
            cfg.setFrameType(ImgFrame.Type.BGR888i)
            cfg.setTimestamp(detections.getTimestamp())
            cfg.setTimestampDevice(detections.getTimestampDevice())
            cfg.setSequenceNum(detections.getSequenceNum())
            node.outputs['image'].send(frame)
            node.outputs['config'].send(cfg)
except Exception as exc:
    node.error(str(exc))
"""
    )

    crop = pipeline.create(dai.node.ImageManip)
    crop.setMaxOutputFrameSize(320 * 320 * 3)
    crop.setNumFramesPool(7)
    crop.inputImage.setMaxSize(5)
    crop.inputConfig.setMaxSize(5)
    crop.inputConfig.setWaitForMessage(True)
    script.outputs["image"].link(crop.inputImage)
    script.outputs["config"].link(crop.inputConfig)
    return crop.out


def build_point_cloud(pipeline, calibration, fps):
    """Share the calibrated stereo cameras with tracking and RGBD."""
    left_socket, right_socket = getStereoSockets(calibration)
    print(f"Using calibrated stereo pair: {left_socket.name} -> {right_socket.name}")
    cameras = {}
    for socket in (left_socket, right_socket):
        camera = pipeline.create(dai.node.Camera).build(socket, sensorFps=fps)
        camera.initialControl.setManualFocus(calibration.getLensPosition(socket))
        cameras[socket] = camera

    # Keep stereo inputs unrotated so they match the calibration. Tracking
    # applies its 180-degree rotation only to its own outputs from CAM_D.
    left_output = cameras[left_socket].requestOutput(
        STEREO_SIZE, type=dai.ImgFrame.Type.BGR888i,
        fps=min(fps, DEPTH_FPS), enableUndistortion=False
    )
    right_output = cameras[right_socket].requestOutput(
        STEREO_SIZE, type=dai.ImgFrame.Type.BGR888i,
        fps=min(fps, DEPTH_FPS), enableUndistortion=False
    )
    stereo = configureNeuralDepth(pipeline, left_output, right_output)
    point_cloud = createPointCloud(pipeline, stereo)
    upright_cloud = pipeline.create(PointCloudForViewer).build(point_cloud)
    return cameras[dai.CameraBoardSocket.CAM_D], upright_cloud.out


def build_tracker(pipeline, fps, camera):
    high_res = rotate_180(
        pipeline,
        camera.requestOutput(
            size=HIGH_RES_SIZE,
            type=FRAME_TYPE,
            fps=fps,
            resizeMode=dai.ImgResizeMode.CROP,
        ),
        HIGH_RES_SIZE,
    )
    # Share one high-res crop between both tracking branches. The point cloud
    # uses rectified stereo images and needs no additional sensor output.
    preview = pipeline.create(dai.node.ImageManip)
    preview.initialConfig.setOutputSize(
        *LOW_RES_SIZE, dai.ImageManipConfig.ResizeMode.STRETCH
    )
    preview.setMaxOutputFrameSize(LOW_RES_SIZE[0] * LOW_RES_SIZE[1] * 3)
    high_res.link(preview.inputImage)
    low_res = preview.out

    # Close faces may fill the frame so completely that person detection fails.
    fallback_detector = pipeline.create(ExtendedNeuralNetwork)
    fallback_detector.build(
        inputImage=low_res,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        nnSource=FACE_MODEL,
    )
    fallback_face = pipeline.create(LargestDetection).build(
        fallback_detector.out
    )

    person_detector = pipeline.create(ExtendedNeuralNetwork)
    person_detector.build(
        inputImage=low_res,
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        nnSource=PERSON_MODEL,
    )
    largest_person = pipeline.create(LargestDetection).build(
        person_detector.out
    )
    head_region = pipeline.create(HeadRegion).build(largest_person.out)

    head_cropper = pipeline.create(FrameCropper)
    face_detector = pipeline.create(ParsingNeuralNetwork).build(
        input=head_cropper.out, nnSource=FACE_MODEL
    )
    head_cropper.fromImgDetections(
        inputImgDetections=head_region.out,
        outputSize=face_detector._nn_archive.getInputSize(),
        resizeMode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
    ).build(inputImage=high_res)
    collected_faces = pipeline.create(GatherData).build(
        inputData=face_detector.out,
        inputReference=head_region.out,
        cameraFps=fps,
    )
    mapped_faces = pipeline.create(CoordinatesMapper).build(
        toTransformationInput=high_res,
        fromTransformationInput=collected_faces.out,
    )
    faces = pipeline.create(FacesFromCollection).build(mapped_faces.out)
    largest_face = pipeline.create(LargestDetection).build(faces.out)
    focused_zoom = build_face_zoom(
        pipeline, largest_face.out, high_res, sparse=True
    )
    fallback_zoom = build_face_zoom(pipeline, fallback_face.out, low_res)
    return (
        low_res,
        person_detector.out,
        largest_face.out,
        fallback_face.out,
        focused_zoom,
        fallback_zoom,
    )


def main():
    load_dotenv(override=True)
    args = parse_args()
    connection = None
    try:
        if not args.no_servos:
            connection = connect_to_arm(args.serial_port)
            servo_targets = initialize_servos(connection)
            last_sent = dict(servo_targets)

        visualizer = dai.RemoteConnection(httpPort=8082)
        device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
        platform = device.getPlatformAsString()
        print(f"Platform: {platform}")
        if platform != "RVC4":
            raise SystemExit("Focused face tracking requires an RVC4 camera.")

        with dai.Pipeline(device) as pipeline:
            camera, point_cloud = build_point_cloud(
                pipeline, device.readCalibration(), args.fps_limit
            )
            (
                video,
                people,
                focused_faces,
                fallback_faces,
                focused_zoom,
                fallback_zoom,
            ) = build_tracker(pipeline, args.fps_limit, camera)
            people_queue = people.createOutputQueue(maxSize=1, blocking=False)
            focused_queue = focused_faces.createOutputQueue(maxSize=1, blocking=False)
            fallback_queue = fallback_faces.createOutputQueue(maxSize=1, blocking=False)
            visualizer.addTopic("Video", video, "images")
            visualizer.addTopic("People", people, "images")
            visualizer.addTopic("Focused faces", focused_faces, "images")
            visualizer.addTopic("Fallback faces", fallback_faces, "images")
            visualizer.addTopic("Focused face zoom", focused_zoom, "focused_face_zoom")
            visualizer.addTopic("Fallback face zoom", fallback_zoom, "fallback_face_zoom")
            visualizer.addTopic("Point cloud (metres)", point_cloud, group="Point cloud")
            pipeline.start()
            visualizer.registerPipeline(pipeline)
            print(
                "Tracking faces and streaming the point cloud. "
                "Open the web viewer on port 8082; press q to stop."
            )

            started = time.monotonic()
            last_report = started
            people_count = focused_count = fallback_count = 0
            last_focused_face = float("-inf")
            while pipeline.isRunning():
                pipeline.processTasks()
                now = time.monotonic()
                people_message = get_latest_message(people_queue)
                if people_message is not None and people_message.detections:
                    people_count += 1
                focused_message = get_latest_message(focused_queue)
                fallback_message = get_latest_message(fallback_queue)
                focused_face = (
                    get_first_face_detection(focused_message)
                    if focused_message is not None
                    else None
                )
                fallback_face = (
                    get_first_face_detection(fallback_message)
                    if fallback_message is not None
                    else None
                )
                if focused_face is not None:
                    focused_count += 1
                    last_focused_face = now
                if fallback_face is not None:
                    fallback_count += 1
                face = focused_face
                if face is None and now - last_focused_face > 1.0:
                    face = fallback_face
                if face is not None and connection is not None:
                    update_servo_targets(face, servo_targets, step=FOCUSED_SERVO_STEP)
                    send_pending_servo_updates(connection, servo_targets, last_sent)

                if now - last_report >= 5:
                    print(
                        f"Frames with people/focused faces/fallback faces "
                        f"in last {now - last_report:.0f}s: "
                        f"{people_count}/{focused_count}/{fallback_count}"
                    )
                    people_count = focused_count = fallback_count = 0
                    last_report = now
                if args.duration is not None and now - started >= args.duration:
                    break
                if visualizer.waitKey(1) == ord("q"):
                    break
            pipeline.stop()
    finally:
        if connection is not None:
            connection.close()
            print("Disconnected from servos.")


if __name__ == "__main__":
    main()
