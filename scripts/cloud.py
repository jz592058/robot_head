#!/usr/bin/env python3

"""Stereo depth and point cloud from the IMX586 cameras on CAM_A and CAM_D."""

import argparse
import time

import cv2
import numpy as np

import depthai as dai
from depthai_nodes.node import ApplyDepthColormap


CAMERA_SOCKETS = {
    dai.CameraBoardSocket.CAM_A,
    dai.CameraBoardSocket.CAM_D,
}
OUTPUT_SIZE = (640, 400)
NEURAL_MODEL = dai.DeviceModelZoo.NEURAL_DEPTH_576X360
DEPTH_FPS = 8.0
COLOR_SIZE = (3840, 2160)
COLOR_DISPLAY_SIZE = (960, 540)
FPS = 10.0
VISUALIZER_PORT = 8082


def parseArgs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-ip",
        default="10.12.222.68",
        help="PoE device IP address (default: %(default)s)",
    )
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def getStereoSockets(calibrationHandler):
    eeprom = calibrationHandler.getEepromData()
    rectification = eeprom.stereoRectificationData
    leftSocket = rectification.leftCameraSocket
    rightSocket = rectification.rightCameraSocket

    if {leftSocket, rightSocket} != CAMERA_SOCKETS:
        raise RuntimeError(
            "Expected the calibrated stereo pair to be CAM_A/CAM_D, but EEPROM "
            f"contains {leftSocket.name}/{rightSocket.name}."
        )

    # This robot's physical stereo order is CAM_D (left), CAM_A (right).
    # EEPROM identifies the calibrated pair but may list it in reverse order.
    return dai.CameraBoardSocket.CAM_D, dai.CameraBoardSocket.CAM_A


def configureNeuralDepth(pipeline, leftOutput, rightOutput):
    neural = pipeline.create(dai.node.NeuralDepth).build(
        leftOutput, rightOutput, NEURAL_MODEL
    )
    neural.initialConfig.setDepthUnit(dai.LengthUnit.MILLIMETER)
    # Camera FPS can exceed inference throughput. Drop old pairs before Sync
    # and bound NN buffering, while keeping left/right inference inputs paired.
    for input_stream in (neural.left, neural.right):
        input_stream.setMaxSize(2)
        input_stream.setBlocking(False)
    for name in ("left", "right"):
        neural.neuralNetwork.inputs[name].setMaxSize(1)
        neural.neuralNetwork.inputs[name].setBlocking(True)
    print(
        f"LENS Neural Stereo: camera inputs {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}, "
        f"native inference {neural.getInputSize(NEURAL_MODEL)}", flush=True
    )
    return neural


class RectifiedRGB(dai.node.HostNode):
    """Convert the matching rectified image to RGB without another alignment."""

    def build(self, frames):
        self.link_args(frames)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, message):
        image = message.getCvFrame()
        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB
        )
        message.setCvFrame(image, dai.ImgFrame.Type.RGB888i)
        return message


class ValidPointCloud(dai.node.HostNode):
    """Remove invalid/background points and report metric cloud health."""

    def build(self, clouds):
        self.link_args(clouds)
        self.inputs["message"].setMaxSize(1)
        self.inputs["message"].setBlocking(False)
        self.sendProcessingToPipeline(True)
        self.last_report = 0.0
        self.frames = 0
        return self

    def process(self, message):
        points, colors = message.getPointsRGB()
        valid = np.isfinite(points).all(axis=1)
        valid &= (points[:, 2] >= 0.2) & (points[:, 2] <= 8.0)
        self.frames += 1
        if time.monotonic() - self.last_report >= 5:
            median = f"{np.median(points[valid, 2]):.2f} m" if valid.any() else "none"
            age_ms = (dai.Clock.now() - message.getTimestamp()).total_seconds() * 1000
            print(
                f"PCL frame {self.frames}: {valid.sum()}/{len(points)} points "
                f"({valid.mean():.1%}), median depth {median}, age {age_ms:.0f} ms",
                flush=True,
            )
            self.last_report = time.monotonic()
        # Empty frames during camera warmup should not reset the viewer bounds.
        if not valid.any():
            return None
        message.setPointsRGB(
            np.ascontiguousarray(points[valid]), np.ascontiguousarray(colors[valid])
        )
        message.setSize(int(valid.sum()), 1)
        message.updateBoundingBox()
        return message


def createPointCloud(pipeline, stereo):
    # NeuralDepth outputs depth in the rectified LEFT camera pixel grid.
    # Feed that same image to RGBD without same-camera ImageAlign.
    color = pipeline.create(RectifiedRGB).build(stereo.rectifiedLeft)
    rgbd = pipeline.create(dai.node.RGBD).build()
    color.out.link(rgbd.inColor)
    stereo.depth.link(rgbd.inDepth)
    rgbd.setDepthUnits(dai.LengthUnit.MILLIMETER)  # Input mm -> output metres.
    return pipeline.create(ValidPointCloud).build(rgbd.pcl).out


def main():
    args = parseArgs()
    visualizer = dai.RemoteConnection(httpPort=VISUALIZER_PORT)
    device = dai.Device(dai.DeviceInfo(args.device_ip))

    with dai.Pipeline(device) as pipeline:
        calibrationHandler = device.readCalibration()
        leftSocket, rightSocket = getStereoSockets(calibrationHandler)
        print(f"Using calibrated stereo pair: {leftSocket.name} -> {rightSocket.name}")

        leftCamera = pipeline.create(dai.node.Camera).build(leftSocket, sensorFps=FPS)
        rightCamera = pipeline.create(dai.node.Camera).build(rightSocket, sensorFps=FPS)

        leftFocus = calibrationHandler.getLensPosition(leftSocket)
        rightFocus = calibrationHandler.getLensPosition(rightSocket)
        leftCamera.initialControl.setManualFocus(leftFocus)
        rightCamera.initialControl.setManualFocus(rightFocus)
        print(
            "Using calibrated focus positions: "
            f"{leftSocket.name}={leftFocus}, {rightSocket.name}={rightFocus}"
        )

        # NeuralDepth rectifies and resizes these 640x400 inputs for LENS.
        leftOutput = leftCamera.requestOutput(
            OUTPUT_SIZE, type=dai.ImgFrame.Type.BGR888i, fps=DEPTH_FPS, enableUndistortion=False
        )
        rightOutput = rightCamera.requestOutput(
            OUTPUT_SIZE, type=dai.ImgFrame.Type.BGR888i, fps=DEPTH_FPS, enableUndistortion=False
        )
        leftColorOutput = leftCamera.requestOutput(
            COLOR_SIZE, fps=FPS, enableUndistortion=False
        )
        rightColorOutput = rightCamera.requestOutput(
            COLOR_SIZE, fps=FPS, enableUndistortion=False
        )

        leftColorPreview = pipeline.create(dai.node.ImageManip)
        rightColorPreview = pipeline.create(dai.node.ImageManip)
        manipBackend = (
            dai.node.ImageManip.Backend.GPU
            if device.hasGPU()
            else dai.node.ImageManip.Backend.CPU
        )
        for colorOutput, preview in (
            (leftColorOutput, leftColorPreview),
            (rightColorOutput, rightColorPreview),
        ):
            preview.setBackend(manipBackend)
            preview.initialConfig.setOutputSize(
                *COLOR_DISPLAY_SIZE, dai.ImageManipConfig.ResizeMode.STRETCH
            )
            colorOutput.link(preview.inputImage)

        stereo = configureNeuralDepth(pipeline, leftOutput, rightOutput)

        depthColor = pipeline.create(ApplyDepthColormap).build(stereo.depth)
        # Process Python callbacks in the main loop so they finish before
        # shutdown closes their output queues.
        depthColor.sendProcessingToPipeline(True)
        pointCloud = createPointCloud(pipeline, stereo)

        visualizer.addTopic("Depth (millimetres)", depthColor.out, group="Stereo")
        visualizer.addTopic("Point cloud (metres)", pointCloud, group="Point cloud")
        visualizer.addTopic(
            "Rectified left", stereo.rectifiedLeft, group="Stereo"
        )
        visualizer.addTopic(
            "Rectified right", stereo.rectifiedRight, group="Stereo"
        )
        # Only transfer the downscaled previews to the host. Sending two raw 4K
        # streams over XLink can backpressure the camera branches and break the
        # synchronization required by StereoDepth.
        visualizer.addTopic(
            f"{leftSocket.name} color 4K", leftColorPreview.out, group="Color"
        )
        visualizer.addTopic(
            f"{rightSocket.name} color 4K", rightColorPreview.out, group="Color"
        )

        pipeline.start()
        visualizer.registerPipeline(pipeline)
        print(
            f"Open http://localhost:{VISUALIZER_PORT} in Chrome to view the streams."
        )
        print("Neural point cloud: metres, 0.2–8 m range, native 576x360 depth.")
        started = time.monotonic()
        try:
            while pipeline.isRunning():
                pipeline.processTasks()
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    break
                if visualizer.waitKey(1) == ord("q"):
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
