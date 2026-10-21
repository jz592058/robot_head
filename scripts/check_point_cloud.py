#!/usr/bin/env python3
"""Check real-device cloud density, metric scale, and sustained frame delivery."""

import argparse
from pathlib import Path
import time

import depthai as dai
import numpy as np

from cloud import NEURAL_MODEL
from focused_person_follow import build_point_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-ip', required=True)
    parser.add_argument('--duration', type=float, default=15)
    parser.add_argument('--output', type=Path, help='Save the last cloud as binary PLY.')
    args = parser.parse_args()
    if args.duration < 5:
        parser.error('--duration must be at least 5 seconds')

    counts = []
    medians = []
    ages_ms = []
    last = None
    with dai.Device(dai.DeviceInfo(args.device_ip)) as device:
        with dai.Pipeline(device) as pipeline:
            _, cloud = build_point_cloud(pipeline, device.readCalibration(), 10)
            neural = next(
                node for node in pipeline.getAllNodes()
                if isinstance(node, dai.node.NeuralDepth)
            )
            expected_size = dai.node.NeuralDepth.getInputSize(NEURAL_MODEL)
            depth_queue = neural.depth.createOutputQueue(maxSize=1, blocking=False)
            queue = cloud.createOutputQueue(maxSize=2, blocking=False)
            pipeline.start()
            started = time.monotonic()
            try:
                while pipeline.isRunning() and time.monotonic() - started < args.duration:
                    pipeline.processTasks()
                    depth = depth_queue.tryGet()
                    if depth is not None:
                        actual_size = (depth.getWidth(), depth.getHeight())
                        if actual_size != expected_size:
                            raise RuntimeError(
                                f'Neural depth size {actual_size}, expected {expected_size}'
                            )
                    message = queue.tryGet()
                    if message is not None:
                        points, colors = message.getPointsRGB()
                        if not np.isfinite(points).all() or not (points[:, 2] > 0).all():
                            raise RuntimeError('Cloud contains invalid points')
                        counts.append(len(points))
                        medians.append(float(np.median(points[:, 2])))
                        ages_ms.append(
                            (dai.Clock.now() - message.getTimestamp()).total_seconds() * 1000
                        )
                        last = points, colors
                    time.sleep(0.001)
            finally:
                pipeline.stop()

    if len(counts) < args.duration * 5:
        raise RuntimeError(f'Cloud stalled: only {len(counts)} frames in {args.duration}s')
    if np.median(counts) < 10000:
        raise RuntimeError(f'Cloud too sparse: median {np.median(counts):.0f} points')
    if not 0.2 <= np.median(medians) <= 8:
        raise RuntimeError('Cloud has an implausible metric scale')
    if np.percentile(ages_ms, 90) > 750:
        raise RuntimeError(f'Cloud is delayed: p90 age {np.percentile(ages_ms, 90):.0f} ms')
    print(
        f'PASS: {len(counts)} frames, median {np.median(counts):.0f} points/frame, '
        f'median Z {np.median(medians):.2f} m, '
        f'p90 age {np.percentile(ages_ms, 90):.0f} ms'
    )
    if args.output:
        points, colors = last
        vertices = np.empty(len(points), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ])
        for index, name in enumerate(('x', 'y', 'z')):
            vertices[name] = points[:, index]
        for index, name in enumerate(('red', 'green', 'blue')):
            vertices[name] = colors[:, index]
        header = (
            'ply\nformat binary_little_endian 1.0\n'
            f'element vertex {len(vertices)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'end_header\n'
        )
        with args.output.open('wb') as output:
            output.write(header.encode('ascii'))
            output.write(vertices.tobytes())
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
