"""Independent geometry and decoded-video acceptance checks for 04."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from ase.io import iread

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / '_tmp/04_deep_potential_md_dynamic_qa'
MOTION = QA / 'motion_cli'


def verify(video):
    video = Path(video)
    trace = np.load(QA / 'deepmd_trace/deepmd_trace.npz')
    geometry = np.load(MOTION / 'display_geometry.npz')
    positions = geometry['positions']
    cell = float(trace['cell'][0, 0])
    source_ids = trace['source_ids']
    focus_ids = geometry['focus_ids']
    endpoint = positions[-1] - trace['positions'][-1]
    endpoint -= cell * np.rint(endpoint / cell)
    endpoint_error = float(np.max(np.abs(endpoint)))
    assert endpoint_error < 1e-10, endpoint_error
    counts = []
    for state, current in enumerate(trace['positions']):
        delta = current - current[126]
        delta -= cell * np.rint(delta / cell)
        distance = np.linalg.norm(delta, axis=1)
        expected = set(np.flatnonzero((distance > 1e-10) & (distance < 6.0)))
        actual = set(trace['nlist'][state][trace['nlist'][state] >= 0])
        assert actual == expected, (state, actual ^ expected)
        counts.append(len(actual))
    for frame, current in enumerate(positions):
        delta = current - current[126]
        delta -= cell * np.rint(delta / cell)
        distance = np.linalg.norm(delta, axis=1)
        neighbors = set(np.flatnonzero((distance > 1e-10) & (distance < 6)))
        assert neighbors.issubset(set(focus_ids)), (frame, neighbors - set(focus_ids))
    source_frames = 0
    for atoms in iread(MOTION / 'water_box.extxyz'):
        assert np.array_equal(atoms.arrays['source_id'], source_ids)
        assert np.array_equal(atoms.arrays['molecule_id'], trace['molecule_ids'])
        assert len(atoms) == 192 and all(atoms.pbc)
        assert np.allclose(atoms.cell.array, trace['cell'])
        source_frames += 1
    assert source_frames == 240
    meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration', '-of', 'json', str(video)], text=True, timeout=20))
    stream = meta['streams'][0]
    assert (stream['width'], stream['height'], stream['avg_frame_rate'], int(stream['nb_frames'])) == (1920, 600, '24/1', 720)
    assert abs(float(meta['format']['duration']) - 30) < 1e-6
    decode = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total = 0
    previous = None
    moving_frames = 0
    motion_pixels = []
    baseline = None
    final_change = 0
    while True:
        raw = decode.stdout.read(1920 * 600 * 3)
        if not raw:
            break
        assert len(raw) == 1920 * 600 * 3
        image = np.frombuffer(raw, np.uint8).reshape(600, 1920, 3)
        # Fixed scene crop excludes counters, formulas, and changing text.
        scene = image[155:545, 345:1082].copy()
        if total == 480:
            baseline = scene.copy()
        if total > 480 and previous is not None:
            count = int(np.count_nonzero(np.max(np.abs(scene.astype(np.int16) - previous.astype(np.int16)), axis=2) > 8))
            motion_pixels.append(count)
            moving_frames += count > 50
        previous = scene
        total += 1
    stderr = decode.stderr.read().decode(errors='replace')
    assert decode.wait(timeout=20) == 0, stderr
    assert total == 720
    final_change = int(np.count_nonzero(np.max(np.abs(previous.astype(np.int16) - baseline.astype(np.int16)), axis=2) > 8))
    assert moving_frames >= 25, moving_frames
    assert final_change > 300, final_change
    report = {'passed': True, 'decoded_frames': total, 'source_frames': source_frames,
              'dimensions': [1920,600], 'fps': 24, 'duration_seconds': 30,
              'neighbor_counts': counts, 'final_coordinate_error_angstrom': endpoint_error,
              'frames_with_scene_motion': moving_frames, 'first_to_final_scene_changed_pixels': final_change,
              'motion_crop': [345,155,1082,545], 'max_consecutive_scene_changed_pixels': max(motion_pixels),
              'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
              'source_identity': '192 unique source IDs and 64 stable molecule IDs across 240 display frames'}
    (QA / 'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]), indent=2))
