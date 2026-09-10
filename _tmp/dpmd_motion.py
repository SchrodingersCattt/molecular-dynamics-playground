"""MatterVis CLI atom layers and a reproducible display schedule for 04."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / '_tmp/04_deep_potential_md_dynamic_qa/motion_cli'
DIRECTION = np.array([1.0, -0.65, 0.45])
DIRECTION /= np.linalg.norm(DIRECTION)
TARGET = np.full(3, 12.429633119082435 / 2)
BOX_SCALE = 10.8
FOCUS_SCALE = 6.8
SIZE = 640
VERSION = 1


def schedule(time):
    if time < 20:
        return 0, 0.0, 0, 'evaluate'
    elapsed = min(time - 20, 10 - 1e-8)
    step = min(int(elapsed // 2), 4)
    local = elapsed - step * 2
    if local < .25:
        return step, 0.0, step, 'half_kick'
    if local < .95:
        return step, float((local - .25) / .70), step, 'drift'
    if local < 1.45:
        return step, 1.0, step + 1, 'evaluate'
    if local < 1.75:
        return step, 1.0, step + 1, 'force'
    return step, 1.0, step + 1, 'final_kick'


def cli_env():
    env = os.environ.copy()
    checkout = ROOT.parent / 'MatterVis'
    if checkout.exists():
        env['PYTHONPATH'] = os.pathsep.join((str(checkout), str(ROOT / '_tmp/.render-runtime')))
    env['PYTHONUTF8'] = '1'
    return env


def invoke(arguments, timeout=120):
    result = subprocess.run(arguments, env=cli_env(), capture_output=True, encoding='utf-8', timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:] or result.stdout[-4000:])
    return result


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    trajectory_path = ROOT / 'product/data/dpmd_water_box_trajectory.npz'
    source_hash = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    manifest_path = DIRECTORY / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('version') == VERSION and old.get('source_sha256') == source_hash and len(list((DIRECTORY / 'box_frames').glob('*.png'))) == 240 and len(list((DIRECTORY / 'focus_frames').glob('*.png'))) == 240:
            return old
    trajectory = np.load(trajectory_path)
    base = np.load(ROOT / 'product/data/water_box_64.npz')
    positions = trajectory['positions']
    elements = trajectory['elements']
    molecule_ids = base['molecule_ids']
    source_ids = base['source_ids']
    cell = trajectory['cell']
    box = float(trajectory['box_length'])
    union = sorted({int(i) for row in trajectory['neighbour_ids'] for i in row if i >= 0})
    focus_ids = np.asarray([126] + [i for i in union if i != 126])
    box_frames, focus_frames, shown_positions, states = [], [], [], []
    for frame in range(240):
        step, alpha, state, phase = schedule(20 + frame / 24)
        displacement = positions[step + 1] - positions[step]
        displacement -= box * np.rint(displacement / box)
        current = positions[step] + alpha * displacement
        # Preserve each water molecule's periodic image throughout its move.
        for molecule in np.unique(molecule_ids):
            members = np.flatnonzero(molecule_ids == molecule)
            oxygen = members[elements[members] == 'O'][0]
            local = current[members] - current[oxygen]
            local -= box * np.rint(local / box)
            current[members] = (current[oxygen] % box) + local
        shown_positions.append(current.copy())
        delta = current[focus_ids] - current[126]
        delta -= box * np.rint(delta / box)
        focus = TARGET + delta
        full_atoms = Atoms(elements.tolist(), positions=current, cell=cell, pbc=True)
        focus_atoms = Atoms(elements[focus_ids].tolist(), positions=focus, cell=cell, pbc=True)
        for atoms, ids in ((full_atoms, np.arange(192)), (focus_atoms, focus_ids)):
            atoms.new_array('source_id', np.asarray(source_ids[ids], dtype=int))
            atoms.new_array('molecule_id', np.asarray(molecule_ids[ids], dtype=int))
            atoms.info['saved_step'] = int(step)
            atoms.info['display_fraction'] = float(alpha)
        box_frames.append(full_atoms)
        focus_frames.append(focus_atoms)
        states.append({'step': int(step), 'alpha': float(alpha), 'state': int(state), 'phase': phase})
    sources = {'box': DIRECTORY / 'water_box.extxyz', 'focus': DIRECTORY / 'local_environment.extxyz'}
    write(sources['box'], box_frames, format='extxyz')
    write(sources['focus'], focus_frames, format='extxyz')
    np.savez_compressed(DIRECTORY / 'display_geometry.npz', positions=np.asarray(shown_positions), focus_ids=focus_ids)
    records = []
    for kind in ('box', 'focus'):
        print(f'MatterVis {kind}: inspect', flush=True)
        inspected = invoke([sys.executable, '-m', 'mat_viewer', 'inspect', str(sources[kind]), '--json'], timeout=30)
        inspect_result = json.loads(inspected.stdout)
        (DIRECTORY / f'{kind}_inspect.json').write_text(json.dumps(inspect_result, indent=2), encoding='utf-8')
        command = [sys.executable, '-m', 'mat_viewer', 'render', str(sources[kind]),
                   '--backend', 'cpu', '--renderer', 'general', '--view', 'unit_cell' if kind == 'box' else 'cluster',
                   '--orthogonal', '--view-direction', *map(str, DIRECTION), '--camera-up', '0', '0', '1',
                   '--camera-target', *map(str, TARGET), '--ortho-scale', str(BOX_SCALE if kind == 'box' else FOCUS_SCALE),
                   '--width', str(SIZE), '--height', str(SIZE), '--scale', '1', '--show-hydrogen',
                   '--show-cell' if kind == 'box' else '--no-cell', '--no-axes', '--no-boundary-replicas',
                   '--atom-scale', '1.0', '--bond-radius', '.075', '--cell-color', '#AAB3B7', '--cell-width', '1.1', '--json']
        for style, output, extra in [('ball_stick', DIRECTORY / f'{kind}_bonded.png', ['--frame', '0']),
                                     ('ball', DIRECTORY / f'{kind}_motion.mp4', ['--fps', '24'])]:
            run_command = [*command, '--style', style, *extra, '-o', str(output)]
            print(f'MatterVis {kind}: {style}', flush=True)
            result = invoke(run_command, timeout=600)
            record = {'command': run_command, 'result': json.loads(result.stdout), 'stderr': result.stderr, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
            records.append(record)
        frame_dir = DIRECTORY / f'{kind}_frames'
        frame_dir.mkdir(exist_ok=True)
        invoke(['ffmpeg', '-v', 'error', '-y', '-i', str(DIRECTORY / f'{kind}_motion.mp4'), str(frame_dir / '%04d.png')], timeout=30)
    manifest = {'version': VERSION, 'source_sha256': source_hash, 'records': records, 'states': states,
                'direction': DIRECTION.tolist(), 'target': TARGET.tolist(), 'box_scale': BOX_SCALE, 'focus_scale': FOCUS_SCALE,
                'source_ids': source_ids.tolist(), 'molecule_ids': molecule_ids.tolist(), 'focus_ids': focus_ids.tolist(),
                'interpolation': 'minimum-image linear display sampling within each saved VV drift; energies and forces only at saved states'}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    prepare()
