"""Inspect and replay the retained compressed water model without TensorFlow.

This is a numeric trace exporter for this one frozen graph, not a new model.
Compression follows DeepMD-kit v3.1.3 source/lib/src/tabulate.cc.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'scripts/submit_calculation/H2O-Phase-Diagram-model_compressed.pb'
MODEL_HASH = '435d8a455edea9908faf07bd7107cca24f28d7ec18e6abd37761324a4dcf7263'


def varint(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7


def fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f'Unexpected protobuf wire type {wire}')
        yield field, value


def constant(attributes):
    attrs = {dict(fields(item))[1].decode(): dict(fields(item))[2] for item in attributes}
    if 'value' not in attrs:
        return None
    attr = dict(fields(attrs['value']))
    if 8 not in attr:
        return None
    tensor_fields = list(fields(attr[8]))
    tensor = dict(tensor_fields)
    dims = tuple(dict(fields(item)).get(1, 0) for key, item in fields(tensor.get(2, b'')) if key == 2)
    if tensor[1] == 7:
        return [item.decode(errors='replace') for key, item in tensor_fields if key == 8]
    dtype = {1: '<f4', 2: '<f8', 3: '<i4', 9: '<i8', 10: '?'}.get(tensor[1])
    if dtype is None:
        return None
    if 4 in tensor:
        return np.frombuffer(tensor[4], dtype=dtype).reshape(dims)
    value_field = {1: 5, 2: 6, 3: 7, 9: 10, 10: 11}[tensor[1]]
    values = []
    for key, value in tensor_fields:
        if key != value_field:
            continue
        if isinstance(value, bytes):
            if tensor[1] in (1, 2):
                values.extend(np.frombuffer(value, dtype=dtype).tolist())
            else:
                offset = 0
                while offset < len(value):
                    number, offset = varint(value, offset)
                    values.append(number)
        else:
            values.append(value)
    if tensor[1] in (3, 9):
        bits = 32 if tensor[1] == 3 else 64
        values = [((int(v) + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1)) for v in values]
    array = np.asarray(values or [0], dtype=dtype)
    size = int(np.prod(dims)) if dims else 1
    if size == 0:
        return np.empty(dims, dtype=dtype)
    if array.size == 1 and size != 1:
        array = np.full(size, array[0], dtype=dtype)
    return array.reshape(dims)


class FrozenWaterModel:
    def __init__(self):
        raw = MODEL.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == MODEL_HASH
        self.nodes = {}
        self.constants = {}
        for key, item in fields(raw):
            if key != 1:
                continue
            node = list(fields(item))
            d = dict(node)
            name, op = d[1].decode(), d[2].decode()
            inputs = [v.decode() for k, v in node if k == 3]
            self.nodes[name] = {'op': op, 'inputs': inputs}
            if op == 'Const':
                self.constants[name] = constant([v for k, v in node if k == 5])
        self.mu = self.constants['descrpt_attr/t_avg'].reshape(2, 600, 4)
        self.sigma = self.constants['descrpt_attr/t_std'].reshape(2, 600, 4)
        self.tables = {}
        for i in range(2):
            for j in range(2):
                name = f'filter_type_{i}/TabulateFusionSeA' + ('_1' if j else '')
                inputs = self.nodes[name]['inputs']
                self.tables[i, j] = (self.constants[inputs[0]].reshape(-1, 100, 6), self.constants[inputs[1]])

    def embedding(self, x, centre_type, neighbor_type):
        table, info = self.tables[centre_type, neighbor_type]
        lower, upper, maximum, stride0, stride1 = info[:5]
        count0 = int((upper - lower) / stride0)
        index = np.where(x < upper, ((x - lower) / stride0).astype(int), count0 + ((x - upper) / stride1).astype(int))
        offset = np.where(x < upper, x - (index * stride0 + lower), x - ((index - count0) * stride1 + upper))
        outside = (x < lower) | (x >= maximum)
        index = np.where(x < lower, 0, index)
        index = np.where(x >= maximum, count0 + int((maximum - upper) / stride1) - 1, index)
        offset = np.where(outside, 0.0, offset)
        c = table[index]
        z = offset[:, None]
        return c[:, :, 0] + z * (c[:, :, 1] + z * (c[:, :, 2] + z * (c[:, :, 3] + z * (c[:, :, 4] + z * c[:, :, 5]))))

    def descriptor(self, positions, types, cell, centre):
        delta = positions - positions[centre]
        delta -= cell * np.rint(delta / cell)
        distance = np.linalg.norm(delta, axis=1)
        ids = np.full(600, -1, dtype=int)
        raw = np.zeros((600, 4), dtype=float)
        for kind, start, size in ((0, 0, 200), (1, 200, 400)):
            chosen = np.flatnonzero((types == kind) & (distance < 6.0) & (np.arange(len(types)) != centre))
            chosen = chosen[np.lexsort((chosen, distance[chosen]))]
            assert len(chosen) <= size
            ids[start:start + len(chosen)] = chosen
            r = distance[chosen]
            u = np.clip((r - 0.5) / 5.5, 0.0, 1.0)
            switch = 1.0 + u**3 * (-6.0 * u**2 + 15.0 * u - 10.0)
            s = switch / r
            raw[start:start + len(chosen), 0] = s
            raw[start:start + len(chosen), 1:] = s[:, None] * delta[chosen] / r[:, None]
        normalized = (raw - self.mu[types[centre]]) / self.sigma[types[centre]]
        g = np.concatenate((self.embedding(normalized[:200, 0], types[centre], 0), self.embedding(normalized[200:, 0], types[centre], 1)))
        a = (normalized[:200].T @ g[:200] + normalized[200:].T @ g[200:]) / 600.0
        d = a.T @ a[:, :12]
        return {'nlist': ids, 'R': raw, 'Rbar': normalized, 'G': g, 'A': a, 'D': d}

    def fitting(self, descriptors, kind):
        x = np.asarray(descriptors).reshape(-1, 1200)
        activations = []
        for layer in range(3):
            name = f'layer_{layer}_type_{kind}'
            y = np.tanh(x @ self.constants[name + '/matrix'] + self.constants[name + '/bias'])
            if layer:
                y = x + y * self.constants[name + '/idt']
            x = y
            activations.append(x.copy())
        name = f'final_layer_type_{kind}'
        energy = x @ self.constants[name + '/matrix'] + self.constants[name + '/bias']
        return energy[:, 0], activations

    def evaluate(self, positions, types, cell):
        descriptors = np.stack([self.descriptor(positions, types, cell, i)['D'] for i in range(len(types))])
        energy = np.empty(len(types))
        for kind in (0, 1):
            energy[types == kind] = self.fitting(descriptors[types == kind], kind)[0]
        return energy


def export_trace(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = FrozenWaterModel()
    source = ROOT / 'product/data/dpmd_water_box_trajectory.npz'
    trajectory = np.load(source)
    base = np.load(ROOT / 'product/data/water_box_64.npz')
    types = np.where(trajectory['elements'] == 'O', 0, 1)
    box = float(trajectory['box_length'])
    centre = 126
    traces = []
    energy_errors = []
    force_errors = []
    numeric_forces = []
    for state, positions in enumerate(trajectory['positions']):
        trace = model.descriptor(positions, types, box, centre)
        traces.append(trace)
        energies = model.evaluate(positions, types, box)
        energy_errors.append(float(np.max(np.abs(energies - trajectory['atomic_energy_ev'][state]))))
        # Independent total-energy finite differences, including every atom's
        # environment dependence on the selected central coordinate.
        force = []
        epsilon = 1.0e-5
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[centre, axis] += epsilon
            minus[centre, axis] -= epsilon
            derivative = (model.evaluate(plus, types, box).sum() - model.evaluate(minus, types, box).sum()) / (2 * epsilon)
            force.append(-float(derivative))
        numeric_forces.append(force)
        force_errors.append(float(np.max(np.abs(force - trajectory['forces_ev_per_angstrom'][state, centre]))))
        print(f'trace {state + 1}/6: energy error {energy_errors[-1]:.3g}; force error {force_errors[-1]:.3g}', flush=True)
    metadata = json.loads((ROOT / 'product/data/dpmd_water_box_trajectory.json').read_text(encoding='utf-8'))
    reference = np.asarray(metadata['model_probe']['descriptor']['o126']).reshape(100, 12)
    d_error = float(np.max(np.abs(traces[0]['D'] - reference)))
    assert d_error < 1e-10, d_error
    assert max(energy_errors) < 1e-8, energy_errors
    assert max(force_errors) < 1e-4, force_errors
    payload = {key: np.stack([t[key] for t in traces]) for key in traces[0]}
    payload.update({
        'positions': trajectory['positions'], 'cell': trajectory['cell'],
        'source_ids': base['source_ids'], 'molecule_ids': base['molecule_ids'],
        'elements': trajectory['elements'], 'velocities': trajectory['velocities'],
        'half_velocities': trajectory['half_velocities'], 'displacements': trajectory['displacements'],
        'atomic_energy_ev': trajectory['atomic_energy_ev'], 'total_energy_ev': trajectory['total_energy_ev'],
        'forces_ev_per_angstrom': trajectory['forces_ev_per_angstrom'],
        'finite_difference_force': np.asarray(numeric_forces), 'mu': model.mu, 'sigma': model.sigma,
    })
    np.savez_compressed(output_dir / 'deepmd_trace.npz', **payload)
    report = {
        'schema': 'deepmd_frozen_trace/v1', 'model': str(MODEL), 'model_sha256': MODEL_HASH,
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'reference_version': '3.1.3',
        'reference_source': 'https://github.com/deepmodeling/deepmd-kit/blob/v3.1.3/source/lib/src/tabulate.cc',
        'compression': 'same fifth-degree table polynomials and O/H typed contractions as TabulateFusionSeA',
        'derivative_check': 'central finite differences of total energy over all 192 atoms; epsilon=1e-5 angstrom',
        'normalization': '(raw_environment - model_mu) / model_sigma; padded slots included',
        'source_id_mapping': 'original NPZ array indices, before the backend sorts atoms by type',
        'shape': {k: list(v.shape) for k, v in payload.items()},
        'neighbor_counts': [int(np.count_nonzero(t['nlist'] >= 0)) for t in traces],
        'D_saved_max_abs_error': d_error, 'atomic_energy_max_abs_error_ev': energy_errors,
        'force_max_abs_error_ev_per_angstrom': force_errors,
        'atomic_sum_max_abs_error_ev': float(np.max(np.abs(trajectory['atomic_energy_ev'].sum(axis=1) - trajectory['total_energy_ev']))),
        'passed': True,
    }
    (output_dir / 'deepmd_trace.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return payload, report


if __name__ == '__main__':
    model = FrozenWaterModel()
    trajectory = np.load(ROOT / 'product/data/dpmd_water_box_trajectory.npz')
    types = np.where(trajectory['elements'] == 'O', 0, 1)
    trace = model.descriptor(trajectory['positions'][0], types, float(trajectory['box_length']), 126)
    metadata = json.loads((ROOT / 'product/data/dpmd_water_box_trajectory.json').read_text(encoding='utf-8'))
    reference = np.asarray(metadata['model_probe']['descriptor']['o126']).reshape(100, 12)
    print('D max error', float(np.max(np.abs(trace['D'] - reference))))
    row = int(np.flatnonzero(trace['nlist'] == 127)[0])
    print('H127 row', row, 'R', trace['R'][row].tolist(), 'Rbar', trace['Rbar'][row].tolist())
    print('G', trace['G'][row, :4].tolist(), 'A first column', trace['A'][:, 0].tolist())
    print('fit constants', [(k, np.shape(v), np.asarray(v).ravel()[:2].tolist()) for k,v in model.constants.items() if ('layer' in k and any(w in k for w in ['matrix', 'bias', 'idt'])) or 'bias_atom' in k])
