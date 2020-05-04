import numpy as np


def marker_sof(f):
    length = int.from_bytes(f.read(2), 'big')
    frame_header = {}
    frame_header['sample_precision'] = int.from_bytes(f.read(1), 'big')
    frame_header['height'] = int.from_bytes(f.read(2), 'big')
    frame_header['width'] = int.from_bytes(f.read(2), 'big')
    frame_header['num_channels'] = int.from_bytes(f.read(1), 'big')
    k = length - 8
    while k > 0:
        n = int.from_bytes(f.read(1), 'big')
        frame_header[f'channel{n}'] = {}
        tmp = int.from_bytes(f.read(1), 'big')
        hn = tmp >> 4
        vn = tmp - (hn << 4)
        frame_header[f'channel{n}']['horizontal_ratio'] = hn
        frame_header[f'channel{n}']['vertical_ratio'] = vn
        frame_header[f'channel{n}']['target_quantization_table'] = \
            int.from_bytes(f.read(1), 'big')
        k -= 3
    return frame_header


def marker_sof0(f):
    frame_header = marker_sof(f)
    frame_header['method'] = 'baseline'
    return frame_header


def marker_sof2(f):
    frame_header = marker_sof(f)
    frame_header['method'] = 'progressive'
    return frame_header


def marker_dqt(f):
    length = int.from_bytes(f.read(2), 'big')
    k = length - 2
    ret = {}
    while k > 0:
        tmp = int.from_bytes(f.read(1), 'big')
        pqn = tmp >> 4
        tqn = tmp - (pqn << 4)
        k -= 65 if pqn == 0 else 129
        ret[f'quantization_table{tqn}'] = np.array([
                int.from_bytes(f.read(1 if pqn == 0 else 2), 'big')
                for _ in range(64)],
                dtype=np.uint8 if pqn == 0 else np.uint16)
    return ret


MARKERS = {
    b'\xff\xc0': marker_sof0,
    b'\xff\xc2': marker_sof2,
    b'\xff\xdb': marker_dqt,
}


def inspect(path: str):
    f = open(str(path), 'rb')
    # SOI(Start of Image)
    assert f.read(2) == b'\xff\xd8'
    info = {}
    telled = 0
    while telled < f.tell():
        telled = f.tell()
        if f.read(1) != b'\xff':
            continue
        marker = b'\xff' + f.read(1)
        info.update(MARKERS.get(marker, lambda _: {})(f))
    return info


if __name__ == '__main__':
    from pathlib import Path
    from pprint import pprint
    assets_dir = Path(__file__).joinpath('../../../assets')
    pprint(inspect(assets_dir.joinpath('q50_420.jpg')))
    pprint(inspect(assets_dir.joinpath('q70_420.jpg')))
    pprint(inspect(assets_dir.joinpath('q90_420.jpg')))
    pprint(inspect(assets_dir.joinpath('q90_444.jpg')))
