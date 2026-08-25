# MapTRv2 Environment Log

This file records environment setup and dependency issues separately from
`x_run.md`, which is reserved for data preprocessing and running commands.

## 1. Server Choice

Initial PPU server was not suitable for native MapTRv2 reproduction:

```text
Python 3.12.3
torch 2.7.0
CUDA 12.8
```

MapTRv2 is an old OpenMMLab 1.x project, so direct reproduction should avoid
Python 3.12 / torch 2.x if possible.

Switched to RTX 2080 Ti server:

```text
GPU: NVIDIA GeForce RTX 2080 Ti, 11 GB
Driver: 535.230.02
NVIDIA-SMI CUDA capability: 12.2
Conda root: /data/huinian/miniconda3
Conda env used: py38
```

## 2. Base Environment

Environment check:

```bash
conda activate py38
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Observed:

```text
Python 3.8.20
torch 1.9.0+cu111
torch CUDA 11.1
```

Installed OpenMMLab versions:

```text
mmcv-full 1.4.0
mmdet 2.14.0
mmsegmentation 0.14.1
```

## 3. AV2 Dependency Note

`requirement.txt` originally contained:

```text
shapely==1.8.5.post1
av2
```

Installing plain `av2` through `pip install -r requirement.txt` attempted to
pull `kornia -> torch` and download `torch 2.4.1`, which would overwrite the
MapTR-compatible torch 1.9.0/cu111 environment and also caused disk pressure.

Initial workaround was to install `av2 --no-deps`, but `av2 0.3.1` was too
new for this old reproduction stack. It pulled newer dependencies and expected
newer numpy features.

Pinned AV2 to the older API generation:

```bash
pip install -r requirement.txt
pip uninstall -y av2
pip install "av2==0.2.1" --no-deps
pip install av click joblib matplotlib nox opencv-python pandas pyarrow pyproj rich scipy
```

`av2==0.2.1` still imports `numpy.typing`, so `numpy==1.19.5` is too old.
Use the following compromise versions:

```bash
pip install "numpy==1.21.6" "pandas==1.3.5"
pip install "numba==0.56.4" "llvmlite==0.39.1"
```

This requires patching one old mmdetection3d import:

```bash
python - <<'PY'
from pathlib import Path

p = Path("/data/huinian/source/MapTR/mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py")
s = p.read_text()
old = "from numba.errors import NumbaPerformanceWarning"
new = """try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning"""
if old not in s:
    print("target line not found, maybe already patched")
else:
    p.write_text(s.replace(old, new))
    print("patched", p)
PY
```

Verify dependency state:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import numpy; import numpy.typing; print(numpy.__version__, 'numpy typing ok')"
python -c "import av2; print('av2 ok')"
python -c "from numba.core.errors import NumbaPerformanceWarning; print('numba core ok')"
```

Expected torch:

```text
1.9.0+cu111 11.1
```

## 4. mmdetection3d Build

Build command:

```bash
cd /data/huinian/source/MapTR/mmdetection3d
python setup.py develop
```

Observed:

- CUDA/C++ ops compiled and copied successfully, including `bev_pool_v2`.
- `mmdet3d` was registered as version `0.17.2`.
- Dependency processing later failed when old `easy_install` tried to process
  `scikit-image 0.26.0`, which has no legacy setup script.

Verification still passed:

```bash
python -c "import mmdet3d; print(mmdet3d.__version__)"
python -c "from mmdet3d.ops import bev_pool_v2; print('bev_pool_v2 ok')"
```

Observed:

```text
0.17.2
bev_pool_v2 ok
```

Decision:

- Treat `mmdetection3d` as usable for now.
- If the `scikit-image` issue matters later, install a compatible version first:

```bash
pip install "scikit-image==0.19.3"
```

or:

```bash
pip install "scikit-image==0.21.0"
```

then re-run:

```bash
cd /data/huinian/source/MapTR/mmdetection3d
python setup.py develop
```

## 5. Next Environment Step

Compile MapTR GKT op:

```bash
cd /data/huinian/source/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```

Verify:

```bash
cd /data/huinian/source/MapTR
python -c "from projects.mmdet3d_plugin.maptr.modules.ops.geometric_kernel_attn.function import geometric_kernel_attn_func; print('gkt func ok')"
```

Observed:

```text
gkt func ok
```

Note: the directory is `function/`, not `functions/`.
