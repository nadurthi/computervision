#!/usr/bin/env bash
# Install everything the benchmark runs, on an Ubuntu 22.04 GPU machine with the NVIDIA driver + CUDA toolkit
# (the AWS "Deep Learning Base OSS Nvidia Driver GPU AMI" has both).
#
#   bash setup/install_models.sh /opt/mvs/models            # idempotent; re-running only fills in what is missing
#
# Environments (all under $MODELS_ROOT/envs):
#   main     uv venv, Python 3.11, torch 2.4.1 cu121 + xformers   -> classical (ours), SGBM, RAFT-Stereo, IGEV,
#                                                                    FoundationStereo, PatchmatchNet, IterMVS
#   monster  uv venv, Python 3.11, torch 2.4.1 cu121, timm 0.6.13 -> MonSter (pins an old timm)
#   mvsa     micromamba env from MVSAnywhere's environment.yml    -> MVSAnywhere, Vis-MVSNet, Robust MVD (robustmvd)
#                                                                    and COLMAP 3.8 (CUDA) for COLMAP PatchMatch
# Code + weights:
#   RAFT-Stereo, IGEV, FoundationStereo, MonSter, PatchmatchNet, IterMVS, ACMMP (built with CUDA), DTUeval-python,
#   robustmvd, mvsanywhere, Vis-MVSNet. Google-Drive weights are mirrored to S3 ($CACHE_S3) after the first
#   successful download, because gdown is regularly rate-limited.
set -uo pipefail
MODELS_ROOT=${1:-/opt/mvs/models}
CACHE_S3=${CACHE_S3:-}                       # e.g. s3://mvs-bench-<account>-us-east-1/cache
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/fetch.sh"
mkdir -p "$MODELS_ROOT"/{envs,weights}
cd "$MODELS_ROOT"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[install_models] $*"; }

# ------------------------------------------------------------------------------------------------ system packages
if ! dpkg -s libopencv-dev p7zip-full >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libopencv-dev cmake build-essential p7zip-full unzip \
    git wget curl >/dev/null
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v micromamba >/dev/null || (curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
  tar -xvj -C "$HOME/.local" bin/micromamba >/dev/null)

clone() {  # clone <url> <dir> [ref]
  [ -d "$2/.git" ] || git clone --depth 1 ${3:+--branch $3} "$1" "$2"
}

# ------------------------------------------------------------------------------------------------ repos
clone https://github.com/princeton-vl/RAFT-Stereo RAFT-Stereo
clone https://github.com/gangweiX/IGEV IGEV
clone https://github.com/NVlabs/FoundationStereo FoundationStereo
clone https://github.com/Junda24/MonSter MonSter
clone https://github.com/FangjinhuaWang/PatchmatchNet PatchmatchNet
clone https://github.com/FangjinhuaWang/IterMVS IterMVS
clone https://github.com/GhiXu/ACMMP ACMMP
clone https://github.com/jzhangbs/DTUeval-python DTUeval-python
clone https://github.com/lmb-freiburg/robustmvd robustmvd
clone https://github.com/nianticlabs/mvsanywhere mvsanywhere
clone https://github.com/jzhangbs/Vis-MVSNet Vis-MVSNet

# ------------------------------------------------------------------------------------------------ envs
PY_COMMON="numpy<2 opencv-python-headless scikit-image scipy matplotlib pyyaml tqdm pillow imageio omegaconf \
  einops plyfile open3d scikit-learn gdown huggingface_hub"
if [ ! -x envs/main/bin/python ]; then
  uv venv -q -p 3.11 envs/main
  uv pip install -q -p envs/main/bin/python torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1 \
    --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
  uv pip install -q -p envs/main/bin/python $PY_COMMON timm opt_einsum ruamel.yaml trimesh joblib albumentations \
    imgaug transformations
fi
if [ ! -x envs/monster/bin/python ]; then
  uv venv -q -p 3.11 envs/monster
  uv pip install -q -p envs/monster/bin/python torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
  uv pip install -q -p envs/monster/bin/python $PY_COMMON timm==0.6.13 tensorboard
fi
if [ ! -x envs/mvsa/bin/python ]; then
  micromamba create -y -q -p "$MODELS_ROOT/envs/mvsa" -f mvsanywhere/environment.yml || \
    log "WARNING: micromamba env from environment.yml failed; MVSAnywhere/Vis-MVSNet/RobustMVD/COLMAP will be skipped"
fi
if [ -x envs/mvsa/bin/python ]; then
  envs/mvsa/bin/pip install -q -e mvsanywhere
  envs/mvsa/bin/pip install -q -e robustmvd
  envs/mvsa/bin/pip install -q pyyaml plyfile "numpy<2"
  cat > "$HOME/rmvd_model_paths.toml" <<EOF
[mvsanywhere]
    root = '$MODELS_ROOT/mvsanywhere'
[vis_mvsnet]
    root = '$MODELS_ROOT/Vis-MVSNet'
EOF
fi

# ------------------------------------------------------------------------------------------------ weights
# RAFT-Stereo: official models.zip (Dropbox)
if [ ! -f RAFT-Stereo/models/raftstereo-middlebury.pth ]; then
  fetch_url raft_stereo_models.zip \
    "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1" /tmp/raft.zip \
    && unzip -qo /tmp/raft.zip -d RAFT-Stereo && rm -f /tmp/raft.zip
fi
# IGEV-Stereo: Google Drive folder (sceneflow / middlebury / kitti / eth3d ...)
[ -n "$(ls weights/igev 2>/dev/null)" ] || fetch_gdrive_folder igev \
  "https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz" weights/igev
# FoundationStereo: Google Drive folder with 23-51-11 (ViT-L, best) and 11-33-40 (ViT-S)
[ -n "$(ls weights/foundationstereo 2>/dev/null)" ] || fetch_gdrive_folder foundationstereo \
  "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf" weights/foundationstereo
# MonSter: Hugging Face (sceneflow + mix_all) and the Depth Anything V2 ViT-L encoder it is built on
mkdir -p weights/monster MonSter/pretrained
for w in sceneflow mix_all; do
  [ -f weights/monster/$w.pth ] || fetch_url monster_$w.pth \
    "https://huggingface.co/cjd24/MonSter/resolve/main/$w.pth?download=true" weights/monster/$w.pth
done
[ -f MonSter/pretrained/depth_anything_v2_vitl.pth ] || fetch_url depth_anything_v2_vitl.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
  MonSter/pretrained/depth_anything_v2_vitl.pth
# PatchmatchNet / IterMVS: checkpoints are inside the repos (checkpoints/)
# MVSAnywhere: Niantic GCS
mkdir -p mvsanywhere/pretrained_weights
for w in mvsanywhere_hero mvsanywhere_dot; do
  [ -f mvsanywhere/pretrained_weights/$w.ckpt ] || fetch_url $w.ckpt \
    "https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/$w.ckpt" mvsanywhere/pretrained_weights/$w.ckpt
done

# ------------------------------------------------------------------------------------------------ ACMMP (CUDA)
if [ ! -x ACMMP/build/ACMMP ]; then
  CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda 2>/dev/null || true)}
  CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  CC_ARCH=${CC_ARCH:-89}
  sed -i "s#set(CUDA_TOOLKIT_ROOT_DIR .*)#set(CUDA_TOOLKIT_ROOT_DIR \"$CUDA_HOME\")#" ACMMP/CMakeLists.txt
  sed -i "s#-gencode arch=compute_[0-9]*,code=sm_[0-9]*#-gencode arch=compute_${CC_ARCH},code=sm_${CC_ARCH}#" ACMMP/CMakeLists.txt
  sed -i "s#-std=c++11#-std=c++14#g" ACMMP/CMakeLists.txt
  mkdir -p ACMMP/build && (cd ACMMP/build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j"$(nproc)" >/dev/null) \
    && log "built ACMMP" || log "WARNING: ACMMP build failed (acmmp will be skipped)"
fi

log "done. envs: $(ls envs | tr '\n' ' ')"
