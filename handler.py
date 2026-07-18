"""
VCKTOR Worker A - RunPod Serverless handler (커스텀 모델 전용)

백엔드 잡 규약(기존 워커와 동일):
  input = {
    "model_id": "<모델 이름>",          # 커스텀 모델 이름 (api.vvckd.ai/static/models/<name>/<name>.pth)
    "params":  { "vc_transform": 0, "f0_method": "rmvpe", "index_rate": 0.4, ... },
    "input_files": ["<presigned S3 다운로드 URL>", ...]
  }
  output = {
    "files": [
      {"status": "success", "output_url": "vcktor/outputs/<...>.wav", "filename": "<...>.wav"},
      ...
    ]
  }

기본 8모델은 기존 워커가 처리하고, 이 워커는 업로드된 커스텀 pth/index 모델만 담당한다.
"""
import os
import sys
import uuid
import tempfile
import traceback

# 엔진 루트를 import 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

# 엔진이 참조하는 경로 env (Dockerfile ENV 로도 세팅하지만 안전하게 기본값 보강)
os.environ.setdefault("weight_root", os.path.join(BASE_DIR, "assets", "weights"))
os.environ.setdefault("index_root", os.path.join(BASE_DIR, "assets", "indices"))
os.environ.setdefault("rmvpe_root", os.path.join(BASE_DIR, "assets", "rmvpe"))

WEIGHT_ROOT = os.environ["weight_root"]
INDEX_ROOT = os.environ["index_root"]

import numpy as np
import requests
import boto3
from scipy.io import wavfile
import runpod

from configs import Config
from infer.modules.vc import VC

# ---- 설정 ----
MODEL_STATIC_BASE = os.environ.get("MODEL_STATIC_BASE", "https://api.vvckd.ai/static/models").rstrip("/")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))
OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "vcktor/outputs").strip("/")

# 자연스러운 사운드용 고정 파라미터 프리셋(피치/transpose만 사용자값 사용)
DEFAULT_PARAMS = {
    "f0_method": "rmvpe",
    "index_rate": 0.4,
    "filter_radius": 3,
    "rms_mix_rate": 0.2,
    "protect": 0.4,
    "resample_sr": 0,
}

_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
)

# VC 엔진(전역 1회 초기화 → 워커 재사용 시 재로딩 없음)
_config = Config()
_vc = VC(_config)
_current_model = None  # get_vc 로 로드된 현재 모델 이름 캐시


def _download(url: str, dest: str, timeout: int = 180):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _ensure_model(name: str):
    """커스텀 모델 pth(+index)를 백엔드 static에서 받아 weight_root/index_root 에 배치."""
    os.makedirs(WEIGHT_ROOT, exist_ok=True)
    os.makedirs(INDEX_ROOT, exist_ok=True)
    pth_path = os.path.join(WEIGHT_ROOT, f"{name}.pth")
    idx_path = os.path.join(INDEX_ROOT, f"{name}.index")

    if not os.path.exists(pth_path):
        _download(f"{MODEL_STATIC_BASE}/{name}/{name}.pth", pth_path)

    if not os.path.exists(idx_path):
        try:
            _download(f"{MODEL_STATIC_BASE}/{name}/{name}.index", idx_path)
        except Exception:
            idx_path = ""  # index 없으면 index_rate 무효 (변환 자체는 가능)
    return pth_path, idx_path


def _load_model(name: str):
    global _current_model
    _ensure_model(name)
    if _current_model != name:
        _vc.get_vc(f"{name}.pth")
        _current_model = name


def handler(job):
    inp = job.get("input", {}) or {}
    name = inp.get("model_id")
    params = inp.get("params", {}) or {}
    input_files = inp.get("input_files", []) or []

    if not name:
        return {"status": "error", "error": "model_id(모델 이름)가 없습니다."}
    if not S3_BUCKET:
        return {"status": "error", "error": "S3_BUCKET_NAME 미설정"}

    try:
        _load_model(name)
    except Exception:
        return {"status": "error", "error": f"모델 로드 실패({name}): {traceback.format_exc()}"}

    idx_path = os.path.join(INDEX_ROOT, f"{name}.index")
    if not os.path.exists(idx_path):
        idx_path = ""

    # 파라미터 (피치=vc_transform/f0_up_key 만 사용자값, 나머지는 자연스러움 프리셋)
    f0_up_key = int(params.get("vc_transform", params.get("f0_up_key", 0)) or 0)
    f0_method = params.get("f0_method", DEFAULT_PARAMS["f0_method"])
    index_rate = float(params.get("index_rate", DEFAULT_PARAMS["index_rate"]))
    filter_radius = int(params.get("filter_radius", DEFAULT_PARAMS["filter_radius"]))
    rms_mix_rate = float(params.get("rms_mix_rate", DEFAULT_PARAMS["rms_mix_rate"]))
    protect = float(params.get("protect", DEFAULT_PARAMS["protect"]))
    resample_sr = int(params.get("resample_sr", DEFAULT_PARAMS["resample_sr"]))

    files = []
    for i, url in enumerate(input_files):
        in_path = None
        try:
            fd, in_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            _download(url, in_path)

            msg, out = _vc.vc_single(
                0,               # sid (speaker id)
                in_path,
                f0_up_key,
                None,            # f0_file
                f0_method,
                idx_path,        # file_index
                "",              # file_index2
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            )

            if out is None:
                files.append({"status": "error", "error": str(msg)})
                continue

            tgt_sr, audio = out
            out_name = f"{name}_{uuid.uuid4().hex}_{i}.wav"
            out_path = os.path.join(tempfile.gettempdir(), out_name)
            wavfile.write(out_path, tgt_sr, audio)

            s3_key = f"{OUTPUT_PREFIX}/{out_name}"
            _s3.upload_file(out_path, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "audio/wav"})

            try:
                os.remove(out_path)
            except OSError:
                pass

            files.append({"status": "success", "output_url": s3_key, "filename": out_name})
        except Exception:
            files.append({"status": "error", "error": traceback.format_exc()})
        finally:
            if in_path and os.path.exists(in_path):
                try:
                    os.remove(in_path)
                except OSError:
                    pass

    return {"files": files}


runpod.serverless.start({"handler": handler})
