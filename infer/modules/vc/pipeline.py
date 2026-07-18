import os
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from time import time

import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from rvc.f0 import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)

# 16000Hz 샘플링 레이트에서 48Hz 이상의 주파수를 통과시키는 5차 버터워스 고역 필터
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate):  # data1은 입력 오디오, data2는 출력 오디오, rate는 data2의 비율
    # print(data1.max(),data2.max())
    # 각 오디오의 RMS(Root Mean Square) 값 계산 (반초마다 하나의 값)
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 매 반초마다 하나의 포인트
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    # RMS 값을 데이터2의 길이에 맞게 선형 보간
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    # RMS2가 너무 작은 값을 갖지 않도록 최소값 설정
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    # RMS 비율에 따라 data2의 볼륨 조절
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        # 설정 값 초기화
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert 입력 샘플링 레이트
        self.window = 160  # 각 프레임당 포인트 수
        self.t_pad = self.sr * self.x_pad  # 각 오디오 앞뒤 패딩 시간
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 쿼리 지점 전후 쿼리 시간
        self.t_center = self.sr * self.x_center  # 쿼리 지점 위치
        self.t_max = self.sr * self.x_max  # 쿼리 면제 시간 임계값
        self.device = config.device

        # F0(기본 주파수) 생성기 초기화
        self.f0_gen = Generator(
            Path(os.environ["rmvpe_root"]),
            self.is_half,
            self.x_pad,
            self.device,
            self.window,
            self.sr,
        )

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # 음성 변환 함수
        # 오디오를 텐서로 변환
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # 듀얼 채널인 경우
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        # 모델 입력 준비
        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = time()
        # 특성 추출
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        # 인덱스 기반 검색 및 특성 혼합
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # 가장 유사한 특성들 검색
            try:
                score, ix = index.search(npy, k=8)
            except:
                raise Exception("index mistatch")
            # 가중치 계산 및 특성 혼합
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            # 검색된 특성과 원본 특성 혼합
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        # 특성 업샘플링 (2배 스케일 팩터)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = time()
        # 프레임 수 계산
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        # 음색 보호 기능 적용
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        # 생성 모델로 오디오 생성
        with torch.no_grad():
            audio1 = (
                (
                    net_g.infer(
                        feats,
                        p_len,
                        sid,
                        pitch=pitch,
                        pitchf=pitchf,
                    )[0, 0]
                )
                .data.cpu()
                .float()
                .numpy()
            )
        del feats, p_len, padding_mask
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        t2 = time()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=None,
    ):
        # 인덱스 파일 로드 (존재하는 경우)
        if (
            file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        # 고역 필터 적용
        audio = signal.filtfilt(bh, ah, audio)
        # 오디오 패딩
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        # 긴 오디오를 처리하기 위한 세그먼트 분할 지점 찾기
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            # 진폭이 가장 낮은 지점(무음에 가까운 지점)을 찾아 세그먼트 경계로 사용
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = time()
        # 오디오 패딩
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        # F0 파일 로드 (제공된 경우)
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    raw_lines = f.read()
                    if len(raw_lines) > 0:
                        lines = raw_lines.strip("\n").split("\n")
                        inp_f0 = []
                        for line in lines:
                            inp_f0.append([float(i) for i in line.split(",")])
                        inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        # F0(기본 주파수) 계산
        if if_f0:
            if if_f0 == 1:
                pitch, pitchf = self.f0_gen.calculate(
                    audio_pad,
                    p_len,
                    f0_up_key,
                    f0_method,
                    filter_radius,
                    inp_f0,
                )
            elif if_f0 == 2:
                pitch, pitchf = f0_method
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = time()
        times[1] += t2 - t1
        # 세그먼트별 음성 변환 처리
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        # 마지막 세그먼트 처리
        if if_f0:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        # 최종 오디오 병합
        audio_opt = np.concatenate(audio_opt)
        # RMS 혼합 적용 (설정된 경우)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        # 샘플링 레이트 변환 (필요한 경우)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        # 오디오 정규화
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        np.multiply(audio_opt, max_int16, audio_opt)
        # 메모리 정리
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return audio_opt
