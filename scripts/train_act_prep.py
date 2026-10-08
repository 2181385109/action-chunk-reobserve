"""T10a 自训练 ACT 准备（WSL 内运行；入口 scripts/train_act_prep.sh）。

子命令：
  download                     下载 lerobot/aloha_sim_transfer_cube_human，记录大小、视频编码、解码后端与解码速度
                               -> results/train_act_own/prep/dataset_info.json
  bench --workers 4 --steps N  用 lerobot 0.3.3 自带训练脚本（lerobot.scripts.train，不改源码）+ 官方超参数跑 N 步，
                               逐行读日志记录到达时间，同时每 0.5 s 采样 WSL 内存 / 进程树 RSS / GPU 显存
                               -> results/train_act_own/prep/bench_w<W>_s<N>.json（+ 原始日志 .txt）
  diffcfg <train_config.json>  与官方 train_config 比较（只列不同的字段）

训练命令行与 scripts/train_act.sh 共用 train_act_tools.train_args()，保证冒烟 / 测速 / 正式训练用的是同一套参数。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_act_tools import (OFFICIAL_CFG, REPO_ID, diff_train_config, parse_train_log,  # noqa: E402
                             train_args)

PREP = ROOT / "results" / "train_act_own" / "prep"


def _du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cmd_download(a):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import get_safe_default_codec

    t0 = time.time()
    ds = LeRobotDataset(REPO_ID, video_backend="pyav")
    t_dl = time.time() - t0
    root = Path(ds.root)
    vids = sorted(root.rglob("*.mp4"))
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt,width,height,avg_frame_rate", "-of", "json", str(vids[0])],
                           capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0] if probe.returncode == 0 else {"error": probe.stderr[-300:]}

    # 解码速度：同一组随机样本，分别用 pyav / torchcodec 取（只读，不影响训练）
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, ds.num_frames, (40,), generator=g).tolist()
    decode = {}
    for backend in ["pyav", "torchcodec"]:
        try:
            d = LeRobotDataset(REPO_ID, video_backend=backend)
            _ = d[idx[0]]                      # 预热
            t = time.time()
            for i in idx:
                item = d[i]
            dt = (time.time() - t) / len(idx)
            img = item["observation.images.top"]
            decode[backend] = dict(ok=True, ms_per_sample=round(dt * 1000, 1), image_shape=list(img.shape),
                                   image_dtype=str(img.dtype), action_shape=list(item["action"].shape))
        except Exception as e:  # noqa: BLE001
            decode[backend] = dict(ok=False, error=f"{type(e).__name__}: {e}"[:400])

    # 两种解码器出的像素是否一样：直接用 PyAV（av.open，不经 torchvision）与 torchcodec 解同一视频的几帧
    pix = {}
    try:
        import av
        import numpy as np
        from torchcodec.decoders import VideoDecoder
        want = [0, 50, 199, 399]
        with av.open(str(vids[0])) as c:
            av_frames = {i: f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(video=0)) if i in want}
        dec = VideoDecoder(str(vids[0]))
        diffs = []
        for i in want:
            tc = dec[i].permute(1, 2, 0).numpy().astype(np.int16)
            diffs.append(np.abs(tc - av_frames[i].astype(np.int16)))
        dd = np.stack(diffs)
        pix = dict(video=vids[0].name, frames=want, max_abs_diff=int(dd.max()), mean_abs_diff=round(float(dd.mean()), 4),
                   frac_pixels_diff=round(float((dd > 0).mean()), 4))
    except Exception as e:  # noqa: BLE001
        pix = dict(error=f"{type(e).__name__}: {e}"[:400])

    info = dict(
        repo_id=REPO_ID, root=str(root), download_s=round(t_dl, 1), pyav_vs_torchcodec_pixels=pix,
        size_bytes=_du(root), size_MB=round(_du(root) / 1e6, 1),
        size_by_dir_MB={p.name: round(_du(p) / 1e6, 1) for p in sorted(root.iterdir()) if p.is_dir()},
        n_video_files=len(vids), num_episodes=ds.num_episodes, num_frames=ds.num_frames, fps=ds.fps,
        features={k: dict(dtype=v["dtype"], shape=list(v["shape"])) for k, v in ds.meta.features.items()},
        codebase_version=ds.meta.info.get("codebase_version"),
        video_info=ds.meta.info["features"]["observation.images.top"].get("info"),
        ffprobe_first_video=stream,
        delta_timestamps_note="ACT 训练时 lerobot 按 chunk_size=100 自动设置 action 的 delta_timestamps（每样本 1 帧图像 + 100 步动作）",
        lerobot_default_video_backend=get_safe_default_codec(),
        official_video_backend=OFFICIAL_CFG["dataset"]["video_backend"],
        decode_bench=decode,
        ffmpeg=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
    )
    try:
        import av
        info["pyav_version"] = av.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torchcodec
        info["torchcodec_version"] = torchcodec.__version__
    except Exception:  # noqa: BLE001
        pass
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------
def _meminfo() -> dict:
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) * 1024
    return d


def _tree_rss(pid: int) -> int:
    """pid 及其所有子孙进程的 RSS 之和（字节），读 /proc。"""
    children = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(p.name))
        except OSError:
            continue
    total, stack = 0, [pid]
    while stack:
        q = stack.pop()
        try:
            for line in open(f"/proc/{q}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
        stack += children.get(q, [])
    return total


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def cmd_bench(a):
    out_dir = Path(a.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)          # 只用于测速的临时目录（/root/train_out/bench_*）
    args = train_args(steps=a.steps, num_workers=a.workers, output_dir=str(out_dir),
                      save_freq=a.save_freq, log_freq=a.log_freq)
    cmd = [sys.executable, "-m", "lerobot.scripts.train", *args]
    print("命令：", " ".join(cmd), flush=True)
    mem0, gpu0 = _meminfo(), _gpu_used_mib()
    used0 = mem0["MemTotal"] - mem0["MemAvailable"]
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            m = _meminfo()
            samples.append(dict(t=round(time.time() - t_start, 2), wsl_used=m["MemTotal"] - m["MemAvailable"],
                                tree_rss=_tree_rss(proc.pid), gpu_mib=_gpu_used_mib()))
            stop.wait(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    lines, arrivals = [], []
    for line in proc.stdout:
        lines.append(line)
        arrivals.append(time.time() - t_start)
        sys.stdout.write(line)
    rc = proc.wait()
    stop.set()
    th.join()
    t_total = time.time() - t_start

    raw = PREP / f"bench_w{a.workers}_s{a.steps}_log.txt"
    PREP.mkdir(parents=True, exist_ok=True)
    raw.write_text("".join(lines), encoding="utf-8")
    recs = parse_train_log("".join(lines), log_freq=a.log_freq)
    # 每条 step 日志行的到达时间
    t_of_step = {}
    k = 0
    for line, t in zip(lines, arrivals):
        if " step:" in line and " loss:" in line:
            if k < len(recs):
                t_of_step[recs[k]["step"]] = t
            k += 1
    steady = None
    warm = a.warmup
    pts = sorted((s, t) for s, t in t_of_step.items() if s >= warm)
    if len(pts) >= 2:
        (s0, t0), (s1, t1) = pts[0], pts[-1]
        steady = dict(from_step=s0, to_step=s1, seconds=round(t1 - t0, 2), steps_per_s=round((s1 - s0) / (t1 - t0), 3))
    first_step_t = min(t_of_step.values()) if t_of_step else None
    peak = lambda key: max((s[key] for s in samples if s[key] is not None), default=None)  # noqa: E731
    res = dict(
        workers=a.workers, steps=a.steps, returncode=rc, wall_s=round(t_total, 1),
        startup_s_to_first_log=round(first_step_t - a.log_freq / steady["steps_per_s"], 1) if (steady and first_step_t) else None,
        steady=steady,
        mean_updt_s=_mean([r["updt_s"] for r in recs if r["step"] >= warm]),
        mean_data_s=_mean([r["data_s"] for r in recs if r["step"] >= warm]),
        wsl_used_before_GB=round(used0 / 2**30, 2), wsl_used_peak_GB=round(peak("wsl_used") / 2**30, 2) if samples else None,
        wsl_total_GB=round(mem0["MemTotal"] / 2**30, 2),
        train_tree_rss_peak_GB=round(peak("tree_rss") / 2**30, 2) if samples else None,
        gpu_used_before_MiB=gpu0, gpu_used_peak_MiB=peak("gpu_mib"),
        gpu_train_peak_MiB=(peak("gpu_mib") - gpu0) if (gpu0 is not None and peak("gpu_mib") is not None) else None,
        n_samples=len(samples), losses=[(r["step"], r["loss"]) for r in recs],
        cmd=cmd,
    )
    (PREP / f"bench_w{a.workers}_s{a.steps}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("losses", "cmd")}, indent=2))
    sys.exit(rc)


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def cmd_diffcfg(a):
    ours = json.loads(Path(a.path).read_text(encoding="utf-8"))
    rows = diff_train_config(OFFICIAL_CFG, ours)
    for key, off, mine in rows:
        print(f"{key}: 官方={off!r}  我们={mine!r}")
    if a.out:
        Path(a.out).write_text(json.dumps([dict(key=k, official=o, ours=m) for k, o, m in rows], indent=2,
                                          ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    b = sub.add_parser("bench")
    b.add_argument("--workers", type=int, required=True)
    b.add_argument("--steps", type=int, default=600)
    b.add_argument("--log-freq", type=int, default=50, dest="log_freq")
    b.add_argument("--save-freq", type=int, default=100000, dest="save_freq")
    b.add_argument("--warmup", type=int, default=100)
    b.add_argument("--out-dir", dest="out_dir")
    d = sub.add_parser("diffcfg")
    d.add_argument("path")
    d.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "bench" and not a.out_dir:
        a.out_dir = f"/root/train_out/bench_w{a.workers}"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    dict(download=cmd_download, bench=cmd_bench, diffcfg=cmd_diffcfg)[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
