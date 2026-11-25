"""Worker XTTS-v2 chạy trong process riêng (env cần coqui-tts, transformers<5).

Cùng giao thức JSON-lines với omnivoice_worker.py. XTTS không hỗ trợ `duration` (server tự atempo).
"""
import argparse
import ctypes
import gc
import json
import os
import sys

proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(obj: dict) -> None:
    proto.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    os.environ["COQUI_TOS_AGREED"] = "1"
    import soundfile as sf
    import torch
    from TTS.api import TTS

    if args.threads:
        torch.set_num_threads(args.threads)
    tts = TTS(args.model).to(args.device)
    # Checkpoint trên CPU (~4GB) còn giữ tới khi gc chạy
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if sys.platform.startswith("linux"):
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    send({"ready": True})

    for line in sys.stdin:
        try:
            req = json.loads(line)
            tts.tts_to_file(text=req["text"], language=req["language"], speaker_wav=req["ref_wav"],
                            file_path=req["out"], split_sentences=True)
            send({"ok": True, "out": req["out"], "seconds": sf.info(req["out"]).duration})
        except Exception as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
