"""Worker OmniVoice chạy trong process riêng (có thể khác Python env với server).

Giao thức JSON-lines: đọc request từ stdin, ghi response ra stdout. Mọi log của thư viện đi ra stderr.
  request : {"text", "language", "ref_wav": [path], "ref_text", "out", "duration"?}
  response: {"ok": true, "out", "seconds"} | {"ok": false, "error"}
"""
import argparse
import gc
import json
import os
import sys
from collections import OrderedDict

proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(obj: dict) -> None:
    proto.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=dtype)
    gc.collect()
    prompts: OrderedDict = OrderedDict()
    send({"ready": True, "sampling_rate": model.sampling_rate})

    for line in sys.stdin:
        try:
            req = json.loads(line)
            ref = req["ref_wav"][0]
            key = (ref, os.path.getmtime(ref), req.get("ref_text"))
            if key not in prompts:
                prompts[key] = model.create_voice_clone_prompt(ref_audio=ref, ref_text=req.get("ref_text"))
                while len(prompts) > 32:
                    prompts.popitem(last=False)
            prompts.move_to_end(key)
            audio = model.generate(
                text=req["text"], language=req.get("language"), voice_clone_prompt=prompts[key],
                duration=req.get("duration"), num_step=args.num_step,
            )[0]
            sf.write(req["out"], audio, model.sampling_rate)
            send({"ok": True, "out": req["out"], "seconds": len(audio) / model.sampling_rate})
        except Exception as exc:  # trả lỗi về server, worker vẫn sống
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
