"""Evaluasi zero-shot ToMMeR (deteksi mention): precision/recall/F1, decoding threshold dan greedy.

Pakai:
  python eval_table1.py <checkpoint HF atau folder model> <out.json> "CoNLL 2003,CrossNER_AI"
  python eval_table1.py --compare <folder berisi *.json>
Hasil disimpan per dataset, jadi kalau terputus cukup dijalankan ulang (dataset yang selesai dilewati).
"""
import os, sys, json
from pathlib import Path


def compare(folder):
    files = sorted(Path(folder).glob("*.json"))
    runs = {f.stem: json.loads(f.read_text()) for f in files}
    if not runs:
        print("belum ada hasil di", folder); return
    for strat in ("threshold", "greedy"):
        print(f"\n== decoding: {strat} (P / R / F1, %) ==")
        names = sorted({k.split("/", 1)[1] for r in runs.values() for k in r if k.startswith(strat + "/")})
        print(f"{'dataset':<22}" + "".join(f"{n:>24}" for n in runs))
        rec = {n: [] for n in runs}
        for d in names:
            row = f"{d:<22}"
            for n, r in runs.items():
                m = r.get(f"{strat}/{d}")
                if m:
                    row += f"{100*m['precision']:>8.1f}{100*m['recall']:>8.1f}{100*m['f1']:>8.1f}"
                    rec[n].append(m["recall"])
                else:
                    row += f"{'-':>24}"
            print(row)
        print(f"{'rata-rata recall':<22}" + "".join(
            f"{(100*sum(v)/len(v) if v else 0):>24.1f}" for v in rec.values()))


if sys.argv[1] == "--compare":
    compare(sys.argv[2]); sys.exit()

import torch
from xpm_torch.huggingface import TorchHFHub
from llm2ner import utils
from llm2ner.data import load_all_splits

CKPT, OUT = sys.argv[1], Path(sys.argv[2])
DATASETS = sys.argv[3].split(",")
DATA = Path(os.environ["NER_DATA"])

own = Path(CKPT).is_dir() and not (Path(CKPT) / "experimaestro.json").exists()
# Hasil training llm2ner hanya menyimpan bobot (model/model.safetensors), tanpa experimaestro.json.
# Model dibangun ulang dengan kelas dan config yang PERSIS sama dengan saat training (dari params.json
# job, mis. TokenMatchingNER dengan mask_bos=False), lalu bobot training dimuat.
if own:
    import importlib
    params = json.loads((Path(CKPT).parent / "params.json").read_text())
    obj = next(o for o in params["objects"] if ".models." in o["typename"] and "layer" in o["fields"])
    mod, cls = obj["typename"].rsplit(".", 1)
    print("kelas model training:", obj["typename"], flush=True)
    print("config training:", obj["fields"], flush=True)
    tommer = getattr(importlib.import_module(mod), cls).C(**obj["fields"]).instance()
    tommer.initialize()
    tommer.load_model(Path(CKPT))  # strict: error kalau bentuk bobot tidak cocok
    print("layer training:", tommer.layer, "| bobot training dimuat dari", CKPT, flush=True)
else:
    tommer = TorchHFHub.from_pretrained(CKPT)
    tommer.initialize()
llm = utils.load_llm(tommer.llm_name, cut_to_layer=tommer.layer).eval()
tommer.to(llm.device)
print("backbone:", tommer.llm_name, "| layer:", tommer.layer, "| device:", llm.device, flush=True)
c_length = utils.get_model_max_length(tommer.llm_name)

res = json.loads(OUT.read_text()) if OUT.exists() else {}
LOADER = "kelas_training" if own else "hf_rilis"
if res and res.get("_loader", None if own else "hf_rilis") != LOADER:  # hasil rilis lama tetap sah
    # hasil lama dibuat dengan cara muat model yang berbeda: disimpan sebagai .bak, evaluasi diulang
    OUT.rename(OUT.with_suffix(".bak")); print("hasil lama dipindah ke", OUT.with_suffix(".bak"), flush=True)
    res = {}
res["_loader"] = LOADER
with torch.no_grad():
    for name in DATASETS:
        if all(f"{s}/{name}" in res for s in ("threshold", "greedy")):
            print("lewati (sudah ada):", name); continue
        if not (DATA / name).is_dir():
            print("LEWATI, folder dataset tidak ada:", DATA / name, flush=True); continue
        ds = load_all_splits(dataset_name=name, data_folder=DATA, model=llm, mode="last",
                             max_ent_length=400, max_length=c_length, verbose=True)
        for strat in ("threshold", "greedy"):
            m = tommer.evaluate(ds.get_loader(batch_size=30), decoding_strategy=strat,
                                threshold=0.5, verbose=True)
            m["n_samples"] = len(ds)
            res[f"{strat}/{name}"] = m
            OUT.write_text(json.dumps(res, indent=2, default=float))
            print(strat, name, {k: round(float(m[k]), 4) for k in ("precision", "recall", "f1")}, flush=True)
