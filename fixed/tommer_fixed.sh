#!/usr/bin/env bash
# ToMMeR versi distilasi aktif. Semua file baru ada di folder skrip ini (tommer_fixed).
# Folder lama (tommer, tommer_repro) hanya dibaca dan tidak pernah ditulis.
# Pakai: bash tommer_fixed.sh setup | train 6 | train 1 | status | eval | save
set -euo pipefail
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SELF="$WORK/$(basename "${BASH_SOURCE[0]}")"
GROUP="$(dirname "$WORK")"; OLD="$GROUP/tommer"; OLD_REPRO="$GROUP/tommer_repro"
[ "$(basename "$WORK")" = tommer_fixed ] || { echo "ERROR: taruh skrip ini di $GROUP/tommer_fixed/"; exit 1; }
REPO="$WORK/llm2ner"; XPM="$WORK/xpm-workspace"; OUT="$WORK/results"; LOG="$WORK/logs"; MODELS="$WORK/models"
JOBS="$XPM/jobs/llm2ner.tasks.learntokenmatching"; COMMIT=d71bab4a9c517952b0cee015abf68403af64be34
DATASETS="CoNLL 2003,CrossNER_AI,CrossNER_literature,CrossNER_music,CrossNER_politics,CrossNER_science,MultiNERD,ncbi,FabNER,WikiNeural,GENIA_NER,ACE 2005,Ontonotes"
# Data dan Llama dari folder lama hanya dibaca, mode offline agar cache lama tidak berubah
export NER_DATA="$OLD/NER_DATA" HF_HOME="$OLD/hf_home" HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$WORK/.cache" XDG_CONFIG_HOME="$WORK/.config" XDG_DATA_HOME="$WORK/.local/share"
export TORCH_HOME="$WORK/.cache/torch" UV_CACHE_DIR="$WORK/uv_cache" UV_TOOL_DIR="$WORK/uv_tools"
export UV_PYTHON_INSTALL_DIR="$WORK/uv_python" PATH="$OLD/bin:$PATH"
die() { echo "ERROR: $*" >&2; exit 1; }
venv() { [ -f "$REPO/.venv/bin/activate" ] || die "jalankan dulu: bash $SELF setup"; source "$REPO/.venv/bin/activate"; cd "$REPO"; }
running() { pgrep -u "$USER" -f -- "run-experiment --workdir $GROUP/tommer" >/dev/null 2>&1; }
gpu() { export CUDA_VISIBLE_DEVICES; CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '); [ -n "$CUDA_VISIBLE_DEVICES" ] || die "GPU tidak terdeteksi"; echo "GPU: $CUDA_VISIBLE_DEVICES"; }
tag() { local l e; l=$(grep -oE '"layer": *[0-9]+' "$1/params.json" | head -1 | grep -oE '[0-9]+$'); e=$(grep -oE '"epochs": *[0-9]+' "$1/params.json" | head -1 | grep -oE '[0-9]+$'); echo "L${l}_E${e}"; }

apply_patches() {
  git apply --check - <<'DIFF'
diff --git a/src/llm2ner/distillation.py b/src/llm2ner/distillation.py
index e751aff..b6de32b 100644
--- a/src/llm2ner/distillation.py
+++ b/src/llm2ner/distillation.py
@@ -8,7 +8,7 @@ from torch.utils.data import DataLoader
 from transformer_lens import HookedTransformer
 from transformers import PreTrainedModel
 
-from llm2ner.models import NERmodel, train_lightning
+from llm2ner.models import NERmodel, manual_train
 from llm2ner.losses import balanced_BCE
 
 
@@ -156,12 +156,12 @@ def distill_step(
         return_tensors="pt",
         truncation=True,
     )
-    tokens = inputs["input_ids"].to(self.device)  # shape (batch, seq)
-    attn_mask = inputs["attention_mask"].to(self.device)
+    tokens = inputs["input_ids"].to(model.device)  # shape (batch, seq)
+    attn_mask = inputs["attention_mask"].to(model.device)
 
     gt_spans = batch[
         "pattern"
-    ].to(self.device)  # data must be in 'last' mode <-> patterns are span labels
+    ].to(model.device)  # data must be in 'last' mode <-> patterns are span labels
 
     # Forward pass through the teacher model
     with torch.no_grad():
@@ -241,9 +241,8 @@ def train_distill(
             "
     )
 
-    return train_lightning(
+    return manual_train(
         student,
-        model=model,
         train_loader=train_loader,
         val_loader=val_loader,
         optimizer=optimizer,
diff --git a/src/llm2ner/models/TokenMatching.py b/src/llm2ner/models/TokenMatching.py
index 8fe5200..b415444 100644
--- a/src/llm2ner/models/TokenMatching.py
+++ b/src/llm2ner/models/TokenMatching.py
@@ -44,6 +44,7 @@ from llm2ner.models.model import (
     DEFAULT_KERNEL_PADDING,
     NERmodel,
     train_lightning,
+    manual_train,
 )
 
 
@@ -1931,9 +1932,8 @@ class TokenMatchingNER(NERmodel):
         gc.collect()
         torch.cuda.empty_cache()
 
-        hist = train_lightning(
+        hist = manual_train(
             self,
-            model=train_loader.dataset.model,
             train_loader=train_loader,
             val_loader=val_loader,
             optimizer=optimizer,
DIFF
  git apply - <<'DIFF'
diff --git a/src/llm2ner/distillation.py b/src/llm2ner/distillation.py
index e751aff..b6de32b 100644
--- a/src/llm2ner/distillation.py
+++ b/src/llm2ner/distillation.py
@@ -8,7 +8,7 @@ from torch.utils.data import DataLoader
 from transformer_lens import HookedTransformer
 from transformers import PreTrainedModel
 
-from llm2ner.models import NERmodel, train_lightning
+from llm2ner.models import NERmodel, manual_train
 from llm2ner.losses import balanced_BCE
 
 
@@ -156,12 +156,12 @@ def distill_step(
         return_tensors="pt",
         truncation=True,
     )
-    tokens = inputs["input_ids"].to(self.device)  # shape (batch, seq)
-    attn_mask = inputs["attention_mask"].to(self.device)
+    tokens = inputs["input_ids"].to(model.device)  # shape (batch, seq)
+    attn_mask = inputs["attention_mask"].to(model.device)
 
     gt_spans = batch[
         "pattern"
-    ].to(self.device)  # data must be in 'last' mode <-> patterns are span labels
+    ].to(model.device)  # data must be in 'last' mode <-> patterns are span labels
 
     # Forward pass through the teacher model
     with torch.no_grad():
@@ -241,9 +241,8 @@ def train_distill(
             "
     )
 
-    return train_lightning(
+    return manual_train(
         student,
-        model=model,
         train_loader=train_loader,
         val_loader=val_loader,
         optimizer=optimizer,
diff --git a/src/llm2ner/models/TokenMatching.py b/src/llm2ner/models/TokenMatching.py
index 8fe5200..b415444 100644
--- a/src/llm2ner/models/TokenMatching.py
+++ b/src/llm2ner/models/TokenMatching.py
@@ -44,6 +44,7 @@ from llm2ner.models.model import (
     DEFAULT_KERNEL_PADDING,
     NERmodel,
     train_lightning,
+    manual_train,
 )
 
 
@@ -1931,9 +1932,8 @@ class TokenMatchingNER(NERmodel):
         gc.collect()
         torch.cuda.empty_cache()
 
-        hist = train_lightning(
+        hist = manual_train(
             self,
-            model=train_loader.dataset.model,
             train_loader=train_loader,
             val_loader=val_loader,
             optimizer=optimizer,
DIFF
  python3 - <<'EOF'
import pathlib
def sub(path, old, new):
    p = pathlib.Path(path); s = p.read_text()
    assert s.count(old) == 1, f"pola tidak ditemukan di {path}: {old!r}"
    p.write_text(s.replace(old, new))
x = "experiments/xp_learnTokenMatching.py"
sub(x, '    eval_launcher = find_launcher(cfg.eval_launcher, tags=["slurm"])\n',
    '    eval_launcher = find_launcher(cfg.eval_launcher, tags=["slurm"])\n'
    '    for _launcher in (gpulauncher, eval_launcher):\n'
    '        _launcher.environ.update({k: os.environ[k] for k in ("HF_HOME", "HF_HUB_OFFLINE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",\n'
    '                                  "XDG_DATA_HOME", "TORCH_HOME", "NER_DATA", "CUDA_VISIBLE_DEVICES")\n'
    '                                  if k in os.environ})\n')
sub(x, "tb_service = helper.xp.add_service(TensorboardService(runpath))", "tb_service = None")
sub(x, "tb_service.add(task, task.runpath)", "pass")
EOF
}

make_config() {
  python3 - "$1" <<'EOF'
import re, sys, pathlib
layer = sys.argv[1]
lines = pathlib.Path("experiments/trainTM_layers.yaml").read_text().splitlines()
scalars = {"id": f"llm2ner_TM_fixed_L{layer}",
           "launcher": "duration=15h & cuda(mem=15G)*1 & cpu(cores=4, mem=10G)",
           "eval_launcher": "duration=10h & cuda(mem=15G)*1 & cpu(cores=4, mem=10G)"}
lists = {"flat_eval_datasets": ["CoNLL 2003", "CrossNER_AI", "ncbi"], "nested_eval_datasets": ["GENIA_NER"], "llm_annotated_datasets": []}
res, skip, block = [], False, None
for l in lines:
    k = re.match(r"^([A-Za-z_]+):", l)
    if skip:
        if re.match(r"^\s+-", l) or re.match(r"^\s*#", l) or not l.strip(): continue
        skip = False
    if k:
        block = k.group(1)
        if block in scalars: res.append(f"{block}: {scalars[block]}"); continue
        if block in lists:
            res.append(f"{block}:" + ("" if lists[block] else " []")); res += [f'    - "{d}"' for d in lists[block]]; skip = True; continue
    if block == "layers" and re.match(r"^\s+(from|to):", l):
        l = re.sub(r"(from|to):\s*\d+", lambda m: f"{m.group(1)}: {layer}", l)
    res.append(l)
pathlib.Path(f"experiments/trainTM_fixed_L{layer}.yaml").write_text("\n".join(res) + "\n")
EOF
}

case "${1:-}" in
setup)
  [ -e "$REPO" ] && die "$REPO sudah ada"
  mkdir -p "$OUT" "$LOG" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
  [ ! -d "$OLD/.config" ] || cp -rn "$OLD/.config/." "$XDG_CONFIG_HOME/"
  cp "$OLD_REPRO/eval_table1.py" "$WORK/eval_table1.py"
  git clone "${REPO_URL:-https://github.com/VictorMorand/llm2ner.git}" "$REPO"
  cd "$REPO"; git checkout -q "$COMMIT"
  apply_patches; make_config 6; make_config 1
  [ "${SKIP_UV:-0}" = 1 ] || uv sync
  echo "== perubahan dari kode penulis =="; git diff --stat; git status --short | grep '^??' || true
  echo "Setup selesai. Lanjut: bash $SELF train 6" ;;
train)
  L="${2:?layer: 6 atau 1}"; venv; Y="experiments/trainTM_fixed_L$L.yaml"
  [ -f "$Y" ] || die "$Y tidak ada"; grep -q "return manual_train" src/llm2ner/distillation.py || die "patch distilasi tidak ada"
  [ -f "$NER_DATA/Pile-NER/train.json" ] || die "Pile-NER tidak ditemukan di $NER_DATA"
  running && die "masih ada training kelompok ini yang berjalan"
  gpu
  setsid nohup bash -c "experimaestro run-experiment --workdir '$XPM' '$Y'; echo \"selesai, exit \$?\"" > "$LOG/train_L$L.log" 2>&1 < /dev/null &
  sleep 5; tail -3 "$LOG/train_L$L.log"; echo "Berjalan di background. Cek: bash $SELF status" ;;
status)
  if running; then echo "STATUS: training berjalan"; else echo "STATUS: tidak ada training berjalan"; fi
  [ -d "$JOBS" ] || { echo "job belum dibuat"; tail -5 "$LOG"/train_L*.log 2>/dev/null; exit 0; }
  for J in "$JOBS"/*/; do
    [ -f "$J/params.json" ] || continue
    if [ -f "$J/learntokenmatching.done" ]; then S=SELESAI; elif [ -f "$J/learntokenmatching.failed" ]; then S=GAGAL; else S=BERJALAN; fi
    P=belum; [ -f "$J/model_phase_0.pth" ] && P=selesai
    echo "$(tag "$J") $S | fase 0: $P | $J"
    tail -c 1500 "$J/learntokenmatching.err" 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -3 | cut -c1-150 || true
  done ;;
eval)
  venv; running && die "tunggu training selesai"; gpu; mkdir -p "$OUT"
  cp "$OLD/results/release_L6.json" "$OUT/release_L6.json"
  for t in L6_E8 L1_E8; do [ ! -f "$OLD/results/own_$t.json" ] || cp "$OLD/results/own_$t.json" "$OUT/lama_$t.json"; done
  for J in "$JOBS"/*/; do
    [ -f "$J/learntokenmatching.done" ] && [ -d "$J/model" ] || continue
    python "$WORK/eval_table1.py" "$J/model" "$OUT/fixed_$(tag "$J").json" "$DATASETS"
  done
  python "$WORK/eval_table1.py" --compare "$OUT" | tee "$OUT/tabel_perbandingan.txt" ;;
save)
  for J in "$JOBS"/*/; do
    [ -f "$J/learntokenmatching.done" ] || continue
    t=$(tag "$J"); l=${t%_*}; e=${t#*_}; D="$MODELS/ToMMeR-Llama-3.2-1B_${l}_R64_${e}_fixed"
    mkdir -p "$D/model"; cp "$J/model/model.safetensors" "$D/model/"; cp "$J/params.json" "$D/"
    sha256sum "$J/model/model.safetensors" "$D/model/model.safetensors"; echo "tersimpan: $D"
  done ;;
*) sed -n 2,4p "$SELF" ;;
esac
