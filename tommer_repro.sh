#!/usr/bin/env bash
# Reproduksi inti ToMMeR (Llama-3.2-1B, layer 6, Pile-NER) di server bersama via SSH.
#
# Taruh folder ini di folder kelompok:  ~/<NAMA_KELOMPOK>/tommer_repro/
# Semua file (kode, data, model, cache, token HF, log) hanya ditulis ke ~/<NAMA_KELOMPOK>/,
# tidak ada yang ditulis ke ~ (home bersama) atau folder kelompok lain.
#
# Pakai:  bash tommer_repro.sh <perintah>
#   check     cek GPU, RAM, disk, internet (hanya baca)
#   setup     clone llm2ner @ d71bab4, uv sync, patch kecil, buat config L6
#   hf-login  login Hugging Face, token disimpan di folder kelompok
#   data      siapkan Pile-NER (dari HF) + dataset evaluasi (NER_datasets.zip)
#   train     jalankan training di background (aman walau SSH putus)
#   status    cek progres: proses, RAM/GPU, validasi P/R/F1, selesai/gagal
#   resume    lanjutkan dari checkpoint: bash tommer_repro.sh resume <file.ckpt>
#   eval      evaluasi checkpoint rilis + model sendiri, cetak tabel P/R/F1
#   snapshot  salin patch, config, hasil ke folder ini untuk di-commit ke GitHub
#
# Variabel opsional, dibaca saat train (contoh: EPOCHS=4 bash tommer_repro.sh train):
#   LAYER (6), EPOCHS (8), BATCH (16), ACCUM (1), CLIP (2.0), NER_ZIP (path zip),
#   GDRIVE_ID (id Google Drive zip), EVAL_DATASETS ("CoNLL 2003,CrossNER_AI,ncbi,GENIA_NER")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
GROUP_DIR="$(dirname "$SCRIPT_DIR")"
if [ "$GROUP_DIR" = "$HOME" ] || [ "$GROUP_DIR" = "/" ]; then
  echo "ERROR: taruh folder tommer_repro di dalam folder kelompok, mis. ~/kelompok3/tommer_repro" >&2; exit 1
fi
WORK="$GROUP_DIR/tommer"
REPO="$WORK/llm2ner"
COMMIT=d71bab4a9c517952b0cee015abf68403af64be34
LOG="$WORK/logs"
RESUME_FILE="$WORK/resume_ckpt_path"
LAYER="${LAYER:-6}"
XP_ID="llm2ner_TM_L${LAYER}"
XP_YAML="experiments/trainTokenMatching_L${LAYER}.yaml"
# Default = config asli penulis di repo (trainTokenMatching.yaml)
# Default = Table 4 paper ToMMeR (layer experiment, Llama-3.2-1B): 8 epoch, batch 16, tanpa accumulation, clip 2.0
EPOCHS="${EPOCHS:-8}"; BATCH="${BATCH:-16}"; ACCUM="${ACCUM:-1}"; CLIP="${CLIP:-2.0}"
# Default: 13 benchmark Table 1 paper (EVAL_DATASETS="CoNLL 2003,ncbi" untuk subset)
EVAL_DATASETS="${EVAL_DATASETS:-CoNLL 2003,CrossNER_AI,CrossNER_literature,CrossNER_music,CrossNER_politics,CrossNER_science,MultiNERD,ncbi,FabNER,WikiNeural,GENIA_NER,ACE 2005,Ontonotes}"
RELEASE_CKPT="llm2ner/ToMMeR-Llama-3.2-1B_L6_R64"
ALL_DATASETS=("CoNLL 2003" CrossNER_AI CrossNER_literature CrossNER_music CrossNER_politics
              CrossNER_science MultiNERD ncbi FabNER WikiNeural GENIA_NER "ACE 2005" Ontonotes)

# Semua cache/config diarahkan ke folder kelompok (akun dipakai bersama)
export NER_DATA="$WORK/NER_DATA"
export HF_HOME="$WORK/hf_home"
export XDG_CACHE_HOME="$WORK/.cache" XDG_CONFIG_HOME="$WORK/.config" XDG_DATA_HOME="$WORK/.local/share"
export TORCH_HOME="$WORK/.cache/torch"
export UV_CACHE_DIR="$WORK/uv_cache" UV_TOOL_DIR="$WORK/uv_tools" UV_PYTHON_INSTALL_DIR="$WORK/uv_python"
export PATH="$WORK/bin:$PATH"
mkdir -p "$WORK/bin" "$LOG" "$HF_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"

now() { TZ=Asia/Jakarta date '+%F %T WIB'; }
say() { echo "[$(now)] $*"; }
die() { echo "[$(now)] ERROR: $*" >&2; exit 1; }
venv() { [ -f "$REPO/.venv/bin/activate" ] || die "jalankan dulu: bash $SELF setup"; source "$REPO/.venv/bin/activate"; }
# Label run dari params.json job, mis. L6_E8
job_tag() {
  local l e
  l=$(grep -oE '"layer": *[0-9]+' "$1/params.json" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
  e=$(grep -oE '"epochs": *[0-9]+' "$1/params.json" 2>/dev/null | head -1 | grep -oE '[0-9]+$')
  echo "L${l:-x}_E${e:-x}"
}
latest_job() { ls -dt "$REPO"/xpm-workspace/jobs/llm2ner.tasks.learntokenmatching/*/ 2>/dev/null | head -1 || true; }
# Hanya proses milik folder kelompok ini (user Linux-nya dipakai bersama kelompok lain)
our_procs() { pgrep -u "$USER" -af -- "$REPO/" | grep -v -e pgrep -e "$SELF" || true; }
# Server bersama: pakai 1 GPU saja (GPU=<index> untuk memilih, default yang memorinya paling kosong)
use_one_gpu() {
  local g="${GPU:-}"
  [ -n "$g" ] || g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
                     | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  [ -n "$g" ] || die "GPU tidak terdeteksi"
  export CUDA_VISIBLE_DEVICES="$g"; echo "$g" > "$WORK/gpu_used"
  say "memakai GPU $g saja ($(nvidia-smi -i "$g" --query-gpu=name,memory.used --format=csv,noheader))"
}
training_running() { pgrep -u "$USER" -f -- "run-experiment --workdir $REPO/xpm-workspace" >/dev/null 2>&1; }

cmd_check() {
  set +e
  echo "folder kelompok: $GROUP_DIR"
  echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || echo "GPU tidak terdeteksi"
  echo "== RAM =="; free -h | head -2
  echo "batas RAM cgroup: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo '?')"
  echo "oom_kill sejauh ini: $(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo '?')"
  echo "== Disk =="; df -h "$GROUP_DIR" | tail -1
  echo "== Tools =="; for c in git curl unzip tmux python3 uv; do command -v "$c" >/dev/null && echo "ok      $c" || echo "TIDAK   $c"; done
  echo "== Internet =="
  curl -sI -m 8 https://huggingface.co >/dev/null && echo "ok      huggingface.co" || echo "TIDAK   huggingface.co"
  curl -sI -m 8 https://github.com >/dev/null && echo "ok      github.com" || echo "TIDAK   github.com"
}

patch_code() {
  "$REPO/.venv/bin/python" - "$RESUME_FILE" <<'EOF'
import pathlib, sys
resume_file = sys.argv[1]
def sub(path, old, new):
    p = pathlib.Path(path); s = p.read_text()
    if new in s:
        print("sudah dipatch:", path); return
    assert s.count(old) == 1, f"pola tidak ditemukan di {path}: {old!r}"
    p.write_text(s.replace(old, new)); print("patch      :", path)

# Resume dari checkpoint: path .ckpt dibaca dari file di folder kelompok (dibuat oleh 'resume')
sub("src/llm2ner/models/model.py",
    "trainer.fit(lightning_model, train_loader, val_loader)",
    "trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=(lambda p: p.read_text().strip() "
    f"if p.exists() else None)(__import__('pathlib').Path({resume_file!r})))")
# Teruskan env folder kelompok (HF_HOME, cache) ke job experimaestro
sub("experiments/xp_learnTokenMatching.py",
    '    eval_launcher = find_launcher(cfg.eval_launcher, tags=["slurm"])\n',
    '    eval_launcher = find_launcher(cfg.eval_launcher, tags=["slurm"])\n'
    '    for _launcher in (gpulauncher, eval_launcher):  # env folder kelompok -> job\n'
    '        _launcher.environ.update({k: os.environ[k] for k in ("HF_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",\n'
    '                                  "XDG_DATA_HOME", "TORCH_HOME", "NER_DATA", "CUDA_VISIBLE_DEVICES")\n'
    '                                  if k in os.environ})\n')
# TensorBoard service experimaestro dimatikan (log tfevents tetap ditulis Lightning)
sub("experiments/xp_learnTokenMatching.py",
    "tb_service = helper.xp.add_service(TensorboardService(runpath))",
    "tb_service = None  # TensorBoard service dimatikan")
sub("experiments/xp_learnTokenMatching.py",
    "tb_service.add(task, task.runpath)",
    "pass  # TensorBoard service dimatikan")
EOF
}

make_config() {
  "$REPO/.venv/bin/python" - "$XP_ID" "$LAYER" "$EPOCHS" "$BATCH" "$ACCUM" "$CLIP" "$XP_YAML" <<'EOF'
import re, sys, pathlib
xp_id, layer, epochs, batch, accum, clip, out = sys.argv[1:]
lines = pathlib.Path("experiments/trainTokenMatching.yaml").read_text().splitlines()
lists = {  # evaluasi otomatis setelah training: cukup 4 dataset (inti)
    "flat_eval_datasets": ["CoNLL 2003", "CrossNER_AI", "ncbi"],
    "nested_eval_datasets": ["GENIA_NER"],
    "llm_annotated_datasets": [],  # LLM-as-judge dilewatkan
}
scalars = {"id": xp_id, "epochs": epochs, "batch_size": batch, "accumulation_steps": accum,
           "grad_clip": clip, "eval_threshold": "0.5", "lr": "1e-2", "patience": "5000"}
res, skip_list, block = [], False, None
for l in lines:
    key = re.match(r"^([A-Za-z_]+):", l)
    if skip_list:
        if re.match(r"^\s+-", l): continue
        skip_list = False
    if key:
        block = key.group(1)
        if block in scalars:
            res.append(f"{block}: {scalars[block]}"); continue
        if block in lists:
            items = lists[block]
            res.append(f"{block}:" + ("" if items else " []"))
            res += [f'    - "{d}"' for d in items]
            skip_list = True; continue
    if block == "model_names" and re.match(r"^\s+-", l) and "Llama-3.2-1B" not in l:
        continue  # hanya Llama-3.2-1B
    if block == "layers" and re.match(r"^\s+(from|to):", l):
        l = re.sub(r"(from|to):\s*\d+", lambda m: f"{m.group(1)}: {layer}", l)
    res.append(l)
pathlib.Path(out).write_text("\n".join(res) + "\n")
EOF
  say "config dibuat: $REPO/$XP_YAML"
  grep -nE "^(id|epochs|batch_size|lr|patience|accumulation_steps|grad_clip|eval_threshold):|^\s+(from|to):|Llama|^\S*_datasets:|^\s+- \"" "$REPO/$XP_YAML"
}

cmd_setup() {
  command -v git >/dev/null || die "git tidak ada"
  if ! command -v uv >/dev/null; then
    say "memasang uv ke $WORK/bin (tanpa mengubah ~/.bashrc bersama)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh
  fi
  [ -d "$REPO/.git" ] || git clone https://github.com/VictorMorand/llm2ner.git "$REPO"
  cd "$REPO"
  git checkout -q "$COMMIT"; say "commit: $(git log --oneline -1)"
  say "uv sync (beberapa menit)"; uv sync
  patch_code
  make_config
  say "setup selesai. Berikutnya: bash $SELF hf-login"
}

cmd_hf_login() {
  venv
  echo "Token disimpan di $HF_HOME (folder kelompok)."
  echo "Jawab 'n' kalau ditanya 'Add token as git credential?' (git credential di home bersama)."
  hf auth login
  python -c "from huggingface_hub import auth_check; auth_check('meta-llama/Llama-3.2-1B'); print('akses Llama-3.2-1B OK')"
}

cmd_data() {
  venv; cd "$REPO"; mkdir -p "$NER_DATA"
  if [ ! -f "$NER_DATA/Pile-NER/train.json" ]; then
    say "proses Pile-NER dari HF (Universal-NER/Pile-NER-type)"
    python data/process_pilener.py
    mv data/Pile-NER "$NER_DATA/"
  fi
  ls -la "$NER_DATA/Pile-NER/train.json"
  ZIP="${NER_ZIP:-$WORK/NER_datasets.zip}"
  if [ ! -f "$ZIP" ]; then
    [ -n "${GDRIVE_ID:-}" ] || die "NER_datasets.zip tidak ada di $ZIP. Salin dulu (lihat langkah) atau set GDRIVE_ID"
    say "unduh NER_datasets.zip dari Google Drive"
    uv run --with gdown gdown "$GDRIVE_ID" -O "$ZIP"
  fi
  sz=$(stat -c %s "$ZIP"); [ "$sz" = 381518734 ] || echo "PERINGATAN: ukuran zip $sz, di server lama 381518734"
  say "ekstrak dataset evaluasi"
  unzip -q -o "$ZIP" -d "$WORK/NER_raw"
  for d in "${ALL_DATASETS[@]}"; do
    [ -d "$NER_DATA/$d" ] || cp -r "$WORK/NER_raw/NER/$d" "$NER_DATA/" || echo "TIDAK ADA di zip: $d"
  done
  say "isi NER_DATA:"; ls "$NER_DATA"
}

cmd_monitor() {
  set +e
  while true; do
    if [ -f /sys/fs/cgroup/memory.current ]; then m=$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))
    else m=$(free -m | awk '/^Mem:/{print $3"/"$2}'); fi
    anon=$(awk '/^anon /{printf "%.1f", $2/2^30}' /sys/fs/cgroup/memory.stat 2>/dev/null)
    oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
    g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    echo "$(now) RAM_MB=$m anon_GB=${anon:-?} oom_kill=${oom:-?} GPU=$g" >> "$LOG/mem.log"
    sleep 60
  done
}

cmd_train() {
  venv; cd "$REPO"
  training_running && die "training kelompok ini sudah berjalan (cek: bash $SELF status)"
  [ -f experiments/trainTokenMatching.yaml ] || die "repo belum siap, jalankan: bash $SELF setup"
  make_config  # selalu dibuat ulang, supaya EPOCHS/BATCH/... saat train yang dipakai
  [ -f "$NER_DATA/Pile-NER/train.json" ] || die "data belum siap, jalankan: bash $SELF data"
  python -c "from huggingface_hub import auth_check; auth_check('meta-llama/Llama-3.2-1B')" \
    || die "belum punya akses Llama-3.2-1B. Jalankan: bash $SELF hf-login"
  use_one_gpu
  say "unduh/cek Llama-3.2-1B lalu uji load sampai layer $LAYER"
  python -c "from llm2ner import utils; m = utils.load_llm('meta-llama/Llama-3.2-1B', cut_to_layer=$LAYER); print('OK', round(sum(p.numel() for p in m.parameters())/1e6), 'M param')"
  [ "${RESUME:-0}" = 1 ] || rm -f "$RESUME_FILE"
  pgrep -u "$USER" -f -- "$SELF monitor" >/dev/null || setsid nohup bash "$SELF" monitor >/dev/null 2>&1 < /dev/null &
  say "mulai training $XP_ID (log: $LOG/train_L${LAYER}.log)"
  setsid nohup bash -c "experimaestro run-experiment --workdir '$REPO/xpm-workspace' '$XP_YAML'; rc=\$?; echo \"[\$(TZ=Asia/Jakarta date '+%F %T WIB')] experimaestro selesai, exit \$rc\"" \
    > "$LOG/train_L${LAYER}.log" 2>&1 < /dev/null &
  sleep 5; tail -3 "$LOG/train_L${LAYER}.log"
  say "berjalan di background. SSH boleh ditutup. Cek progres: bash $SELF status"
}

cmd_resume() {
  ck="${1:-}"; [ -f "$ck" ] || die "pakai: bash $SELF resume /path/ke/file.ckpt"
  cp "$ck" "$WORK/resume.ckpt"            # salinan aman, folder job bisa ditimpa
  echo "$WORK/resume.ckpt" > "$RESUME_FILE"
  say "akan resume dari $WORK/resume.ckpt"
  RESUME=1 cmd_train
}

cmd_status() {
  set +e
  [ -f "$RESUME_FILE" ] && echo "(mode resume aktif: $(cat "$RESUME_FILE"))"
  [ -f "$WORK/gpu_used" ] && echo "(GPU kelompok ini: $(cat "$WORK/gpu_used"))"
  echo "== Proses (kelompok ini) =="
  p=$(our_procs); [ -n "$p" ] && echo "$p" | cut -c1-150 || echo "TIDAK ADA proses training"
  echo "== RAM/GPU (tiap menit) =="; tail -3 "$LOG/mem.log" 2>/dev/null
  echo "== Log launcher =="; tail -3 "$LOG/train_L${LAYER}.log" 2>/dev/null | cut -c1-160
  J=$(latest_job); [ -n "$J" ] || { echo "job belum dibuat"; return; }
  echo "== Job: $J"
  if   [ -f "$J/learntokenmatching.done" ];   then echo ">>> TRAINING SELESAI (sukses). Lanjut: bash $SELF eval"
  elif [ -f "$J/learntokenmatching.failed" ]; then echo ">>> TRAINING GAGAL. Lihat: tail -50 $J/learntokenmatching.err"
  fi
  echo "== Akhir stderr job =="
  tail -c 3000 "$J/learntokenmatching.err" 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -4 | cut -c1-160
  echo "== Validasi (P/R/F1) =="
  "$REPO/.venv/bin/python" - "$J" <<'EOF'
import sys, glob, datetime as dt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs = sorted(glob.glob(sys.argv[1] + "/runs/**/events.out.tfevents*", recursive=True))
if not fs: print("belum ada log validasi"); sys.exit()
for f in fs[-2:]:
    ea = EventAccumulator(f); ea.Reload()
    if "val/f1" not in ea.Tags()["scalars"]: print(f.split("/runs/")[1][:60], ": belum ada validasi"); continue
    P, R, F = (ea.Scalars(t) for t in ("val/precision", "val/recall", "val/f1"))
    print("--", f.split("/runs/")[1].rsplit("/", 1)[0])
    for p, r, x in list(zip(P, R, F))[-6:]:
        t = dt.datetime.utcfromtimestamp(x.wall_time) + dt.timedelta(hours=7)
        print(f"{t:%m-%d %H:%M} WIB  step {x.step:>6}  P={p.value:.3f} R={r.value:.3f} F1={x.value:.3f}")
    ago = (dt.datetime.now(dt.timezone.utc).timestamp() - F[-1].wall_time) / 60
    done = glob.glob(sys.argv[1] + "/*.done")
    note = "" if done or ago <= 10 else " (lebih dari 10 menit tanpa validasi baru: wajar saat distilasi/evaluasi, cek GPU & proses di atas)"
    print(f"total validasi {len(F)}; terakhir {ago:.0f} menit lalu{note}")
EOF
  echo "== Checkpoint =="; ls -la "$J"/runs/*/*/version_*/checkpoints/ 2>/dev/null | tail -3
}

cmd_eval() {
  venv; cd "$REPO"; mkdir -p "$WORK/results"
  training_running && die "training masih berjalan; tunggu selesai supaya GPU tidak rebutan"
  use_one_gpu
  say "evaluasi checkpoint RILIS $RELEASE_CKPT di: $EVAL_DATASETS"
  python "$SCRIPT_DIR/eval_table1.py" "$RELEASE_CKPT" "$WORK/results/release_L6.json" "$EVAL_DATASETS"
  # Semua training yang sudah .done (layer & epoch berapa pun), satu file hasil per run
  n=0
  for J in "$REPO"/xpm-workspace/jobs/llm2ner.tasks.learntokenmatching/*/; do
    [ -f "$J/learntokenmatching.done" ] && [ -d "$J/model" ] || continue
    tag=$(job_tag "$J")
    say "evaluasi model SENDIRI ($tag): $J/model"
    python "$SCRIPT_DIR/eval_table1.py" "$J/model" "$WORK/results/own_${tag}.json" "$EVAL_DATASETS"
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || echo "model sendiri belum siap (training belum .done), hanya checkpoint rilis yang dievaluasi"
  python "$SCRIPT_DIR/eval_table1.py" --compare "$WORK/results"
}

cmd_snapshot() {
  # Semua hasil kecil ke repo GitHub: patch, config, hasil eval, log, dan per run training:
  # params.json, hist.json, log TensorBoard, stdout/stderr job, bobot ToMMeR (~1 MB).
  # Data, Llama, best_model.ckpt (1,4 GB, berisi Llama), token, dan cache TIDAK ikut.
  mkdir -p "$SCRIPT_DIR/patches" "$SCRIPT_DIR/configs" "$SCRIPT_DIR/results" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/runs"
  [ -d "$REPO/.git" ] && git -C "$REPO" diff > "$SCRIPT_DIR/patches/llm2ner_d71bab4.diff"
  cp "$REPO"/experiments/trainTokenMatching_L*.yaml "$SCRIPT_DIR/configs/" 2>/dev/null || true
  cp "$WORK"/results/*.json "$SCRIPT_DIR/results/" 2>/dev/null || true
  "$REPO/.venv/bin/python" "$SCRIPT_DIR/eval_table1.py" --compare "$WORK/results" > "$SCRIPT_DIR/results/tabel_perbandingan.txt" 2>/dev/null || true
  cp "$LOG"/*.log "$SCRIPT_DIR/logs/" 2>/dev/null || true
  for J in "$REPO"/xpm-workspace/jobs/llm2ner.tasks.learntokenmatching/*/; do
    [ -f "$J/params.json" ] || continue
    D="$SCRIPT_DIR/runs/$(job_tag "$J")_$(basename "$J" | cut -c1-8)"; mkdir -p "$D"
    [ -f "$J/learntokenmatching.done" ] && touch "$D/SELESAI" || rm -f "$D/SELESAI"
    cp "$J/params.json" "$J"/hist.json "$D/" 2>/dev/null || true
    for f in "$J"/*.out "$J"/*.err; do [ -f "$f" ] && tail -c 2000000 "$f" > "$D/$(basename "$f")"; done
    [ -f "$J/model/model.safetensors" ] && cp "$J/model/model.safetensors" "$D/"
    (cd "$J" && find runs -name 'events.out.tfevents*' -size -50M 2>/dev/null) | while read -r f; do
      mkdir -p "$D/$(dirname "$f")"; cp "$J/$f" "$D/$f"; done
  done
  # Hasil evaluasi otomatis llm2ner (Eval.json) per job eval
  for E in "$REPO"/xpm-workspace/jobs/llm2ner.models.model.nermodel.eval/*/; do
    [ -f "$E/Eval.json" ] || continue
    mkdir -p "$SCRIPT_DIR/runs/auto_eval"; cp "$E/Eval.json" "$SCRIPT_DIR/runs/auto_eval/$(basename "$E" | cut -c1-8).json"
    cp "$E/params.json" "$SCRIPT_DIR/runs/auto_eval/$(basename "$E" | cut -c1-8)_params.json" 2>/dev/null || true
  done
  du -sh "$SCRIPT_DIR/runs" "$SCRIPT_DIR/results" "$SCRIPT_DIR/logs" 2>/dev/null
  say "tersalin ke $SCRIPT_DIR. Lanjut commit & push:"
  echo "  cd $SCRIPT_DIR && git add -A && git commit -m 'update repro' && git push"
  git -C "$SCRIPT_DIR" status --short 2>/dev/null | head -20 || echo "(folder ini belum repo git, lihat langkah backup GitHub)"
}

case "${1:-}" in
  check) cmd_check ;; setup) cmd_setup ;; hf-login) cmd_hf_login ;; data) cmd_data ;;
  train) cmd_train ;; status) cmd_status ;; resume) shift; cmd_resume "${1:-}" ;;
  eval) cmd_eval ;; snapshot) cmd_snapshot ;; monitor) cmd_monitor ;;
  *) sed -n '2,23p' "$SELF" ;;
esac
