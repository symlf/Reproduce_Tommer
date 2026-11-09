#!/usr/bin/env bash
# Uji zero-shot ToMMeR pada WikiANN non-Latin. Semua file baru hanya di folder skrip ini (tommer_nonlatin).
# Folder lama (tommer, tommer_fixed) hanya dibaca: venv, data CoNLL, cache Llama, dan model fixed.
#
# Pakai (urut):
#   bash run_nonlatin.sh siapkan   ekstrak data uji (+ WikiANN-en jika parquet_en/ ada)
#   bash run_nonlatin.sh cek       CPU saja, ~5 menit: cek jumlah entitas yang dinilai vs paper
#   bash run_nonlatin.sh start     GPU, background: gerbang CoNLL lalu inferensi semua dataset
#   bash run_nonlatin.sh status    lihat progres
#   bash run_nonlatin.sh analyze   hitung semua tabel, lalu kemas hasil ke hasil_nonlatin.zip
set -euo pipefail
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SELF="$WORK/$(basename "${BASH_SOURCE[0]}")"
GROUP="$(dirname "$WORK")"; OLD="$GROUP/tommer"; FIXED="$GROUP/tommer_fixed"
[ "$(basename "$WORK")" = tommer_nonlatin ] || { echo "ERROR: taruh skrip ini di $GROUP/tommer_nonlatin/"; exit 1; }
DATA="$WORK/data"; OUT="$WORK/hasil"; LOG="$WORK/logs"
MODEL_FIXED="$FIXED/models/ToMMeR-Llama-3.2-1B_L6_R64_E8_fixed/model"
DATASETS="WikiANN_zh_spasi,WikiANN_zh_alami,WikiANN_ja_spasi,WikiANN_ja_alami,WikiANN_ar_alami"
export NER_DATA="$OLD/NER_DATA" HF_HOME="$OLD/hf_home" HF_HUB_OFFLINE=1 TQDM_DISABLE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$WORK/.cache" XDG_CONFIG_HOME="$WORK/.config" XDG_DATA_HOME="$WORK/.local/share" TORCH_HOME="$WORK/.cache/torch"
U="${USER:-$(id -un)}"
die() { echo "ERROR: $*" >&2; exit 1; }
PY="$FIXED/llm2ner/.venv/bin/python"; [ -x "$PY" ] || die "venv tommer_fixed tidak ada: $PY"
running() { pgrep -u "$U" -f -- "$WORK/eval_nonlatin.py" >/dev/null 2>&1; }
mkdir -p "$OUT" "$LOG" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
[ ! -d "$FIXED/.config" ] || cp -rn "$FIXED/.config/." "$XDG_CONFIG_HOME/" 2>/dev/null || true
ds() { local d="$DATASETS"; [ -d "$DATA/WikiANN_id_spasi_huruf_kecil" ] && d="WikiANN_id_spasi_huruf_kecil,$d"; [ -d "$DATA/WikiANN_id_spasi" ] && d="WikiANN_id_spasi,$d"; [ -d "$DATA/WikiANN_en_spasi" ] && d="WikiANN_en_spasi,$d"; echo "$d"; }

case "${1:-}" in
siapkan)
  [ -f "$WORK/nonlatin_ner_json.zip" ] || die "taruh nonlatin_ner_json.zip di $WORK"
  mkdir -p "$DATA"; T=$(mktemp -d "$WORK/tmp.XXXX"); unzip -q "$WORK/nonlatin_ner_json.zip" -d "$T"
  cp -rn "$T"/nonlatin_ner_json/. "$DATA/" 2>/dev/null; rm -rf "$T"
  if [ -d "$WORK/parquet_en" ]; then "$PY" "$WORK/eval_nonlatin.py" siapkan --parquet "$WORK/parquet_en" --kode en --data-root "$DATA"; else echo "parquet_en/ tidak ada: WikiANN-en (kontrol) dilewati"; fi
  [ -d "$NER_DATA/CoNLL 2003" ] || die "data gerbang tidak ada: $NER_DATA/CoNLL 2003"
  [ -f "$MODEL_FIXED/model.safetensors" ] || die "model fixed tidak ada: $MODEL_FIXED"
  ls "$DATA"; echo "Siap. Lanjut: bash $SELF cek" ;;
cek)
  "$PY" "$WORK/eval_nonlatin.py" cek --data-root "$DATA" --data "$(ds)" --out "$OUT" 2>&1 | grep -v Warning | tee "$LOG/cek.log" ;;
start)
  running && die "masih berjalan, cek: bash $SELF status"
  pgrep -u "$U" -f -- "run-experiment --workdir $GROUP/tommer" >/dev/null 2>&1 && die "ada training kelompok yang berjalan"
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || die "GPU tidak terdeteksi"; echo "GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
  setsid nohup "$PY" "$WORK/eval_nonlatin.py" infer --data-root "$DATA" --out "$OUT" --data "$(ds)" --models "release,$MODEL_FIXED" \
    > "$LOG/infer.log" 2>&1 < /dev/null &
  sleep 60; grep -E "GERBANG|model |ERROR|Error" "$LOG/infer.log" || tail -5 "$LOG/infer.log"
  echo "Berjalan di background. Cek: bash $SELF status" ;;
status)
  if running; then echo "STATUS: berjalan"; else echo "STATUS: tidak berjalan"; fi
  grep -E "GERBANG|^model|^infer|lewati|Error|error" "$LOG/infer.log" 2>/dev/null | tail -20
  tail -2 "$LOG/infer.log" 2>/dev/null; echo "file prediksi selesai: $(ls "$OUT/pred" 2>/dev/null | grep -c 'jsonl.gz$' || true)" ;;
analyze)
  running && die "inferensi masih berjalan"
  "$PY" "$WORK/eval_nonlatin.py" analyze --data-root "$DATA" --out "$OUT" 2>&1 | grep -v Warning | tee "$LOG/analyze.log"
  cd "$WORK"; rm -f hasil_nonlatin.zip
  zip -qr hasil_nonlatin.zip hasil/analisis hasil/cek_entitas_dinilai.json logs
  echo "Kirim file ini: $WORK/hasil_nonlatin.zip ($(du -h hasil_nonlatin.zip | cut -f1))" ;;
*) sed -n 2,10p "$SELF" ;;
esac
