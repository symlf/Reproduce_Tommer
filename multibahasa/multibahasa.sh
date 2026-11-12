#!/usr/bin/env bash
# ToMMeR multibahasa, langkah 2 (memperbaiki alat ukur). Model RILIS saja.
# Taruh folder ini di ~/kelompok14/tommer_multibahasa/. tommer/ dan tommer_fixed/ hanya dibaca.
#
#   bash multibahasa.sh pindah            lihat rencana merapikan folder lama (tidak mengubah apa pun)
#   bash multibahasa.sh pindah --yakin    jalankan rencana itu
#   bash multibahasa.sh siapkan           bangun set baru ke data/
#   bash multibahasa.sh tes-lokal         CPU: uji kode tanpa GPU
#   bash multibahasa.sh cek               CPU: jumlah subtoken
#   bash multibahasa.sh start             GPU, background: model rilis di semua set baru
#   bash multibahasa.sh status
#   bash multibahasa.sh juri-sampel       CPU: sampel tebakan untuk LLM juri + lembar anotasi manusia
#   bash multibahasa.sh analisis          CPU: metrik, format, paralel, (precision juri jika jawaban sudah ada), lalu zip
set -euo pipefail
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SELF="$WORK/$(basename "${BASH_SOURCE[0]}")"
GROUP="$(dirname "$WORK")"; OLD="$GROUP/tommer"; FIXED="$GROUP/tommer_fixed"; NL="$GROUP/tommer_nonlatin"; LJ="$GROUP/tommer_lanjut"
[ "$(basename "$WORK")" = tommer_multibahasa ] || { echo "ERROR: taruh folder ini sebagai $GROUP/tommer_multibahasa/"; exit 1; }
export NER_DATA="$OLD/NER_DATA" HF_HOME="$OLD/hf_home" HF_HUB_OFFLINE=1 TQDM_DISABLE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$WORK/.cache" XDG_CONFIG_HOME="$WORK/.config"
U="${USER:-$(id -un)}"; LOG="$WORK/logs"; mkdir -p "$LOG" "$WORK/hasil/pred" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
[ ! -d "$FIXED/.config" ] || cp -rn "$FIXED/.config/." "$XDG_CONFIG_HOME/" 2>/dev/null || true
die() { echo "ERROR: $*" >&2; exit 1; }
PY="$FIXED/llm2ner/.venv/bin/python"; [ -x "$PY" ] || die "venv tidak ada: $PY"
M() { "$PY" "$WORK/multibahasa.py" "$@" 2>&1 | grep -v -i "warning"; }
running() { pgrep -u "$U" -f -- "$WORK/multibahasa.py infer" >/dev/null 2>&1; }
ada() { local o=""; for d in "$@"; do [ -f "$WORK/data/$d/test.json" ] && o="$o${o:+,}$d"; done; echo "$o"; }
SET_BARU="KLUE_ko Stockmark_ja MSRA_zh UNER_GSDSIMP_zh NER_UI_id NER_UGM_id ANERcorp_ar UNER_PUD_en UNER_PUD_sv UNER_PUD_ru UNER_PUD_zh WikiANN_ko_alami WikiANN_ko_karakter WikiANN_en_karakter WikiANN_ar_karakter WikiANN_zh_spasi_tanpa_pagar"

pindah() {
  local Y="${1:-}" A="$WORK/arsip/nonlatin_v1"
  run() { if [ "$Y" = --yakin ]; then "$@"; else echo "  [rencana] $*"; fi; }
  mv_ke() { local dst="$1"; shift; for f in "$@"; do [ -e "$f" ] || continue
      if [ -e "$dst/$(basename "$f")" ]; then echo "  [lewati, sudah ada] $dst/$(basename "$f")"; else run mkdir -p "$dst"; run mv "$f" "$dst/"; fi; done; }
  echo "== 1. tommer_nonlatin: yang masih dipakai dipindah, yang tidak dipakai dihapus =="
  if [ -d "$NL" ]; then
    shopt -s nullglob
    # tommer_nonlatin/data/ sudah dihapus saat pembersihan uji fixed_baru: ekstrak ulang dari zip aslinya (isi sama) langsung ke folder ini.
    if [ ! -d "$NL/data" ] && [ -f "$NL/nonlatin_ner_json.zip" ]; then
      if [ -d "$WORK/data/WikiANN_en_spasi" ]; then echo "  [lewati, sudah ada] $WORK/data/WikiANN_*"
      else run mkdir -p "$WORK/data" "$WORK/.tmp_wikiann"; run unzip -q "$NL/nonlatin_ner_json.zip" -d "$WORK/.tmp_wikiann"
           run cp -rn "$WORK/.tmp_wikiann/nonlatin_ner_json/." "$WORK/data/"; run rm -rf "$WORK/.tmp_wikiann"; fi
    fi
    for f in "$NL"/data/WikiANN_*; do case "$f" in *_titik) run rm -rf "$f";; *) mv_ke "$WORK/data" "$f";; esac; done
    mv_ke "$A" "$NL/data/metadata_entitas_nonlatin.csv"
    for f in "$NL"/hasil/pred/*.jsonl.gz; do case "$(basename "$f")" in release__*_titik*|fixed__*|lama__*) mv_ke "$A/pred_lain" "$f";; release__*) mv_ke "$WORK/hasil/pred" "$f";; *) mv_ke "$A/pred_lain" "$f";; esac; done
    mv_ke "$A" "$NL/hasil/analisis" "$NL/hasil/LAPORAN_HASIL.md" "$NL/hasil/cek_entitas_dinilai.json" "$NL/logs" "$NL/hasil_nonlatin.zip" "$NL/hasil/validasi"
    # skrip lama: bukti cara hasil di arsip dibuat, tidak dijalankan lagi
    mv_ke "$A/skrip" "$NL"/*.py "$NL"/*.sh "$NL"/*.md
    for f in "$NL"/*.zip "$NL"/.cache "$NL"/.config "$NL"/__pycache__; do [ -e "$f" ] && run rm -rf "$f"; done
    shopt -u nullglob
    if [ "$Y" = --yakin ]; then
      find "$NL" -depth -type d -empty -delete 2>/dev/null || true
      if [ -d "$NL" ]; then echo "  SISA di $NL (tidak dihapus, periksa manual):"; find "$NL" -maxdepth 3 | head -30; else echo "  $NL sudah kosong dan dihapus"; fi
    fi
  else echo "  tidak ada"; fi
  echo "== 2. tommer_lanjut (paket lama yang belum dijalankan) =="
  if [ -d "$LJ" ]; then
    for f in "$LJ"/hasil/pred/release__*.jsonl.gz; do [ -e "$f" ] && mv_ke "$WORK/hasil/pred" "$f"; done
    run rm -rf "$LJ"        # rm tidak mengikuti symlink hf_home, cache lama aman
  else echo "  tidak ada"; fi
  echo "== 3. tommer_fixed/fixed_baru.sh (penggantian model fixed, belum dipakai) =="
  if [ -f "$FIXED/fixed_baru.sh" ]; then
    if grep -q TOMMER_RUN "$FIXED"/llm2ner/*/xp_learnTokenMatching.py "$FIXED"/xp_learnTokenMatching.py 2>/dev/null || [ -d "$FIXED/results/kandidat" ]; then
      echo "  fixed_baru.sh SUDAH dipakai (patch TOMMER_RUN atau results/kandidat ada): tidak disentuh, tanyakan Syml"
    else run rm -f "$FIXED/fixed_baru.sh"; fi
  else echo "  tidak ada"; fi
  [ "$Y" = --yakin ] || echo "Belum ada yang diubah. Jalankan: bash $SELF pindah --yakin"
}

case "${1:-}" in
pindah) running && die "inferensi sedang berjalan"; pindah "${2:-}" ;;
siapkan) M siapkan | tee "$LOG/siapkan.log" ;;
tes-lokal) M tes-lokal --out "$WORK/hasil" | grep -E "BANGUN|PARITAS|TES_LOKAL|Error|Traceback" | tee "$LOG/tes_lokal.log" ;;
cek) M cek --sets "$(ada $SET_BARU)" | tee "$LOG/cek.log" ;;
start)
  running && die "masih berjalan: bash $SELF status"
  [ -f "$WORK/hasil/pred/release__CoNLL_2003.jsonl.gz" ] || echo "PERINGATAN: prediksi CoNLL lama tidak ada; gerbang akan menghitung ulang CoNLL (lebih lama)."
  D="${HANYA_DATA:-$(ada $SET_BARU)}"; [ -n "$D" ] || die "data/ kosong: jalankan siapkan"
  pgrep -u "$U" -f -- "eval_table1|run-experiment|eval_nonlatin" >/dev/null 2>&1 && echo "PERINGATAN: ada proses GPU lain milikmu."
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || die "GPU tidak terdeteksi"
  echo "GPU $CUDA_VISIBLE_DEVICES | set: $D"
  setsid nohup "$PY" "$WORK/multibahasa.py" infer --models release --sets "$D" > "$LOG/infer.log" 2>&1 < /dev/null &
  sleep 60; grep -E "GERBANG|^model|^infer|Error" "$LOG/infer.log" || tail -5 "$LOG/infer.log"; echo "Cek: bash $SELF status" ;;
status)
  running && echo "STATUS: berjalan" || echo "STATUS: tidak berjalan"
  [ -f "$LOG/infer.log" ] && { grep -E "GERBANG|^model|^infer|lewati|Error|Traceback" "$LOG/infer.log" | tail -30; tail -1 "$LOG/infer.log"; }
  echo "prediksi rilis: $(ls "$WORK/hasil/pred" | grep -c '^release__' || true) file" ;;
juri-sampel)
  M juri-sampel | tee "$LOG/juri_sampel.log"
  M juri-manusia | tee -a "$LOG/juri_sampel.log"
  echo "Kirim ke laptop: hasil/juri/sampel.jsonl (untuk juri) dan hasil/juri/manusia/*.csv (untuk penilai)" ;;
analisis)
  running && die "inferensi masih berjalan"
  M metrik | tee "$LOG/metrik.log"; M format | tee "$LOG/format.log"; M paralel | tee "$LOG/paralel.log"
  if [ -f "$WORK/hasil/juri/jawaban_juri.jsonl" ]; then M juri-hitung | tee "$LOG/juri_hitung.log"; else echo "jawaban juri belum ada: precision juri dilewati"; fi
  cd "$WORK"; rm -f hasil_langkah2.zip; zip -qr hasil_langkah2.zip hasil/analisis hasil/cek_subtoken.json hasil/LAPORAN_LANGKAH2.md logs data/*/meta.txt hasil/juri/sampel_ringkasan.json 2>/dev/null || true
  echo "Kirim: $WORK/hasil_langkah2.zip" ;;
*) sed -n 2,14p "$SELF" ;;
esac
