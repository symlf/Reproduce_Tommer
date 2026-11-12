"""ToMMeR multibahasa, LANGKAH 2: memperbaiki alat ukur sebelum mencari penyebab dan memperbaiki model.

Satu file untuk semua tahap langkah 2 (menggantikan eval_nonlatin.py, test_lokal.py, buat_data_validasi.py,
analisis_validasi.py, dan paket tommer_lanjut). Skrip uji non-Latin pertama diarsipkan di arsip/, tidak dipakai lagi. Model yang diuji: RILIS llm2ner/ToMMeR-Llama-3.2-1B_L6_R64.

Tahap (python multibahasa.py <tahap> --help):
  siapkan        bangun set baru dari raw/ dan parquet/ (data manusia, data paralel UNER-PUD, WikiANN ko, ANERcorp, kontrol format)
  cek            CPU: jumlah subtoken, kalimat melebihi konteks
  tes-lokal      CPU tanpa GPU/HF: Llama mini + forward palsu; cek loader, inferensi, dan paritas metrik dengan hitungan resmi
  infer          GPU: jalankan model, simpan semua span > 0.1, greedy, probabilitas emas, offset, teks
  metrik         CPU: metrik persis + longgar, varian PLO, kalimat tanpa entitas
  format         CPU: uji ulang poin 2 (huruf dipecah spasi vs teks normal), transisi per entitas, lift
  paralel        CPU: kalimat yang SAMA dalam en/sv/ru/zh (UNER-PUD), selisih berpasangan terhadap en
  juri-sampel    CPU: ambil sampel tebakan untuk dinilai LLM juri (cara paper bagian 4.2)
  juri           di mana saja yang punya internet: kirim sampel ke API kompatibel OpenAI, simpan jawaban (bisa dilanjutkan)
  juri-manusia   buat lembar anotasi manusia buta (2 penilai) dari sampel yang sudah dinilai juri
  juri-hitung    precision "sebenarnya": koreksi PPI (Angelopoulos dkk. 2023) memakai label manusia, kappa Cohen
"""
import argparse, gzip, json, os, random, re, sys, time, unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BATCH = 30                     # sama dengan eval_table1.py dan tasks.py milik penulis
SIMPAN_MIN = 0.1               # probabilitas minimum yang disimpan, untuk kurva threshold
THR = 0.5                      # threshold resmi paper
GERBANG = "CoNLL 2003"
RILIS = "llm2ner/ToMMeR-Llama-3.2-1B_L6_R64"
# Hasil resmi (tp, fp, total) dari eval_table1.py. Inferensi tanpa training harus sama persis.
ACUAN = {"release": {"threshold": (32968, 65294, 34766), "greedy": (30047, 37180, 34766)},
         "fixed": {"threshold": (32772, 79017, 34766), "greedy": (29840, 40713, 34766)},
         "lama": {"threshold": (33659, 109418, 34766), "greedy": (30647, 49086, 34766)}}

# Tipe yang dianggap PER/ORG/LOC per set (setara skema WikiANN). Tipe lain diabaikan pada varian PLO.
PLO = {"KLUE_ko": ["PS", "OG", "LC"], "Stockmark_ja": ["人名", "法人名", "政治的組織名", "その他の組織名", "地名"],
       "MSRA_zh": ["PER", "ORG", "LOC"],
       "NER_UI_id": ["PERSON", "ORGANIZATION", "LOCATION"], "NER_UGM_id": ["PERSON", "ORGANIZATION", "LOCATION"],
       "ANERcorp_ar": ["PERS", "ORG", "LOC"]}
PUD = ("en", "sv", "ru", "zh")       # bahasa UNER-PUD yang kalimatnya bisa dipasangkan lewat sent_id UD (998 kalimat)


# =====================================================================================================================
# Utilitas umum
# =====================================================================================================================
def baca(folder):
    """Baca semua split *.json di satu folder (urutan sama dengan load_dataset_splits). meta.txt sengaja bukan .json."""
    return [x for f in sorted(Path(folder).glob("*.json")) for x in json.loads(f.read_text(encoding="utf8"))]


def baca_meta(folder):
    p = Path(folder) / "meta.txt"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else {}


def baca_pred(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(l) for l in f]


def daftar_pred(dirs, model="release"):
    files, lihat = [], set()
    for d in dirs:
        for f in sorted(Path(d).glob("*.jsonl.gz")):
            m = f.name[:-9].split("__")[0]
            if f.name in lihat or (model and m != model): continue
            lihat.add(f.name); files.append(f)
    return files


def kandidat_sah(L, window=25):
    """Jumlah span sah dalam satu kalimat: awal >= 1 (bukan BOS), akhir < L, panjang <= window + 1."""
    return sum(min(window + 1, L - s) for s in range(1, L))


def sah(p, L):
    s, e = p
    return 1 <= s <= e < L


def tebakan(x, decoding):
    return [(c[0], c[1]) for c in x["cand"] if c[3]] if decoding == "threshold" else [tuple(g) for g in x["greedy"]]


def prf(tp, fp, tot):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / tot if tot else 0.0
    return 100 * P, 100 * R, 100 * (2 * P * R / (P + R) if P + R else 0.0)


# =====================================================================================================================
# siapkan: semua set dibangun dari sumber aslinya, teks ditulis seperti aslinya (protokol input alami)
# =====================================================================================================================
def tulis_set(out_root, nama, rows, meta):
    """rows: list (kalimat, [(awal, akhir, tipe)]) posisi karakter, akhir eksklusif. Spasi di tepi entitas dibuang."""
    out = []
    for i, (kal, ents) in enumerate(rows):
        e2 = []
        for a, b, ty in ents:
            while a < b and kal[a].isspace(): a += 1
            while b > a and kal[b - 1].isspace(): b -= 1
            if a < b: e2.append(dict(pos=[a, b], type=ty, text=kal[a:b]))
        out.append(dict(data_id=i, sentence=kal, entities=e2))
    assert all(x["sentence"][e["pos"][0]:e["pos"][1]] == e["text"] for x in out for e in x["entities"])
    p = Path(out_root) / nama; p.mkdir(parents=True, exist_ok=True)
    (p / "test.json").write_text(json.dumps(out, ensure_ascii=False))
    meta = dict(meta, kalimat=len(out), kalimat_tanpa_entitas=sum(not x["entities"] for x in out), entitas=sum(len(x["entities"]) for x in out))
    (p / "meta.txt").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{nama:30s} {meta['kalimat']:6d} kalimat ({meta['kalimat_tanpa_entitas']:5d} tanpa entitas) {meta['entitas']:6d} entitas | {out[min(1, len(out) - 1)]['sentence'][:50]}", flush=True)
    return out


def bio_ke_span(tags):
    """BIO (juga I- tanpa B-) -> [(awal, akhir, tipe)] dalam indeks unit."""
    h, a = [], None
    for i, t in enumerate(tags):
        if t == "O" or not t:
            if a: h.append(tuple(a)); a = None
        elif t.startswith("B-") or not a or a[2] != t[2:]:
            if a: h.append(tuple(a))
            a = [i, i + 1, t[2:]]
        else:
            a[1] = i + 1
    if a: h.append(tuple(a))
    return h


def gabung(units, sep):
    awal, c = [], 0
    for u in units: awal.append(c); c += len(u) + len(sep)
    return sep.join(units), awal


def dari_unit(units, tags, sep):
    kal, awal = gabung(units, sep)
    return kal, [(awal[a], awal[b - 1] + len(units[b - 1]), ty) for a, b, ty in bio_ke_span(tags)]


def baca_conll_karakter(path):
    """Satu karakter per baris 'char<TAB>tag' (KLUE, MSRA). Spasi teks juga satu baris di KLUE."""
    sents, cur = [], []
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("##"): continue
        if "\t" not in line and not line.strip():
            if cur: sents.append(cur); cur = []
            continue
        ch, tag = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
        cur.append((ch, tag))
    if cur: sents.append(cur)
    return [dari_unit([c for c, _ in s], [t for _, t in s], "") for s in sents]


def baca_uner(path):
    """File .iob2 Universal NER: '# text = ...' memberi teks asli; token dicocokkan berurutan ke teks itu."""
    hasil, gagal, blok, teks, sid = [], 0, [], None, None
    def selesai():
        nonlocal gagal
        if teks is None or not blok: return
        pos, c = [], 0
        for w, _ in blok:
            k = teks.find(w, c)
            if k < 0: gagal += 1; return
            pos.append((k, k + len(w))); c = k + len(w)
        ents = [(pos[a][0], pos[b - 1][1], ty) for a, b, ty in bio_ke_span([t for _, t in blok])]
        hasil.append((teks, ents, sid))
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("# text = "): teks = line[9:].strip()
        elif line.startswith("# sent_id"): sid = line.split("=", 1)[1].strip()
        elif line.startswith("#"): continue
        elif not line.strip():
            selesai(); blok, teks = [], None
        else:
            k = line.split("\t"); blok.append((k[1], k[2]))
    selesai()
    return hasil, gagal


def wikiann_parquet(folder):
    import pyarrow.parquet as pq
    out = []
    for s in ("train", "validation", "test"):
        t = pq.read_table(Path(folder) / f"{s}.parquet")
        nama = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["ner_tags"]["feature"]["names"]
        out += [(x["tokens"], [nama[i] for i in x["ner_tags"]]) for x in t.select(["tokens", "ner_tags"]).to_pylist()]
    return out


def ke_karakter(tok, tag):
    """Kontrol format: kata -> huruf, '#' di antara kata (meniru persis WikiANN zh versi spasi)."""
    ct, cg = [], []
    ent_of = [None] * len(tok)
    for a, b, ty in bio_ke_span(tag):
        for i in range(a, b): ent_of[i] = (ty, a, b)
    for i, w in enumerate(tok):
        if i > 0:
            same = ent_of[i] is not None and ent_of[i - 1] == ent_of[i]
            ct.append("#"); cg.append(("I-" + ent_of[i][0]) if same else "O")
        for j, c in enumerate(w):
            e = ent_of[i]
            cg.append("O" if e is None else (("B-" if (i == e[1] and j == 0) else "I-") + e[0]))
            ct.append(c)
    return ct, cg


def siapkan(a):
    raw, parq, out = Path(a.paket) / "raw", Path(a.paket) / "parquet", Path(a.data)
    out.mkdir(parents=True, exist_ok=True)
    m = lambda **k: dict(ikut_kalimat_kosong=True, **k)
    # --- data berlabel manusia, satu atau lebih per bahasa ---
    tulis_set(out, "KLUE_ko", baca_conll_karakter(raw / "klue-ner-v1.1_dev.tsv"),
              m(bahasa="ko", anotasi="manusia", sumber="KLUE-NER v1.1 dev (test tidak dirilis)", lisensi="CC BY-SA 4.0", tipe_PLO=PLO["KLUE_ko"]))
    d = json.loads((raw / "stockmark_ner.json").read_text(encoding="utf8"))
    tulis_set(out, "Stockmark_ja", [(x["text"], [(e["span"][0], e["span"][1], e["type"]) for e in x["entities"]]) for x in d],
              m(bahasa="ja", anotasi="manusia", sumber="stockmarkteam/ner-wikipedia-dataset", lisensi="CC BY-SA 3.0", tipe_PLO=PLO["Stockmark_ja"]))
    tulis_set(out, "MSRA_zh", baca_conll_karakter(raw / "msra_test_bio.txt"),
              m(bahasa="zh", anotasi="manusia", sumber="MSRA NER test (SIGHAN 2006)", lisensi="riset, jangan dipublikasikan ulang", tipe_PLO=PLO["MSRA_zh"]))
    r, g = baca_uner(raw / "uner" / "zh_gsdsimp-ud-test.iob2")
    tulis_set(out, "UNER_GSDSIMP_zh", [(t, e) for t, e, _ in r], m(bahasa="zh", anotasi="manusia", sumber="Universal NER zh_gsdsimp test (Hanzi sederhana)",
              lisensi="CC BY-SA 4.0", tipe_PLO=["PER", "ORG", "LOC"], token_gagal_dicocokkan=g))
    for src, dst in (("NER_UI", "NER_UI_id"), ("NER_UGM", "NER_UGM_id")):
        d = json.loads((raw / "indo" / src / "test.json").read_text(encoding="utf8"))
        tulis_set(out, dst, [(x["sentence"], [(e["pos"][0], e["pos"][1], e["type"]) for e in x["entities"]]) for x in d],
                  m(bahasa="id", anotasi="manusia", sumber=f"{src} test (indo_ner_json_v4)", tipe_PLO=PLO[dst]))
    f = sorted((parq / "anercorp").glob("test*.parquet"))
    if f:
        import pyarrow.parquet as pq
        sents, cur = [], []
        for r in pq.read_table(f[0]).to_pylist():
            w, tag = (r.get("word") or "").strip(), (r.get("tag") or "O").strip()
            if not w:
                if cur: sents.append(cur); cur = []
                continue
            cur.append((w, tag))
            if set(w) <= {"."}: sents.append(cur); cur = []
        if cur: sents.append(cur)
        tulis_set(out, "ANERcorp_ar", [dari_unit([w for w, _ in s], [t for _, t in s], " ") for s in sents],
                  m(bahasa="ar", anotasi="manusia", sumber="asas-ai/ANERCorp test (split CAMeL)", lisensi="CC BY-SA 4.0", tipe_PLO=PLO["ANERcorp_ar"]))
    else:
        print("parquet/anercorp/test*.parquet tidak ada: ANERcorp_ar dilewati")
    # --- data paralel: kalimat yang sama dalam 4 bahasa (UNER-PUD), dipasangkan lewat sent_id UD ---
    peta = json.loads((raw / "uner" / "pud_sent_id.json").read_text())
    for k in PUD:
        r, g = baca_uner(raw / "uner" / f"{k}_pud-ud-test.iob2")
        assert g == 0 and len(r) == len(peta[k]) == 1000, (k, g, len(r))
        tulis_set(out, f"UNER_PUD_{k}", [(t, e) for t, e, _ in r],
                  m(bahasa=k, anotasi="manusia", sumber=f"Universal NER {k}_pud test (Parallel UD)", lisensi="CC BY-SA 4.0",
                    tipe_PLO=["PER", "ORG", "LOC"], sent_id_ud=peta[k], catatan="zh = Hanzi tradisional" if k == "zh" else ""))
    # --- WikiANN Korea (perak) + kontrol format ---
    if (parq / "wikiann_ko" / "test.parquet").exists():
        rows = wikiann_parquet(parq / "wikiann_ko")
        mk = dict(bahasa="ko", anotasi="perak", sumber="unimelb-nlp/wikiann ko, 3 split digabung", tipe_PLO=["PER", "ORG", "LOC"], ikut_kalimat_kosong=False)
        tulis_set(out, "WikiANN_ko_alami", [dari_unit(t, g, " ") for t, g in rows], mk)
        tulis_set(out, "WikiANN_ko_karakter", [dari_unit(*ke_karakter(t, g), " ") for t, g in rows], dict(mk, catatan="kontrol format"))
    else:
        print("parquet/wikiann_ko/ tidak ada: WikiANN_ko dilewati")
    # --- kontrol format en/ar/zh (sudah dibangun dari parquet WikiANN dengan ke_karakter yang sama) ---
    for n in ("WikiANN_en_karakter", "WikiANN_ar_karakter", "WikiANN_zh_spasi_tanpa_pagar"):
        if (out / n / "test.json").exists(): print(f"{n:30s} sudah ada"); continue
        p = out / n; p.mkdir(parents=True, exist_ok=True)
        (p / "test.json").write_bytes((Path(a.paket) / "data_validasi" / n / "test.json").read_bytes())
        print(f"{n:30s} disalin")
    for n in ("WikiANN_en_karakter", "WikiANN_ar_karakter", "WikiANN_zh_spasi_tanpa_pagar"):
        (out / n / "meta.txt").write_text(json.dumps(dict(bahasa=n.split("_")[1], anotasi="perak", tipe_PLO=["PER", "ORG", "LOC"],
                                                          ikut_kalimat_kosong=False, catatan="kontrol format"), indent=1))


# =====================================================================================================================
# cek
# =====================================================================================================================
def urutan_resmi(panjang, batch=BATCH):
    """Tiru BatchedDataset: urutkan menurut panjang subtoken, potong per batch, buang batch terakhir yang tidak penuh."""
    idx = sorted(range(len(panjang)), key=lambda i: panjang[i])
    penuh = len(idx) // batch
    return [idx[i * batch:(i + 1) * batch] for i in range(penuh)], idx[penuh * batch:]


def cek(a):
    from transformers import AutoTokenizer
    from llm2ner.data import JSONDataSet
    from llm2ner import utils
    tok = AutoTokenizer.from_pretrained(a.llm)
    c_len = utils.get_model_max_length(a.llm)
    hasil = {}
    for name in a.sets.split(","):
        if not (Path(a.data) / name).exists(): continue
        ds = JSONDataSet(baca(Path(a.data) / name), model=None, tokenizer=tok, mode="last", max_ent_length=400, max_length=10 ** 9)
        pjg = [len(x["str_tokens"]) for x in ds.data]
        r = dict(kalimat_berentitas=len(pjg), entitas_terpetakan=sum(len(x["entities"]) for x in ds.data), subtoken_maks=max(pjg),
                 subtoken_median=float(np.median(pjg)), batas_konteks=c_len, kalimat_melebihi_konteks=sum(p > c_len for p in pjg))
        hasil[name] = r; print(name, json.dumps(r), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "cek_subtoken.json").write_text(json.dumps(hasil, indent=1))


# =====================================================================================================================
# infer
# =====================================================================================================================
def load_tommer(spec):
    """'release' = checkpoint penulis di HF. Selain itu folder model hasil training (cara muat sama dengan eval_table1.py)."""
    import importlib
    from xpm_torch.huggingface import TorchHFHub
    if spec == "release":
        t = TorchHFHub.from_pretrained(RILIS); t.initialize(); return t
    params = json.loads((Path(spec).parent / "params.json").read_text())
    obj = next(o for o in params["objects"] if ".models." in o["typename"] and "layer" in o["fields"])
    mod, cls = obj["typename"].rsplit(".", 1)
    t = getattr(importlib.import_module(mod), cls).C(**obj["fields"]).instance()
    t.initialize(); t.load_model(Path(spec))
    return t


def bangun_dataset(records, llm, c_length, ikut_kosong=False):
    """Loader resmi (JSONDataSet). Loader resmi MEMBUANG kalimat tanpa entitas (data.py); ikut_kosong memasukkannya kembali."""
    from llm2ner.data import JSONDataSet
    for x in records:
        for k, e in enumerate(x["entities"]):
            e["idx"] = k
    ds = JSONDataSet(records, model=llm, mode="last", max_ent_length=400, max_length=c_length)
    tambah = 0
    if ikut_kosong:
        ada = {x["data_id"] for x in ds.data}
        for x in records:
            if not x["entities"] and x["data_id"] not in ada:
                it = ds.extract_from_item(x)
                if len(it["str_tokens"]) <= c_length: ds.data.append(it); tambah += 1
    return ds, tambah


def infer_dataset(tommer, llm, ds, out_path):
    import torch
    from llm2ner import decoders
    tok = llm.tokenizer
    pjg = [len(x["str_tokens"]) for x in ds.data]
    penuh, sisa = urutan_resmi(pjg)
    batches = [(b, True) for b in penuh] + ([(sisa, False)] if sisa else [])
    tmp = Path(str(out_path) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf8") as f, torch.no_grad():
        for nb, (ids, resmi) in enumerate(batches):
            items = [ds.data[i] for i in ids]
            inp = tok([x["text"] for x in items], padding=True, padding_side="right", return_tensors="pt",
                      return_offsets_mapping=True, truncation=True)          # sama dengan evaluate_batch
            tokens = inp["input_ids"].to(llm.device); off = inp.pop("offset_mapping")
            mask = inp["attention_mask"].to(llm.device)
            probs = tommer.forward(tokens, llm, attn_mask=mask).detach()     # (batch, akhir, awal), sama dengan infer_entities
            b_i, last, first = torch.where(probs > SIMPAN_MIN)
            pv = probs[b_i, last, first]
            cand = defaultdict(list)
            for b, l, s, p in zip(b_i.tolist(), last.tolist(), first.tolist(), pv.tolist()):
                cand[b].append([s, l, round(p, 4), int(p > THR)])   # kolom 4: > 0.5 dihitung tanpa pembulatan
            for b, x in enumerate(items):
                greedy = [[int(s.start), int(s.end)] for s in decoders.decode(probs[b:b + 1], decoding_strategy="greedy", threshold=THR)]
                gold = [[int(e["tok_pos"][0]), int(e["tok_pos"][1]), round(float(probs[b, e["tok_pos"][1], e["tok_pos"][0]]), 4), e.get("idx", -1)]
                        for e in x["entities"]]
                L = int(mask[b].sum())
                f.write(json.dumps(dict(data_id=x["data_id"], batch=nb, resmi=resmi, L=L, bos=int(tokens[b, 0]) == tok.bos_token_id,
                                        gold=gold, cand=cand[b], greedy=greedy, off=off[b][:L].tolist(), text=x["text"]),
                                   ensure_ascii=False) + "\n")
            if nb % 200 == 0:
                print(f"  batch {nb}/{len(batches)}", flush=True)
    tmp.rename(out_path)


def hitung_resmi(rows):
    """Metrik persis seperti tommer.evaluate: hanya batch penuh, semua prediksi (termasuk padding) ikut."""
    r = {}
    for strat in ("threshold", "greedy"):
        tp = fp = tot = 0
        for x in rows:
            if not x["resmi"]: continue
            gold = [(g[0], g[1]) for g in x["gold"]]
            pred = tebakan(x, strat)
            tot += len(gold); tp += sum(p in gold for p in pred); fp += sum(p not in gold for p in pred)
        r[strat] = (tp, fp, tot)
    return r


def infer(a):
    import torch
    from llm2ner import utils
    from llm2ner.data import load_all_splits
    out = Path(a.out); (out / "pred").mkdir(parents=True, exist_ok=True)
    for spec in [s for s in a.models.split(",") if s]:
        nama_m = "release" if spec == "release" else ("fixed" if "fixed" in spec else "lama")
        tommer = load_tommer(spec)
        llm = utils.load_llm(tommer.llm_name, cut_to_layer=tommer.layer).eval()
        tommer.to(llm.device)
        c_length = utils.get_model_max_length(tommer.llm_name)
        print(f"model {nama_m}: {spec} | layer {tommer.layer} | mask_bos {getattr(tommer, 'mask_bos', None)} | device {llm.device} | c_length {c_length}", flush=True)
        g = out / "pred" / f"{nama_m}__{GERBANG.replace(' ', '_')}.jsonl.gz"
        if not g.exists():
            ds = load_all_splits(dataset_name=GERBANG, data_folder=Path(os.environ["NER_DATA"]), model=llm, mode="last",
                                 max_ent_length=400, max_length=c_length)
            infer_dataset(tommer, llm, ds, g)
        for strat, (tp, fp, tot) in hitung_resmi(baca_pred(g)).items():
            ac = ACUAN[nama_m][strat]
            ok = tot == ac[2] and abs(tp - ac[0]) <= 0.002 * ac[2] and abs(fp - ac[1]) <= 0.002 * (ac[0] + ac[1])
            print(f"GERBANG {nama_m} {strat}: tp {tp} fp {fp} total {tot} | acuan {ac} | {'LOLOS' if ok else 'GAGAL'}", flush=True)
            if not ok:
                sys.exit(f"Gerbang gagal untuk {nama_m}: hasil tidak sama dengan eval_table1.py. Berhenti, jangan lanjut.")
        for name in [s for s in a.sets.split(",") if s]:
            p = out / "pred" / f"{nama_m}__{name}.jsonl.gz"
            if p.exists(): print("lewati (sudah ada):", p.name, flush=True); continue
            if not (Path(a.data) / name).exists(): print("set tidak ada, lewati:", name, flush=True); continue
            ds, tambah = bangun_dataset(baca(Path(a.data) / name), llm, c_length, baca_meta(Path(a.data) / name).get("ikut_kalimat_kosong", False))
            print(f"infer {nama_m} {name}: {len(ds.data)} kalimat ({tambah} tanpa entitas ditambahkan)", flush=True)
            infer_dataset(tommer, llm, ds, p)
        del tommer, llm; torch.cuda.empty_cache()


# =====================================================================================================================
# metrik: persis + longgar, PLO, kalimat tanpa entitas
# =====================================================================================================================
KRITERIA = ("persis", "tepi_bersih", "batas_1", "tumpang")


def tepi(s, a, b):
    bersih = lambda c: c.isspace() or unicodedata.category(c)[0] in "PZ"
    while a < b and bersih(s[a]): a += 1
    while b > a and bersih(s[b - 1]): b -= 1
    return a, b


def cocok(kriteria, p, g, x):
    (ps, pe), (gs, ge) = p, g
    if kriteria == "persis": return p == g
    if kriteria == "batas_1": return abs(ps - gs) <= 1 and abs(pe - ge) <= 1
    if kriteria == "tumpang": return ps <= ge and gs <= pe
    o, t = x["off"], x["text"]
    if pe >= len(o) or ge >= len(o): return False
    return tepi(t, o[ps][0], o[pe][1]) == tepi(t, o[gs][0], o[ge][1])


def pasangkan(pred, gold, kriteria, x):
    """Pemasangan satu-satu, serakah: pasangan persis dulu, lalu irisan terbesar. Satu entitas hanya 'dibayar' satu tebakan."""
    calon = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if cocok(kriteria, p, g, x):
                iris = min(p[1], g[1]) - max(p[0], g[0]) + 1; gab = max(p[1], g[1]) - min(p[0], g[0]) + 1
                calon.append((p != g, -iris / gab, i, j))
    pakai_p, pakai_g = set(), set()
    for _, _, i, j in sorted(calon):
        if i not in pakai_p and j not in pakai_g: pakai_p.add(i); pakai_g.add(j)
    return pakai_p, pakai_g


def hitung_kalimat(rows, rec_by_id, meta, decoding, tipe, kriteria):
    """Per kalimat: tp, jumlah tebakan, jumlah emas."""
    plo = set(meta.get("tipe_PLO") or ["PER", "ORG", "LOC"])
    T, NP, NG = [], [], []
    for x in rows:
        r = rec_by_id.get(x["data_id"])
        gold, abaikan = [], set()
        for g in x["gold"]:
            ty = r["entities"][g[3]]["type"] if r is not None and 0 <= g[3] < len(r["entities"]) else None
            if tipe == "PLO" and ty not in plo: abaikan.add((g[0], g[1])); continue
            gold.append((g[0], g[1]))
        pred = [p for p in tebakan(x, decoding) if p not in abaikan or p in gold]
        pp, _ = pasangkan(pred, gold, kriteria, x)
        T.append(len(pp)); NP.append(len(pred)); NG.append(len(gold))
    arr = lambda v: np.array(v, dtype=np.int64)
    return arr(T), arr(NP), arr(NG)


def prf_n(tp, npred, ngold):
    P = tp / npred if npred else 0.0; R = tp / ngold if ngold else 0.0
    return 100 * P, 100 * R, 100 * (2 * P * R / (P + R) if P + R else 0.0)


def ci_n(T, NP, NG, B=500, seed=0):
    rng = np.random.default_rng(seed); i = rng.integers(0, len(T), size=(B, len(T)))
    t, p, g = T[i].sum(1), NP[i].sum(1), NG[i].sum(1)
    P = t / np.maximum(1, p); R = t / np.maximum(1, g); F = 2 * P * R / np.maximum(1e-12, P + R)
    h = lambda v: 100 * (np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2
    return h(P), h(R), h(F)


def metrik(a):
    import pandas as pd
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    baris = []
    for f in daftar_pred(a.pred.split(",")):
        m, nama = f.name[:-9].split("__")
        if "CoNLL" in nama or (a.set and not re.search(a.set, nama)): continue
        if not (Path(a.data) / nama).exists(): print("lewati, data tidak ada:", nama); continue
        rec = baca(Path(a.data) / nama); meta = baca_meta(Path(a.data) / nama)
        rows = baca_pred(f); rec_by_id = {x["data_id"]: x for x in rec}
        print(f"{nama:34s} {len(rows):6d} kalimat", flush=True)
        for kalimat in (("semua", "ber_entitas") if any(not x["gold"] for x in rows) else ("semua",)):
            rr = rows if kalimat == "semua" else [x for x in rows if x["gold"]]
            for decoding in ("threshold", "greedy"):
                for tipe in ("semua_tipe", "PLO"):
                    for kriteria in KRITERIA:
                        T, NP, NG = hitung_kalimat(rr, rec_by_id, meta, decoding, tipe, kriteria)
                        P, R, F = prf_n(T.sum(), NP.sum(), NG.sum())
                        cP, cR, cF = ci_n(T, NP, NG) if kriteria == "persis" else (np.nan,) * 3
                        baris.append(dict(model=m, set=nama, bahasa=meta.get("bahasa", ""), anotasi=meta.get("anotasi", ""), kalimat=kalimat,
                                          decoding=decoding, tipe=tipe, kriteria=kriteria, P=P, R=R, F1=F, ci_P=cP, ci_R=cR, ci_F1=cF,
                                          tebakan=int(NP.sum()), emas=int(NG.sum()), n_kalimat=len(rr), tebakan_per_kalimat=NP.sum() / max(1, len(rr))))
    D = pd.DataFrame(baris)
    D.round(2).to_csv(out / "metrik.csv", index=False)
    with open(out / "ringkasan_metrik.txt", "w", encoding="utf8") as f:
        def tulis(judul, df): f.write(f"\n== {judul} ==\n{df.to_string(index=False) if len(df) else '(kosong)'}\n")
        if len(D):
            u = D[(D.kalimat == "semua") & (D.kriteria == "persis") & (D.decoding == "threshold")]
            tulis("Utama: threshold 0.5, cocok persis, semua kalimat", u[["set", "bahasa", "anotasi", "tipe", "P", "R", "F1", "ci_F1", "tebakan_per_kalimat", "emas"]].round(2))
            pv = D[(D.kalimat == "semua") & (D.tipe == "semua_tipe")].pivot_table(index=["set", "decoding"], columns="kriteria", values=["R", "P", "F1"]).round(1)
            f.write("\n== Kriteria cocok longgar (semua kalimat, semua tipe) ==\n" + pv.to_string() + "\n")
            k = D[(D.kriteria == "persis") & (D.decoding == "threshold") & (D.tipe == "semua_tipe")]
            k = k[k.set.isin(D[D.kalimat == "ber_entitas"].set.unique())]
            tulis("Pengaruh kalimat tanpa entitas (loader resmi membuangnya)", k[["set", "kalimat", "P", "R", "F1", "n_kalimat"]].round(2))
    print((out / "ringkasan_metrik.txt").read_text(encoding="utf8"))


# =====================================================================================================================
# format: uji ulang poin 2
# =====================================================================================================================
PASANGAN_FORMAT = [("WikiANN_zh_spasi", "WikiANN_zh_alami"), ("WikiANN_ja_spasi", "WikiANN_ja_alami"),
                   ("WikiANN_en_spasi", "WikiANN_en_karakter"), ("WikiANN_ar_alami", "WikiANN_ar_karakter"),
                   ("WikiANN_ko_alami", "WikiANN_ko_karakter"), ("WikiANN_zh_spasi", "WikiANN_zh_spasi_tanpa_pagar")]


def format_(a):
    dirs = a.pred.split(",")
    def muat(nama):
        for d in dirs:
            p = Path(d) / f"release__{nama}.jsonl.gz"
            if p.exists(): return {x["data_id"]: x for x in baca_pred(p)}
        return None
    kena = lambda x, g: any(c[0] == g[0] and c[1] == g[1] and c[3] for c in x["cand"])
    lines = ["== A. Transisi per entitas (kalimat dan entitas sama): versi pertama -> versi kedua =="]
    for p1, p2 in PASANGAN_FORMAT[:5]:
        A, B = muat(p1), muat(p2)
        if not A or not B: lines.append(f"  {p1} / {p2}: prediksi belum ada"); continue
        kel = defaultdict(lambda: defaultdict(int)); beda = 0
        for i in A.keys() & B.keys():
            xa, xb = A[i], B[i]
            if len(xa["gold"]) != len(xb["gold"]): beda += 1; continue
            ga = {g[3]: g for g in xa["gold"]}; gb = {g[3]: g for g in xb["gold"]}
            for k in ga.keys() & gb.keys():
                st = {(True, True): "keduanya", (True, False): "hanya_pertama", (False, True): "hanya_kedua", (False, False): "tidak_keduanya"}[(kena(xa, ga[k]), kena(xb, gb[k]))]
                jenis = "seluruh_baris" if (ga[k][0] == 1 and ga[k][1] == xa["L"] - 1) else "dalam_kalimat"
                kel["semua"][st] += 1; kel[jenis][st] += 1
        lines.append(f"  {p1} -> {p2}" + (f"  (dilewati {beda} kalimat, jumlah entitas beda)" if beda else ""))
        for k, d in kel.items():
            n = sum(d.values())
            lines.append(f"     {k:14s} n {n:6d} | " + "  ".join(f"{s} {100 * d[s] / n:5.1f}%" for s in ("keduanya", "hanya_pertama", "hanya_kedua", "tidak_keduanya")))
    lines += ["", "== B. Kontrol format (threshold 0.5, semua kalimat). lift = P model / P acak ==",
              f"  {'set':34s} {'P':>6s} {'R':>6s} {'F1':>6s} {'tebakan/kal':>11s} {'P_acak':>7s} {'lift':>6s}"]
    for p1, p2 in PASANGAN_FORMAT:
        for n in (p1, p2):
            r = muat(n)
            if r is None: lines.append(f"  {n:34s} (belum ada prediksi)"); continue
            tp = fp = tot = kand = 0
            for x in r.values():
                gold = {(g[0], g[1]) for g in x["gold"]}; pred = tebakan(x, "threshold")
                t = sum(p in gold for p in pred); tp += t; fp += len(pred) - t; tot += len(x["gold"]); kand += kandidat_sah(x["L"])
            P, R, F = prf(tp, fp, tot); Pa = 100 * tot / max(1, kand)
            lines.append(f"  {n:34s} {P:6.1f} {R:6.1f} {F:6.1f} {(tp + fp) / len(r):11.2f} {Pa:7.2f} {P / Pa if Pa else 0:6.1f}")
        lines.append("")
    teks = "\n".join(lines); print(teks)
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "validasi_format.txt").write_text(teks, encoding="utf8")


# =====================================================================================================================
# paralel: kalimat yang sama dalam beberapa bahasa (UNER-PUD)
# =====================================================================================================================
def paralel(a):
    import pandas as pd
    dirs = a.pred.split(","); data = {}
    for k in PUD:
        f = next((Path(d) / f"release__UNER_PUD_{k}.jsonl.gz" for d in dirs if (Path(d) / f"release__UNER_PUD_{k}.jsonl.gz").exists()), None)
        if f is None: print("prediksi belum ada:", k); continue
        sid = baca_meta(Path(a.data) / f"UNER_PUD_{k}")["sent_id_ud"]
        data[k] = {sid[x["data_id"]]: x for x in baca_pred(f) if sid[x["data_id"]]}
    if "en" not in data: print("butuh UNER_PUD_en"); return
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    print(f"kalimat paralel yang ada di semua bahasa: {len(ids)}")
    hasil, rng = [], np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(1000, len(ids)))
    def per(k, dec, krit):
        T, NP, NG = [], [], []
        for i in ids:
            x = data[k][i]; gold = [(g[0], g[1]) for g in x["gold"]]; pred = tebakan(x, dec)
            pp, _ = pasangkan(pred, gold, krit, x); T.append(len(pp)); NP.append(len(pred)); NG.append(len(gold))
        return np.array(T), np.array(NP), np.array(NG)
    def met(X, ii):
        t, p, g = X[0][ii].sum(-1), X[1][ii].sum(-1), X[2][ii].sum(-1)
        P = t / np.maximum(1, p); R = t / np.maximum(1, g); return 100 * P, 100 * R, 100 * 2 * P * R / np.maximum(1e-12, P + R)
    for dec in ("threshold", "greedy"):
        for krit in ("persis", "tumpang"):
            E = per("en", dec, krit); e0 = met(E, np.arange(len(ids))); eb = met(E, idx)
            for k in data:
                X = per(k, dec, krit); x0 = met(X, np.arange(len(ids))); xb = met(X, idx)
                h = dict(bahasa=k, decoding=dec, kriteria=krit, kalimat=len(ids), emas=int(X[2].sum()), tebakan_per_kalimat=X[1].sum() / len(ids),
                         P=x0[0], R=x0[1], F1=x0[2])
                for n, j in (("P", 0), ("R", 1), ("F1", 2)):
                    d = xb[j] - eb[j]
                    h[f"d{n}_vs_en"] = x0[j] - e0[j]; h[f"ci95_d{n}"] = (np.percentile(d, 97.5) - np.percentile(d, 2.5)) / 2
                hasil.append(h)
    T = pd.DataFrame(hasil).round(2); Path(a.out).mkdir(parents=True, exist_ok=True)
    T.to_csv(Path(a.out) / "paralel.csv", index=False); print(T.to_string(index=False))


# =====================================================================================================================
# juri: precision sebenarnya, cara paper bagian 4.2 + Lampiran D (LLM juri, divalidasi manusia, koreksi PPI)
# =====================================================================================================================
PROMPT_SISTEM = """You are an expert in entity mention annotation.
A mention is defined as: "something that exists as itself. It does not need to be of material existence."
In particular, abstractions and legal fictions are usually regarded as entities. In general, there is also no presumption that an entity is animate, or present. It may refer to animals; natural features such as mountains; inanimate objects such as tables; numbers or sets as symbols written on a paper; human contrivances such as laws, corporations and academic disciplines; or supernatural beings such as gods and spirits."

## Instructions
- For each text span provided in [[...]], quickly determine if it is a valid mention as defined above, regardless of its type, length, or style, but ensuring it is not a fragment.
- Briefly explain in one concise sentence whether the span fits the definition. Then answer with a clear "yes" or "no"."""
CONTOH_USER = '"... here that she met her future [[second husband]], Gottfried Lessing ..."'
CONTOH_JAWAB = 'The span "second husband" refers to a specific person as a distinct entity, fitting the definition of a mention. Yes'
SET_JURI = "UNER_PUD_en,UNER_PUD_sv,UNER_PUD_ru,UNER_PUD_zh,NER_UI_id,KLUE_ko,Stockmark_ja,MSRA_zh,ANERcorp_ar"


def kategori(p, x):
    gold = [(g[0], g[1]) for g in x["gold"]]
    if p in gold: return "emas"
    s, e = p
    if any(gs <= s and e <= ge for gs, ge in gold): return "di_dalam_entitas"
    if any(s <= gs and ge <= e for gs, ge in gold): return "memuat_entitas"
    if any(s <= ge and gs <= e for gs, ge in gold): return "bersilangan"
    return "lepas"


def juri_sampel(a):
    """Sampel acak dari tebakan threshold yang sah. Tebakan greedy adalah himpunan bagian (greedy memakai threshold yang sama)."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0); semua, ring = [], {}
    for nama in a.sets.split(","):
        f = next((Path(d) / f"release__{nama}.jsonl.gz" for d in a.pred.split(",") if (Path(d) / f"release__{nama}.jsonl.gz").exists()), None)
        if f is None: print("lewati, prediksi belum ada:", nama); continue
        rows = baca_pred(f); emas_list, bukan = [], []
        n_pred = {"threshold": 0, "greedy": 0}; n_emas = {"threshold": 0, "greedy": 0}
        for x in rows:
            gs = {tuple(g) for g in x["greedy"]}
            for p in tebakan(x, "threshold"):
                if not sah(p, x["L"]): continue
                k = kategori(p, x); g = p in gs
                n_pred["threshold"] += 1; n_pred["greedy"] += g; n_emas["threshold"] += k == "emas"; n_emas["greedy"] += g and k == "emas"
                o = x["off"]; a0, b0 = o[p[0]][0], o[p[1]][1]
                while a0 < b0 and x["text"][a0].isspace(): a0 += 1
                item = dict(set=nama, data_id=x["data_id"], span=[p[0], p[1]], kategori=k, greedy=g, teks_span=x["text"][a0:b0],
                            kalimat_tanda=x["text"][:a0] + "[[" + x["text"][a0:b0] + "]]" + x["text"][b0:])
                (emas_list if k == "emas" else bukan).append(item)
        pilih = rng.sample(bukan, min(a.n, len(bukan))) + rng.sample(emas_list, min(a.n_emas, len(emas_list)))
        for i, it in enumerate(pilih): it["id"] = f"{nama}#{i}"
        semua += pilih
        ring[nama] = dict(tebakan=n_pred, tebakan_emas=n_emas, bukan_emas_tersedia=len(bukan), disampel_bukan_emas=min(a.n, len(bukan)),
                          disampel_emas=min(a.n_emas, len(emas_list)))
        print(f"{nama:22s} tebakan {n_pred['threshold']:7d}, persis emas {n_emas['threshold']:7d}, disampel {len(pilih)}", flush=True)
    with open(out / "sampel.jsonl", "w", encoding="utf8") as f:
        for it in semua: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "sampel_ringkasan.json").write_text(json.dumps(ring, indent=1))


def juri(a):
    """Kirim tiap span ke API kompatibel OpenAI (default gpt-4.1-mini seperti paper). Kunci dibaca dari env, tidak disimpan."""
    import urllib.request
    url = os.environ.get("JURI_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    model = os.environ.get("JURI_MODEL", "gpt-4.1-mini"); key = os.environ.get("JURI_KEY", "")
    if not key and "api.openai.com" in url: sys.exit("Isi kunci dulu: read -s JURI_KEY; export JURI_KEY")
    masuk = [json.loads(l) for l in open(a.masuk, encoding="utf8")]
    keluar = Path(a.keluar); sudah = set()
    if keluar.exists(): sudah = {json.loads(l)["id"] for l in open(keluar, encoding="utf8")}
    print(f"model {model} | {len(masuk)} span, {len(sudah)} sudah dijawab", flush=True)
    with open(keluar, "a", encoding="utf8") as f:
        for n, it in enumerate(masuk):
            if it["id"] in sudah: continue
            kal = it["kalimat_tanda"]; k = kal.find("[["); kal = ("..." if k > 300 else "") + kal[max(0, k - 300):k + 300 + len(it["teks_span"])] + "..."
            body = dict(model=model, temperature=0, messages=[dict(role="system", content=PROMPT_SISTEM), dict(role="user", content=CONTOH_USER),
                                                               dict(role="assistant", content=CONTOH_JAWAB), dict(role="user", content=f'"{kal}"')])
            for coba in range(5):
                try:
                    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                    teks = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]; break
                except Exception as e:
                    print(f"  ulang ({coba + 1}) {it['id']}: {str(e)[:120]}", flush=True); time.sleep(2 ** coba)
            else:
                sys.exit("API gagal 5 kali berturut-turut; jalankan ulang perintah yang sama untuk melanjutkan.")
            jaw = re.findall(r"\b(yes|no)\b", teks.lower())
            f.write(json.dumps(dict(id=it["id"], jawaban=teks, valid=None if not jaw else jaw[-1] == "yes", model=model), ensure_ascii=False) + "\n"); f.flush()
            if n % 100 == 0: print(f"  {n}/{len(masuk)}", flush=True)


def juri_manusia(a):
    """Lembar anotasi buta: penilai tidak melihat jawaban LLM, emas, maupun penilai lain (protokol Lampiran D.2)."""
    import csv
    sampel = [json.loads(l) for l in open(a.masuk, encoding="utf8")]
    rng = random.Random(1); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for nama in a.sets.split(","):
        bukan = [s for s in sampel if s["set"] == nama and s["kategori"] != "emas"]
        pilih = rng.sample(bukan, min(a.n, len(bukan)))
        for p in ("A", "B"):
            with open(out / f"anotasi_{nama}_{p}.csv", "w", newline="", encoding="utf8") as f:
                w = csv.writer(f); w.writerow(["id", "kalimat_dengan_span", "valid (ya/tidak)", "catatan"])
                for s in rng.sample(pilih, len(pilih)): w.writerow([s["id"], s["kalimat_tanda"], "", ""])
        print(f"{nama}: {len(pilih)} span -> anotasi_{nama}_A.csv dan _B.csv (isi kolom 'valid' dengan ya/tidak)")


def baca_anotasi(path):
    import csv
    ok = {"ya": True, "y": True, "yes": True, "tidak": False, "t": False, "no": False, "n": False}
    with open(path, encoding="utf8") as f:
        return {r["id"]: ok.get(r["valid (ya/tidak)"].strip().lower()) for r in csv.DictReader(f) if r["valid (ya/tidak)"].strip()}


def kappa(x, y):
    x, y = np.array(x, float), np.array(y, float)
    po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def juri_hitung(a):
    """P_sebenarnya = (tebakan persis emas + tebakan bukan-emas yang valid) / semua tebakan.
    theta = bagian tebakan bukan-emas yang valid. Juri: theta_juri = rata-rata jawaban LLM di sampel.
    PPI (Angelopoulos dkk. 2023): theta_PPI = rata-rata LLM (sampel besar) + rata-rata (manusia - LLM) di sampel kecil berlabel manusia.
    Variansi = var(LLM)/N + var(manusia - LLM)/n. Tanpa label manusia hanya angka juri yang dilaporkan (belum tervalidasi)."""
    import pandas as pd
    d = Path(a.dir); ring = json.loads((d / "sampel_ringkasan.json").read_text())
    sampel = {s["id"]: s for s in (json.loads(l) for l in open(d / "sampel.jsonl", encoding="utf8"))}
    jaw = {j["id"]: j["valid"] for j in (json.loads(l) for l in open(d / "jawaban_juri.jsonl", encoding="utf8")) if j["valid"] is not None}
    manusia = {}
    for f in sorted((d / "manusia").glob("anotasi_*_A.csv")) if (d / "manusia").exists() else []:
        nama = f.name[len("anotasi_"):-len("_A.csv")]; A = baca_anotasi(f)
        fb = f.with_name(f"anotasi_{nama}_B.csv"); B = baca_anotasi(fb) if fb.exists() else {}
        fx = f.with_name(f"anotasi_{nama}_final.csv"); X = baca_anotasi(fx) if fx.exists() else {}
        manusia[nama] = (A, B, X)
    hasil = []
    for nama, r in ring.items():
        for dec in ("threshold", "greedy"):
            ids = [i for i, s in sampel.items() if s["set"] == nama and s["kategori"] != "emas" and (dec == "threshold" or s["greedy"]) and i in jaw]
            ide = [i for i, s in sampel.items() if s["set"] == nama and s["kategori"] == "emas" and i in jaw]
            if not ids: continue
            npred, nemas = r["tebakan"][dec], r["tebakan_emas"][dec]; p_emas = nemas / max(1, npred)
            f_ = np.array([jaw[i] for i in ids], float); th = f_.mean(); var = f_.var(ddof=1) / len(f_)
            h = dict(set=nama, decoding=dec, tebakan=npred, P_resmi_persis=100 * p_emas, n_juri=len(ids), juri_terima_emas_pct=100 * np.mean([jaw[i] for i in ide]) if ide else np.nan,
                     theta_juri=100 * th, P_juri=100 * (p_emas + (1 - p_emas) * th),
                     **{f"juri_valid_{k}_pct": 100 * np.mean([jaw[i] for i in ids if sampel[i]["kategori"] == k]) if any(sampel[i]["kategori"] == k for i in ids) else np.nan
                        for k in ("lepas", "di_dalam_entitas", "memuat_entitas", "bersilangan")})
            if nama in manusia:
                A, B, X = manusia[nama]; Y = X or A                     # label final hasil diskusi; kalau belum ada, penilai A
                lab = [i for i in ids if i in Y]
                if len(lab) >= 20:
                    y = np.array([Y[i] for i in lab], float); fl = np.array([jaw[i] for i in lab], float)
                    th_ppi = th + (y - fl).mean(); var_ppi = var + (y - fl).var(ddof=1) / len(lab)
                    both = [i for i in lab if i in A and i in B]
                    h.update(n_manusia=len(lab), setuju_manusia_juri_pct=100 * (y == fl).mean(), kappa_manusia_juri=kappa(y, fl),
                             setuju_A_B_pct=100 * np.mean([A[i] == B[i] for i in both]) if both else np.nan,
                             kappa_A_B=kappa([A[i] for i in both], [B[i] for i in both]) if both else np.nan,
                             bias_juri=100 * (fl - y).mean(), theta_PPI=100 * th_ppi,
                             P_sebenarnya_PPI=100 * (p_emas + (1 - p_emas) * th_ppi), ci95_P_PPI=100 * 1.96 * (1 - p_emas) * np.sqrt(var_ppi))
            h["ci95_P_juri"] = 100 * 1.96 * (1 - p_emas) * np.sqrt(var)
            hasil.append(h)
    T = pd.DataFrame(hasil).round(2); T.to_csv(Path(a.out) / "precision_juri.csv", index=False); print(T.to_string(index=False))


# =====================================================================================================================
# tes-lokal: tanpa GPU, tanpa HF
# =====================================================================================================================
def tes_lokal(a):
    import shutil, warnings
    warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, Regex
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaModel
    from llm2ner import utils
    from llm2ner.models import TokenMatchingNER
    utils.get_model_dim = lambda n: 32
    torch.manual_seed(0)
    data, tmp = Path(a.data), Path(a.out) / "tes_lokal"
    shutil.rmtree(tmp, ignore_errors=True); (tmp / "data").mkdir(parents=True); (tmp / "pred").mkdir()
    sets = {"KLUE_ko": 100, "Stockmark_ja": 120, "NER_UGM_id": 120, "MSRA_zh": 100, "UNER_PUD_en": 80, "UNER_PUD_zh": 80, "UNER_PUD_ru": 80, "UNER_PUD_sv": 80}
    for n, k in sets.items():
        (tmp / "data" / n).mkdir(); d = json.loads((data / n / "test.json").read_text(encoding="utf8"))[:k]
        (tmp / "data" / n / "test.json").write_text(json.dumps(d, ensure_ascii=False))
        (tmp / "data" / n / "meta.txt").write_text(json.dumps(baca_meta(data / n), ensure_ascii=False))
    chars = sorted({c for n in sets for x in baca(tmp / "data" / n) for c in x["sentence"]})
    vocab = {"<s>": 0, "<pad>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); t.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    t.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", pad_token="<pad>", unk_token="<unk>")
    llm = LlamaModel(LlamaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                 max_position_embeddings=4096, pad_token_id=1, bos_token_id=0, eos_token_id=0)).eval(); llm.tokenizer = tok
    ner = TokenMatchingNER.C(llm_name="x", layer=1, rank=8, method="cl_fn_minmaxpool", causal_mask=True, sliding_window=25,
                             use_cosine=True, normalize_scores="").instance(); ner.initialize()
    def palsu(tokens, model, attn_mask=None, **k):
        b, n = tokens.shape; i = torch.arange(n); last, first = i.view(1, n, 1), i.view(1, 1, n)
        v = torch.sin(tokens.view(b, n, 1).float() * 3.1 + tokens.view(b, 1, n).float() * 7.7 + (last - first) * 0.9)
        return torch.sigmoid(4 * v) * ((first <= last) & (last - first < 25)).float()
    ner.forward = palsu
    semua_ok = True
    for n in sets:
        rec = baca(tmp / "data" / n); meta = baca_meta(tmp / "data" / n); kosong = sum(not x["entities"] for x in rec)
        ds, tambah = bangun_dataset(rec, llm, 4096, meta.get("ikut_kalimat_kosong", False))
        p = tmp / "pred" / f"release__{n}.jsonl.gz"; infer_dataset(ner, llm, ds, p); rows = baca_pred(p)
        ok = len(rows) == len(rec); semua_ok &= ok
        print(f"BANGUN {n}: {len(rec)} kalimat, {kosong} tanpa entitas, {tambah} ditambahkan, {len(rows)} diprediksi | {'OK' if ok else 'BEDA'}")
        res = hitung_resmi([dict(x, resmi=True) for x in rows]); rb = {x["data_id"]: x for x in rec}
        for dec in ("threshold", "greedy"):
            T, NP, NG = hitung_kalimat(rows, rb, meta, dec, "semua_tipe", "persis")
            x1, x2 = (int(T.sum()), int(NP.sum() - T.sum()), int(NG.sum())), res[dec]; semua_ok &= x1 == x2
            print(f"PARITAS_METRIK {n} {dec}: {x1} vs resmi {x2} | {'SAMA' if x1 == x2 else 'BEDA'}")
    ns = argparse.Namespace(pred=str(tmp / "pred"), data=str(tmp / "data"), out=str(tmp / "analisis"), set="")
    metrik(ns); paralel(ns); format_(ns)
    js = argparse.Namespace(pred=str(tmp / "pred"), out=str(tmp / "juri"), sets="UNER_PUD_en,KLUE_ko", n=30, n_emas=5); juri_sampel(js)
    # jawaban juri palsu dan anotasi manusia palsu, untuk menguji juri-hitung
    with open(tmp / "juri" / "jawaban_juri.jsonl", "w") as f:
        for l in open(tmp / "juri" / "sampel.jsonl"):
            s = json.loads(l); f.write(json.dumps(dict(id=s["id"], jawaban="... yes", valid=s["kategori"] in ("emas", "lepas"))) + "\n")
    juri_manusia(argparse.Namespace(masuk=str(tmp / "juri" / "sampel.jsonl"), out=str(tmp / "juri" / "manusia"), sets="UNER_PUD_en", n=25))
    import csv
    for p in ("A", "B"):
        f = tmp / "juri" / "manusia" / f"anotasi_UNER_PUD_en_{p}.csv"; r = list(csv.reader(open(f, encoding="utf8")))
        for k, row in enumerate(r[1:]): row[2] = "ya" if (k % 3) else "tidak"
        csv.writer(open(f, "w", newline="", encoding="utf8")).writerows(r)
    juri_hitung(argparse.Namespace(dir=str(tmp / "juri"), out=str(tmp / "juri")))
    print("TES_LOKAL", "SEMUA OK" if semua_ok else "ADA YANG BEDA")


# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="tahap", required=True)
    def tambah(nama, fn, **opsi):
        p = sp.add_parser(nama); p.set_defaults(fn=fn)
        for k, v in opsi.items(): p.add_argument("--" + k.replace("_", "-"), dest=k, default=v, type=type(v) if v is not None else str)
    tambah("siapkan", siapkan, paket=str(HERE), data=str(HERE / "data"))
    tambah("cek", cek, data=str(HERE / "data"), sets="", out=str(HERE / "hasil"), llm="meta-llama/Llama-3.2-1B")
    tambah("tes-lokal", tes_lokal, data=str(HERE / "data"), out=str(HERE / "hasil"))
    tambah("infer", infer, models="release", sets="", data=str(HERE / "data"), out=str(HERE / "hasil"))
    for nama, fn in (("metrik", metrik), ("format", format_), ("paralel", paralel)):
        tambah(nama, fn, pred=str(HERE / "hasil" / "pred"), data=str(HERE / "data"), out=str(HERE / "hasil" / "analisis"), set="")
    tambah("juri-sampel", juri_sampel, pred=str(HERE / "hasil" / "pred"), out=str(HERE / "hasil" / "juri"), sets=SET_JURI, n=500, n_emas=50)
    tambah("juri", juri, masuk=str(HERE / "hasil" / "juri" / "sampel.jsonl"), keluar=str(HERE / "hasil" / "juri" / "jawaban_juri.jsonl"))
    tambah("juri-manusia", juri_manusia, masuk=str(HERE / "hasil" / "juri" / "sampel.jsonl"), out=str(HERE / "hasil" / "juri" / "manusia"),
           sets="NER_UI_id,UNER_PUD_en", n=150)
    tambah("juri-hitung", juri_hitung, dir=str(HERE / "hasil" / "juri"), out=str(HERE / "hasil" / "analisis"))
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
