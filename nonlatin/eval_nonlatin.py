"""Uji zero-shot ToMMeR pada WikiANN non-Latin (zh, ja, ar) dengan semua prediksi disimpan, supaya hasilnya bisa dijelaskan.

Dijalankan lewat run_nonlatin.sh. Tahapan:
  siapkan  ubah parquet WikiANN (misalnya en) menjadi test.json llm2ner
  cek      CPU saja: jumlah entitas yang benar-benar dinilai evaluasi resmi (batch terakhir yang tidak penuh dibuang)
  infer    jalankan model, simpan probabilitas span > 0.1, prediksi greedy, dan probabilitas setiap entitas emas
  analyze  hitung P/R/F1, CI bootstrap, uraian kesalahan, kurva threshold, recall per kelompok, dan bukti per hipotesis
"""
import argparse, gzip, json, os, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BATCH = 30                     # sama dengan eval_table1.py dan tasks.py milik penulis
SIMPAN_MIN = 0.1               # probabilitas minimum yang disimpan, untuk kurva threshold
THR = 0.5                      # threshold resmi paper
GERBANG = "CoNLL 2003"
# Hasil resmi yang sudah kita punya (tp, fp, total) dari eval_table1.py. Inferensi tanpa training harus sama persis.
ACUAN = {"release": {"threshold": (32968, 65294, 34766), "greedy": (30047, 37180, 34766)},
         "fixed": {"threshold": (32772, 79017, 34766), "greedy": (29840, 40713, 34766)}}
PAPER = {"WikiANN_zh": dict(R=40.7, P=6.3, F1=10.9, gR=24.0, gP=8.8, gF1=12.9, n=40000, ent=50033),
         "WikiANN_en": dict(R=84.2, P=33.4, F1=47.8, gR=75.5, gP=46.2, gF1=57.3, n=40000, ent=56035)}
PENGALIHAN = re.compile(r"^\s*(#?\s*redi(rect)?\b|alih\b|重定向|تحويل)", re.I)
_sp = lambda w: r"\s?".join(map(re.escape, w))
# Penanda pengalihan yang juga cocok untuk versi spasi (R E D I R E C T, 重 定 向).
PENANDA = re.compile(r"^[\s#]*(" + "|".join([_sp("redirect") + r"\b", _sp("redi") + r"\b", r"alih\b", _sp("重定向"), "تحويل"]) + r")\s*", re.I)
# Kurung tutup yang pasangannya ada di dalam isi baris adalah bagian dari nama (judul Wikipedia, mis. 'دايو ( شركة )').
PASANGAN = {")": "(", "]": "[", "）": "（", "」": "「", "』": "『", "》": "《", "】": "【"}


def isi_baris(s):
    """Posisi karakter isi baris: tanpa penanda pengalihan, spasi, dan tanda baca di kedua ujung. Dipakai baseline seluruh-baris.
    Kurung tutup yang berpasangan tidak dibuang; kalau dibuang, 23.8% entitas Arab (berakhir ')') tidak pernah cocok."""
    m = PENANDA.match(s); a = m.end() if m else 0; b = len(s)
    tepi = lambda c: c.isspace() or unicodedata.category(c)[0] in "PZ"
    while a < b and tepi(s[a]): a += 1
    while b > a and tepi(s[b - 1]) and not (s[b - 1] in PASANGAN and PASANGAN[s[b - 1]] in s[a:b - 1]): b -= 1
    return a, b


def baseline_seluruh_baris(records):
    """Satu tebakan per kalimat: seluruh isi baris. Mengukur berapa skor yang bisa didapat dari bentuk data saja."""
    tp = ent = 0
    for x in records:
        a, b = isi_baris(x["sentence"]); ent += len(x["entities"])
        tp += any(e["pos"] == [a, b] for e in x["entities"])
    return prf(tp, len(records) - tp, ent)


def kandidat_sah(L, window=25):
    """Jumlah span sah dalam satu kalimat: awal >= 1 (bukan BOS), akhir < L, panjang <= window + 1 (mask sliding window)."""
    return sum(min(window + 1, L - s) for s in range(1, L))


def baca(folder):
    """Baca semua split di satu folder dataset, urutan sama dengan load_dataset_splits."""
    return [x for f in sorted(Path(folder).glob("*.json")) for x in json.loads(f.read_text(encoding="utf8"))]


# ---------------- siapkan ----------------
def siapkan(parquet_dir, kode, out_root):
    """Parquet WikiANN -> test.json versi spasi, dedup, tanpa_alih. Logikanya sama dengan notebook EDA."""
    import pyarrow.parquet as pq

    def spans(tags):
        h, a = [], None
        for i, t in enumerate(tags):
            if t == "O":
                if a: h.append(tuple(a)); a = None
            elif t.startswith("B-") or not a or a[0] != t[2:]:
                if a: h.append(tuple(a))
                a = [t[2:], i, i + 1]
            else:
                a[2] = i + 1
        if a: h.append(tuple(a))
        return h

    semua = []
    for split in ("train", "validation", "test"):
        t = pq.read_table(Path(parquet_dir) / f"{split}.parquet")
        nama = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["ner_tags"]["feature"]["names"]
        for x in t.select(["tokens", "ner_tags"]).to_pylist():
            tok, tag = x["tokens"], [nama[i] for i in x["ner_tags"]]
            awal, c = [], 0
            for w in tok:
                awal.append(c); c += len(w) + 1
            kal = " ".join(tok)
            ent = [dict(pos=[awal[a], awal[b - 1] + len(tok[b - 1])], type=ty, text=" ".join(tok[a:b])) for ty, a, b in spans(tag)]
            semua.append(dict(data_id=len(semua), sentence=kal, entities=ent))
    lihat, unik = set(), []
    for x in semua:
        if x["sentence"] not in lihat: lihat.add(x["sentence"]); unik.append(x)
    bersih = [x for x in unik if not PENGALIHAN.match(x["sentence"])]
    for sub, isi in ((f"WikiANN_{kode}_spasi", semua), (f"WikiANN_{kode}_spasi_dedup", unik), (f"WikiANN_{kode}_spasi_tanpa_alih", bersih)):
        p = Path(out_root) / sub; p.mkdir(parents=True, exist_ok=True)
        (p / "test.json").write_text(json.dumps(isi, ensure_ascii=False))
        print(f"{sub}: {len(isi)} kalimat, {sum(len(x['entities']) for x in isi)} entitas", flush=True)


# ---------------- cek ----------------
def urutan_resmi(panjang, batch=BATCH):
    """Tiru BatchedDataset: urutkan menurut panjang subtoken, potong per batch, buang batch terakhir yang tidak penuh."""
    idx = sorted(range(len(panjang)), key=lambda i: panjang[i])
    penuh = len(idx) // batch
    return [idx[i * batch:(i + 1) * batch] for i in range(penuh)], idx[penuh * batch:]


def cek(data_root, names, llm_name, out):
    from transformers import AutoTokenizer
    from llm2ner.data import JSONDataSet
    tok = AutoTokenizer.from_pretrained(llm_name)
    hasil = {}
    for name in names:
        ds = JSONDataSet(baca(Path(data_root) / name), model=None, tokenizer=tok, mode="last", max_ent_length=400, max_length=10 ** 9)
        pjg = [len(x["str_tokens"]) for x in ds.data]
        n_ent = [len(x["entities"]) for x in ds.data]
        from llm2ner import utils
        try: c_len = utils.get_model_max_length(llm_name)
        except Exception: c_len = None
        r = dict(kalimat=len(pjg), entitas=sum(n_ent), subtoken_maks=max(pjg), batas_konteks=c_len,
                 kalimat_melebihi_konteks=sum(p > c_len for p in pjg) if c_len else None)
        for b in (1, 8, 16, 20, 30, 32, 64, 128):
            _, sisa = urutan_resmi(pjg, b)
            r[f"entitas_dinilai_batch{b}"] = sum(n_ent) - sum(n_ent[i] for i in sisa)
        hasil[name] = r
        ref = PAPER.get(re.sub(r"_(spasi|alami).*", "", name))
        print(name, json.dumps(r), f"| paper: {ref['ent']}" if ref and name.endswith(("_spasi", "_alami")) else "", flush=True)
    Path(out).write_text(json.dumps(hasil, indent=1))


# ---------------- infer ----------------
def load_tommer(spec):
    """'release' = checkpoint penulis di HF, selain itu folder model hasil training (cara muat sama dengan eval_table1.py)."""
    import importlib
    from xpm_torch.huggingface import TorchHFHub
    if spec == "release":
        t = TorchHFHub.from_pretrained("llm2ner/ToMMeR-Llama-3.2-1B_L6_R64"); t.initialize(); return t
    params = json.loads((Path(spec).parent / "params.json").read_text())
    obj = next(o for o in params["objects"] if ".models." in o["typename"] and "layer" in o["fields"])
    mod, cls = obj["typename"].rsplit(".", 1)
    t = getattr(importlib.import_module(mod), cls).C(**obj["fields"]).instance()
    t.initialize(); t.load_model(Path(spec))
    return t


def bangun_dataset(records, llm, c_length):
    """Loader resmi (JSONDataSet). Indeks entitas disisipkan supaya bisa digabung dengan metadata EDA."""
    from llm2ner.data import JSONDataSet
    for x in records:
        for k, e in enumerate(x["entities"]):
            e["idx"] = k
    return JSONDataSet(records, model=llm, mode="last", max_ent_length=400, max_length=c_length)


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


def baca_pred(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(l) for l in f]


def hitung_resmi(rows):
    """Hitung ulang metrik persis seperti tommer.evaluate: hanya batch penuh, semua prediksi (termasuk padding) ikut."""
    r = {}
    for strat in ("threshold", "greedy"):
        tp = fp = tot = 0
        for x in rows:
            if not x["resmi"]: continue
            gold = [(g[0], g[1]) for g in x["gold"]]
            pred = [(c[0], c[1]) for c in x["cand"] if c[3]] if strat == "threshold" else [tuple(g) for g in x["greedy"]]
            tot += len(gold); tp += sum(p in gold for p in pred); fp += sum(p not in gold for p in pred)
        r[strat] = (tp, fp, tot)
    return r


def infer(args):
    import torch
    from llm2ner import utils
    from llm2ner.data import load_all_splits
    out = Path(args.out); (out / "pred").mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        nama_m = "release" if spec == "release" else "fixed"
        tommer = load_tommer(spec)
        llm = utils.load_llm(tommer.llm_name, cut_to_layer=tommer.layer).eval()
        tommer.to(llm.device)
        c_length = utils.get_model_max_length(tommer.llm_name)
        print(f"model {nama_m}: {spec} | layer {tommer.layer} | mask_bos {getattr(tommer, 'mask_bos', None)} | device {llm.device} | c_length {c_length}", flush=True)
        # Gerbang: CoNLL 2003 harus sama dengan hasil eval_table1.py yang sudah ada.
        g = out / "pred" / f"{nama_m}__{GERBANG.replace(' ', '_')}.jsonl.gz"
        if not g.exists():
            ds = load_all_splits(dataset_name=GERBANG, data_folder=Path(os.environ["NER_DATA"]), model=llm, mode="last",
                                 max_ent_length=400, max_length=c_length)
            infer_dataset(tommer, llm, ds, g)
        r = hitung_resmi(baca_pred(g))
        for strat, (tp, fp, tot) in r.items():
            a = ACUAN[nama_m][strat]
            ok = tot == a[2] and abs(tp - a[0]) <= 0.002 * a[2] and abs(fp - a[1]) <= 0.002 * (a[0] + a[1])
            print(f"GERBANG {nama_m} {strat}: tp {tp} fp {fp} total {tot} | acuan {a} | {'LOLOS' if ok else 'GAGAL'}", flush=True)
            if not ok:
                sys.exit(f"Gerbang gagal untuk {nama_m}: hasil tidak sama dengan eval_table1.py. Berhenti, jangan lanjut.")
        for name in args.data:
            p = out / "pred" / f"{nama_m}__{name}.jsonl.gz"
            if p.exists():
                print("lewati (sudah ada):", p.name, flush=True); continue
            print(f"infer {nama_m} {name}", flush=True)
            ds = bangun_dataset(baca(Path(args.data_root) / name), llm, c_length)
            infer_dataset(tommer, llm, ds, p)
        del tommer, llm; torch.cuda.empty_cache()


# ---------------- analyze ----------------
def prf(tp, fp, tot):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / tot if tot else 0.0
    return 100 * P, 100 * R, 100 * (2 * P * R / (P + R) if P + R else 0.0)


def per_kalimat(rows, thr=THR, strat="threshold", hanya_sah=False):
    """tp, fp, total per kalimat. hanya_sah=True membuang prediksi di BOS, padding, atau awal > akhir."""
    tp, fp, tot = [], [], []
    for x in rows:
        gold = {(g[0], g[1]) for g in x["gold"]}
        if strat == "threshold":
            pred = [(c[0], c[1]) for c in x["cand"] if (c[3] if thr == THR else c[2] > thr)]
        else:
            pred = [tuple(g) for g in x["greedy"]]
        if hanya_sah:
            pred = [p for p in pred if sah(p, x["L"])]
        t = sum(p in gold for p in pred)
        tp.append(t); fp.append(len(pred) - t); tot.append(len(x["gold"]))
    return np.array(tp), np.array(fp), np.array(tot)


def sah(p, L):
    s, e = p
    return 1 <= s <= e < L


def bootstrap(tp, fp, tot, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(tp)
    i = rng.integers(0, n, size=(B, n))
    T, F, N = tp[i].sum(1), fp[i].sum(1), tot[i].sum(1)
    P = T / np.maximum(1, T + F); R = T / np.maximum(1, N); F1 = 2 * P * R / np.maximum(1e-12, P + R)
    return {k: 100 * (np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2 for k, v in (("P", P), ("R", R), ("F1", F1))}


def uraian(rows, thr=THR):
    """Kelompokkan setiap entitas emas dan setiap prediksi salah ke jenis kesalahannya."""
    g_jenis, f_jenis = Counter(), Counter()
    teks_lepas = Counter(); prob_emas = []
    for x in rows:
        gold = [(g[0], g[1]) for g in x["gold"]]
        pred = [(c[0], c[1]) for c in x["cand"] if (c[3] if thr == THR else c[2] > thr)]
        valid = [p for p in pred if sah(p, x["L"])]
        for (s, e), g in zip(gold, x["gold"]):
            prob_emas.append(g[2])
            if (s, e) in pred: g_jenis["tepat"] += 1
            elif any(ps <= e and s <= pe for ps, pe in valid): g_jenis["batas_meleset"] += 1
            else: g_jenis["terlewat"] += 1
        for p in pred:
            if p in gold: continue
            s, e = p
            if not sah(p, x["L"]):
                f_jenis["tidak_sah_BOS_padding"] += 1
                if s == 0: f_jenis["cek_awal_di_BOS"] += 1
                continue
            if any(gs <= s and e <= ge for gs, ge in gold): f_jenis["di_dalam_entitas"] += 1
            elif any(s <= gs and ge <= e for gs, ge in gold): f_jenis["memuat_entitas"] += 1
            elif any(s <= ge and gs <= e for gs, ge in gold): f_jenis["bersilangan"] += 1
            else:
                f_jenis["lepas_1_subtoken" if s == e else "lepas_2+_subtoken"] += 1
                o = x["off"]
                teks_lepas[x["text"][o[s][0]:o[e][1]].strip()] += 1
    return g_jenis, f_jenis, teks_lepas, np.array(prob_emas)


def recall_seluruh_baris(rows):
    """Recall dipisah: entitas yang menutupi seluruh isi baris vs entitas di dalam kalimat biasa."""
    hit = {True: [0, 0], False: [0, 0]}
    for x in rows:
        a, b = isi_baris(x["text"]); o = x["off"]
        pred = {(c[0], c[1]) for c in x["cand"] if c[3]}
        for g in x["gold"]:
            ga, gb = o[g[0]][0], o[g[1]][1]
            while ga < gb and x["text"][ga].isspace(): ga += 1
            k = (ga, gb) == (a, b)
            hit[k][0] += (g[0], g[1]) in pred; hit[k][1] += 1
    return {k: (100 * v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in hit.items()}


def recall_berkurung(rows):
    """Recall dipisah: entitas yang diakhiri ')' (judul Wikipedia dengan keterangan, misalnya 'دايو ( شركة )') vs lainnya."""
    hit = {True: [0, 0], False: [0, 0]}
    for x in rows:
        o = x["off"]; pred = {(c[0], c[1]) for c in x["cand"] if c[3]}
        for g in x["gold"]:
            k = x["text"][o[g[0]][0]:o[g[1]][1]].rstrip().endswith(")")
            hit[k][0] += (g[0], g[1]) in pred; hit[k][1] += 1
    return {k: (100 * v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in hit.items()}


def analyze(args):
    import pandas as pd
    out = Path(args.out); rep = out / "analisis"; rep.mkdir(exist_ok=True)
    preds = sorted((out / "pred").glob("*.jsonl.gz"))
    tabel, urai, kurva, lepas = [], [], [], {}
    DATA = {}
    for p in preds:
        m, name = p.name.replace(".jsonl.gz", "").split("__")
        rows = baca_pred(p); DATA[(m, name)] = rows
        r = hitung_resmi(rows)
        for strat in ("threshold", "greedy"):
            P, R, F1 = prf(*r[strat])
            tp, fp, tot = per_kalimat(rows, strat=strat)
            ci = bootstrap(tp, fp, tot)
            Ps, _, _ = prf(*(a.sum() for a in per_kalimat(rows, strat=strat, hanya_sah=True)))
            tabel.append(dict(model=m, dataset=name, decoding=strat, P=P, R=R, F1=F1, ci_P=ci["P"], ci_R=ci["R"], ci_F1=ci["F1"],
                              P_tanpa_tidak_sah=Ps, tebakan_per_kalimat=(tp.sum() + fp.sum()) / len(rows), emas_per_kalimat=tot.sum() / len(rows),
                              kalimat=len(rows), kalimat_dinilai_resmi=sum(x["resmi"] for x in rows), entitas_dinilai_resmi=r[strat][2]))
        gj, fj, tl, pe = uraian(rows)
        ng, nf = sum(gj.values()), sum(fj.values())
        urai.append(dict(model=m, dataset=name, **{f"emas_{k}_%": 100 * v / ng for k, v in gj.items()},
                         **{f"salah_{k}_%": 100 * v / max(1, nf) for k, v in fj.items()},
                         prob_emas_median=float(np.median(pe)), emas_prob_0_1_sampai_0_5_pct=100 * float(((pe > 0.1) & (pe <= 0.5)).mean())))
        lepas[f"{m}__{name}"] = tl.most_common(40)
        for thr in np.round(np.arange(0.1, 0.91, 0.05), 2):
            P, R, F1 = prf(*(a.sum() for a in per_kalimat(rows, thr=thr)))
            kurva.append(dict(model=m, dataset=name, threshold=thr, P=P, R=R, F1=F1))
    T = pd.DataFrame(tabel); U = pd.DataFrame(urai); K = pd.DataFrame(kurva)
    T.round(2).to_csv(rep / "metrik_utama.csv", index=False); U.round(2).to_csv(rep / "uraian_kesalahan.csv", index=False)
    K.round(2).to_csv(rep / "kurva_threshold.csv", index=False)
    (rep / "teks_salah_lepas_teratas.json").write_text(json.dumps(lepas, ensure_ascii=False, indent=1))

    # Subset dedup dan tanpa_alih: saring data_id dari prediksi versi lengkap.
    sub = []
    for (m, name), rows in DATA.items():
        for akhiran in ("_dedup", "_tanpa_alih"):
            f = Path(args.data_root) / (name + akhiran)
            if not f.exists(): continue
            keep = {x["data_id"] for x in baca(f)}
            rr = [x for x in rows if x["data_id"] in keep]
            for strat in ("threshold", "greedy"):
                tp, fp, tot = per_kalimat(rr, strat=strat); ci = bootstrap(tp, fp, tot)
                P, R, F1 = prf(tp.sum(), fp.sum(), tot.sum())
                sub.append(dict(model=m, dataset=name + akhiran, decoding=strat, P=P, R=R, F1=F1, ci_R=ci["R"], ci_P=ci["P"], kalimat=len(rr)))
    S = pd.DataFrame(sub); S.round(2).to_csv(rep / "metrik_subset.csv", index=False)

    # Recall per kelompok dari metadata EDA (digabung lewat bahasa, data_id, dan urutan entitas).
    G = []
    meta_p = Path(args.data_root) / "metadata_entitas_nonlatin.csv"
    if meta_p.exists():
        meta = pd.read_csv(meta_p); meta["idx"] = meta.groupby(["bahasa", "data_id"]).cumcount()
        meta["panjang_karakter"] = pd.cut(meta.karakter, [0, 2, 4, 8, 16, 10 ** 6], labels=["1-2", "3-4", "5-8", "9-16", "17+"])
        for (m, name), rows in DATA.items():
            k = re.search(r"WikiANN_(zh|ja|ar)_", name)
            if not k: continue
            hit = pd.DataFrame([dict(data_id=x["data_id"], idx=g[3], kena=any(c[0] == g[0] and c[1] == g[1] and c[3] for c in x["cand"]),
                                     kena_greedy=[g[0], g[1]] in x["greedy"], prob=g[2], subtoken=g[1] - g[0] + 1)
                                for x in rows for g in x["gold"]])
            j = hit.merge(meta[meta.bahasa == k.group(1)], on=["data_id", "idx"], how="left", validate="one_to_one")
            assert j.tipe.notna().all(), f"metadata tidak cocok untuk {name}"
            for kol in ("ada_latin", "aksara_utama", "pengalihan", "tipe", "di_pile", "panjang_karakter"):
                for v, d in j.groupby(kol, observed=True):
                    rb = np.random.default_rng(0).integers(0, len(d), size=(1000, len(d)))
                    rs = 100 * d.kena.to_numpy()[rb].mean(1)
                    G.append(dict(model=m, dataset=name, kelompok=kol, nilai=str(v), entitas=len(d), recall=100 * d.kena.mean(),
                                  ci_95=(np.percentile(rs, 97.5) - np.percentile(rs, 2.5)) / 2, recall_greedy=100 * d.kena_greedy.mean(),
                                  prob_median=d.prob.median()))
    Gd = pd.DataFrame(G); Gd.round(2).to_csv(rep / "recall_per_kelompok.csv", index=False)

    # Baseline: seluruh-baris (dari data) dan peluang acak (dari jumlah span sah), plus recall entitas seluruh-baris vs biasa.
    bl = []
    for (m, name), rows in DATA.items():
        for akhiran in ("", "_dedup", "_tanpa_alih"):
            f = Path(args.data_root) / (name + akhiran)
            if not f.exists(): continue
            P, R, F1 = baseline_seluruh_baris(baca(f))
            keep = {x["data_id"] for x in baca(f)}; rr = [x for x in rows if x["data_id"] in keep]
            emas = sum(len(x["gold"]) for x in rr); kand = sum(kandidat_sah(x["L"]) for x in rr)
            tp, fp, tot = (a.sum() for a in per_kalimat(rr))
            rs = recall_seluruh_baris(rr); rk = recall_berkurung(rr)
            bl.append(dict(model=m, dataset=name + akhiran, baseline_baris_P=P, baseline_baris_R=R, baseline_baris_F1=F1,
                           P_acak=100 * emas / max(1, kand), P_model=prf(tp, fp, tot)[0], lift_P=prf(tp, fp, tot)[0] / max(1e-9, 100 * emas / max(1, kand)),
                           recall_entitas_seluruh_baris=rs[True][0], n_seluruh_baris=rs[True][1], recall_entitas_dalam_kalimat=rs[False][0], n_dalam_kalimat=rs[False][1],
                           recall_entitas_berkurung=rk[True][0], n_berkurung=rk[True][1], recall_entitas_tanpa_kurung=rk[False][0]))
    B = pd.DataFrame(bl); B.round(2).to_csv(rep / "baseline.csv", index=False)

    # Perbandingan berpasangan spasi vs alami (data_id sama), untuk H1.
    pas = []
    for m in {k[0] for k in DATA}:
        for lang in ("zh", "ja"):
            a, b = DATA.get((m, f"WikiANN_{lang}_spasi")), DATA.get((m, f"WikiANN_{lang}_alami"))
            if not a or not b: continue
            ia, ib = {x["data_id"]: x for x in a}, {x["data_id"]: x for x in b}
            ids = sorted(set(ia) & set(ib))
            for strat in ("threshold", "greedy"):
                A = np.stack(per_kalimat([ia[i] for i in ids], strat=strat)); Bm = np.stack(per_kalimat([ib[i] for i in ids], strat=strat))
                rng = np.random.default_rng(0); idx = rng.integers(0, len(ids), size=(1000, len(ids)))
                def met(X, ii):
                    T, F, N = X[0][ii].sum(-1), X[1][ii].sum(-1), X[2][ii].sum(-1)
                    P = T / np.maximum(1, T + F); R = T / np.maximum(1, N); return 100 * P, 100 * R, 100 * 2 * P * R / np.maximum(1e-12, P + R)
                da = [x - y for x, y in zip(met(Bm, idx), met(A, idx))]
                d0 = [x - y for x, y in zip(met(Bm, np.arange(len(ids))), met(A, np.arange(len(ids))))]
                pas.append(dict(model=m, bahasa=lang, decoding=strat, dP_alami_minus_spasi=d0[0], dR=d0[1], dF1=d0[2],
                                **{f"ci95_{n}": (np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2 for n, v in zip(("dP", "dR", "dF1"), da)}))
    Pd = pd.DataFrame(pas); Pd.round(2).to_csv(rep / "spasi_vs_alami.csv", index=False)

    # Ringkasan yang mudah dibaca.
    with open(rep / "ringkasan.txt", "w", encoding="utf8") as f:
        def tulis(judul, df):
            f.write(f"\n== {judul} ==\n{df.to_string(index=False) if len(df) else '(kosong)'}\n")
        kol = ["model", "dataset", "decoding", "P", "R", "F1", "ci_R", "ci_P", "tebakan_per_kalimat", "emas_per_kalimat", "entitas_dinilai_resmi"]
        tulis("Metrik utama (cara hitung resmi: batch penuh, threshold 0.5)", T[kol].round(2))
        ref = pd.DataFrame([dict(dataset=k, **v) for k, v in PAPER.items()])
        tulis("Angka paper (model rilis penulis)", ref)
        tulis("Subset dedup dan tanpa pengalihan", S.round(2))
        tulis("Spasi vs alami (alami dikurangi spasi, bootstrap berpasangan)", Pd.round(2))
        tulis("Baseline: seluruh-baris, peluang acak, dan recall entitas seluruh-baris vs dalam kalimat", B.round(2))
        tulis("Uraian kesalahan (threshold 0.5)", U.round(1))
        if len(K):
            best = K.loc[K.groupby(["model", "dataset"]).F1.idxmax()]
            tulis("Threshold dengan F1 terbaik", best.round(2))
        if len(Gd):
            tulis("Recall per kelompok (threshold 0.5)", Gd.round(1))
    print((rep / "ringkasan.txt").read_text(encoding="utf8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tahap", choices=["siapkan", "cek", "infer", "analyze"])
    ap.add_argument("--out", default="hasil"); ap.add_argument("--data-root", default="data")
    ap.add_argument("--data", default=""); ap.add_argument("--models", default="")
    ap.add_argument("--parquet", default=""); ap.add_argument("--kode", default="en")
    ap.add_argument("--llm", default="meta-llama/Llama-3.2-1B")
    a = ap.parse_args()
    a.data = [d for d in a.data.split(",") if d]; a.models = [m for m in a.models.split(",") if m]
    if a.tahap == "siapkan": siapkan(a.parquet, a.kode, a.data_root)
    elif a.tahap == "cek": Path(a.out).mkdir(parents=True, exist_ok=True); cek(a.data_root, a.data, a.llm, Path(a.out) / "cek_entitas_dinilai.json")
    elif a.tahap == "infer": infer(a)
    else: analyze(a)


if __name__ == "__main__":
    main()
