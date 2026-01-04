import os
import json
import hashlib
import numpy as np
import faiss
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import re

# -------------------------
# Config
# -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STORE_DIR = "store"
USE_GPU = True
DEFAULT_NLIST = 128
DEFAULT_NPROBE = 8
IVF_THRESHOLD = 500
SUMMARY_SENTENCES = 5


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def atomic_write_json(path: str, data: Dict[str, Any]):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def meta_paths(doc_id: str, store_dir: str):
    return os.path.join(store_dir, f"{doc_id}.meta.json"), os.path.join(store_dir, f"{doc_id}.index")

def read_pdf_text(pdf_source) -> str:
    reader = PdfReader(pdf_source if isinstance(pdf_source, str) else pdf_source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def batch_encode(model, sentences: List[str]) -> np.ndarray:
    return model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True).astype("float32")


# -------------------------
# Metadata utilities
# -------------------------
def save_meta(meta_path: str, sentences: List[str], ids: List[int], config: Dict[str, Any]):
    atomic_write_json(meta_path, {"sentences": sentences, "ids": ids, "config": config})

def load_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def stable_doc_id(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{h}"


# -------------------------
# Index utilities
# -------------------------
def _maybe_to_gpu(index: faiss.Index) -> faiss.Index:
    if USE_GPU:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            return index
    return index

def build_index(embeddings: np.ndarray, use_ivf: bool = False, nlist: int = DEFAULT_NLIST, nprobe: int = DEFAULT_NPROBE) -> faiss.IndexIDMap2:
    dim = embeddings.shape[1]
    if not use_ivf or embeddings.shape[0] < 2 * nlist:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    try:
        index_ivf.train(embeddings)
        index_ivf.nprobe = max(1, min(nprobe, nlist))
        return faiss.IndexIDMap2(index_ivf)
    except Exception:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

def save_index(index: faiss.Index, path: str):
    cpu_index = faiss.index_gpu_to_cpu(index) if USE_GPU else index
    faiss.write_index(cpu_index, path)

def load_index(path: str) -> Optional[faiss.Index]:
    return faiss.read_index(path) if os.path.exists(path) else None


# -------------------------
# Analyzer
# -------------------------
class EfficientPDFAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, store_dir: str = STORE_DIR):
        self.model = SentenceTransformer(model_name)
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

    def index_pdf(self, pdf_source, doc_id: Optional[str] = None, reindex: bool = False) -> dict:
        inferred_id = stable_doc_id(getattr(pdf_source, "name", pdf_source)) if isinstance(pdf_source, (str, type(""))) else "uploaded-pdf"
        doc_id = doc_id or inferred_id
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)

        if not reindex:
            meta, index = load_meta(meta_path), load_index(idx_path)
            if index is not None and meta.get("sentences"):
                return {"status": "loaded", "doc_id": doc_id, "count": len(meta["sentences"]), "index_type": meta.get("config", {}).get("index_type", "Flat")}

        sentences = split_sentences(read_pdf_text(pdf_source))
        if not sentences:
            raise ValueError("No readable sentences extracted")

        embeddings = batch_encode(self.model, sentences)
        use_ivf = len(sentences) > IVF_THRESHOLD
        nlist = min(1024, max(64, int(len(sentences) ** 0.5) * 4))
        nprobe = min(64, max(8, nlist // 16)) if use_ivf else 0

        base_index = build_index(embeddings, use_ivf=use_ivf, nlist=nlist, nprobe=nprobe)
        index = _maybe_to_gpu(base_index)
        ids = np.arange(len(sentences), dtype="int64")
        index.add_with_ids(embeddings, ids)

        save_index(base_index, idx_path)
        save_meta(meta_path, sentences, ids.tolist(), {"index_type": "IVF" if use_ivf else "Flat", "nlist": nlist, "nprobe": nprobe})
        return {"status": "indexed", "doc_id": doc_id, "count": len(sentences), "index_type": "IVF" if use_ivf else "Flat"}

    def search(self, query: str, doc_id: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        meta_path, idx_path = meta_paths(doc_id, self.store_dir)
        meta, sentences = load_meta(meta_path), load_meta(meta_path).get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        index = load_index(idx_path)
        if index is None:
            raise ValueError("Index file missing or unreadable")

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        k = min(top_k, index.ntotal)
        if k == 0: return []
        D, I = index.search(q_emb, k)
        return [(int(idx), float(score), sentences[int(idx)]) for score, idx in zip(D[0], I[0]) if idx != -1 and 0 <= idx < len(sentences)]

    def extractive_summary(self, doc_id: str, num_sentences: int = SUMMARY_SENTENCES) -> str:
        meta_path, _ = meta_paths(doc_id, self.store_dir)
        sentences = load_meta(meta_path).get("sentences", [])
        if not sentences:
            raise ValueError("Document not indexed")
        embeddings = batch_encode(self.model, sentences)
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(centroid)
        tmp = faiss.IndexFlatIP(embeddings.shape[1])
        tmp.add(embeddings)
        _, I = tmp.search(centroid, min(num_sentences, len(sentences)))
        return " ".join(sentences[int(i)] for i in I[0])

    def generate_answer(self, question: str, doc_id: str, top_k: int = 3, use_generator: bool = False) -> str:
        retrieved = self.search(question, doc_id, top_k=top_k)
        if not retrieved: return "No relevant passages found."
        context = "\n\n".join([r[2] for r in retrieved])
        return f"[GENERATIVE ANSWER PLACEHOLDER]\n\nContext:\n{context}" if use_generator else context


# -------------------------
# Gradio UI
# -------------------------
analyzer = EfficientPDFAnalyzer()

def ui_load_pdf(file_obj):
    if file_obj is None: return "❌ Please upload a PDF", ""
    try:
        meta = analyzer.index_pdf(file_obj, reindex=False)
        return f"✅ {meta['status'].capitalize()}. {meta['count']} sentences indexed. Index type: {meta.get('index_type
