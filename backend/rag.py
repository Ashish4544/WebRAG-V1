import requests
import trafilatura
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class WebRAG:

    def __init__(self):

        # ==========================================
        # Embedding model
        # ==========================================

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # ==========================================
        # Text splitter
        # ==========================================

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300
        )

        # ==========================================
        # Cross-encoder reranker
        # ==========================================

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


    def load_webpage(self, url):
        """Fetch and extract meaningful content from a webpage."""

        response = requests.get(
            url,
            headers={
                "User-Agent": "WebRAG/1.0"
            },
            timeout=15
        )

        response.raise_for_status()

        text = trafilatura.extract(
            response.text,
            include_links=False,
            include_images=False,
            include_tables=True
        )

        if not text:
            raise ValueError(
                "Could not extract webpage content"
            )

        return text


    def create_chunks(self, text):
        """Split webpage text into overlapping chunks."""

        chunks = self.splitter.create_documents(
            [text]
        )

        return chunks


    def create_indexes(self, chunks):
        """Create FAISS and BM25 indexes."""

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        # ==========================================
        # Create embeddings
        # ==========================================

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        # ==========================================
        # FAISS index
        # ==========================================

        dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatIP(
            dimension
        )

        faiss_index.add(
            embeddings
        )

        # ==========================================
        # BM25 index
        # ==========================================

        tokenized_chunks = [
            text.lower().split()
            for text in texts
        ]

        bm25_index = BM25Okapi(
            tokenized_chunks
        )

        return (
            texts,
            faiss_index,
            bm25_index
        )


    def retrieve(self, url, query, k=5):
        """
        Retrieve relevant webpage chunks using:

        FAISS + BM25
            ↓
        RRF
            ↓
        Cross-encoder reranking
            ↓
        Top-k results
        """

        # ==========================================
        # 1. Load webpage
        # ==========================================

        text = self.load_webpage(
            url
        )

        # ==========================================
        # 2. Create chunks
        # ==========================================

        chunks = self.create_chunks(
            text
        )

        if not chunks:
            return []

        # ==========================================
        # 3. Create indexes
        # ==========================================

        (
            texts,
            faiss_index,
            bm25_index
        ) = self.create_indexes(
            chunks
        )

        # ==========================================
        # Candidate count
        # ==========================================

        # Retrieve more candidates than we finally return.
        # These candidates will be reranked later.

        candidate_k = min(
            10,
            len(texts)
        )

        # ==========================================
        # 4. FAISS retrieval
        # ==========================================

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        _, faiss_indices = faiss_index.search(
            query_embedding,
            candidate_k
        )

        # ==========================================
        # 5. BM25 retrieval
        # ==========================================

        tokenized_query = query.lower().split()

        bm25_scores = bm25_index.get_scores(
            tokenized_query
        )

        bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:candidate_k]

        # ==========================================
        # 6. Reciprocal Rank Fusion
        # ==========================================

        rrf_scores = {}

        rrf_k = 60

        # ------------------------------------------
        # FAISS ranking
        # ------------------------------------------

        for rank, idx in enumerate(
            faiss_indices[0],
            start=1
        ):

            rrf_scores[idx] = (
                rrf_scores.get(idx, 0)
                + 1 / (rrf_k + rank)
            )

        # ------------------------------------------
        # BM25 ranking
        # ------------------------------------------

        for rank, idx in enumerate(
            bm25_indices,
            start=1
        ):

            rrf_scores[idx] = (
                rrf_scores.get(idx, 0)
                + 1 / (rrf_k + rank)
            )

        # ==========================================
        # 7. Sort by RRF score
        # ==========================================

        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # ==========================================
        # 8. Select candidates for reranking
        # ==========================================

        candidate_indices = [
            idx
            for idx, _ in ranked_results[:candidate_k]
        ]

        # ==========================================
        # 9. Cross-encoder reranking
        # ==========================================

        pairs = [
            [query, texts[idx]]
            for idx in candidate_indices
        ]

        reranker_scores = self.reranker.predict(
            pairs
        )

        # ==========================================
        # 10. Sort by reranker score
        # ==========================================

        reranked = sorted(
            zip(
                candidate_indices,
                reranker_scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        # ==========================================
        # 11. Return final top-k results
        # ==========================================

        results = []

        final_k = min(
            k,
            len(reranked)
        )

        for idx, score in reranked[:final_k]:

            results.append({
                "text": texts[idx],
                "reranker_score": float(score),
                "source": url
            })

        return results