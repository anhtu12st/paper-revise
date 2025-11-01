"""
Chunked Dataset Loader
=======================

Efficient loader for chunked preprocessed datasets.

Features:
- Fast loading from Arrow chunks (2-5 min vs 30-60 min preprocessing)
- Streaming support for large datasets
- Partial loading (test with few examples, train with full dataset)
- Compatible with existing TimeStepMajorDataLoader
- Memory-efficient (only loads needed chunks)

Usage:
    from memxlnet.data import ChunkedDataset

    # Load first 1000 examples (instant)
    dataset = ChunkedDataset.from_manifest(
        "./preprocessed_data/squad_v2/train_manifest.json",
        mode="first_n",
        num_examples=1000
    )

    # Load full dataset (streaming)
    dataset = ChunkedDataset.from_manifest(
        "./preprocessed_data/squad_v2/train_manifest.json",
        mode="streaming"
    )

    # Load specific chunks
    dataset = ChunkedDataset.from_manifest(
        "./preprocessed_data/squad_v2/train_manifest.json",
        mode="chunks",
        chunk_indices=[0, 1, 2]
    )
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk

logger = logging.getLogger(__name__)


class DocumentMetadata(TypedDict):
    """Metadata for a document in streaming mode."""

    doc_idx: int
    chunk_id: int
    loaded: bool


class ChunkedDataset:
    """Dataset that loads from chunked Arrow files with flexible loading modes.

    Note: This dataset flattens documents into individual segments for compatibility
    with TimeStepMajorDataLoader. Each index corresponds to a single segment, and
    document_map tracks which segments belong to which document.
    """

    documents: list[list[dict[str, Any]]]  # For non-streaming modes
    features: list[dict[str, Any]]  # Flattened segments for TimeStepMajorDataLoader
    document_map: dict[str, list[int]]  # Maps doc_id -> list of segment indices

    # Streaming mode attributes
    _doc_metadata: dict[str, DocumentMetadata]  # Document metadata for lazy loading
    _loaded_chunks: dict[Any, list[list[dict[str, Any]]]]  # Cached loaded chunks
    _document_to_chunk: dict[int, Any]  # Maps doc index to chunk ID
    _total_documents: int  # Total number of documents in streaming mode
    _max_loaded_chunks: int  # LRU cache size limit
    _chunk_access_order: list[Any]  # LRU tracking

    def __init__(
        self,
        manifest_path: str | Path,
        mode: str = "streaming",
        num_examples: int | None = None,
        chunk_indices: list[int] | None = None,
        max_n_segs: int | None = None,
    ):
        """Initialize chunked dataset.

        Args:
            manifest_path: Path to manifest.json file
            mode: Loading mode: "streaming", "first_n", "chunks", "full"
            num_examples: Number of examples to load (for "first_n" mode)
            chunk_indices: List of chunk indices to load (for "chunks" mode)
            max_n_segs: Maximum segments per document (for filtering)
        """
        self.manifest_path = Path(manifest_path)
        self.mode = mode
        self.num_examples = num_examples
        self.chunk_indices = chunk_indices
        self.max_n_segs = max_n_segs

        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

        self.dataset_name = self.manifest["dataset_name"]
        self.split = self.manifest["split"]
        self.chunks = self.manifest["chunks"]
        self.total_examples = self.manifest["total_examples"]

        # Base directory for chunks
        self.base_dir = self.manifest_path.parent

        # Initialize document_map
        self.document_map = {}

        # Load data based on mode
        self._load_data()

        # Flatten documents into features and build document_map
        self._flatten_to_features()

        logger.info(
            f"✅ ChunkedDataset loaded: {len(self.features)} segments, {len(self.document_map)} documents, mode={mode}"
        )

    def _load_data(self):
        """Load data based on specified mode."""
        if self.mode == "streaming":
            # Lazy loading - load chunks on demand
            self._init_streaming()
        elif self.mode == "first_n":
            # Load first N examples
            self._load_first_n()
        elif self.mode == "chunks":
            # Load specific chunks
            self._load_specific_chunks()
        elif self.mode == "full":
            # Load all chunks into memory
            self._load_all()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _init_streaming(self) -> None:
        """Initialize streaming mode (lazy loading)."""
        # Don't load any data yet - load on-the-fly in __getitem__
        self._loaded_chunks: dict[Any, list[list[dict[str, Any]]]] = {}  # Cache for loaded chunks
        self._document_to_chunk: dict[int, Any] = {}  # Map document index to chunk

        # Build document-to-chunk mapping
        doc_idx = 0
        for chunk in self.chunks:
            for _ in range(chunk["num_documents"]):
                self._document_to_chunk[doc_idx] = chunk["chunk_id"]
                doc_idx += 1

        self._total_documents: int = doc_idx

        # Memory management for streaming mode
        self._max_loaded_chunks = 5  # Keep at most 5 chunks in memory
        self._chunk_access_order: list[Any] = []  # Track access order for LRU eviction

        logger.info(f"📊 Streaming mode: {self._total_documents} documents across {len(self.chunks)} chunks")
        logger.info(f"💾 Chunk cache limit: {self._max_loaded_chunks} chunks")

    def _load_first_n(self) -> None:
        """Load first N examples."""
        if self.num_examples is None:
            raise ValueError("num_examples must be specified for 'first_n' mode")

        logger.info(f"📥 Loading first {self.num_examples} documents...")

        documents: list[list[dict[str, Any]]] = []
        docs_loaded = 0

        for chunk in self.chunks:
            if docs_loaded >= self.num_examples:
                break

            # Load chunk
            chunk_path = self.base_dir / chunk["path"]
            chunk_data = self._load_chunk(chunk_path)

            # Group segments by document
            chunk_documents = self._group_segments_to_documents(chunk_data)

            # Add documents until we reach the limit
            for doc in chunk_documents:
                if docs_loaded >= self.num_examples:
                    break
                documents.append(doc)
                docs_loaded += 1

            logger.info(f"  Loaded chunk {chunk['chunk_id']}: {docs_loaded}/{self.num_examples} documents")

        self.documents = documents
        logger.info(f"✅ Loaded {len(self.documents)} documents")

    def _load_specific_chunks(self) -> None:
        """Load specific chunks by index."""
        if self.chunk_indices is None:
            raise ValueError("chunk_indices must be specified for 'chunks' mode")

        logger.info(f"📥 Loading chunks: {self.chunk_indices}...")

        documents: list[list[dict[str, Any]]] = []
        for chunk_idx in self.chunk_indices:
            # Find chunk in manifest
            chunk = next((c for c in self.chunks if c["chunk_id"] == chunk_idx), None)
            if chunk is None:
                logger.warning(f"⚠️ Chunk {chunk_idx} not found in manifest")
                continue

            # Load chunk
            chunk_path = self.base_dir / chunk["path"]
            chunk_data = self._load_chunk(chunk_path)

            # Group segments to documents
            chunk_documents = self._group_segments_to_documents(chunk_data)
            documents.extend(chunk_documents)

            logger.info(f"  Loaded chunk {chunk_idx}: {len(chunk_documents)} documents")

        self.documents = documents
        logger.info(f"✅ Loaded {len(self.documents)} documents from {len(self.chunk_indices)} chunks")

    def _load_all(self) -> None:
        """Load all chunks into memory."""
        logger.info(f"📥 Loading all {len(self.chunks)} chunks...")

        documents: list[list[dict[str, Any]]] = []
        for chunk in self.chunks:
            chunk_path = self.base_dir / chunk["path"]
            chunk_data = self._load_chunk(chunk_path)
            chunk_documents = self._group_segments_to_documents(chunk_data)
            documents.extend(chunk_documents)

            logger.info(f"  Loaded chunk {chunk['chunk_id']}: {len(chunk_documents)} documents")

        self.documents = documents
        logger.info(f"✅ Loaded {len(self.documents)} documents total")

    def _load_chunk(self, chunk_path: Path) -> HFDataset:
        """Load a single chunk from disk.

        Args:
            chunk_path: Path to chunk directory

        Returns:
            HuggingFace dataset
        """
        try:
            return load_from_disk(str(chunk_path))
        except Exception as e:
            logger.error(f"❌ Failed to load chunk {chunk_path}: {e}")
            raise

    def _group_segments_to_documents(self, chunk_data: HFDataset) -> list[list[dict[str, Any]]]:
        """Group segments by document ID.

        Args:
            chunk_data: HuggingFace dataset with flattened segments

        Returns:
            List of documents, where each document is a list of segments
        """
        documents_dict: dict[Any, list[dict[str, Any]]] = defaultdict(list)

        for i in range(len(chunk_data)):
            item = chunk_data[i]
            doc_id = item["document_id"]

            segment = {
                "example_id": item["example_id"],
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "token_type_ids": item["token_type_ids"],
                "offset_mapping": item["offset_mapping"],
                "context": item["context"],
                "start_positions": item["start_positions"],
                "end_positions": item["end_positions"],
                # Metadata fields for evaluation
                "question": item.get("question", ""),
                "cls_index": item.get("cls_index", 0),
                "has_answer": item.get("has_answer", False),
                "chosen_answer_text": item.get("chosen_answer_text", ""),
                "chosen_answer_char_span": item.get("chosen_answer_char_span", [-1, -1]),
                # Segment selection metadata
                "segment_index_in_doc": item.get("segment_index_in_doc", 0),
                "has_answer_in_segment": item.get("has_answer_in_segment", False),
            }

            documents_dict[doc_id].append(segment)

        # Convert to list and sort by document ID
        documents = [documents_dict[doc_id] for doc_id in sorted(documents_dict.keys())]

        # Note: max_n_segs is no longer applied here for truncation
        # Segment selection will happen at dataloader level via SegmentSelector
        # Keep all segments to allow smart selection (answer-centered, etc.)

        return documents

    def _flatten_to_features(self) -> None:
        """Flatten documents into individual segments for TimeStepMajorDataLoader compatibility."""
        self.features = []
        self.document_map = {}

        if self.mode == "streaming":
            # For streaming mode, build lightweight document map without loading all features
            # Features will be loaded on-demand in TimeStepMajorDataLoader
            logger.info(f"📊 Streaming mode: Building document map for {self._total_documents} documents...")

            # Build document map by reading chunk manifests only (no data loading)
            self._doc_metadata = {}  # Initialize metadata storage
            for doc_idx in range(self._total_documents):
                doc_id = f"doc_{doc_idx}"

                # Find which chunk this document is in
                chunk_id = self._document_to_chunk.get(doc_idx)
                if chunk_id is None:
                    logger.warning(f"⚠️ Document {doc_idx} has no chunk mapping")
                    continue

                chunk = next((c for c in self.chunks if c["chunk_id"] == chunk_id), None)
                if chunk is None:
                    logger.warning(f"⚠️ Document {doc_idx} not found in any chunk")
                    continue

                # Create placeholder mapping that will be filled on first access
                self.document_map[doc_id] = []

                # Store metadata for lazy loading
                self._doc_metadata[doc_id] = {
                    "doc_idx": doc_idx,
                    "chunk_id": chunk_id,
                    "loaded": False,
                }

            logger.info(f"✅ Document map built: {len(self.document_map)} documents (features will load on-demand)")

        else:
            # For non-streaming modes, flatten pre-loaded documents
            for doc_idx, doc_segments in enumerate(self.documents):
                doc_id = f"doc_{doc_idx}"
                self.document_map[doc_id] = []

                for segment in doc_segments:
                    segment_idx = len(self.features)
                    self.features.append(segment)
                    self.document_map[doc_id].append(segment_idx)

    def _get_document_by_index(self, idx: int) -> list[dict[str, Any]]:
        """Get a document by index (internal method for streaming mode).

        Args:
            idx: Document index

        Returns:
            List of segments for the document
        """
        if self.mode == "streaming":
            # Lazy load the chunk containing this document
            chunk_id = self._document_to_chunk.get(idx)
            if chunk_id is None:
                raise IndexError(f"Document index {idx} out of range")

            # Check if chunk is already loaded
            if chunk_id not in self._loaded_chunks:
                # Implement LRU eviction if we hit the cache limit
                if len(self._loaded_chunks) >= self._max_loaded_chunks:
                    # Remove oldest chunk (first in access order)
                    oldest_chunk_id = self._chunk_access_order.pop(0)
                    if oldest_chunk_id in self._loaded_chunks:
                        del self._loaded_chunks[oldest_chunk_id]
                        logger.debug(f"💾 Evicted chunk {oldest_chunk_id} from cache (LRU)")

                chunk = next((c for c in self.chunks if c["chunk_id"] == chunk_id), None)
                if chunk is None:
                    raise ValueError(f"Chunk {chunk_id} not found in manifest")

                chunk_path = self.base_dir / chunk["path"]
                chunk_data = self._load_chunk(chunk_path)
                self._loaded_chunks[chunk_id] = self._group_segments_to_documents(chunk_data)

                logger.debug(
                    f"📥 Loaded chunk {chunk_id} on-demand (cache: {len(self._loaded_chunks)}/{self._max_loaded_chunks})"
                )

            # Update access order for LRU
            if chunk_id in self._chunk_access_order:
                self._chunk_access_order.remove(chunk_id)
            self._chunk_access_order.append(chunk_id)

            # Find document within chunk
            chunk = next(c for c in self.chunks if c["chunk_id"] == chunk_id)
            doc_start: int = chunk["document_range"][0]
            local_idx = idx - doc_start

            return self._loaded_chunks[chunk_id][local_idx]
        else:
            # Direct access for pre-loaded data
            return self.documents[idx]

    def __len__(self) -> int:
        """Return number of segments.

        Note: In streaming mode, this returns the number of documents (not segments)
        since segments are loaded on-demand. TimeStepMajorDataLoader works at
        document level, so this is fine.
        """
        if self.mode == "streaming":
            # Return number of documents for streaming mode
            # TimeStepMajorDataLoader iterates over documents, not individual segments
            return len(self.document_map)
        else:
            return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single segment by index.

        Args:
            idx: Segment index

        Returns:
            Single segment (feature dictionary) with tensors
        """
        feature = self.features[idx]

        # Convert to format expected by TimeStepMajorDataLoader
        # (similar to SquadLikeQADataset.__getitem__)
        item = {}

        # Convert tensor fields from lists to tensors
        tensor_keys = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
        for key in tensor_keys:
            if key in feature:
                if isinstance(feature[key], torch.Tensor):
                    item[key] = feature[key]
                else:
                    item[key] = torch.tensor(feature[key])

        # Keep metadata as-is
        metadata_keys = [
            "example_id",
            "segment_index",
            "total_segments",
            "offset_mapping",
            "context",
            "question",
            "cls_index",
            "has_answer",
            "chosen_answer_text",
            "chosen_answer_char_span",
            "segment_index_in_doc",  # Position within document (for segment selection)
            "has_answer_in_segment",  # Whether this segment contains the answer
            "boundary_info",
        ]
        for key in metadata_keys:
            if key in feature:
                item[key] = feature[key]

        return item

    def get_all_documents(self) -> list[str]:
        """Get all document IDs for TimeStepMajorDataLoader compatibility.

        Returns:
            List of document identifiers
        """
        return list(self.document_map.keys())

    def get_document_segments(self, example_id: str) -> list[int]:
        """Get segment indices for a document for TimeStepMajorDataLoader compatibility.

        Args:
            example_id: Document identifier (e.g., "doc_0")

        Returns:
            List of segment indices belonging to this document
        """
        if self.mode == "streaming":
            # Lazy load document segments on first access
            if hasattr(self, "_doc_metadata") and example_id in self._doc_metadata:
                metadata = self._doc_metadata[example_id]
                if not metadata["loaded"]:
                    # Load this document's segments
                    doc_idx = metadata["doc_idx"]
                    doc_segments = self._get_document_by_index(doc_idx)

                    # Add segments to features list and update document_map
                    segment_indices = []
                    for segment in doc_segments:
                        segment_idx = len(self.features)
                        self.features.append(segment)
                        segment_indices.append(segment_idx)

                    self.document_map[example_id] = segment_indices
                    metadata["loaded"] = True

        return list(self.document_map.get(example_id, []))

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        split: str | None = None,
        mode: str = "streaming",
        num_examples: int | None = None,
        chunk_indices: list[int] | None = None,
        max_n_segs: int | None = None,
    ) -> "ChunkedDataset":
        """Create ChunkedDataset from manifest file.

        Args:
            manifest_path: Path to manifest.json (main or split-specific)
            split: Split name (train/validation) if using main manifest
            mode: Loading mode: "streaming", "first_n", "chunks", "full"
            num_examples: Number of examples (for "first_n" mode)
            chunk_indices: Chunk indices (for "chunks" mode)
            max_n_segs: Maximum segments per document

        Returns:
            ChunkedDataset instance
        """
        manifest_path = Path(manifest_path)

        # If main manifest provided, get split-specific manifest
        if split is not None:
            with open(manifest_path) as f:
                main_manifest = json.load(f)

            # Check if this is main manifest
            if "splits" in main_manifest:
                # This is main manifest, get split-specific one
                split_manifest_path = manifest_path.parent / f"{split}_manifest.json"
                if not split_manifest_path.exists():
                    raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
                manifest_path = split_manifest_path

        result: ChunkedDataset = cls(
            manifest_path=manifest_path,
            mode=mode,
            num_examples=num_examples,
            chunk_indices=chunk_indices,
            max_n_segs=max_n_segs,
        )
        return result


# Convenience function for backward compatibility
def load_chunked_dataset(
    dataset_dir: str | Path,
    split: str = "train",
    mode: str = "streaming",
    num_examples: int | None = None,
    chunk_indices: list[int] | None = None,
    max_n_segs: int | None = None,
) -> ChunkedDataset:
    """Load chunked dataset from directory.

    Args:
        dataset_dir: Directory containing manifest.json and chunks
        split: Split name (train/validation)
        mode: Loading mode
        num_examples: Number of examples (for "first_n" mode)
        chunk_indices: Chunk indices (for "chunks" mode)
        max_n_segs: Maximum segments per document

    Returns:
        ChunkedDataset instance
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return ChunkedDataset.from_manifest(
        manifest_path=manifest_path,
        split=split,
        mode=mode,
        num_examples=num_examples,
        chunk_indices=chunk_indices,
        max_n_segs=max_n_segs,
    )
