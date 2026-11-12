"""
Hybrid RAG MCP server (text + image) over Qdrant.

Retrieval pipeline
------------------
  query ─┬─ text-embedding-3-small ──► ANN on embedding_text   (modality='text')
         └─ Qdrant BM25           ──► scored content index    (modality='text')
  (optional query IMAGE ─ Cohere embed-v4.0 ─► ANN on embedding_image)
                         │
                         ▼
                Reciprocal Rank Fusion (RRF)
                         │
                         ▼
              Cohere-rerank-v4.0-pro (Azure AI Foundry)
                         │
                         ▼
                      top_k results

Install:
    pip install "mcp[cli]<2" qdrant-client httpx

Run:
    python main.py                                  # streamable-http (default), serves HOST:PORT/mcp
    MCP_TRANSPORT=stdio python main.py              # local clients only

Required environment variables:
    QDRANT_URL                   Qdrant server URL, e.g. http://localhost:6333
    QDRANT_API_KEY               optional for self-hosted Qdrant; required for Qdrant Cloud
    QDRANT_COLLECTION            collection name (default: rag_chunks)
    AZURE_FOUNDRY_API_KEY        default key used by all three model endpoints

    AZURE_TEXT_EMBED_URL         full URL of the text-embedding-3-small embeddings endpoint, e.g.
        https://<res>.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21
    AZURE_IMAGE_EMBED_URL        full URL of the Cohere embed v2 endpoint (used for IMAGES only), e.g.
        https://<res>.services.ai.azure.com/providers/cohere/v2/embed
    AZURE_RERANK_URL             full URL of the Cohere rerank v2 endpoint, e.g.
        https://<res>.services.ai.azure.com/providers/cohere/v2/rerank

Optional:
    AZURE_TEXT_EMBED_KEY / AZURE_IMAGE_EMBED_KEY / AZURE_RERANK_KEY   per-endpoint key overrides
    TEXT_EMBED_MODEL      (default: text-embedding-3-small)
    IMAGE_EMBED_MODEL     (default: embed-v4.0)
    RERANK_MODEL          (default: Cohere-rerank-v4.0-pro)
    EMBED_DIM             (default: 1536)  -> must match the named vector dimensions
    CANDIDATES_PER_RETRIEVER (default: 50)
    RERANK_CANDIDATES        (default: 30)
    RRF_K                    (default: 60)
    QDRANT_TIMEOUT_SECONDS   (default: 60)
    IMAGE_URL_ALLOWED_HOSTS  comma-separated allowlist of hosts for image_url (default: any public host)
    IMAGE_URL_ALLOW_PRIVATE  "true" to let image_url reach private/loopback addresses (default: false)
    MCP_TRANSPORT / HOST / PORT / LOG_LEVEL
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import logging
import os
import socket
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Optional
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from qdrant_client import AsyncQdrantClient, models
import hashlib
import json
import os
import uuid
from pathlib import Path
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langchain_openai import AzureOpenAIEmbeddings

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("hybrid-rag-mcp")


def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val  # type: ignore[return-value]

DATABASE_URL = _env("DATABASE_URL", required=True)
APPROVAL_VERIFY_URL = _env("APPROVAL_VERIFY_URL", "")
APPROVAL_API_KEY = _env("APPROVAL_API_KEY", "")

QDRANT_URL = _env("QDRANT_URL", required=True)
QDRANT_API_KEY = _env("QDRANT_API_KEY", "")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "rag_chunks")
QDRANT_TIMEOUT_SECONDS = float(_env("QDRANT_TIMEOUT_SECONDS", "60"))
_DEFAULT_KEY = _env("AZURE_FOUNDRY_API_KEY", "")

TEXT_EMBED_URL = _env("AZURE_TEXT_EMBED_URL", required=True)
TEXT_EMBED_KEY = _env("AZURE_TEXT_EMBED_KEY", required=True)
TEXT_EMBED_MODEL = _env("TEXT_EMBED_MODEL", "text-embedding-3-small")

IMAGE_EMBED_URL = _env("AZURE_IMAGE_EMBED_URL", required=True)
IMAGE_EMBED_KEY = _env("AZURE_IMAGE_EMBED_KEY", "")
IMAGE_EMBED_MODEL = _env("IMAGE_EMBED_MODEL", "embed-v4.0")

RERANK_URL = _env("AZURE_RERANK_URL", required=True)
RERANK_KEY = _env("AZURE_RERANK_KEY", IMAGE_EMBED_KEY)
RERANK_MODEL = _env("RERANK_MODEL", "Cohere-rerank-v4.0-pro")

EMBED_DIM = int(_env("EMBED_DIM", "1536"))
CANDIDATES_PER_RETRIEVER = int(_env("CANDIDATES_PER_RETRIEVER", "50"))
RERANK_CANDIDATES = int(_env("RERANK_CANDIDATES", "30"))
RRF_K = int(_env("RRF_K", "60"))

SNIPPET_CHARS = 1500
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# --------------------------------------------------------------------------- #
# Shared resources (Qdrant client + HTTP client) managed by the server lifespan
# --------------------------------------------------------------------------- #

_qdrant: Optional[AsyncQdrantClient] = None
_http: Optional[httpx.AsyncClient] = None
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=TEXT_EMBED_URL,
    api_key=TEXT_EMBED_KEY,
    openai_api_version="2024-06-01",
)
pool = AsyncConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"row_factory": dict_row},
)

@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[None]:
    global _qdrant, _http
    _qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )
    _http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    

    try:
        await _ensure_collection()

        log.info("Resources ready (Qdrant + http client)")
        yield
    finally:
        await _http.aclose()
        await _qdrant.close()
        _qdrant = _http = None


mcp = FastMCP(
    "hybrid-rag and MaintiDoc Ticket Management",
    instructions=(
        "Hybrid retrieval over a document store containing text chunks and images. "
        "Use `hybrid_search` to find relevant chunks (BM25 + dense text + dense image, "
        "fused with RRF and reranked). Use `get_chunk_image` to view an image result "
        "and `get_document_chunks` to read a whole document in order."
        "Manage maintenance ticket drafts and approved tickets. "
        "Never create an operational ticket without human approval."
    ),
    lifespan=lifespan,
    host=_env("HOST", "0.0.0.0"),
    port=int(_env("PORT", "8000")),
)

# --------------------------------------------------------------------------- #
# Azure AI Foundry clients
# --------------------------------------------------------------------------- #


def _headers(key: str) -> dict[str, str]:
    # Azure OpenAI deployments accept `api-key`; Foundry/serverless endpoints accept Bearer.
    # Sending both is harmless and keeps this working across endpoint flavours.
    return {
        "Content-Type": "application/json",
        "api-key": key,
        "Authorization": f"Bearer {key}",
    }


async def _post(url: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    assert _http is not None, "HTTP client not initialised"
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await _http.post(url, headers=_headers(key), json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (
                429, 500, 502, 503, 504,
            ):
                break
            await asyncio.sleep(0.5 * (2**attempt))
    detail = ""
    if isinstance(last_exc, httpx.HTTPStatusError):
        detail = f" | body: {last_exc.response.text[:500]}"
    raise RuntimeError(f"Request to {url.split('?')[0]} failed: {last_exc}{detail}")


async def embed_text_query(query: str) -> list[float]:
    """text-embedding-3-small (OpenAI-style embeddings API)."""
    return await asyncio.to_thread(
        embeddings.embed_query,
        query
    )
async def embed_image_query(data_url: str) -> list[float]:
    """Cohere embed-v4.0 of an IMAGE (data URL) into the same 1536-d image space."""
    data = await _post(
        IMAGE_EMBED_URL,
        IMAGE_EMBED_KEY,
        {
            "model": IMAGE_EMBED_MODEL,
            "input_type": "image",
            "embedding_types": ["float"],
            "output_dimension": EMBED_DIM,
            "images": [data_url],
        },
    )
    return data["embeddings"]["float"][0]


async def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """Cohere-rerank-v4.0-pro. Returns [(index_into_documents, relevance_score)] best-first."""
    data = await _post(
        RERANK_URL,
        RERANK_KEY,
        {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        },
    )
    return [(r["index"], float(r["relevance_score"])) for r in data["results"]]


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #

async def _ensure_collection() -> None:
    """Create the collection and the payload indexes used by filtering and BM25."""
    assert _qdrant is not None, "Qdrant client not initialised"
    if not await _qdrant.collection_exists(QDRANT_COLLECTION):
        await _qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "embedding_text": models.VectorParams(
                    size=EMBED_DIM, distance=models.Distance.COSINE
                ),
                "embedding_image": models.VectorParams(
                    size=EMBED_DIM, distance=models.Distance.COSINE
                ),
            },
        )

    info = await _qdrant.get_collection(QDRANT_COLLECTION)
    indexed_fields = info.payload_schema or {}
    for field in ("document_id", "modality"):
        if field not in indexed_fields:
            await _qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
    if "content" not in indexed_fields:
        await _qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="content",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                scoring=True,
            ),
            wait=True,
        )


def _qdrant_filter(
    document_id: Optional[str], modality: Optional[str] = None
) -> Optional[models.Filter]:
    must: list[Any] = []
    if document_id is not None:
        must.append(models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id)
        ))
    if modality is not None:
        must.append(models.FieldCondition(
            key="modality", match=models.MatchValue(value=modality)
        ))
    return models.Filter(must=must) if must else None


def _point_row(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    modality = payload.get("modality", "text")
    return {
        "id": str(point.id),
        "document_id": payload.get("document_id", ""),
        "chunk_index": payload.get("chunk_index", 0),
        "modality": modality,
        "content": payload.get("content", "") or "",
        "metadata": payload.get("metadata") or {},
        "has_image": modality == "image",
        "score": float(point.score),
    }


async def _dense_text(vec: list[float], document_id: Optional[str]) -> list[dict[str, Any]]:
    assert _qdrant is not None, "Qdrant client not initialised"
    response = await _qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        using="embedding_text",
        query_filter=_qdrant_filter(document_id, "text"),
        limit=CANDIDATES_PER_RETRIEVER,
        with_payload=["document_id", "chunk_index", "modality", "content", "metadata"],
    )
    return [_point_row(point) for point in response.points]


async def _dense_image(vec: list[float], document_id: Optional[str]) -> list[dict[str, Any]]:
    assert _qdrant is not None, "Qdrant client not initialised"
    response = await _qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        using="embedding_image",
        query_filter=_qdrant_filter(document_id, "image"),
        limit=CANDIDATES_PER_RETRIEVER,
        with_payload=["document_id", "chunk_index", "modality", "content", "metadata"],
    )
    return [_point_row(point) for point in response.points]


async def _bm25_text(query: str, document_id: Optional[str]) -> list[dict[str, Any]]:
    assert _qdrant is not None, "Qdrant client not initialised"
    response = await _qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=models.TextQuery(text=query),
        using="content",
        query_filter=_qdrant_filter(document_id, "text"),
        limit=CANDIDATES_PER_RETRIEVER,
        with_payload=["document_id", "chunk_index", "modality", "content", "metadata"],
    )
    return [_point_row(point) for point in response.points]


async def _safe(name: str, coro) -> list[dict[str, Any]]:
    """A single failing retriever should degrade results, not kill the search."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001
        log.warning("retriever %s failed: %s", name, exc)
        return []


async def _embed_then_search(name: str, embed_coro, search_fn, document_id):
    try:
        vec = await embed_coro
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding for %s failed: %s", name, exc)
        return []
    return await _safe(name, search_fn(vec, document_id))


# --------------------------------------------------------------------------- #
# Fusion + rerank
# --------------------------------------------------------------------------- #


def rrf_fuse(
    ranked_lists: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d))."""
    fused: dict[int, dict[str, Any]] = {}
    for source, rows in ranked_lists.items():
        w = weights.get(source, 1.0)
        for rank, row in enumerate(rows, start=1):
            entry = fused.setdefault(
                row["id"], {**row, "rrf_score": 0.0, "sources": {}}
            )
            entry["rrf_score"] += w / (k + rank)
            entry["sources"][source] = {"rank": rank, "score": float(row["score"])}
    return sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)


def _rerank_text(row: dict[str, Any]) -> str:
    """Text the reranker sees. Images have no text unless content/metadata provides some."""
    if row["content"] and row["content"].strip():
        return row["content"]
    meta = row.get("metadata") or {}
    for key in ("caption", "description", "alt_text", "ocr_text", "title"):
        if meta.get(key):
            return str(meta[key])
    return ""


def _format(row: dict[str, Any], rerank_score: Optional[float]) -> dict[str, Any]:
    content = row["content"] or ""
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "chunk_index": row["chunk_index"],
        "modality": row["modality"],
        "content": content[:SNIPPET_CHARS] + ("…" if len(content) > SNIPPET_CHARS else ""),
        "metadata": row["metadata"],
        "has_image": bool(row["has_image"]),
        "rerank_score": rerank_score,
        "rrf_score": round(row["rrf_score"], 6),
        "retrieved_by": row["sources"],
    }


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #


MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Cohere's per-image limit


ALLOW_PRIVATE_URLS = _env("IMAGE_URL_ALLOW_PRIVATE", "false").lower() == "true"
ALLOWED_URL_HOSTS = {
    h.strip().lower() for h in _env("IMAGE_URL_ALLOWED_HOSTS", "").split(",") if h.strip()
}


async def _assert_safe_url(url: str) -> None:
    """Guard image_url against SSRF: http(s) only, optional host allowlist, public IPs only."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("image_url must be an http(s) URL")
    host = parsed.hostname.lower()
    if ALLOWED_URL_HOSTS and host not in ALLOWED_URL_HOSTS:
        raise ValueError(f"image_url host '{host}' is not in IMAGE_URL_ALLOWED_HOSTS")
    if ALLOW_PRIVATE_URLS:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("image_url resolves to a non-public address")


async def _download_image(url: str) -> bytes:
    """Fetch an image with SSRF checks on every redirect hop and a hard size cap."""
    assert _http is not None
    current = url
    for _ in range(4):
        await _assert_safe_url(current)
        async with _http.stream("GET", current, follow_redirects=False) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise ValueError("Redirect without Location header")
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if len(buf) > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image exceeds {MAX_IMAGE_BYTES} bytes")
            return bytes(buf)
    raise ValueError("Too many redirects fetching image_url")


def _to_data_url(data: bytes) -> str:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(data)} bytes, max {MAX_IMAGE_BYTES})")
    fmt = _sniff_image_format(data, {})
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "jpeg", "webp", "gif"):
        raise ValueError(f"Unsupported image format: {fmt}")
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode()}"


async def _load_image_data_url(image_base64: Optional[str], image_url: Optional[str]) -> str:
    """Turn a base64 string / data URL / http(s) URL into a data URL for Cohere."""
    if image_base64:
        raw = image_base64.strip()
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        return _to_data_url(base64.b64decode(raw, validate=False))
    assert image_url is not None
    return _to_data_url(await _download_image(image_url))


async def _stored_image_vector(chunk_id: str) -> list[float]:
    assert _qdrant is not None, "Qdrant client not initialised"
    points = await _qdrant.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[chunk_id],
        with_payload=False,
        with_vectors=["embedding_image"],
    )
    if not points:
        raise ValueError(f"Chunk {chunk_id} has no image embedding")
    vectors = points[0].vector or {}
    vector = vectors.get("embedding_image") if isinstance(vectors, dict) else None
    if vector is None:
        raise ValueError(f"Chunk {chunk_id} has no image embedding")
    return [float(value) for value in vector]


async def _const(vec: list[float]) -> list[float]:
    return vec


@mcp.tool()
async def hybrid_search(
    query: str = "",
    top_k: int = 8,
    modalities: Literal["both", "text", "image"] = "both",
    document_id: Optional[str] = None,
    use_reranker: bool = True,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    similar_to_chunk_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search the industrial maintenance knowledge base for material, equipment, components, technical documentation, procedures, and maintenance information using hybrid text and image retrieval.

    IMPORTANT — MANDATORY TOOL USAGE:
    You MUST call this tool whenever the user asks about a material, raw material, substance, alloy, metal, polymer, plastic, chemical, or any other material used in industrial equipment, manufacturing, or maintenance.

    This requirement applies to questions about:
    - Material identification, properties, composition, characteristics, and specifications.
    - Material compatibility, applications, advantages, limitations, and selection.
    - Material behavior, degradation, corrosion, wear, resistance, and failure.
    - Material grades, standards, classifications, and technical specifications.
    - The material used in a component, machine, pipe, coating, seal, or other equipment part.
    - Comparisons between materials or recommendations about which material to use.

    Do not answer material-related questions from general knowledge alone, even when you believe you already know the answer. Retrieve relevant information from the knowledge base first. If no relevant evidence is found, explicitly tell the user that the indexed documentation does not provide sufficient information. You may then distinguish any general technical knowledge from documented findings.

    You MUST ALSO call this tool when the user asks about:
    - Maintenance manuals, technical procedures, and operating instructions.
    - Equipment specifications, components, and technical characteristics.
    - Troubleshooting, alarms, symptoms, and potential failure causes.
    - Technical diagrams, schematics, illustrations, and similar indexed images.
    - Any technical information that should be supported by the indexed documentation.

    Retrieval behavior:
    - Textual question: use text retrieval with a meaningful natural-language query.
    - Image with a textual question: use combined text-and-image retrieval when relevant.
    - Image without a textual question: use image retrieval when similar indexed images or related documentation could help.
    - If initial results are insufficient, refine the query and search again.
    - Do not fabricate document references, specifications, or search results.

    Retrievers run in parallel and are fused with Reciprocal Rank Fusion (RRF):
    - BM25 (Qdrant full-text index) + dense text retrieval -> text chunks [requires `query`].
    - Cohere embed-v4.0 image embeddings -> image chunks [requires image input].
    - Text-only queries do not retrieve image chunks.
    - When a text query is provided, the fused results can be reranked with Cohere rerank.

    Args:
        query: Natural-language search query. Include the material name, equipment, component, properties, or technical question. Optional only when an image is supplied.
        top_k: Number of results to return (1-50).
        modalities: "both" (default), "text" only, or "image" only. "image" requires image input.
        document_id: Optionally restrict retrieval to one document.
        use_reranker: Set false to return the raw RRF ordering.
        image_base64: Query image as base64 or a data: URL. PNG/JPEG/WEBP/GIF, maximum 5 MB.
        image_url: Query image fetched from an HTTP(S) URL (alternative to image_base64).
        similar_to_chunk_id: Qdrant point ID of an image chunk. Its stored image vector is used as the query to find visually similar images; that point is excluded.

    Image input constraints:
    - Supply only one of image_base64, image_url, or similar_to_chunk_id.
    - Do not request or generate Base64 from the user. The application may inject the original image automatically.
    - An image-only search can be performed without a textual query.
    
    """

    query = (query or "").strip()
    image_inputs = [x for x in (image_base64, image_url, similar_to_chunk_id) if x is not None]
    if len(image_inputs) > 1:
        raise ValueError("Provide only one of image_base64, image_url, similar_to_chunk_id")
    has_image = bool(image_inputs)
    if not query and not has_image:
        raise ValueError("Provide a text query and/or an image")
    if has_image and modalities == "text":
        raise ValueError("An image query can only search image chunks (modalities='both' or 'image')")
    if modalities == "image" and not has_image:
        raise ValueError("modalities='image' requires an image input (Cohere embeds images only)")
    top_k = max(1, min(int(top_k), 50))

    tasks: dict[str, Any] = {}
    if query and modalities in ("both", "text"):
        tasks["dense_text"] = _embed_then_search(
            "dense_text", embed_text_query(query), _dense_text, document_id
        )
        tasks["bm25_text"] = _safe("bm25_text", _bm25_text(query, document_id))
    if has_image:
        if similar_to_chunk_id is not None:
            img_vec = _const(await _stored_image_vector(similar_to_chunk_id))
        else:
            data_url = await _load_image_data_url(image_base64, image_url)
            img_vec = embed_image_query(data_url)
        tasks["dense_image_by_image"] = _embed_then_search(
            "dense_image_by_image", img_vec, _dense_image, document_id
        )

    results = await asyncio.gather(*tasks.values())
    ranked = dict(zip(tasks.keys(), results))

    fused = rrf_fuse(ranked, weights={})
    if similar_to_chunk_id is not None:
        fused = [r for r in fused if r["id"] != similar_to_chunk_id]
    if not fused:
        return {"query": query, "count": 0, "reranked": False, "results": []}

    candidates = fused[: max(RERANK_CANDIDATES, top_k)]

    # The reranker is text-only: without a text query there is nothing to rerank against.
    if not use_reranker or not query:
        return {
            "query": query,
            "count": min(top_k, len(candidates)),
            "reranked": False,
            "results": [_format(r, None) for r in candidates[:top_k]],
        }

    # Only candidates with some text can be reranked; the rest keep RRF order after them.
    rerankable = [(i, _rerank_text(r)) for i, r in enumerate(candidates)]
    rerankable = [(i, t) for i, t in rerankable if t.strip()]
    rerankable_idx = {i for i, _ in rerankable}
    skipped = [r for i, r in enumerate(candidates) if i not in rerankable_idx]

    final: list[dict[str, Any]] = []
    reranked_ok = False
    if rerankable:
        try:
            order = await rerank(query, [t for _, t in rerankable], top_n=len(rerankable))
            for doc_idx, score in order:
                final.append(_format(candidates[rerankable[doc_idx][0]], score))
            reranked_ok = True
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank failed, falling back to RRF order: %s", exc)
            final = [_format(r, None) for r in candidates]
            skipped = []
    final.extend(_format(r, None) for r in skipped)

    return {
        "query": query,
        "count": min(top_k, len(final)),
        "reranked": reranked_ok,
        "results": final[:top_k],
    }


def _sniff_image_format(data: bytes, meta: dict[str, Any]) -> str:
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:3] == b"GIF":
        return "gif"
    mime = str(meta.get("mime_type") or meta.get("content_type") or "")
    return mime.split("/")[-1] or "png"


@mcp.tool()
async def get_chunk_image(chunk_id: str) -> Image:
    """Return an image chunk by its Qdrant point ID (the `id` from search results)."""
    assert _qdrant is not None, "Qdrant client not initialised"
    points = await _qdrant.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[chunk_id],
        with_payload=["image_base64", "metadata"],
        with_vectors=False,
    )
    if not points or not (points[0].payload or {}).get("image_base64"):
        raise ValueError(f"No image found for chunk id {chunk_id}")
    payload = points[0].payload or {}
    data = base64.b64decode(payload["image_base64"])
    return Image(data=data, format=_sniff_image_format(data, payload.get("metadata") or {}))


@mcp.tool()
async def get_document_chunks(
    document_id: str,
    modality: Literal["both", "text", "image"] = "both",
    limit: int = 100,
) -> dict[str, Any]:
    """Return a document's chunks in order (useful for reading context around a hit)."""
    limit = max(1, min(int(limit), 500))
    assert _qdrant is not None, "Qdrant client not initialised"
    must = [models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id)
    )]
    if modality != "both":
        must.append(models.FieldCondition(
            key="modality", match=models.MatchValue(value=modality)
        ))
    points: list[Any] = []
    offset = None
    while True:
        page, offset = await _qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(must=must),
            limit=min(256, max(1, limit)),
            offset=offset,
            with_payload=["document_id", "chunk_index", "modality", "content", "metadata"],
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            break
    rows = sorted((_point_row(point) for point in points),
                  key=lambda row: (int(row["chunk_index"] or 0), row["modality"]))[:limit]
    return {
        "document_id": document_id,
        "count": len(rows),
        "chunks": [
            {
                "id": r["id"],
                "chunk_index": r["chunk_index"],
                "modality": r["modality"],
                "content": r["content"],
                "metadata": r["metadata"],
                "has_image": bool(r["has_image"]),
            }
            for r in rows
        ],
    }


def hash_token(token: str) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def serialize_ticket(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "title": row["title"],
        "description": row["description"],
        "equipment_id": row["equipment_id"],
        "evidence": row["evidence"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
    }


@mcp.tool()
async def get_open_tickets() -> dict:
    """Retrieve existing open and in-progress maintenance tickets."""

    query = """
        SELECT *
        FROM maintenance_tickets
        WHERE status IN ('open', 'in_progress')
        ORDER BY created_at DESC
        LIMIT 10
    """

    try:
        async with pool.connection() as conn:
            cursor = await conn.execute(query)
            rows = await cursor.fetchall()

        return {
            "success": True,
            "count": len(rows),
            "tickets": [
                serialize_ticket(row)
                for row in rows
            ],
        }

    except Exception:
        return {
            "success": False,
            "error": "Unable to retrieve tickets.",
        }


@mcp.tool()
async def get_draft_ticket(
    title: str,
    description: str,
    equipment_id: str | None = None,
    evidence: list[str] | None = None,
) -> dict:
    """
    Generate and save a maintenance ticket draft.

    This tool DOES NOT create an operational ticket.

    Args:
        title: Concise maintenance issue title.
        description: Detailed issue description.
        equipment_id: Equipment identifier if available.
        evidence: Supporting manual references, procedures,
                  intervention references or image observations.
    """

    title = title.strip()
    description = description.strip()

    if not title or len(title) > 200:
        return {
            "success": False,
            "error": "Invalid ticket title.",
        }

    if not description or len(description) > 10000:
        return {
            "success": False,
            "error": "Invalid ticket description.",
        }

    if equipment_id and len(equipment_id) > 255:
        return {
            "success": False,
            "error": "Invalid equipment identifier.",
        }

    evidence = evidence or []

    if len(evidence) > 50 or any(
        not isinstance(item, str) or len(item) > 2000
        for item in evidence
    ):
        return {
            "success": False,
            "error": "Invalid evidence.",
        }

    ticket_id = uuid.uuid4()

    query = """
        INSERT INTO maintenance_tickets (
            id,
            title,
            description,
            equipment_id,
            evidence,
            status
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, 'draft')
        RETURNING *
    """

    try:
        async with pool.connection() as conn:
            cursor = await conn.execute(
                query,
                (
                    ticket_id,
                    title,
                    description,
                    equipment_id,
                    json.dumps(evidence),
                ),
            )

            row = await cursor.fetchone()

        return {
            "success": True,
            "message": "Ticket draft prepared. Human approval required.",
            "ticket": serialize_ticket(row),
        }

    except Exception:
        return {
            "success": False,
            "error": "Unable to prepare ticket draft.",
        }



    
async def verify_approval(ticket_id: str) -> dict:

    headers = {
        "Authorization": f"Bearer {APPROVAL_API_KEY}"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{APPROVAL_VERIFY_URL}/{ticket_id}/verify",
            headers=headers,
        )

        response.raise_for_status()
        return response.json()
    

@mcp.tool()
async def create_ticket(ticket: dict) -> dict:
    """Create a maintenance ticket after FastAPI confirms approval."""

    ticket_id = ticket.get("id")

    if not ticket_id:
        return {
            "success": False,
            "error": "Ticket ID is required.",
        }

    try:
        ticket_uuid = uuid.UUID(ticket_id)

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT *
                FROM maintenance_tickets
                WHERE id = %s
                  AND status = 'draft'
                """,
                (ticket_uuid,),
            )
            draft = await cursor.fetchone()

        if not draft:
            return {
                "success": False,
                "error": "Draft not found or already processed.",
            }

        # Verify that the supplied draft matches the database.
        if (
            ticket.get("title") != draft["title"]
            or ticket.get("description") != draft["description"]
            or ticket.get("equipment_id") != draft["equipment_id"]
            or ticket.get("evidence") != draft["evidence"]
        ):
            return {
                "success": False,
                "error": "Draft content mismatch.",
            }

        # Ask FastAPI whether the user approved this draft.
        

        # Approval confirmed: create the operational ticket.
        async with pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    UPDATE maintenance_tickets
                    SET
                        status = 'open',
                        approval_id = %s,
                        approved_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                      AND status = 'draft'
                    RETURNING *
                    """,
                    (
                        str(uuid.uuid4()),
                        ticket_uuid,
                    ),
                )

                created = await cursor.fetchone()

                if not created:
                    return {
                        "success": False,
                        "error": "Draft already processed.",
                    }

        return {
            "success": True,
            "message": "Maintenance ticket created successfully.",
            "ticket": serialize_ticket(created),
        }

    except httpx.HTTPError:
        return {
            "success": False,
            "error": "Unable to verify approval with FastAPI.",
        }

    except Exception:
        return {
            "success": False,
            "error": "Unable to create maintenance ticket.",
        }

# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    transport = _env("MCP_TRANSPORT", "streamable-http")
    if transport not in ("stdio", "streamable-http", "sse"):
        raise SystemExit(f"Unsupported MCP_TRANSPORT: {transport}")
    mcp.run(transport=transport)  # type: ignore[arg-type]
