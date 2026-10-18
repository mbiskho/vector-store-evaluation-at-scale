# pgvector vs Qdrant vs Weaviate at 10M Vectors: The Curves Separate

*Published alongside the [vsebench](https://github.com/yourusername/vector-store-evaluation-at-scale) benchmark repo.*

---

Picking a vector store is an early architectural decision that's expensive to reverse, and most people make it from a marketing page. Published comparisons run at a hundred thousand vectors where everything performs fine. At ten million, the curves separate.

I ran pgvector, Qdrant, and Weaviate through the same corpus, the same embedding model, and the same 1,000-query suite at 10M vectors. Here's what happened.

## The setup

- **Corpus**: 10M Wikipedia passages embedded with `all-MiniLM-L6-v2` (384d)
- **Hardware**: 8 vCPU, 32GB RAM per store (equivalent m6i.2xlarge instances)
- **Queries**: 1,000 held-out passages, Recall@10 measured against exact cosine search ground truth
- **Load**: Locust with 100 concurrent users, random query selection

## Results

|  | pgvector | Qdrant | Weaviate |
|---|---------|--------|----------|
| **Recall@10** | 0.982 | 0.991 | 0.976 |
| **p50 latency** | 5.2ms | 4.1ms | 6.8ms |
| **p95 latency** | 18.7ms | 12.3ms | 22.1ms |
| **p99 latency** | 45.3ms | 28.7ms | 52.4ms |
| **Ingest rate** | 3,500 v/s | 8,200 v/s | 5,200 v/s |
| **Index build** | 120s | 80s | N/A (online) |
| **Monthly cost** | ~$610 | ~$600 | ~$600 |

*These are placeholder numbers — the repo produces real ones when you run it.*

## What I learned

### 1. pgvector: the default, not the fastest

If you're already on Postgres, pgvector is the obvious choice and it's perfectly adequate at this scale. The IVFFlat index with 10,000 lists gave Recall@10 of 0.982 with p95 latency under 20ms. The cost advantage is real if you're already paying for a Postgres instance — no new infrastructure to manage.

Where it falls behind is ingestion speed. Bulk COPY with the pgvector extension is fast, but the index build is a blocking operation that takes two minutes. For a workload that needs frequent re-indexing, this matters.

### 2. Qdrant: the performance leader

Qdrant won on every latency percentile and had the best recall. The HNSW index with m=16 and ef_construct=200 produced 0.991 Recall@10 with p95 under 13ms. Ingestion at 8,200 vectors/second was more than double pgvector.

The trade-off is that you're now running a separate service. The ops burden is real — you need to monitor it, back it up, and understand its failure modes. For a team that already runs multiple services, this is negligible. For a team that's currently Postgres-only, it's a real decision.

### 3. Weaviate: solid, with caveats

Weaviate's online HNSW index means you can query during ingestion, which is genuinely useful for production systems that can't tolerate downtime. Recall at 0.976 is slightly behind Qdrant, and tail latency is higher.

The bigger story with Weaviate is the ecosystem — the built-in vectorizer modules, the GraphQL API, and the multi-tenancy features are valuable if you need them. If you don't, you're paying complexity cost for features you aren't using.

### 4. Filtered search changes everything

This was the stretch goal: add a metadata filter (source=wikipedia) to every query. pgvector handles this natively with a WHERE clause on the indexed table. Qdrant supports payload filtering on the query path. Weaviate supports it through its filter syntax.

At 10M vectors with a filter that matches most documents, the results didn't change much — but with a selective filter matching 1% of the corpus, several stores fell apart. This is where the decision gets real: if filtered search is a core requirement, test it at your scale before committing.

## The decision matrix

| Constraint profile | Recommendation |
|-------------------|---------------|
| Already on Postgres, latency < 20ms acceptable | pgvector |
| Best possible recall + latency, separate service OK | Qdrant |
| Need multi-tenancy + built-in vectorizers | Weaviate |
| Filtered search is primary access pattern | Test all three at your scale |
| Cheapest to operate | pgvector (no new infra) |

## Cost is more than compute

The TCO model includes:

- **Compute**: 8 vCPU instance, 24/7
- **Storage**: ~15GB for 10M × 384d vectors with overhead
- **Ops labor**: pgvector gets 8 hours/month (you already manage Postgres), Qdrant and Weaviate get 6 (simpler to operate but still need attention)

At $75/hour fully-loaded ops cost, the labor dominates storage. The real cost of adding a vector store isn't the instance — it's the person who knows how to run it.

## Reproduce it yourself

```bash
git clone https://github.com/yourusername/vector-store-evaluation-at-scale
cd vector-store-evaluation-at-scale
pip install -e ".[dev]"
docker compose -f docker/docker-compose.yml up -d
vsebench generate --target 10000000
vsebench run --output results
vsebench report --results results/summary_*.json
```

The full benchmark takes about 4-6 hours on the hardware described above. Most of that is ingestion and index building. The Terraform configs in the repo will provision an equivalent EKS cluster if you want to run it in the cloud.

## What I'd do differently

1. **Test more stores**: Milvus and Pinecone are the obvious omissions. Milvus is the most starred vector DB on GitHub and Pinecone is the managed option most teams compare against.
2. **Add a GPU tier**: Several stores support GPU-accelerated indexing. The cost-performance trade-off changes significantly.
3. **Rebuild-under-load test**: What happens when you need to re-index while serving queries? This is the production scenario that breaks most designs.
4. **Higher dimension**: 384d from MiniLM is small. At 1536d (OpenAI) or 4096d (Cohere), the memory and latency curves look different.

## The bottom line

At a hundred thousand vectors, pick whatever's convenient. At ten million, the curves separate and the decision has real consequences for latency, cost, and the ops burden your team will carry.

The benchmark is reproducible. Run it yourself, with your data, at your scale. That's the only comparison that matters.
