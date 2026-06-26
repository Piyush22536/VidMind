import 'dotenv/config';
import { OpenAIEmbeddings } from '@langchain/openai';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Document } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import pg from 'pg';

const { Pool } = pg;

export const pool = new Pool({
  connectionString: process.env.DB_URL,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
  keepAlive: true,
  keepAliveInitialDelayMillis: 10000,
});
setInterval(async () => {
  try {
    await pool.query('SELECT 1');
    console.log('[pool] keepalive ping');
  } catch (e) {
    console.error('[pool] keepalive failed:', e.message);
  }
}, 4 * 60 * 1000);


pool.on('error', (err) => {
  console.error('[pool] idle client error:', err.message);
});

// ── Embeddings model ──────────────────────────────────────────────────────────
// const embeddings = new OpenAIEmbeddings({
//   model: 'text-embedding-3-large', // 3072-dim
// });
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dims
  apiKey: process.env.GOOGLE_API_KEY,
});



export const vectorStore = await PGVectorStore.initialize(embeddings, {
  postgresConnectionOptions: { connectionString: process.env.DB_URL },
  tableName: 'transcripts',
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'vector',
    contentColumnName: 'content',
    metadataColumnName: 'metadata',
  },
  distanceStrategy: 'cosine',
});

// ── Hybrid search via Postgres hybrid_search() function ───────────────────────
/**
 * @param {string} queryText  
 * @param {number} topK       
 * @param {string|null} videoIdFilter
 * @returns {Promise<Array<{content: string, metadata: object, rrfScore: number, vecRank: number, bm25Rank: number}>>}
 */
export const hybridSearch = async (queryText, topK = 5, videoIdFilter = null) => {
  // Embed the query for the vector leg
  const [queryVector] = await embeddings.embedDocuments([queryText]);

  const vectorLiteral = `[${queryVector.join(',')}]`;

  const { rows } = await pool.query(
    `SELECT
       id,
       content,
       metadata,
       rrf_score,
       vec_rank,
       bm25_rank
     FROM hybrid_search($1, $2::vector, $3, $4, 60, 0.7, 0.3)`,
    [queryText, vectorLiteral, topK, videoIdFilter]
  );

  return rows.map((r) => ({
    content:   r.content,
    metadata:  r.metadata,
    rrfScore:  parseFloat(r.rrf_score),
    vecRank:   r.vec_rank,
    bm25Rank:  r.bm25_rank,
  }));
};

export const isVideoIndexed = async (videoId) => {
  const { rows } = await pool.query(
    `SELECT 1 FROM transcripts WHERE metadata->>'video_id' = $1 LIMIT 1`,
    [videoId]
  );
  return rows.length > 0;
};

// ── Ingest a YouTube video into the vector store ──────────────────────────────
export const addYTVideoToVectorStore = async (videoData) => {
  const { transcript, video_id, url } = videoData;

  if (await isVideoIndexed(video_id)) {
    console.log(`[embeddings] video ${video_id} already indexed — skipping`);
    return { skipped: true, video_id };
  }

  const docs = [
    new Document({
      pageContent: transcript,
      metadata: { video_id, url: url ?? '' },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const chunks = await splitter.splitDocuments(docs);
  await vectorStore.addDocuments(chunks);

  console.log(`[embeddings] indexed ${chunks.length} chunks for video ${video_id}`);
  return { skipped: false, video_id, chunks: chunks.length };
};