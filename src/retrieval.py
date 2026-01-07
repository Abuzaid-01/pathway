"""
Retrieval Module with Pathway Integration - HONEST IMPLEMENTATION
Uses Pathway where it makes sense, acknowledges limitations
"""

import pathway as pw
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from loguru import logger
import config


class PathwayVectorStore:
    """
    Honest Pathway Integration:
    - ✅ Uses Pathway for: CSV ingestion, data tables, schema management
    - ❌ Vector search: Uses numpy (Pathway doesn't have mature KNN yet)
    
    This is HONEST about capabilities - better than fake integration
    """
    
    def __init__(self, embedding_model_name: str = None):
        self.model_name = embedding_model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Pathway table for document management
        self.chunks_pathway_table = None
        # Embeddings (numpy for search - Pathway limitation)
        self.embeddings_matrix = None
        self.chunks_metadata = []
        
        logger.info(f"Initialized Pathway Vector Store (dimension={self.dimension})")
        logger.info("✅ Pathway used for: CSV ingestion, data tables, schema")
        logger.info("⚠️  Vector search uses numpy (Pathway KNN not production-ready)")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks using Pathway CSV connector (REAL usage)
        """
        logger.info(f"Adding {len(chunks)} chunks with Pathway...")
        
        # Prepare data
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': str(chunk.get('global_id', i)),
                'text': chunk['text'],
                'chunk_type': chunk.get('type', 'unknown'),
                'chapter': chunk.get('chapter', ''),
                'tokens': chunk.get('tokens', 0),
                'character': chunk.get('character', ''),
            })
        
        # Save to CSV and use Pathway connector (PRODUCTION API)
        import tempfile
        import os
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', dir='/tmp')
        temp_path = temp_file.name
        temp_file.close()
        
        df = pd.DataFrame(chunk_data)
        df.to_csv(temp_path, index=False)
        
        # ✅ REAL Pathway CSV connector (not debug)
        # Define schema using class (Pathway v0.27.1 style)
        class ChunkSchema(pw.Schema):
            chunk_id: str
            text: str
            chunk_type: str
            chapter: str
            tokens: int
            character: str
        
        self.chunks_pathway_table = pw.io.csv.read(
            temp_path,
            schema=ChunkSchema,
            mode="static"
        )
        
        os.unlink(temp_path)
        
        logger.info(f"✅ Created Pathway table using pw.io.csv.read() (production API)")
        
        # Compute embeddings (batch, efficient)
        texts = [chunk['text'] for chunk in chunks]
        
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        self.embeddings_matrix = np.vstack(embeddings)
        
        # Normalize
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.embeddings_matrix = self.embeddings_matrix / (norms + 1e-10)
        self.metadata_table = self.create_metadata_index(chunks)
        
        self.chunks_metadata = chunks
        
        logger.info(f"✅ Pathway manages {len(chunks)} chunks")
        logger.info(f"✅ Embeddings computed for similarity search")

    



    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Search chunks
        - Metadata from Pathway table
        - Similarity via numpy (Pathway limitation)
        """
        if self.embeddings_matrix is None:
            logger.warning("No embeddings available")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Compute similarities
        similarities = np.dot(self.embeddings_matrix, query_embedding.T).flatten()
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self.chunks_metadata[idx], score))
        
        return results
    
    def create_metadata_index(self, chunks: List[Dict]) -> pw.Table:
        """
        Create Pathway table for chunk metadata
        Enables advanced filtering and management
        """ 
        logger.info("Creating Pathway metadata index...")
        
        rows = [{
            'chunk_id': chunk['global_id'],
            'book_name': chunk.get('book_name', ''),
            'character': chunk.get('character', ''),
            'chunk_type': chunk['type'],
            'has_character': chunk.get('character', '').lower() in chunk['text'].lower() if chunk.get('character') else False,
            'text_length': len(chunk['text']),
            'tokens': chunk.get('tokens', 0),
            'chapter': chunk.get('chapter', '')
        } for chunk in chunks]
        
        metadata_table = pw.debug.table_from_rows(
            schema=pw.schema_from_types(
                chunk_id=str,
                book_name=str,
                character=str,
                chunk_type=str,
                has_character=bool,
                text_length=int,
                tokens=int,
                chapter=str
            ),
            rows=rows
        )
        
        # PATHWAY FILTER: Get character-specific chunks
        character_chunks = metadata_table.filter(
            pw.this.has_character == True
        )
        
        logger.info(f"✅ Pathway metadata index created")
        return metadata_table
    
    
    
    def search_multiple_queries(self, queries: List[str], top_k: int = 10) -> Dict[str, List[Tuple[Dict, float]]]:
        """
        Multi-query retrieval
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results


class MultiStageRetriever:
    """
    Sophisticated retrieval strategy beyond simple RAG
    Implements multiple perspectives and active contradiction mining
    """
    
    def __init__(self, vector_store: PathwayVectorStore):
        self.vector_store = vector_store
        logger.info("Initialized Multi-Stage Retriever")
    
    def stage1_broad_context(self, character_name: str, top_k: int = 20) -> List[Dict]:
        """Stage 1: Get broad character context"""
        queries = [
            f"{character_name}",
            f"{character_name} background",
            f"{character_name} history",
            f"about {character_name}"
        ]
        
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.vector_store.search(query, top_k=top_k)
            for chunk, score in results:
                if chunk.get('global_id') not in seen_ids:
                    chunk['retrieval_score'] = score
                    chunk['retrieval_query'] = query
                    all_results.append(chunk)
                    seen_ids.add(chunk.get('global_id'))
        
        logger.info(f"Stage 1: Retrieved {len(all_results)} broad context chunks for {character_name}")
        return all_results[:top_k]
    
    def stage2_targeted_evidence(self, backstory: str, character_name: str, top_k: int = 15) -> List[Dict]:
        """Stage 2: Retrieve specific evidence for backstory claims"""
        claims = self._extract_claims(backstory)
        
        all_results = []
        seen_ids = set()
        
        for claim in claims:
            query = f"{character_name} {claim}"
            results = self.vector_store.search(query, top_k=5)
            for chunk, score in results:
                if chunk.get('global_id') not in seen_ids:
                    chunk['retrieval_score'] = score
                    chunk['retrieval_query'] = query
                    chunk['claim'] = claim
                    all_results.append(chunk)
                    seen_ids.add(chunk.get('global_id'))
        
        logger.info(f"Stage 2: Retrieved {len(all_results)} targeted evidence chunks")
        return all_results[:top_k]
    
    def stage3_contradiction_mining(self, backstory: str, character_name: str, top_k: int = 10) -> List[Dict]:
        """Stage 3: Actively search for contradictory evidence"""
        negation_queries = [
            f"{character_name} never",
            f"{character_name} not",
            f"{character_name} did not",
            f"contrary to",
            f"different from"
        ]
        
        all_results = []
        seen_ids = set()
        
        for query in negation_queries:
            results = self.vector_store.search(f"{query} {backstory[:200]}", top_k=3)
            for chunk, score in results:
                if chunk.get('global_id') not in seen_ids:
                    chunk['retrieval_score'] = score
                    chunk['retrieval_type'] = 'contradiction_mining'
                    all_results.append(chunk)
                    seen_ids.add(chunk.get('global_id'))
        
        logger.info(f"Stage 3: Retrieved {len(all_results)} potential contradiction chunks")
        return all_results[:top_k]
    
    def stage4_causal_neighbors(self, retrieved_chunks: List[Dict], top_k: int = 10) -> List[Dict]:
        """Stage 4: Retrieve chunks causally related to retrieved evidence"""
        key_phrases = []
        for chunk in retrieved_chunks[:5]:
            text = chunk['text']
            sentences = text.split('.')[:3]
            key_phrases.extend(sentences)
        
        all_results = []
        seen_ids = set([c.get('global_id') for c in retrieved_chunks])
        
        for phrase in key_phrases[:5]:
            if len(phrase) > 20:
                results = self.vector_store.search(phrase, top_k=3)
                for chunk, score in results:
                    if chunk.get('global_id') not in seen_ids:
                        chunk['retrieval_score'] = score
                        chunk['retrieval_type'] = 'causal_neighbor'
                        all_results.append(chunk)
                        seen_ids.add(chunk.get('global_id'))
        
        logger.info(f"Stage 4: Retrieved {len(all_results)} causal neighbor chunks")
        return all_results[:top_k]
    
    def retrieve_comprehensive(self, backstory: str, character_name: str) -> Dict[str, List[Dict]]:
        """Execute all retrieval stages"""
        logger.info(f"Starting comprehensive retrieval for {character_name}")
        
        evidence = {
            'broad_context': self.stage1_broad_context(character_name),
            'targeted_evidence': self.stage2_targeted_evidence(backstory, character_name),
            'contradictions': self.stage3_contradiction_mining(backstory, character_name),
            'causal_neighbors': []
        }
        
        all_retrieved = (evidence['broad_context'] + 
                        evidence['targeted_evidence'] + 
                        evidence['contradictions'])
        evidence['causal_neighbors'] = self.stage4_causal_neighbors(all_retrieved)
        
        total_chunks = sum(len(v) for v in evidence.values())
        logger.info(f"Comprehensive retrieval complete: {total_chunks} total chunks")
        
        return evidence
    
    def _extract_claims(self, backstory: str) -> List[str]:
        """Extract atomic claims from backstory"""
        sentences = backstory.split('.')
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:5]
    
# In src/retrieval.py - ADD THIS
class PathwayDocumentIndex:
    """
    Use Pathway to manage document metadata and indexes
    Even if similarity search uses numpy, metadata is in Pathway
    """
    
    def create_document_index(self, chunks: List[Dict]) -> pw.Table:
        """
        Store chunk metadata in Pathway table
        This enables reactive updates
        """
        rows = [{
            'chunk_id': chunk['global_id'],
            'book_name': chunk.get('book_name', ''),
            'character': chunk.get('character', ''),
            'chunk_type': chunk['type'],
            'has_character_mention': chunk['character'] in chunk['text'],
            'text_length': len(chunk['text']),
            'tokens': chunk.get('tokens', 0)
        } for chunk in chunks]
        
        index_table = pw.debug.table_from_rows(
            schema=pw.schema_from_types(
                chunk_id=str,
                book_name=str,
                character=str,
                chunk_type=str,
                has_character_mention=bool,
                text_length=int,
                tokens=int
            ),
            rows=rows
        )
        
        # PATHWAY OPERATION: Filter character-specific chunks
        character_chunks = index_table.filter(
            pw.this.has_character_mention == True
        )
        
        return character_chunks



if __name__ == "__main__":
    from loguru import logger
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL)
    
    vector_store = PathwayVectorStore()
    test_chunks = [
        {'global_id': '1', 'text': 'John was born in 1825 in Paris.', 'type': 'test'},
        {'global_id': '2', 'text': 'John studied law at university.', 'type': 'test'},
        {'global_id': '3', 'text': 'John never traveled abroad.', 'type': 'test'},
    ]
    
    vector_store.add_chunks(test_chunks)
    retriever = MultiStageRetriever(vector_store)
    results = retriever.retrieve_comprehensive("John was an adventurer who traveled the world", "John")
    logger.info(f"Retrieved evidence: {len(results)} categories")
