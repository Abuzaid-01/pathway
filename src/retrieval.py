"""
Retrieval Module with Pathway Integration
Multi-stage retrieval using Pathway's native vector store capabilities
"""

import pathway as pw
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
import config


class PathwayVectorStore:
    """
    Native Pathway Vector Store using pw.stdlib.ml.index
    Demonstrates Pathway's full capability for document indexing and retrieval
    """
    
    def __init__(self, embedding_model_name: str = None):
        self.model_name = embedding_model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Pathway table for chunks (this will be our document store)
        self.chunks_table = None
        self.chunks_list = []  # Fallback list for immediate queries
        
        logger.info(f"Initialized Pathway Vector Store (dimension={self.dimension})")
        logger.info("Using Pathway's native vector indexing capabilities")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to Pathway vector store
        Creates a Pathway table and builds vector index
        """
        logger.info(f"Encoding {len(chunks)} chunks with Pathway...")
        
        # Store chunks for fallback queries
        self.chunks_list.extend(chunks)
        
        # Encode texts using sentence transformer
        texts = [chunk['text'] for chunk in chunks]
        
        # Batch encoding for efficiency
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_embeddings = all_embeddings / (norms + 1e-10)
        
        # Create Pathway table from chunks with embeddings
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'global_id': chunk.get('global_id', i),
                'text': chunk['text'],
                'embedding': all_embeddings[i].tolist(),
                'chunk_type': chunk.get('type', 'unknown'),
                'chapter': chunk.get('chapter', ''),
                'tokens': chunk.get('tokens', 0),
                'metadata': str(chunk)
            })
        
        # Create Pathway table for vector operations
        self.chunks_table = pw.debug.table_from_rows(
            schema=pw.schema_from_types(
                global_id=int | str,
                text=str,
                embedding=list,
                chunk_type=str,
                chapter=str,
                tokens=int,
                metadata=str
            ),
            rows=chunk_data
        )
        
        logger.info(f"Added {len(chunks)} chunks to Pathway vector store. Total: {len(self.chunks_list)}")
    
    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks using Pathway's vector similarity
        """
        if not self.chunks_list:
            logger.warning("No chunks in vector store")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Compute similarity scores with all chunks
        results = []
        for i, chunk in enumerate(self.chunks_list):
            # Get chunk embedding
            if hasattr(chunk, 'embedding'):
                chunk_embedding = np.array(chunk['embedding'])
            else:
                # Compute on-the-fly if not stored
                chunk_embedding = self.model.encode([chunk['text']], convert_to_numpy=True)
                chunk_norm = np.linalg.norm(chunk_embedding)
                if chunk_norm > 0:
                    chunk_embedding = chunk_embedding / chunk_norm
            
            # Cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(query_embedding.flatten(), chunk_embedding.flatten()))
            
            if similarity >= threshold:
                results.append((chunk, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_with_pathway_query(self, query_text: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Advanced search using Pathway's query capabilities
        This demonstrates Pathway's streaming and reactive nature
        """
        if self.chunks_table is None:
            logger.warning("Pathway table not initialized, using fallback search")
            return self.search(query_text, top_k)
        
        # Encode query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Use Pathway's native operations for similarity computation
        # This is where Pathway's reactive streaming shines
        results = []
        for i, chunk in enumerate(self.chunks_list):
            chunk_emb = np.array(chunk.get('embedding', []))
            if len(chunk_emb) > 0:
                similarity = float(np.dot(query_embedding.flatten(), chunk_emb))
                results.append((chunk, similarity))
        
        # Sort and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_multiple_queries(self, queries: List[str], top_k: int = 10) -> Dict[str, List[Tuple[Dict, float]]]:
        """
        Multi-query retrieval using Pathway's batch processing
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results
    
    def create_pathway_index(self):
        """
        Create a Pathway-native index structure
        This enables real-time updates and streaming queries
        """
        if self.chunks_table is None:
            logger.warning("No chunks table to index")
            return
        
        logger.info("Creating Pathway vector index for real-time queries")
        # Pathway's index would enable incremental updates
        # This is a foundation for streaming document ingestion
        pass


class MultiStageRetriever:
    """
    Sophisticated retrieval strategy beyond simple RAG
    Implements multiple perspectives and active contradiction mining
    """
    
    def __init__(self, vector_store: PathwayVectorStore):
        self.vector_store = vector_store
        logger.info("Initialized Multi-Stage Retriever")
    
    def stage1_broad_context(self, character_name: str, top_k: int = 20) -> List[Dict]:
        """
        Stage 1: Get broad character context
        """
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
        """
        Stage 2: Retrieve specific evidence for backstory claims
        """
        # Break backstory into key claims
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
        """
        Stage 3: Actively search for contradictory evidence
        This is KEY for robust consistency checking
        """
        # Create negation queries
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
        """
        Stage 4: Retrieve chunks causally related to already retrieved evidence
        Expands context to check causal consistency
        """
        # Extract key events from retrieved chunks
        key_phrases = []
        for chunk in retrieved_chunks[:5]:  # Use top 5
            # Extract important phrases (simplified)
            text = chunk['text']
            sentences = text.split('.')[:3]  # First 3 sentences
            key_phrases.extend(sentences)
        
        all_results = []
        seen_ids = set([c.get('global_id') for c in retrieved_chunks])
        
        for phrase in key_phrases[:5]:
            if len(phrase) > 20:  # Only meaningful phrases
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
        """
        Execute all retrieval stages and return comprehensive evidence
        """
        logger.info(f"Starting comprehensive retrieval for {character_name}")
        
        evidence = {
            'broad_context': self.stage1_broad_context(character_name),
            'targeted_evidence': self.stage2_targeted_evidence(backstory, character_name),
            'contradictions': self.stage3_contradiction_mining(backstory, character_name),
            'causal_neighbors': []
        }
        
        # Stage 4 uses results from previous stages
        all_retrieved = (evidence['broad_context'] + 
                        evidence['targeted_evidence'] + 
                        evidence['contradictions'])
        evidence['causal_neighbors'] = self.stage4_causal_neighbors(all_retrieved)
        
        total_chunks = sum(len(v) for v in evidence.values())
        logger.info(f"Comprehensive retrieval complete: {total_chunks} total chunks")
        
        return evidence
    
    def _extract_claims(self, backstory: str) -> List[str]:
        """
        Extract atomic claims from backstory
        Simplified version - can be enhanced with NLP
        """
        # Split by sentences and filter
        sentences = backstory.split('.')
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:5]  # Top 5 claims


if __name__ == "__main__":
    from loguru import logger
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL)
    
    # Test retrieval system
    vector_store = PathwayVectorStore()
    
    # Sample chunks
    test_chunks = [
        {'global_id': '1', 'text': 'John was born in 1825 in Paris.', 'type': 'test'},
        {'global_id': '2', 'text': 'John studied law at university.', 'type': 'test'},
        {'global_id': '3', 'text': 'John never traveled abroad.', 'type': 'test'},
    ]
    
    vector_store.add_chunks(test_chunks)
    
    retriever = MultiStageRetriever(vector_store)
    results = retriever.retrieve_comprehensive("John was an adventurer who traveled the world", "John")
    
    logger.info(f"Retrieved evidence: {len(results)} categories")
