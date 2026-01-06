"""
Retrieval Module with Pathway Integration
Multi-stage retrieval using vector stores and strategic querying
"""

import pathway as pw
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from loguru import logger
import config


class PathwayVectorStore:
    """
    Vector store integration with Pathway
    Demonstrates Pathway's capability for indexing and retrieval
    """
    
    def __init__(self, embedding_model_name: str = None):
        self.model_name = embedding_model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.chunks = []
        self.chunk_embeddings = []
        
        logger.info(f"Initialized Pathway Vector Store (dimension={self.dimension})")
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to the vector store"""
        logger.info(f"Encoding {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Encode in batches for efficiency
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.chunk_embeddings.append(embeddings)
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks using semantic similarity
        """
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_multiple_queries(self, queries: List[str], top_k: int = 10) -> Dict[str, List[Tuple[Dict, float]]]:
        """
        Multi-query retrieval for comprehensive evidence gathering
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
