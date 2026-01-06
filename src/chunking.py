"""
Intelligent Chunking Module
Multi-strategy text segmentation: semantic, structural, and character-centric
"""

import re
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
import config


class MultiStrategyChunker:
    """
    Advanced chunking beyond simple token splits
    Implements semantic, structural, and character-centric strategies
    """
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.overlap = config.CHUNK_OVERLAP
        logger.info("Initialized Multi-Strategy Chunker")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def chunk_by_structure(self, text: str, book_name: str) -> List[Dict]:
        """
        Structural chunking: preserve narrative boundaries
        Splits at chapter boundaries, then scenes
        """
        chunks = []
        
        # Try to split by chapters
        chapter_pattern = r'(Chapter \d+[^\n]*\n)'
        parts = re.split(chapter_pattern, text)
        
        current_chunk = ""
        current_chapter = "Intro"
        chunk_id = 0
        
        for i, part in enumerate(parts):
            if re.match(chapter_pattern, part):
                # Save previous chunk if it exists
                if current_chunk:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk,
                        'chapter': current_chapter,
                        'type': 'structural',
                        'tokens': self.estimate_tokens(current_chunk)
                    })
                    chunk_id += 1
                current_chapter = part.strip()
                current_chunk = ""
            else:
                current_chunk += part
                # If chunk too large, split it
                if self.estimate_tokens(current_chunk) > self.chunk_size:
                    # Split at paragraph boundaries
                    paragraphs = current_chunk.split('\n\n')
                    temp_chunk = ""
                    for para in paragraphs:
                        if self.estimate_tokens(temp_chunk + para) < self.chunk_size:
                            temp_chunk += para + "\n\n"
                        else:
                            if temp_chunk:
                                chunks.append({
                                    'chunk_id': chunk_id,
                                    'text': temp_chunk,
                                    'chapter': current_chapter,
                                    'type': 'structural',
                                    'tokens': self.estimate_tokens(temp_chunk)
                                })
                                chunk_id += 1
                            temp_chunk = para + "\n\n"
                    current_chunk = temp_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk,
                'chapter': current_chapter,
                'type': 'structural',
                'tokens': self.estimate_tokens(current_chunk)
            })
        
        logger.info(f"Structural chunking: {len(chunks)} chunks")
        return chunks
    
    def chunk_by_character(self, text: str, character_name: str) -> List[Dict]:
        """
        Character-centric chunking: extract passages mentioning the character
        Critical for retrieval efficiency
        """
        chunks = []
        
        # Find all mentions of the character (case-insensitive)
        # Handle variations: full name, first name, last name
        name_parts = character_name.split()
        patterns = [re.escape(name_parts[0])]
        if len(name_parts) > 1:
            patterns.append(re.escape(name_parts[-1]))
        
        pattern = '|'.join(patterns)
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        chunk_id = 0
        window_size = 5  # sentences before and after mention
        
        for i, sent in enumerate(sentences):
            if re.search(pattern, sent, re.IGNORECASE):
                # Extract context window
                start = max(0, i - window_size)
                end = min(len(sentences), i + window_size + 1)
                context = ' '.join(sentences[start:end])
                
                if self.estimate_tokens(context) > config.MIN_CHUNK_SIZE:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': context,
                        'character': character_name,
                        'type': 'character-centric',
                        'tokens': self.estimate_tokens(context),
                        'sentence_idx': i
                    })
                    chunk_id += 1
        
        logger.info(f"Character-centric chunking for {character_name}: {len(chunks)} chunks")
        return chunks
    
    def chunk_overlapping_windows(self, text: str) -> List[Dict]:
        """
        Sliding window chunking with overlap
        Prevents information loss at boundaries
        """
        chunks = []
        words = text.split()
        
        # Approximate: 1 token ≈ 0.75 words
        words_per_chunk = int(self.chunk_size * 0.75)
        words_overlap = int(self.overlap * 0.75)
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'type': 'overlapping',
                'tokens': self.estimate_tokens(chunk_text),
                'start_idx': start,
                'end_idx': end
            })
            
            chunk_id += 1
            start += words_per_chunk - words_overlap
            
            if end >= len(words):
                break
        
        logger.info(f"Overlapping window chunking: {len(chunks)} chunks")
        return chunks
    
    def chunk_hybrid(self, text: str, book_name: str, character_name: str) -> List[Dict]:
        """
        Hybrid approach: combine multiple strategies
        This is our main chunking method
        """
        all_chunks = []
        
        # 1. Structural chunks (for global context)
        structural = self.chunk_by_structure(text, book_name)
        all_chunks.extend(structural)
        
        # 2. Character-centric chunks (for targeted retrieval)
        character = self.chunk_by_character(text, character_name)
        all_chunks.extend(character)
        
        # 3. Overlapping windows (for dense coverage)
        overlapping = self.chunk_overlapping_windows(text)
        # Sample to avoid too many chunks
        overlapping_sampled = overlapping[::3]  # Take every 3rd chunk
        all_chunks.extend(overlapping_sampled)
        
        # Add unique IDs
        for i, chunk in enumerate(all_chunks):
            chunk['global_id'] = f"{book_name}_{character_name}_{i}"
        
        logger.info(f"Hybrid chunking total: {len(all_chunks)} chunks")
        return all_chunks
    
    def extract_temporal_markers(self, text: str) -> List[Dict]:
        """
        Extract temporal information from text
        Used for timeline reasoning
        """
        temporal_patterns = [
            (r'at (\d+)', 'age'),
            (r'(\d+) years? old', 'age'),
            (r'in (\d{4})', 'year'),
            (r'when (?:he|she) was (\d+)', 'age'),
            (r'(before|after) (.+)', 'sequence'),
            (r'during (.+)', 'period')
        ]
        
        markers = []
        for pattern, marker_type in temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                markers.append({
                    'type': marker_type,
                    'value': match.group(1) if match.groups() else match.group(0),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return markers


if __name__ == "__main__":
    from loguru import logger
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL)
    
    chunker = MultiStrategyChunker()
    
    # Test with sample text
    sample_text = """
    Chapter 1. The Beginning
    
    John was a young man of twenty-five years old. He lived in Paris during 1850.
    Before the revolution, he studied at the university.
    
    Chapter 2. The Journey
    
    At 30, John decided to travel. He met Mary in Rome.
    """
    
    chunks = chunker.chunk_hybrid(sample_text, "Test Book", "John")
    logger.info(f"Created {len(chunks)} chunks")
    
    markers = chunker.extract_temporal_markers(sample_text)
    logger.info(f"Found {len(markers)} temporal markers")
