"""
Main Pipeline Runner
Orchestrates the entire narrative consistency checking system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import pathway as pw
from tqdm import tqdm
from loguru import logger
from typing import Dict, List
import config

from src.ingest import NarrativeDataIngester
from src.chunking import MultiStrategyChunker
from src.retrieval import PathwayVectorStore, MultiStageRetriever
from src.reasoning import ConsistencyScoringEngine
from src.decision import DecisionAggregator


class NarrativeConsistencyPipeline:
    """
    Complete pipeline for narrative consistency verification
    Integrates all components with Pathway
    """
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("Initializing Narrative Consistency Pipeline with Pathway")
        logger.info("=" * 80)
        
        # Initialize components
        self.ingester = NarrativeDataIngester()
        self.chunker = MultiStrategyChunker()
        self.vector_store = PathwayVectorStore()
        self.retriever = MultiStageRetriever(self.vector_store)
        self.scorer = ConsistencyScoringEngine()
        self.decision_maker = DecisionAggregator()
        
        # Cache for book chunks
        self.book_chunks_cache = {}
        
        logger.info("Pipeline initialization complete!")
    
    def prepare_book_index(self, book_name: str, character_name: str):
        """
        Prepare and index a book for retrieval
        This is done once per book-character pair
        """
        cache_key = f"{book_name}_{character_name}"
        
        if cache_key in self.book_chunks_cache:
            logger.info(f"Using cached chunks for {book_name} - {character_name}")
            return
        
        logger.info(f"Preparing index for {book_name} - {character_name}")
        
        # Load book text
        book_text = self.ingester.load_book_text(book_name)
        
        if not book_text:
            logger.error(f"Could not load book: {book_name}")
            return
        
        # Create hybrid chunks
        chunks = self.chunker.chunk_hybrid(book_text, book_name, character_name)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        # Cache
        self.book_chunks_cache[cache_key] = chunks
        
        logger.info(f"Indexed {len(chunks)} chunks for {book_name}")
    
    def process_single_example(self, example: Dict) -> Dict:
        """
        Process a single backstory-novel pair
        Returns prediction and all intermediate results
        """
        book_name = example['book_name']
        character = example['character']
        backstory = example['backstory']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {book_name} - {character}")
        logger.info(f"{'='*60}")
        
        # Step 1: Prepare book index
        self.prepare_book_index(book_name, character)
        
        # Step 2: Retrieve evidence
        logger.info("Stage: Multi-stage retrieval")
        evidence = self.retriever.retrieve_comprehensive(backstory, character)
        
        # Step 3: Reasoning and scoring
        logger.info("Stage: Adversarial reasoning & scoring")
        reasoning_result = self.scorer.compute_ensemble_score(backstory, evidence, character)
        
        # Step 4: Final decision
        logger.info("Stage: Final decision")
        decision = self.decision_maker.make_decision(reasoning_result,evidence=evidence)
        
        # Combine all results
        result = {
            'id': example['id'],
            'book_name': book_name,
            'character': character,
            'backstory': backstory[:100] + '...',  # Truncate for logging
            'prediction': decision['prediction'],
            'label': decision['label'],
            'confidence': decision['confidence'],
            'scores': reasoning_result['individual_scores'],
            'explanation': decision['explanation']
        }
        
        if 'label' in example:
            result['true_label'] = example['label']
            result['correct'] = (decision['label'] == example['label'])
        
        logger.info(f"Result: {decision['label']} (confidence: {decision['confidence']:.3f})")
        
        return result
    
    def run_on_train(self, limit: int = None):
        """
        Run pipeline on training data for validation and threshold tuning
        """
        logger.info("\n" + "=" * 80)
        logger.info("Running on TRAINING data")
        logger.info("=" * 80 + "\n")
        
        # Load training data
        train_data = self.ingester.load_train_data()
        
        if limit:
            train_data = train_data[:limit]
        
        results = []
        correct = 0
        total = 0
        
        for example in tqdm(train_data, desc="Processing training examples"):
            try:
                result = self.process_single_example(example)
                results.append(result)
                
                if result.get('correct'):
                    correct += 1
                total += 1
                
            except Exception as e:
                logger.error(f"Error processing example {example['id']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Accuracy: {accuracy:.3f} ({correct}/{total})")
        logger.info(f"{'='*80}\n")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_path = config.BASE_DIR / "train_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Training results saved to: {results_path}")
        
        return results
    
    def run_on_test(self):
        """
        Run pipeline on test data and generate submission file
        """
        logger.info("\n" + "=" * 80)
        logger.info("Running on TEST data")
        logger.info("=" * 80 + "\n")
        
        # Load test data
        test_data = self.ingester.load_test_data()
        
        results = []
        
        for example in tqdm(test_data, desc="Processing test examples"):
            try:
                result = self.process_single_example(example)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing example {example['id']}: {e}")
                import traceback
                traceback.print_exc()
                # Add default prediction on error
                results.append({
                    'id': example['id'],
                    'prediction': 1,  # Default to consistent
                    'label': 'consistent'
                })
        
        # Create submission file
        submission = pd.DataFrame([
        {
            'id': r['id'],
            'label': r['label'],
            'rationale': r.get('rationale', '')  # ADD THIS
        }
        for r in results
    ])
        
        submission.to_csv(config.RESULTS_CSV, index=False)
        logger.info(f"\n{'='*80}")
        logger.info(f"Test results saved to: {config.RESULTS_CSV}")
        logger.info(f"Total predictions: {len(submission)}")
        logger.info(f"{'='*80}\n")
        
        # Save detailed results
        detailed_results_df = pd.DataFrame(results)
        detailed_path = config.BASE_DIR / "test_results_detailed.csv"
        detailed_results_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed results saved to: {detailed_path}")
        
        return results
# In src/run.py - ADD THIS
def aggregate_results_with_pathway(self, results: List[Dict]) -> pw.Table:
    """
    Use Pathway to aggregate and analyze results
    Shows orchestration capability
    """
    results_table = pw.debug.table_from_rows(
        schema=pw.schema_from_types(
            id=int,
            book_name=str,
            prediction=int,
            confidence=float
        ),
        rows=results
    )
    
    # PATHWAY AGGREGATION
    summary = results_table.groupby(pw.this.book_name).reduce(
        book_name=pw.this.book_name,
        total_predictions=pw.reducers.count(),
        avg_confidence=pw.reducers.avg(pw.this.confidence),
        consistent_count=pw.reducers.sum(
            pw.if_else(pw.this.prediction == 1, 1, 0)
        )
    )
    
    return summary    


def main():
    """Main entry point"""
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL, rotation="10 MB")
    logger.info("Starting Narrative Consistency System")
    logger.info(f"Configuration: {config.__file__}")
    
    # Create pipeline
    pipeline = NarrativeConsistencyPipeline()
    
    # Option 1: Run on training data first (for validation)
    logger.info("\nRun mode selection:")
    logger.info("1. Test on training data (with accuracy)")
    logger.info("2. Generate predictions for test data")
    logger.info("3. Both\n")
    
    mode = input("Select mode (1/2/3): ").strip()
    
    if mode == "1":
        # Run on training data
        pipeline.run_on_train(limit=None)  # Remove limit to use all data
    
    elif mode == "2":
        # Run on test data
        pipeline.run_on_test()
    
    elif mode == "3":
        # Both
        logger.info("First, running on training data for validation...")
        pipeline.run_on_train(limit=10)  # Small sample for speed
        
        logger.info("\nNow running on test data...")
        pipeline.run_on_test()
    
    else:
        logger.warning("Invalid mode selected. Running on test data by default.")
        pipeline.run_on_test()
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline execution complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
