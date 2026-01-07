"""
Decision Module
Final classification using ensemble aggregation
"""

from typing import Dict, List
import numpy as np
from loguru import logger
import config


class DecisionAggregator:
    """
    Aggregates all evidence and scores to make final binary decision
    """
    
    def __init__(self):
        self.threshold = config.CONSISTENCY_THRESHOLD
        logger.info(f"Initialized Decision Aggregator (threshold={self.threshold})")
    
    def make_decision(self, reasoning_result: Dict) -> Dict:
        """
        Make final binary decision based on all evidence
        
        Returns:
            {
                'prediction': int (0 or 1),
                'confidence': float,
                'explanation': str
            }
        """
        # Get combined score from reasoning
        combined_score = reasoning_result['combined_score']
        adversarial_score = reasoning_result['adversarial_score']
        ensemble_score = reasoning_result['ensemble_score']
        
        # Make decision
        is_consistent = combined_score >= self.threshold
        prediction = 1 if is_consistent else 0
        
        # Calculate confidence
        # Distance from threshold indicates confidence
        confidence = abs(combined_score - self.threshold)
        confidence = min(confidence * 2, 1.0)  # Scale to 0-1
        
        # Generate explanation
        explanation = self._generate_explanation(reasoning_result, prediction)
        
        logger.info(f"Decision: {prediction} ({'consistent' if prediction == 1 else 'contradict'}), "
                   f"Confidence: {confidence:.3f}")
        
        if config.GENERATE_RATIONALE and evidence: 

            explanation = self.generate_comprehensive_rationale(
            reasoning_result, evidence
        )
        return {
            'prediction': prediction,
            'label': 'consistent' if prediction == 1 else 'contradict',
            'confidence': float(confidence),
            'rationale': explanation,
            'combined_score': float(combined_score),
            'adversarial_score': float(adversarial_score),
            'ensemble_score': float(ensemble_score),
            'explanation': explanation
        }
    
    def _generate_explanation(self, reasoning_result: Dict, prediction: int) -> str:
        """
        Generate human-readable explanation for the decision
        """
        scores = reasoning_result['individual_scores']
        judgment = reasoning_result['judgment']
        
        if prediction == 1:
            # Consistent
            strongest_support = max(scores.items(), key=lambda x: x[1])
            explanation = (f"Backstory is CONSISTENT with the narrative. "
                          f"Strongest support: {strongest_support[0]} (score: {strongest_support[1]:.2f}). "
                          f"No strong contradictions detected.")
        else:
            # Contradict
            weakest_score = min(scores.items(), key=lambda x: x[1])
            explanation = (f"Backstory CONTRADICTS the narrative. "
                          f"Primary issue: {weakest_score[0]} (score: {weakest_score[1]:.2f}). ")
            
            if judgment.get('has_strong_contradiction'):
                explanation += "Strong direct contradictions found in text."
        
        return explanation
    


    def generate_comprehensive_rationale(self, reasoning_result: Dict, evidence: Dict) -> str:
        """Generate comprehensive evidence rationale as per problem statement
        Format: Excerpts + Linkage + Analysis"""
        prosecution = reasoning_result.get('judgment', {}).get('prosecution_strength', 0)
        defense = reasoning_result.get('judgment', {}).get('defense_strength', 0)
        
        rationale_parts = []
        
        # 1. KEY CONTRADICTIONS (if any)
        if prosecution > 0.3:
            rationale_parts.append("CONTRADICTIONS FOUND:")
            contradictions = evidence.get('contradictions', [])[:3]
            for i, chunk in enumerate(contradictions, 1):
                excerpt = chunk['text'][:200] + "..."
                rationale_parts.append(f"{i}. Excerpt: \"{excerpt}\"")
                rationale_parts.append(f"   Analysis: This contradicts the backstory claim")
        
        # 2. SUPPORTING EVIDENCE (if any)
        if defense > 0.2:
            rationale_parts.append("\nSUPPORTING EVIDENCE:")
            supports = evidence.get('broad_context', [])[:3]
            for i, chunk in enumerate(supports, 1):
                excerpt = chunk['text'][:200] + "..."
                score = chunk.get('retrieval_score', 0)
                rationale_parts.append(f"{i}. Excerpt: \"{excerpt}\"")
                rationale_parts.append(f"   Relevance: {score:.2f}")
                rationale_parts.append(f"   Analysis: Consistent with backstory")
        
        # 3. OVERALL VERDICT
        verdict = reasoning_result.get('judgment', {}).get('verdict', 'unknown')
        scores = reasoning_result.get('individual_scores', {})
        
        rationale_parts.append(f"\nFINAL ANALYSIS:")
        rationale_parts.append(f"- Contradiction Score: {scores.get('contradiction', 0):.2f}")
        rationale_parts.append(f"- Causal Plausibility: {scores.get('causal', 0):.2f}")
        rationale_parts.append(f"- Character Consistency: {scores.get('character', 0):.2f}")
        rationale_parts.append(f"- Temporal Coherence: {scores.get('temporal', 0):.2f}")
        rationale_parts.append(f"- Verdict: {verdict.upper()}")
        
        return "\n".join(rationale_parts)

# ADD THIS METHOD TO DecisionAggregator class in src/decision.py
    
    def batch_decision(self, reasoning_results: List[Dict]) -> List[Dict]:
        """
        Make decisions for multiple examples
        """
        decisions = []
        for result in reasoning_results:
            decision = self.make_decision(result)
            decisions.append(decision)
        
        logger.info(f"Batch decisions complete: {len(decisions)} examples")
        return decisions
    
    def calibrate_threshold(self, validation_data: List[Dict], validation_labels: List[int]) -> float:
        """
        Optimize decision threshold on validation data
        """
        # Try different thresholds
        thresholds = np.linspace(0.3, 0.7, 20)
        best_accuracy = 0
        best_threshold = self.threshold
        
        for thresh in thresholds:
            self.threshold = thresh
            correct = 0
            
            for result, true_label in zip(validation_data, validation_labels):
                decision = self.make_decision(result)
                if decision['prediction'] == true_label:
                    correct += 1
            
            accuracy = correct / len(validation_labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = thresh
        
        self.threshold = best_threshold
        logger.info(f"Calibrated threshold: {best_threshold:.3f} (accuracy: {best_accuracy:.3f})")
        
        return best_threshold


class ConfidenceCalibrator:
    """
    Calibrate confidence scores to match actual accuracy
    """
    
    def __init__(self):
        self.calibration_map = {}
        logger.info("Initialized Confidence Calibrator")
    
    def fit(self, predicted_confidences: List[float], actual_correctness: List[bool]):
        """
        Learn calibration mapping from validation data
        """
        # Bin confidences and calculate actual accuracy per bin
        bins = np.linspace(0, 1, 11)
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i+1]
            
            # Find predictions in this bin
            in_bin = [(conf, correct) for conf, correct in zip(predicted_confidences, actual_correctness)
                     if bin_start <= conf < bin_end]
            
            if in_bin:
                actual_accuracy = sum(correct for _, correct in in_bin) / len(in_bin)
                self.calibration_map[(bin_start, bin_end)] = actual_accuracy
        
        logger.info(f"Calibration map created with {len(self.calibration_map)} bins")
    
    def calibrate(self, confidence: float) -> float:
        """
        Return calibrated confidence score
        """
        if not self.calibration_map:
            return confidence
        
        # Find the right bin
        for (bin_start, bin_end), actual_acc in self.calibration_map.items():
            if bin_start <= confidence < bin_end:
                return actual_acc
        
        return confidence


if __name__ == "__main__":
    from loguru import logger
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL)
    
    # Test decision aggregator
    aggregator = DecisionAggregator()
    
    test_result = {
        'individual_scores': {
            'contradiction': 0.8,
            'causal': 0.7,
            'character': 0.6,
            'temporal': 0.9,
            'narrative': 0.7
        },
        'ensemble_score': 0.75,
        'adversarial_score': 0.70,
        'combined_score': 0.725,
        'judgment': {
            'verdict': 'consistent',
            'has_strong_contradiction': False
        }
    }
    
    decision = aggregator.make_decision(test_result)
    logger.info(f"Test decision: {decision}")
