"""
Advanced Reasoning Module
Multi-perspective consistency checking with adversarial agents
This is the CORE INNOVATION beyond basic RAG
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from loguru import logger
import config

try:
    from transformers import pipeline
    NLI_AVAILABLE = True
except:
    NLI_AVAILABLE = False

try:
    import openai
    if config.OPENAI_API_KEY:
        openai.api_key = config.OPENAI_API_KEY
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

try:
    from groq import Groq
    if config.GROQ_API_KEY:
        GROQ_CLIENT = Groq(api_key=config.GROQ_API_KEY)
        GROQ_AVAILABLE = True
    else:
        GROQ_AVAILABLE = False
except:
    GROQ_AVAILABLE = False


class AdversarialReasoningFramework:
    """
    Prosecutor-Defense-Judge framework for robust consistency checking
    Three agents debate the consistency of the backstory
    """
    
    def __init__(self):
        self.nli_model = None
        self.nli_type = None  # Track if using cross-encoder or zero-shot
        
        if NLI_AVAILABLE and getattr(config, 'USE_NLI_MODEL', False):
            try:
                logger.info(f"Loading NLI model: {config.NLI_MODEL}...")
                
                # Check if it's a cross-encoder model
                if 'cross-encoder' in config.NLI_MODEL.lower():
                    # Cross-encoders work differently - they need sentence-transformers
                    from sentence_transformers import CrossEncoder
                    self.nli_model = CrossEncoder(config.NLI_MODEL)
                    self.nli_type = 'cross-encoder'
                    logger.info("Loaded cross-encoder NLI model")
                else:
                    # Standard zero-shot classification pipeline
                    self.nli_model = pipeline("text-classification", model=config.NLI_MODEL, device=-1)
                    self.nli_type = 'zero-shot'
                    logger.info("Loaded zero-shot NLI model")
                    
            except Exception as e:
                logger.warning(f"Could not load NLI model: {e}")
                logger.info("Using fallback methods for contradiction detection")
        else:
            logger.info("NLI model disabled - using fallback methods for memory efficiency")
        
        logger.info("Initialized Adversarial Reasoning Framework")
    
    def prosecutor_agent(self, backstory: str, evidence: Dict[str, List[Dict]]) -> Dict:
        """
        PROSECUTOR: Actively searches for contradictions and inconsistencies
        Returns evidence AGAINST the backstory
        """
        logger.info("🔴 Prosecutor: Searching for contradictions...")
        
        contradictions = []
        suspicions = []
        
        # Check contradiction-mined chunks first
        for chunk in evidence.get('contradictions', []):
            score = self._check_contradiction(backstory, chunk['text'])
            if score > 0.6:  # High contradiction score
                contradictions.append({
                    'chunk': chunk,
                    'contradiction_score': score,
                    'reason': 'Direct contradiction detected'
                })
        
        # Check targeted evidence for inconsistencies
        for chunk in evidence.get('targeted_evidence', []):
            score = self._check_contradiction(backstory, chunk['text'])
            if score > 0.4:  # Medium contradiction score
                suspicions.append({
                    'chunk': chunk,
                    'contradiction_score': score,
                    'reason': 'Potential inconsistency'
                })
        
        # Check for timeline violations
        temporal_issues = self._check_temporal_consistency(backstory, evidence)
        
        prosecution_strength = len(contradictions) * 0.5 + len(suspicions) * 0.2 + len(temporal_issues) * 0.3
        
        logger.info(f"🔴 Prosecutor found: {len(contradictions)} contradictions, {len(suspicions)} suspicions")
        
        return {
            'contradictions': contradictions,
            'suspicions': suspicions,
            'temporal_issues': temporal_issues,
            'strength': min(prosecution_strength, 1.0)
        }
    
    def defense_agent(self, backstory: str, evidence: Dict[str, List[Dict]]) -> Dict:
        """
        DEFENSE: Searches for supporting evidence and consistency
        Returns evidence FOR the backstory
        """
        logger.info("🟢 Defense: Searching for supporting evidence...")
        
        supports = []
        plausible_links = []
        
        # Check broad context for support
        for chunk in evidence.get('broad_context', []):
            score = self._check_entailment(backstory, chunk['text'])
            if score > 0.5:  # Good support score
                supports.append({
                    'chunk': chunk,
                    'support_score': score,
                    'reason': 'Consistent with narrative'
                })
        
        # Check causal neighbors for plausibility
        for chunk in evidence.get('causal_neighbors', []):
            score = self._check_entailment(backstory, chunk['text'])
            if score > 0.3:  # Moderate support
                plausible_links.append({
                    'chunk': chunk,
                    'support_score': score,
                    'reason': 'Plausible connection'
                })
        
        defense_strength = len(supports) * 0.4 + len(plausible_links) * 0.2
        
        logger.info(f"🟢 Defense found: {len(supports)} supports, {len(plausible_links)} plausible links")
        
        return {
            'supports': supports,
            'plausible_links': plausible_links,
            'strength': min(defense_strength, 1.0)
        }
    
    def judge_agent(self, prosecution: Dict, defense: Dict, backstory: str, evidence: Dict) -> Dict:
        """
        JUDGE: Weighs both sides and makes final judgment
        Uses weighted scoring from both perspectives
        """
        logger.info("⚖️  Judge: Weighing evidence...")
        
        # Calculate weighted scores
        prosecution_weight = prosecution['strength']
        defense_weight = defense['strength']
        
        # Strong contradictions should outweigh general support
        has_strong_contradiction = any(c['contradiction_score'] > 0.7 for c in prosecution['contradictions'])
        
        if has_strong_contradiction:
            prosecution_weight *= 1.5
        
        # Calculate final score (higher = more consistent)
        consistency_score = (defense_weight - prosecution_weight + 1.0) / 2.0
        consistency_score = np.clip(consistency_score, 0.0, 1.0)
        
        # Determine verdict
        verdict = "consistent" if consistency_score >= config.CONSISTENCY_THRESHOLD else "contradict"
        
        logger.info(f"⚖️  Judge verdict: {verdict} (score: {consistency_score:.3f})")
        
        return {
            'verdict': verdict,
            'consistency_score': consistency_score,
            'prosecution_strength': prosecution_weight,
            'defense_strength': defense_weight,
            'has_strong_contradiction': has_strong_contradiction
        }
    
    def _check_contradiction(self, premise: str, hypothesis: str) -> float:
        """
        Check if hypothesis contradicts premise using NLI model
        Returns contradiction score (0-1)
        """
        if not self.nli_model:
            # Fallback: simple keyword-based check
            return self._fallback_contradiction_check(premise, hypothesis)
        
        try:
            if self.nli_type == 'cross-encoder':
                # Cross-encoder: returns raw score for entailment
                scores = self.nli_model.predict([[premise, hypothesis]])
                # Handle numpy array
                if hasattr(scores, 'ndim'):
                    if scores.ndim == 0:
                        score = float(scores)
                    else:
                        score = float(scores.flatten()[0])
                else:
                    score = float(scores)
                # THIS MODEL: Positive score = contradiction, Negative = entailment
                # So we need to use positive scores as contradiction
                if score > 0:
                    return min(1.0, score / 10.0)  # Normalize and cap at 1.0
                else:
                    return 0.0  # Negative = entailment/no contradiction
            else:
                # Zero-shot classification
                result = self.nli_model(f"{premise} [SEP] {hypothesis}")[0]
                
                # Map labels to scores
                if result['label'] == 'CONTRADICTION':
                    return result['score']
                elif result['label'] == 'ENTAILMENT':
                    return 0.0
                else:  # NEUTRAL
                    return 0.3
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            return self._fallback_contradiction_check(premise, hypothesis)
    
    def _check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Check if hypothesis is entailed by premise
        Returns entailment score (0-1)
        """
        if not self.nli_model:
            return self._fallback_entailment_check(premise, hypothesis)
        
        try:
            if self.nli_type == 'cross-encoder':
                # Cross-encoder: check for entailment
                scores = self.nli_model.predict([[premise, hypothesis]])
                # Handle numpy array
                if hasattr(scores, 'ndim'):
                    if scores.ndim == 0:
                        score = float(scores)
                    else:
                        score = float(scores.flatten()[0])
                else:
                    score = float(scores)
                # THIS MODEL: Negative score = entailment, Positive = contradiction
                # We want entailment score, so use negative scores
                if score < 0:
                    return min(1.0, abs(score) / 10.0)  # Normalize
                else:
                    return 0.0  # Positive = contradiction/no entailment
            else:
                # Zero-shot classification
                result = self.nli_model(f"{premise} [SEP] {hypothesis}")[0]
                
                if result['label'] == 'ENTAILMENT':
                    return result['score']
                elif result['label'] == 'CONTRADICTION':
                    return 0.0
                else:  # NEUTRAL
                    return 0.3
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            return self._fallback_entailment_check(premise, hypothesis)
            return self._fallback_entailment_check(premise, hypothesis)
        
        try:
            result = self.nli_model(f"{premise} [SEP] {hypothesis}")[0]
            
            if result['label'] == 'ENTAILMENT':
                return result['score']
            elif result['label'] == 'CONTRADICTION':
                return 0.0
            else:  # NEUTRAL
                return 0.3
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            return self._fallback_entailment_check(premise, hypothesis)
    
    def _check_temporal_consistency(self, backstory: str, evidence: Dict) -> List[Dict]:
        """
        Check for timeline violations
        """
        issues = []
        
        # Extract ages/dates from backstory
        backstory_ages = re.findall(r'(\d+)\s+years?\s+old', backstory)
        backstory_years = re.findall(r'in\s+(\d{4})', backstory)
        
        # Check against evidence
        for chunk in evidence.get('broad_context', [])[:5]:
            chunk_ages = re.findall(r'(\d+)\s+years?\s+old', chunk['text'])
            chunk_years = re.findall(r'in\s+(\d{4})', chunk['text'])
            
            # Simple check: if ages differ significantly
            for b_age in backstory_ages:
                for c_age in chunk_ages:
                    if abs(int(b_age) - int(c_age)) > 10:
                        issues.append({
                            'type': 'age_mismatch',
                            'backstory_age': b_age,
                            'novel_age': c_age,
                            'chunk': chunk
                        })
        
        return issues
    
    def _fallback_contradiction_check(self, premise: str, hypothesis: str) -> float:
        """
        Fallback contradiction detection using keywords
        """
        negation_words = ['never', 'not', 'no', 'cannot', 'didn\'t', 'wasn\'t', 'weren\'t']
        
        score = 0.0
        for word in negation_words:
            if word in hypothesis.lower():
                score += 0.2
        
        return min(score, 0.8)
    
    def _fallback_entailment_check(self, premise: str, hypothesis: str) -> float:
        """
        Fallback entailment check using simple overlap
        """
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        
        overlap = len(premise_words & hypothesis_words)
        score = overlap / max(len(hypothesis_words), 1)
        
        return min(score, 0.7)


class ConsistencyScoringEngine:
    """
    Multi-metric scoring system
    Beyond single-pass LLM calls - uses multiple specialized scores
    """
    
    def __init__(self):
        self.adversarial_framework = AdversarialReasoningFramework()
        logger.info("Initialized Consistency Scoring Engine")
    
    def score_direct_contradiction(self, backstory: str, evidence: Dict) -> float:
        """
        Score 1: Direct contradiction detection
        Weight: 30%
        """
        prosecution = self.adversarial_framework.prosecutor_agent(backstory, evidence)
        score = 1.0 - prosecution['strength']  # Invert: lower contradictions = higher score
        return score
    
    def score_causal_plausibility(self, backstory: str, evidence: Dict) -> float:
        """
        Score 2: Causal plausibility
        Weight: 25%
        Checks if backstory makes later events plausible
        """
        # Check if causal neighbors support the backstory
        causal_chunks = evidence.get('causal_neighbors', [])
        
        if not causal_chunks:
            return 0.5  # Neutral if no causal evidence
        
        support_count = 0
        for chunk in causal_chunks:
            # Simple check: does chunk mention character positively?
            if any(word in chunk['text'].lower() for word in ['because', 'therefore', 'thus', 'led to']):
                support_count += 1
        
        score = support_count / max(len(causal_chunks), 1)
        return score
    
    def score_character_consistency(self, backstory: str, evidence: Dict, character_name: str) -> float:
        """
        Score 3: Character trait consistency
        Weight: 20%
        """
        # Extract character mentions from broad context
        character_chunks = evidence.get('broad_context', [])
        
        if not character_chunks:
            return 0.5
        
        # Simple consistency: check if character traits align
        defense = self.adversarial_framework.defense_agent(backstory, evidence)
        score = defense['strength']
        
        return score
    
    def score_temporal_coherence(self, backstory: str, evidence: Dict) -> float:
        """
        Score 4: Temporal coherence
        Weight: 15%
        """
        temporal_issues = self.adversarial_framework._check_temporal_consistency(backstory, evidence)
        
        if not temporal_issues:
            return 1.0  # Perfect score if no temporal issues
        
        # Penalize each issue
        score = max(0.0, 1.0 - len(temporal_issues) * 0.3)
        return score
    
    def score_narrative_fit(self, backstory: str, evidence: Dict) -> float:
        """
        Score 5: Overall narrative fit
        Weight: 10%
        """
        # Check if backstory tone matches novel tone
        # Simplified: check retrieval scores
        all_chunks = []
        for category in evidence.values():
            all_chunks.extend(category)
        
        if not all_chunks:
            return 0.5
        
        avg_retrieval_score = np.mean([c.get('retrieval_score', 0.5) for c in all_chunks])
        return float(avg_retrieval_score)
    
    def compute_ensemble_score(self, backstory: str, evidence: Dict, character_name: str) -> Dict:
        """
        Compute weighted ensemble of all scores
        """
        logger.info("Computing ensemble consistency score...")
        
        scores = {
            'contradiction': self.score_direct_contradiction(backstory, evidence),
            'causal': self.score_causal_plausibility(backstory, evidence),
            'character': self.score_character_consistency(backstory, evidence, character_name),
            'temporal': self.score_temporal_coherence(backstory, evidence),
            'narrative': self.score_narrative_fit(backstory, evidence)
        }
        
        # Weighted average
        weights = {
            'contradiction': config.WEIGHT_CONTRADICTION,
            'causal': config.WEIGHT_CAUSAL,
            'character': config.WEIGHT_CHARACTER,
            'temporal': config.WEIGHT_TEMPORAL,
            'narrative': config.WEIGHT_NARRATIVE
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        
        # Also get adversarial judgment
        prosecution = self.adversarial_framework.prosecutor_agent(backstory, evidence)
        defense = self.adversarial_framework.defense_agent(backstory, evidence)
        judgment = self.adversarial_framework.judge_agent(prosecution, defense, backstory, evidence)
        
        logger.info(f"Ensemble scores: {scores}")
        logger.info(f"Final score: {final_score:.3f}, Adversarial score: {judgment['consistency_score']:.3f}")
        
        # Combine both approaches
        combined_score = (final_score + judgment['consistency_score']) / 2.0
        
        return {
            'individual_scores': scores,
            'ensemble_score': final_score,
            'adversarial_score': judgment['consistency_score'],
            'combined_score': combined_score,
            'judgment': judgment,
            'is_consistent': combined_score >= config.CONSISTENCY_THRESHOLD
        }


if __name__ == "__main__":
    from loguru import logger
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL)
    
    # Test reasoning engine
    scorer = ConsistencyScoringEngine()
    
    test_backstory = "John was a peaceful scholar who never traveled."
    test_evidence = {
        'broad_context': [{'text': 'John was known for his extensive travels across Europe.', 'retrieval_score': 0.8}],
        'targeted_evidence': [],
        'contradictions': [{'text': 'John was an adventurer', 'retrieval_score': 0.7}],
        'causal_neighbors': []
    }
    
    result = scorer.compute_ensemble_score(test_backstory, test_evidence, "John")
    logger.info(f"Test result: {result['is_consistent']}")
