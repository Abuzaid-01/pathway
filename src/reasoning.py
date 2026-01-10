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
    GROQ_CLIENTS = []
    for key in config.API_KEYS.get('groq', []):
        if key:
            GROQ_CLIENTS.append(Groq(api_key=key))
    GROQ_AVAILABLE = len(GROQ_CLIENTS) > 0
except:
    GROQ_AVAILABLE = False
    GROQ_CLIENTS = []

try:
    import google.generativeai as genai
    GEMINI_CLIENTS = []
    for key in config.API_KEYS.get('gemini', []):
        if key:
            genai.configure(api_key=key)
            GEMINI_CLIENTS.append(genai.GenerativeModel(config.GEMINI_MODEL))
    GEMINI_AVAILABLE = len(GEMINI_CLIENTS) > 0
except:
    GEMINI_AVAILABLE = False
    GEMINI_CLIENTS = []


class AdversarialReasoningFramework:
    """
    Prosecutor-Defense-Judge framework for robust consistency checking
    ENHANCED with LLM API for deep reasoning
    """
    
    def __init__(self):
        self.nli_model = None
        self.nli_type = None  # Track if using cross-encoder or zero-shot
        self.use_llm_api = getattr(config, 'USE_LLM_API', False)
        self.api_rotation_index = 0  # For rotating between API keys
        
        if NLI_AVAILABLE and getattr(config, 'USE_NLI_MODEL', False):
            try:
                logger.info(f"Loading NLI model: {config.NLI_MODEL}...")
                
                # Check if it's a cross-encoder model
                if 'cross-encoder' in config.NLI_MODEL.lower():
                    # Cross-encoders work differently - they need sentence-transformers
                    from sentence_transformers import CrossEncoder
                    self.nli_model = CrossEncoder(config.NLI_MODEL)
                    self.nli_type = 'cross-encoder'
                    logger.info(f"✅ Loaded cross-encoder NLI model: {config.NLI_MODEL}")
                else:
                    # Standard zero-shot classification pipeline
                    self.nli_model = pipeline("text-classification", model=config.NLI_MODEL, device=-1)
                    self.nli_type = 'zero-shot'
                    logger.info("Loaded zero-shot NLI model")
                    
            except Exception as e:
                logger.warning(f"Could not load NLI model: {e}")
                logger.info("Using fallback methods for contradiction detection")
        else:
            logger.info("NLI model disabled - using fallback methods")
        
        # Initialize LLM API with rotation support
        if self.use_llm_api:
            self.available_apis = []
            
            # Setup Groq clients
            if GROQ_AVAILABLE and config.LLM_PROVIDER == "groq":
                for i, client in enumerate(GROQ_CLIENTS):
                    self.available_apis.append(('groq', client, i))
                logger.info(f"✅ LLM API enabled: {len(GROQ_CLIENTS)} Groq key(s) ({config.LLM_MODEL})")
            
            # Setup Gemini as fallback
            if GEMINI_AVAILABLE:
                for i, client in enumerate(GEMINI_CLIENTS):
                    self.available_apis.append(('gemini', client, i))
                logger.info(f"✅ Gemini API available: {len(GEMINI_CLIENTS)} key(s) as fallback")
            
            # Setup OpenAI
            if LLM_AVAILABLE and config.LLM_PROVIDER == "openai":
                self.available_apis.append(('openai', openai, 0))
                logger.info(f"✅ LLM API enabled: OpenAI ({config.LLM_MODEL})")
            
            if not self.available_apis:
                logger.warning("LLM API requested but no keys available - disabling")
                self.use_llm_api = False
        
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
    
    # def defense_agent(self, backstory: str, evidence: Dict[str, List[Dict]]) -> Dict:
    #     """
    #     DEFENSE: Searches for supporting evidence and consistency
    #     Returns evidence FOR the backstory
    #     """
    #     logger.info("🟢 Defense: Searching for supporting evidence...")
        
    #     supports = []
    #     plausible_links = []
        
    #     # Check broad context for support
    #     for chunk in evidence.get('broad_context', []):
    #         score = self._check_entailment(backstory, chunk['text'])
    #         if score > 0.5:  # Good support score
    #             supports.append({
    #                 'chunk': chunk,
    #                 'support_score': score,
    #                 'reason': 'Consistent with narrative'
    #             })
        
    #     # Check causal neighbors for plausibility
    #     for chunk in evidence.get('causal_neighbors', []):
    #         score = self._check_entailment(backstory, chunk['text'])
    #         if score > 0.3:  # Moderate support
    #             plausible_links.append({
    #                 'chunk': chunk,
    #                 'support_score': score,
    #                 'reason': 'Plausible connection'
    #             })
        
    #     defense_strength = len(supports) * 0.4 + len(plausible_links) * 0.2
        
    #     logger.info(f"🟢 Defense found: {len(supports)} supports, {len(plausible_links)} plausible links")
        
    #     return {
    #         'supports': supports,
    #         'plausible_links': plausible_links,
    #         'strength': min(defense_strength, 1.0)
    #     }
    
    def defense_agent(self, backstory: str, evidence: Dict[str, List[Dict]]) -> Dict:
        """
        DEFENSE: Searches for supporting evidence and consistency
        FIXED: More aggressive in finding support
        """
        logger.info("🟢 Defense: Searching for supporting evidence...")
        
        supports = []
        plausible_links = []
        
        # EXPANDED SEARCH: Check ALL evidence categories
        all_evidence = []
        for category in ['broad_context', 'targeted_evidence', 'causal_neighbors']:
            all_evidence.extend(evidence.get(category, []))
        
        for chunk in all_evidence:
            # LOWERED THRESHOLD: More lenient support detection
            score = self._check_entailment(backstory, chunk['text'])
            
            # Strong support
            if score > 0.3:  # CHANGED from 0.5 to 0.3
                supports.append({
                    'chunk': chunk,
                    'support_score': score,
                    'reason': 'Consistent with narrative'
                })
            # Plausible connection
            elif score > 0.2:  # CHANGED from 0.3 to 0.2
                plausible_links.append({
                    'chunk': chunk,
                    'support_score': score,
                    'reason': 'Plausible connection'
                })
        
        # ENHANCED: Check for semantic similarity too
        for chunk in all_evidence[:10]:
            if not any(s['chunk'].get('global_id') == chunk.get('global_id') for s in supports):
                # Simple word overlap check
                backstory_words = set(backstory.lower().split())
                chunk_words = set(chunk['text'].lower().split())
                overlap = len(backstory_words & chunk_words)
                if overlap > 5:  # At least 5 common words
                    plausible_links.append({
                        'chunk': chunk,
                        'support_score': min(overlap / 20.0, 1.0),
                        'reason': 'Semantic similarity'
                    })
        
        # ADJUSTED SCORING: Give more weight to defense to balance 95% contradict issue
        # Increased multipliers to strengthen defense agent
        defense_strength = len(supports) * 0.7 + len(plausible_links) * 0.4
        
        # Bonus for having any support at all (addresses "0 supports" problem)
        if supports:
            defense_strength += 0.3
        if plausible_links:
            defense_strength += 0.2
        
        logger.info(f"🟢 Defense found: {len(supports)} supports, {len(plausible_links)} plausible links (strength: {defense_strength:.2f})")
        
        return {
            'supports': supports,
            'plausible_links': plausible_links,
            'strength': min(defense_strength, 1.0)
        }



    def judge_agent(self, prosecution: Dict, defense: Dict, backstory: str, evidence: Dict) -> Dict:
        """
        JUDGE: Weighs both sides and makes final judgment
        Uses weighted scoring from both perspectives
        REBALANCED to fix 95% contradict issue
        """
        logger.info("⚖️  Judge: Weighing evidence...")
        
        # Calculate weighted scores
        prosecution_weight = prosecution['strength']
        defense_weight = defense['strength']
        
        # REBALANCED: Only significantly boost prosecution for VERY strong contradictions
        has_strong_contradiction = any(c['contradiction_score'] > 0.8 for c in prosecution['contradictions'])
        
        if has_strong_contradiction:
            prosecution_weight *= 1.3  # Reduced from 1.5 to balance
        
        # REBALANCED: Give more weight to defense
        if defense_weight > 0:
            defense_weight *= 1.2  # Boost defense slightly
        
        # Calculate final score (higher = more consistent)
        consistency_score = (defense_weight - prosecution_weight + 1.0) / 2.0
        consistency_score = np.clip(consistency_score, 0.0, 1.0)
        
        # Determine verdict using updated threshold from config
        verdict = "consistent" if consistency_score >= config.CONSISTENCY_THRESHOLD else "contradict"
        
        logger.info(f"⚖️  Judge verdict: {verdict} (score: {consistency_score:.3f}, threshold: {config.CONSISTENCY_THRESHOLD})")
        logger.info(f"   Prosecution: {prosecution_weight:.2f}, Defense: {defense_weight:.2f}")
        
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
                # Truncate long sequences to avoid warning (512 tokens ≈ 2000 chars)
                max_chars = 2000  # Conservative: ~500 tokens per text
                if len(premise) > max_chars:
                    premise = premise[:max_chars]
                if len(hypothesis) > max_chars:
                    hypothesis = hypothesis[:max_chars]
                
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
                # Truncate long sequences to avoid warning (512 tokens ≈ 2000 chars)
                max_chars = 2000  # Conservative: ~500 tokens per text
                if len(premise) > max_chars:
                    premise = premise[:max_chars]
                if len(hypothesis) > max_chars:
                    hypothesis = hypothesis[:max_chars]
                
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
    
    def llm_deep_analysis(self, backstory: str, evidence: Dict, character_name: str) -> Dict:
        """
        NEW: Use LLM API for deep reasoning and analysis with rotation
        This goes beyond simple pattern matching to understand narrative coherence
        """
        if not self.use_llm_api or not self.available_apis:
            return {'score': 0.5, 'reasoning': 'LLM API not available', 'verdict': 'neutral'}
        
        # Prepare context from top evidence
        evidence_texts = []
        for category in ['targeted_evidence', 'contradictions', 'broad_context']:
            chunks = evidence.get(category, [])[:3]  # Top 3 from each
            for chunk in chunks:
                evidence_texts.append(f"[{category}] {chunk['text'][:300]}")
        
        context = "\n\n".join(evidence_texts[:8])  # Max 8 pieces of evidence
        
        # Construct prompt for deep analysis
        prompt = f"""You are an expert literary analyst. Analyze whether a proposed character backstory is consistent with evidence from a novel.

CHARACTER: {character_name}

PROPOSED BACKSTORY:
{backstory}

EVIDENCE FROM NOVEL:
{context}

TASK: Determine if the backstory is CONSISTENT or CONTRADICTS the novel's narrative.

Consider:
1. Direct contradictions (facts that cannot both be true)
2. Causal coherence (does the backstory make later events plausible?)
3. Character consistency (personality, beliefs, motivations)
4. Temporal logic (timeline makes sense)
5. Narrative constraints (implicit rules of the story world)

Respond in this format:
VERDICT: [CONSISTENT or CONTRADICT]
CONFIDENCE: [0.0-1.0]
REASONING: [2-3 sentences explaining your judgment with specific evidence]"""

        # Try APIs with rotation and fallback
        max_retries = len(self.available_apis)
        for attempt in range(max_retries):
            try:
                # Get next API in rotation
                api_type, client, key_idx = self.available_apis[self.api_rotation_index % len(self.available_apis)]
                self.api_rotation_index += 1
                
                logger.debug(f"Calling {api_type} API (key #{key_idx})")
                
                # Call appropriate API
                if api_type == "groq":
                    response = client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=getattr(config, 'LLM_TEMPERATURE', 0.1),
                        max_tokens=getattr(config, 'LLM_MAX_TOKENS', 500)
                    )
                    result = response.choices[0].message.content
                    
                elif api_type == "gemini":
                    response = client.generate_content(prompt)
                    result = response.text
                    
                elif api_type == "openai":
                    response = client.ChatCompletion.create(
                        model=config.LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=getattr(config, 'LLM_TEMPERATURE', 0.1),
                        max_tokens=getattr(config, 'LLM_MAX_TOKENS', 500)
                    )
                    result = response.choices[0].message.content
                else:
                    continue
                
                # Successfully got response, parse it
                break
                
            except Exception as e:
                logger.warning(f"{api_type} API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # All attempts failed
                    return {'score': 0.5, 'reasoning': f'All API calls failed: {str(e)}', 'verdict': 'neutral'}
                continue
        
        
        # Parse LLM response
        try:
            verdict = 'consistent' if 'CONSISTENT' in result.upper() and 'CONTRADICT' not in result.split('VERDICT:')[1].split('\n')[0].upper() else 'contradict'
            
            # Extract confidence
            confidence = 0.5
            if 'CONFIDENCE:' in result:
                conf_line = result.split('CONFIDENCE:')[1].split('\n')[0].strip()
                try:
                    confidence = float(conf_line)
                except:
                    # Try to extract number from text
                    import re
                    numbers = re.findall(r'0\.\d+|1\.0', conf_line)
                    if numbers:
                        confidence = float(numbers[0])
            
            # Extract reasoning
            reasoning = result
            if 'REASONING:' in result:
                reasoning = result.split('REASONING:')[1].strip()
            
            # Convert verdict to score
            score = confidence if verdict == 'consistent' else (1.0 - confidence)
            
            logger.info(f"🤖 LLM Analysis: {verdict.upper()} (score: {score:.3f})")
            
            return {
                'score': score,
                'reasoning': reasoning[:500],
                'verdict': verdict,
                'confidence': confidence,
                'raw_response': result
            }
            
        except Exception as e:
            logger.error(f"LLM response parsing failed: {e}")
            return {'score': 0.5, 'reasoning': f'Parse error: {str(e)}', 'verdict': 'neutral'}


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
    
    def score_llm_judgment(self, backstory: str, evidence: Dict, character_name: str) -> float:
        """
        Score 6: LLM Deep Reasoning (NEW)
        Weight: 15%
        Uses API-powered language model for sophisticated analysis
        """
        llm_result = self.adversarial_framework.llm_deep_analysis(backstory, evidence, character_name)
        return llm_result['score']
    
    def compute_ensemble_score(self, backstory: str, evidence: Dict, character_name: str) -> Dict:
        """
        Compute weighted ensemble of all scores
        ENHANCED with LLM judgment
        """
        logger.info("Computing ensemble consistency score...")
        
        scores = {
            'contradiction': self.score_direct_contradiction(backstory, evidence),
            'causal': self.score_causal_plausibility(backstory, evidence),
            'character': self.score_character_consistency(backstory, evidence, character_name),
            'temporal': self.score_temporal_coherence(backstory, evidence),
            'narrative': self.score_narrative_fit(backstory, evidence),
        }
        
        # Add LLM score if enabled
        llm_analysis = None
        if getattr(config, 'USE_LLM_API', False):
            llm_analysis = self.adversarial_framework.llm_deep_analysis(backstory, evidence, character_name)
            scores['llm_judgment'] = llm_analysis['score']
        
        # Weighted average
        weights = {
            'contradiction': config.WEIGHT_CONTRADICTION,
            'causal': config.WEIGHT_CAUSAL,
            'character': config.WEIGHT_CHARACTER,
            'temporal': config.WEIGHT_TEMPORAL,
            'narrative': config.WEIGHT_NARRATIVE
        }
        
        # Add LLM weight if used
        if 'llm_judgment' in scores:
            weights['llm_judgment'] = getattr(config, 'WEIGHT_LLM_JUDGMENT', 0.15)
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        
        # Also get adversarial judgment
        prosecution = self.adversarial_framework.prosecutor_agent(backstory, evidence)
        defense = self.adversarial_framework.defense_agent(backstory, evidence)
        judgment = self.adversarial_framework.judge_agent(prosecution, defense, backstory, evidence)
        
        logger.info(f"Ensemble scores: {scores}")
        logger.info(f"Final score: {final_score:.3f}, Adversarial score: {judgment['consistency_score']:.3f}")
        if llm_analysis:
            logger.info(f"LLM Analysis: {llm_analysis['verdict']} (score: {llm_analysis['score']:.3f})")
        
        # Combine both approaches
        combined_score = (final_score + judgment['consistency_score']) / 2.0
        
        return {
            'individual_scores': scores,
            'ensemble_score': final_score,
            'adversarial_score': judgment['consistency_score'],
            'combined_score': combined_score,
            'judgment': judgment,
            'llm_analysis': llm_analysis,
            'is_consistent': combined_score >= config.CONSISTENCY_THRESHOLD,
            'evidence': evidence  # Pass through for rationale generation
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
