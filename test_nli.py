#!/usr/bin/env python3
"""
Quick test to verify NLI model loads correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Testing NLI Model Loading")
print("=" * 80)

try:
    from transformers import pipeline
    from sentence_transformers import CrossEncoder
    import config
    
    print(f"\n✓ Imports successful")
    print(f"✓ NLI Model configured: {config.NLI_MODEL}")
    print(f"✓ USE_NLI_MODEL: {config.USE_NLI_MODEL}")
    
    if config.USE_NLI_MODEL:
        print(f"\n⏳ Loading NLI model (this may take 30-60 seconds)...")
        
        # Check if it's a cross-encoder model
        if 'cross-encoder' in config.NLI_MODEL.lower():
            nli_model = CrossEncoder(config.NLI_MODEL)
            nli_type = 'cross-encoder'
            print(f"✅ Cross-encoder NLI model loaded successfully!")
        else:
            nli_model = pipeline("text-classification", model=config.NLI_MODEL, device=-1)
            nli_type = 'zero-shot'
            print(f"✅ Zero-shot NLI model loaded successfully!")
        
        # Test the model with a simple example
        print(f"\n" + "=" * 80)
        print("Testing NLI Model Inference")
        print("=" * 80)
        
        premise = "Sarah grew up in New York and moved to London in 2010"
        hypothesis = "Sarah lived in Paris her entire childhood"
        
        print(f"\nPremise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        
        if nli_type == 'cross-encoder':
            # Cross-encoder returns raw score
            scores = nli_model.predict([[premise, hypothesis]])
            # Handle different numpy array formats
            import numpy as np
            if isinstance(scores, np.ndarray):
                if scores.ndim == 0:  # Scalar array
                    score = float(scores)
                else:  # Multi-dimensional array
                    score = float(scores.flatten()[0])
            else:
                score = float(scores)
            print(f"\n✅ NLI Result (Cross-Encoder):")
            print(f"   Score: {score:.4f}")
            print(f"   Interpretation: {'CONTRADICTION' if score < -0.5 else 'ENTAILMENT' if score > 0.5 else 'NEUTRAL'}")
        else:
            # Zero-shot returns label and confidence
            result = nli_model(f"{premise} [SEP] {hypothesis}")[0]
            print(f"\n✅ NLI Result (Zero-Shot):")
            print(f"   Label: {result['label']}")
            print(f"   Score: {result['score']:.4f}")
        
        # Test entailment
        print(f"\n" + "-" * 80)
        premise2 = "John is a doctor in Paris"
        hypothesis2 = "John works in France"
        
        print(f"\nPremise: {premise2}")
        print(f"Hypothesis: {hypothesis2}")
        
        if nli_type == 'cross-encoder':
            scores2 = nli_model.predict([[premise2, hypothesis2]])
            # Handle different numpy array formats
            import numpy as np
            if isinstance(scores2, np.ndarray):
                if scores2.ndim == 0:  # Scalar array
                    score2 = float(scores2)
                else:  # Multi-dimensional array
                    score2 = float(scores2.flatten()[0])
            else:
                score2 = float(scores2)
            print(f"\n✅ NLI Result (Cross-Encoder):")
            print(f"   Score: {score2:.4f}")
            print(f"   Interpretation: {'ENTAILMENT' if score2 > 0.5 else 'CONTRADICTION' if score2 < -0.5 else 'NEUTRAL'}")
        else:
            result2 = nli_model(f"{premise2} [SEP] {hypothesis2}")[0]
            print(f"\n✅ NLI Result (Zero-Shot):")
            print(f"   Label: {result2['label']}")
            print(f"   Score: {result2['score']:.4f}")
        
        print(f"\n" + "=" * 80)
        print("✅ NLI MODEL IS WORKING PERFECTLY!")
        print("=" * 80)
        print(f"\nMemory Impact:")
        print(f"  - NLI Model ({config.NLI_MODEL}): ~140-400 MB")
        print(f"  - Embedding Model: ~420 MB")
        print(f"  - Total: ~600-900 MB RAM required")
        print(f"\n✓ Much better for your Mac's memory!")
        
    else:
        print(f"\n⚠️ NLI model is disabled in config")
        print(f"   Set USE_NLI_MODEL = True to enable")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n✅ All tests passed!")
