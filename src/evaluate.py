"""
Enhanced Evaluation Module
Provides comprehensive metrics beyond simple accuracy:
- Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix
- Per-class analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class ComprehensiveEvaluator:
    """
    Evaluate model performance with multiple metrics
    """
    
    def __init__(self):
        logger.info("Initialized Comprehensive Evaluator")
    
    def evaluate_predictions(self, results_df: pd.DataFrame, save_path: Path = None) -> dict:
        """
        Comprehensive evaluation of predictions
        
        Args:
            results_df: DataFrame with 'true_label'/'label' and 'prediction' columns
            save_path: Optional path to save detailed results
            
        Returns:
            dict with all metrics
        """
        # Get labels and predictions - handle different column names
        if 'true_label' in results_df.columns:
            y_true = results_df['true_label'].values
        elif 'label' in results_df.columns:
            y_true = results_df['label'].values
        else:
            raise ValueError("DataFrame must have 'true_label' or 'label' column")
            
        y_pred = results_df['prediction'].values
        
        # Convert to numeric if needed (handle string labels)
        def to_numeric(val):
            if isinstance(val, str):
                return 1 if val.lower() == 'consistent' else 0
            return int(val)
        
        y_true = np.array([to_numeric(v) for v in y_true])
        y_pred = np.array([to_numeric(v) for v in y_pred])
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'per_class': {
                'contradict': {
                    'precision': precision_per_class[0],
                    'recall': recall_per_class[0],
                    'f1': f1_per_class[0],
                    'support': int(support_per_class[0])
                },
                'consistent': {
                    'precision': precision_per_class[1],
                    'recall': recall_per_class[1],
                    'f1': f1_per_class[1],
                    'support': int(support_per_class[1])
                }
            },
            'total_samples': len(y_true),
            'correct_predictions': int((y_true == y_pred).sum()),
            'incorrect_predictions': int((y_true != y_pred).sum())
        }
        
        # Print comprehensive report
        self._print_report(metrics, y_true, y_pred)
        
        # Save detailed analysis if requested
        if save_path:
            self._save_detailed_analysis(results_df, metrics, save_path)
        
        return metrics
    
    def _print_report(self, metrics: dict, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Print human-readable evaluation report
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EVALUATION REPORT")
        logger.info("="*80)
        
        logger.info(f"\n📊 OVERALL METRICS:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1 Score:  {metrics['f1_score']:.4f}")
        
        logger.info(f"\n📈 CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        logger.info(f"                 Predicted")
        logger.info(f"              Contradict | Consistent")
        logger.info(f"   Actual   --------------------------------")
        logger.info(f"   Contradict |    {cm['true_negatives']:4d}    |    {cm['false_positives']:4d}")
        logger.info(f"   Consistent |    {cm['false_negatives']:4d}    |    {cm['true_positives']:4d}")
        
        logger.info(f"\n📋 PER-CLASS METRICS:")
        for label, label_name in [(0, 'Contradict'), (1, 'Consistent')]:
            pc = metrics['per_class'][label_name.lower()]
            logger.info(f"\n   {label_name}:")
            logger.info(f"      Precision: {pc['precision']:.4f}")
            logger.info(f"      Recall:    {pc['recall']:.4f}")
            logger.info(f"      F1 Score:  {pc['f1']:.4f}")
            logger.info(f"      Support:   {pc['support']} examples")
        
        logger.info(f"\n💯 SUMMARY:")
        logger.info(f"   Total:     {metrics['total_samples']} examples")
        logger.info(f"   Correct:   {metrics['correct_predictions']} ({metrics['accuracy']*100:.1f}%)")
        logger.info(f"   Incorrect: {metrics['incorrect_predictions']} ({(1-metrics['accuracy'])*100:.1f}%)")
        logger.info("="*80 + "\n")
    
    def _save_detailed_analysis(self, results_df: pd.DataFrame, metrics: dict, save_path: Path):
        """
        Save detailed per-example analysis
        """
        # Add correctness column
        results_df['correct'] = results_df['label'] == results_df['prediction']
        
        # Add error type
        def get_error_type(row):
            if row['correct']:
                return 'correct'
            elif row['label'] == 1 and row['prediction'] == 0:
                return 'false_negative'  # Missed a consistent backstory
            else:
                return 'false_positive'  # Incorrectly said contradict was consistent
        
        results_df['error_type'] = results_df.apply(get_error_type, axis=1)
        
        # Save
        analysis_path = save_path.parent / f"{save_path.stem}_detailed_analysis.csv"
        results_df.to_csv(analysis_path, index=False)
        logger.info(f"💾 Detailed analysis saved to: {analysis_path}")
        
        # Save metrics summary
        metrics_path = save_path.parent / f"{save_path.stem}_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION METRICS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(f"  TN: {cm['true_negatives']}, FP: {cm['false_positives']}\n")
            f.write(f"  FN: {cm['false_negatives']}, TP: {cm['true_positives']}\n")
        
        logger.info(f"💾 Metrics summary saved to: {metrics_path}")
    
    def analyze_errors(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze which examples were misclassified
        Returns DataFrame of errors with details
        """
        errors = results_df[results_df['label'] != results_df['prediction']].copy()
        
        if len(errors) == 0:
            logger.info("🎉 Perfect predictions! No errors to analyze.")
            return errors
        
        logger.info(f"\n🔍 ERROR ANALYSIS:")
        logger.info(f"   Total errors: {len(errors)}")
        
        # False positives (predicted consistent, actually contradict)
        fp = errors[errors['label'] == 0]
        logger.info(f"   False Positives: {len(fp)} (predicted consistent, actually contradict)")
        
        # False negatives (predicted contradict, actually consistent)
        fn = errors[errors['label'] == 1]
        logger.info(f"   False Negatives: {len(fn)} (predicted contradict, actually consistent)")
        
        return errors


def evaluate_model_predictions(train_results_path: Path, test_results_path: Path = None):
    """
    Main evaluation function
    """
    evaluator = ComprehensiveEvaluator()
    
    # Evaluate training set
    if train_results_path.exists():
        logger.info(f"Evaluating training predictions from: {train_results_path}")
        train_df = pd.read_csv(train_results_path)
        
        if 'label' in train_df.columns and 'prediction' in train_df.columns:
            metrics = evaluator.evaluate_predictions(train_df, save_path=train_results_path)
            
            # Analyze errors
            errors = evaluator.analyze_errors(train_df)
            if len(errors) > 0:
                error_path = train_results_path.parent / f"{train_results_path.stem}_errors.csv"
                errors.to_csv(error_path, index=False)
                logger.info(f"💾 Error analysis saved to: {error_path}")
        else:
            logger.warning("Training results missing 'label' or 'prediction' columns")
    else:
        logger.warning(f"Training results not found: {train_results_path}")
    
    # Test set doesn't have labels, so just report statistics
    if test_results_path and test_results_path.exists():
        logger.info(f"\n📊 Test set predictions from: {test_results_path}")
        test_df = pd.read_csv(test_results_path)
        
        if 'prediction' in test_df.columns:
            pred_counts = test_df['prediction'].value_counts()
            logger.info(f"   Total test examples: {len(test_df)}")
            logger.info(f"   Predicted Contradict (0): {pred_counts.get(0, 0)}")
            logger.info(f"   Predicted Consistent (1): {pred_counts.get(1, 0)}")
            logger.info(f"   Distribution: {pred_counts.get(1, 0)/len(test_df)*100:.1f}% consistent")


if __name__ == "__main__":
    from loguru import logger
    logger.add(config.LOG_FILE, level=config.LOG_LEVEL)
    
    # Evaluate
    train_path = config.BASE_DIR / "train_results.csv"
    test_path = config.BASE_DIR / "results.csv"
    
    evaluate_model_predictions(train_path, test_path)
