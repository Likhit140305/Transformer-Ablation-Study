import numpy as np
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

def welchs_t_test(group1, group2, alpha=0.05, bonferroni_n=1):
    """
    Performs Welch's t-test for unequal variances between two groups.
    Args:
        group1: List or array of values (e.g., perplexities of baseline)
        group2: List or array of values (e.g., perplexities of ablation)
        alpha: Significance level
        bonferroni_n: Number of comparisons for Bonferroni correction
    Returns:
        dict containing t-statistic, p-value, and significance flag
    """
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Bonferroni correction
    adjusted_alpha = alpha / bonferroni_n
    is_significant = p_val < adjusted_alpha
    
    return {
        "t_stat": t_stat,
        "p_val": p_val,
        "adjusted_alpha": adjusted_alpha,
        "is_significant": is_significant
    }

def cohens_d(group1, group2):
    """
    Computes Cohen's d effect size.
    Since variances might be unequal (Welch's), this uses the pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    return (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

def report_statistics(baseline_scores, ablation_scores, ablation_name="Ablation", bonferroni_n=1):
    """
    Prints a formatted statistical report comparing an ablation to the baseline.
    """
    t_test_results = welchs_t_test(baseline_scores, ablation_scores, bonferroni_n=bonferroni_n)
    effect_size = cohens_d(baseline_scores, ablation_scores)
    
    logger.info(f"--- Statistical Report: {ablation_name} ---")
    logger.info(f"Baseline Mean: {np.mean(baseline_scores):.4f} (± {np.std(baseline_scores, ddof=1):.4f})")
    logger.info(f"Ablation Mean: {np.mean(ablation_scores):.4f} (± {np.std(ablation_scores, ddof=1):.4f})")
    logger.info(f"t-statistic: {t_test_results['t_stat']:.4f}, p-value: {t_test_results['p_val']:.4e}")
    logger.info(f"Significant (alpha={t_test_results['adjusted_alpha']:.4f}): {t_test_results['is_significant']}")
    logger.info(f"Effect Size (Cohen's d): {effect_size:.4f}")
    
    return t_test_results, effect_size
