import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict
from .statistics import EdgeTypeStatistics

class KEstimationModel(ABC):
    """
    Abstract base class for k_vector estimation models.
    Goal: Predict how many vectors to fetch to get 'k_final' graph results.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def estimate_k_vector(
        self,
        k_final: int,
        stats: EdgeTypeStatistics
    ) -> int:
        """
        Estimate how many vectors to fetch to achieve k_final targets
        """
        pass

class SimpleInverseDegreeModel(KEstimationModel):
    """
    Model 1: Simple inverse of average degree
    k_vector = k_final / avg_degree
    
    TUNING: Increased safety_margin to 2.0 (was 1.2) to handle sparsity.
    """
    @property
    def name(self) -> str:
        return "simple_inverse_degree"
    
    def __init__(self, safety_margin: float = 2.0):
        self.safety_margin = safety_margin
    
    def estimate_k_vector(self, k_final: int, stats: EdgeTypeStatistics) -> int:
        if stats.avg_degree <= 0:
            return k_final
        
        k_vector = k_final / stats.avg_degree
        k_vector *= self.safety_margin
        
        # FIX: Ensure we never request fewer vectors than the target results
        return max(int(np.ceil(k_vector)), k_final)

class OverlapAwareModel(KEstimationModel):
    """
    Model 2: Accounts for target overlap (Jaccard)
    k_vector = (k_final / avg_degree) * overlap_factor
    
    TUNING: Increased safety_margin to 2.5 (was 1.1)
    """
    @property
    def name(self) -> str:
        return "overlap_aware"
    
    def __init__(self, safety_margin: float = 2.5):
        self.safety_margin = safety_margin
    
    def estimate_k_vector(self, k_final: int, stats: EdgeTypeStatistics) -> int:
        if stats.avg_degree <= 0:
            return k_final
        
        base_k = k_final / stats.avg_degree
        adjusted_k = base_k * stats.overlap_factor
        adjusted_k *= self.safety_margin
        
        return max(int(np.ceil(adjusted_k)), k_final)

class ProbabilisticModel(KEstimationModel):
    """
    Model 3: Probabilistic model based on coupon collector problem logic.
    Formula: k = log(1 - T/N) / log(1 - d/N)
    
    TUNING: Increased safety_margin to 2.5 (was 1.15)
    """
    @property
    def name(self) -> str:
        return "probabilistic"
    
    def __init__(self, safety_margin: float = 2.5):
        self.safety_margin = safety_margin
    
    def estimate_k_vector(self, k_final: int, stats: EdgeTypeStatistics) -> int:
        N = stats.unique_nodes # Total unique targets in graph
        d = stats.avg_degree
        T = min(k_final, N - 1)  # Can't exceed total graph size
        
        # Edge case handling
        if T <= 0 or N <= 0 or d <= 0:
            return k_final
        
        # Avoid log(0) or division by zero errors
        ratio_target = max(1 - T/N, 0.001)
        ratio_degree = max(1 - d/N, 0.001)
        
        if ratio_degree >= 1:
            return k_final
        
        # The core probability formula
        k_vector = np.log(ratio_target) / np.log(ratio_degree)
        
        # Apply overlap adjustment
        k_vector *= stats.overlap_factor
        
        # Apply safety margin
        k_vector *= self.safety_margin
        
        return max(int(np.ceil(k_vector)), k_final)

class DistributionAwareModel(KEstimationModel):
    """
    Model 4: Adjusts based on Coefficient of Variation (CV).
    If graph is "Heavy Tailed" (CV > 1), averages are misleading, so we boost aggressive.
    """
    @property
    def name(self) -> str:
        return "distribution_aware"
    
    def __init__(self, base_model: Optional[KEstimationModel] = None):
        # Wraps the Probabilistic model with our tuned settings
        self.base_model = base_model or ProbabilisticModel(safety_margin=2.5)
    
    def estimate_k_vector(self, k_final: int, stats: EdgeTypeStatistics) -> int:
        # Get base estimate
        base_k = self.base_model.estimate_k_vector(k_final, stats)
        
        # Adjust based on distribution shape (CV)
        cv = stats.coefficient_of_variation
        
        if cv > 1.5: 
            # Highly skewed (Power Law) - Averages lie! Boost significantly.
            adjustment = 1 + (cv - 1) * 0.5  # Increased scaling factor
        elif cv > 1.0:
            # Moderately skewed
            adjustment = 1 + (cv - 1) * 0.3
        elif cv < 0.5:
            # Very uniform - base estimate is reliable
            adjustment = 1.0
        else:
            # Normal range
            adjustment = 1 + (cv - 0.5) * 0.1
        
        adjusted_k = base_k * adjustment
        
        return max(int(np.ceil(adjusted_k)), k_final)

class EnsembleModel(KEstimationModel):
    """
    Model 5: Weighted ensemble of multiple models.
    Combines the strengths of all approaches.
    """
    @property
    def name(self) -> str:
        return "ensemble"
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Initialize sub-models with the new tuned parameters
        self.models = {
            'simple': SimpleInverseDegreeModel(safety_margin=2.0),
            'overlap': OverlapAwareModel(safety_margin=2.5),
            'probabilistic': ProbabilisticModel(safety_margin=2.5),
            'distribution': DistributionAwareModel()
        }
        
        # Default weights favoring the smarter models
        self.weights = weights or {
            'simple': 0.1,      # Lower weight for naive model
            'overlap': 0.2,
            'probabilistic': 0.3,
            'distribution': 0.4 # Higher weight for distribution-aware model
        }
    
    def estimate_k_vector(self, k_final: int, stats: EdgeTypeStatistics) -> int:
        weighted_sum = 0.0
        
        for model_name, model in self.models.items():
            estimate = model.estimate_k_vector(k_final, stats)
            weight = self.weights.get(model_name, 0.0)
            weighted_sum += estimate * weight
        
        return max(int(np.ceil(weighted_sum)), k_final)

def get_all_models() -> Dict[str, KEstimationModel]:
    """
    Returns dictionary of all K-estimation models for the experiment runner.
    """
    return {
        'simple_inverse_degree': SimpleInverseDegreeModel(),
        'overlap_aware': OverlapAwareModel(),
        'probabilistic': ProbabilisticModel(),
        'distribution_aware': DistributionAwareModel(),
        'ensemble': EnsembleModel()
    }