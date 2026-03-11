"""Product Classifier using DevelopMap Taxonomy.

Hybrid approach combining keyword matching and semantic embeddings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .developmap import DEVELOPMAP


class ProductClassifier:
    """Classifies products into developmental domains."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        keyword_weight: float = 0.3,
        embedding_weight: float = 0.7
    ):
        """Initialize the classifier.
        
        Args:
            embedding_model: Name of the sentence transformer model
            use_embeddings: Whether to use semantic embeddings
            keyword_weight: Weight for keyword matching score
            embedding_weight: Weight for embedding similarity score
        """
        self.developmap = DEVELOPMAP
        self.use_embeddings = use_embeddings
        self.keyword_weight = keyword_weight
        self.embedding_weight = embedding_weight
        
        if use_embeddings:
            self.model = SentenceTransformer(embedding_model)
            self._precompute_domain_embeddings()
        else:
            self.model = None
            self.domain_embeddings = None
    
    def _precompute_domain_embeddings(self):
        """Precompute embeddings for all domain descriptions and keywords."""
        self.domain_embeddings = {}
        
        for domain_name, domain in self.developmap.domains.items():
            # Combine description and keywords for richer representation
            text = f"{domain.description}. {' '.join(domain.keywords)}"
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.domain_embeddings[domain_name] = embedding
    
    def _keyword_match_score(self, product_text: str, domain_name: str) -> float:
        """Calculate keyword matching score for a domain.
        
        Args:
            product_text: Product title/description
            domain_name: Name of the developmental domain
            
        Returns:
            Score between 0 and 1
        """
        product_text_lower = product_text.lower()
        domain = self.developmap.get_domain(domain_name)
        
        if domain is None:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for keyword in domain.keywords if keyword in product_text_lower)
        
        # Normalize by number of keywords
        if len(domain.keywords) > 0:
            return min(matches / 5.0, 1.0)  # Cap at 5 matches for score of 1.0
        return 0.0
    
    def _embedding_similarity_score(self, product_text: str, domain_name: str) -> float:
        """Calculate embedding similarity score for a domain.
        
        Args:
            product_text: Product title/description
            domain_name: Name of the developmental domain
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not self.use_embeddings or self.model is None:
            return 0.0
        
        product_embedding = self.model.encode(product_text, convert_to_numpy=True)
        domain_embedding = self.domain_embeddings[domain_name]
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            product_embedding.reshape(1, -1),
            domain_embedding.reshape(1, -1)
        )[0, 0]
        
        # Normalize to [0, 1]
        return (similarity + 1) / 2
    
    def classify(self, product_text: str, threshold: float = 0.3) -> Dict[str, float]:
        """Classify a product into developmental domains.
        
        Args:
            product_text: Product title/description
            threshold: Minimum score to include a domain
            
        Returns:
            Dictionary mapping domain names to scores
        """
        scores = {}
        
        for domain_name in self.developmap.get_domain_names():
            # Keyword matching score
            keyword_score = self._keyword_match_score(product_text, domain_name)
            
            # Embedding similarity score
            if self.use_embeddings:
                embedding_score = self._embedding_similarity_score(product_text, domain_name)
                # Weighted combination
                combined_score = (
                    self.keyword_weight * keyword_score +
                    self.embedding_weight * embedding_score
                )
            else:
                combined_score = keyword_score
            
            if combined_score >= threshold:
                scores[domain_name] = combined_score
        
        return scores
    
    def classify_batch(self, product_texts: List[str], threshold: float = 0.3) -> List[Dict[str, float]]:
        """Classify multiple products.
        
        Args:
            product_texts: List of product titles/descriptions
            threshold: Minimum score to include a domain
            
        Returns:
            List of dictionaries mapping domain names to scores
        """
        return [self.classify(text, threshold) for text in product_texts]
    
    def get_primary_domain(self, product_text: str) -> Optional[Tuple[str, float]]:
        """Get the primary (highest scoring) domain for a product.
        
        Args:
            product_text: Product title/description
            
        Returns:
            Tuple of (domain_name, score) or None if no domains match
        """
        scores = self.classify(product_text, threshold=0.0)
        
        if not scores:
            return None
        
        primary_domain = max(scores.items(), key=lambda x: x[1])
        return primary_domain
    
    def calibrate(
        self,
        labeled_products: List[Tuple[str, str]],
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        weight_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 10
    ) -> Dict[str, float]:
        """Calibrate classification thresholds and keyword-vs-embedding weights.

        Uses labeled product data to find the threshold and keyword_weight
        that maximize macro-averaged F1 score across all domains.

        Args:
            labeled_products: List of (product_text, true_domain) tuples
            threshold_range: Min and max threshold to search
            weight_range: Min and max keyword_weight to search
            n_steps: Number of grid search steps per parameter

        Returns:
            Dict with best_threshold, best_keyword_weight, best_f1, and
            per-domain precision/recall
        """
        from sklearn.metrics import f1_score as sklearn_f1

        texts = [t for t, _ in labeled_products]
        true_labels = [d for _, d in labeled_products]

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        weights = np.linspace(weight_range[0], weight_range[1], n_steps)

        best_f1 = -1.0
        best_threshold = self.default_threshold if hasattr(self, 'default_threshold') else 0.3
        best_kw_weight = self.keyword_weight

        for thresh in thresholds:
            for kw_w in weights:
                # Temporarily set weights
                old_kw = self.keyword_weight
                self.keyword_weight = kw_w

                preds = []
                for text in texts:
                    scores = self.classify(text, threshold=thresh)
                    if scores:
                        preds.append(max(scores, key=scores.get))
                    else:
                        preds.append('unmapped')

                # Restore
                self.keyword_weight = old_kw

                # Macro F1
                all_labels = list(set(true_labels + preds))
                f1 = sklearn_f1(true_labels, preds, labels=all_labels,
                                average='macro', zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
                    best_kw_weight = kw_w

        # Apply best parameters
        self.keyword_weight = best_kw_weight
        if hasattr(self, 'default_threshold'):
            self.default_threshold = best_threshold

        # Compute per-domain metrics with best params
        preds = []
        for text in texts:
            scores = self.classify(text, threshold=best_threshold)
            if scores:
                preds.append(max(scores, key=scores.get))
            else:
                preds.append('unmapped')

        domain_metrics = {}
        for domain in set(true_labels):
            tp = sum(1 for t, p in zip(true_labels, preds) if t == domain and p == domain)
            fp = sum(1 for t, p in zip(true_labels, preds) if t != domain and p == domain)
            fn = sum(1 for t, p in zip(true_labels, preds) if t == domain and p != domain)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            domain_metrics[domain] = {'precision': precision, 'recall': recall}

        result = {
            'best_threshold': float(best_threshold),
            'best_keyword_weight': float(best_kw_weight),
            'best_f1': float(best_f1),
            'per_domain_metrics': domain_metrics,
        }

        print(f"Calibration: threshold={best_threshold:.2f}, "
              f"keyword_weight={best_kw_weight:.2f}, F1={best_f1:.3f}")

        return result

    def get_domain_vector(self, product_text: str) -> np.ndarray:
        """Get a vector representation of domain scores.
        
        Args:
            product_text: Product title/description
            
        Returns:
            Numpy array of scores for all domains (in consistent order)
        """
        scores = self.classify(product_text, threshold=0.0)
        domain_names = self.developmap.get_domain_names()
        
        vector = np.array([scores.get(domain, 0.0) for domain in domain_names])
        return vector