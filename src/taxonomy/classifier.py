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