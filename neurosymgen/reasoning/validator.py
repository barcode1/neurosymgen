import torch
import re
from typing import List, Dict, Tuple


class SymbolicValidator:
    """
    Symbolic Constraint Validator with Prolog-style reasoning.
    """

    def __init__(self, kg_integrator=None):
        self.kg = kg_integrator
        self.constraints = []

    def add_constraint(self, name: str, func):
        """Add custom validation rule"""
        self.constraints.append((name, func))

    def validate(self, content: str, target_concepts: List[str]) -> Tuple[float, str]:
        """
        Validate content against symbolic constraints.

        Returns:
            (confidence_score, feedback_message)
        """
        if not isinstance(content, str):
            return 1.0, "Non-text content"

        scores = []
        failed_constraints = []

        # 1. Length constraint
        length_score = min(len(content) / 100, 1.0)
        if length_score < 0.3:
            failed_constraints.append("length")
        scores.append(length_score)

        # 2. Keyword constraint
        keyword_score = self._check_keywords(content, target_concepts)
        if keyword_score < 0.5:
            failed_constraints.append("keywords")
        scores.append(keyword_score)

        # 3. Logic consistency (simple pattern matching)
        logic_score = self._check_logic_patterns(content)
        scores.append(logic_score)

        # 4. KG alignment (if available)
        if self.kg:
            kg_score = self._check_kg_alignment(content, target_concepts)
            scores.append(kg_score)

        # Final score
        final_score = sum(scores) / len(scores)

        if failed_constraints:
            feedback = f"Failed constraints: {failed_constraints}. Score: {final_score:.2f}"
        else:
            feedback = f"✅ All constraints passed. Score: {final_score:.2f}"

        return final_score, feedback

    def _check_keywords(self, content: str, concepts: List[str]) -> float:
        """Check if target concepts are mentioned"""
        content_lower = content.lower()
        matches = sum(1 for concept in concepts if concept.lower() in content_lower)
        return matches / len(concepts) if concepts else 1.0

    def _check_logic_patterns(self, content: str) -> float:
        """Check for logical consistency patterns"""
        # Look for contradictions like "A is B" and "A is not B"
        contradictions = 0

        # Simple pattern: "X is Y" vs "X is not Y"
        is_patterns = re.findall(r"(\w+) is (\w+)", content)
        is_not_patterns = re.findall(r"(\w+) is not (\w+)", content)

        for subject, obj in is_patterns:
            if (subject, obj) in is_not_patterns:
                contradictions += 1

        return max(0, 1.0 - contradictions * 0.5)

    def _check_kg_alignment(self, content: str, concepts: List[str]) -> float:
        """Check alignment with Knowledge Graph"""
        # Placeholder for KG-based validation
        return 0.8