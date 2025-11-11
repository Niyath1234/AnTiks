"""
Automatic relationship detection for multi-table databases.
Finds potential foreign key relationships between tables.
"""

import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import difflib
import re

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Profile of a column for relationship detection"""
    table_name: str
    column_name: str
    data_type: str
    is_primary_key: bool
    is_nullable: bool
    distinct_count: int
    total_count: int
    sample_values: List[Any]
    value_set: Set[Any]
    
    @property
    def cardinality_ratio(self) -> float:
        """Ratio of distinct values to total values"""
        if self.total_count == 0:
            return 0.0
        return self.distinct_count / self.total_count
    
    @property
    def is_likely_key(self) -> bool:
        """Check if column likely represents a key (high cardinality)"""
        return self.cardinality_ratio > 0.95 or self.is_primary_key


@dataclass
class RelationshipCandidate:
    """A potential relationship between two tables"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    confidence: float  # 0.0 to 1.0
    reason: str
    match_ratio: float  # Percentage of values that match
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "match_ratio": round(self.match_ratio, 2),
        }


class RelationshipDetector:
    """Detects relationships between tables in a database"""
    
    # Configuration
    MAX_SAMPLE_SIZE = 10000
    NAME_SIMILARITY_THRESHOLD = 0.7
    VALUE_MATCH_THRESHOLD = 0.7  # 70% of values must match
    MIN_CONFIDENCE = 0.5
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.profiles: Dict[Tuple[str, str], ColumnProfile] = {}
        
    def profile_tables(self, table_names: List[str]) -> None:
        """Profile all columns in the given tables"""
        logger.info(f"Profiling {len(table_names)} tables for relationship detection")
        
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            for table_name in table_names:
                self._profile_table(conn, table_name)
        finally:
            conn.close()
            
        logger.info(f"Profiled {len(self.profiles)} columns across {len(table_names)} tables")
    
    def _profile_table(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
        """Profile all columns in a single table"""
        try:
            # Get table info
            info_query = f"PRAGMA table_info('{table_name}')"
            table_info = conn.execute(info_query).fetchall()
            
            for cid, col_name, data_type, not_null, default_val, is_pk in table_info:
                profile = self._profile_column(conn, table_name, col_name, data_type, bool(is_pk), not bool(not_null))
                if profile:
                    self.profiles[(table_name, col_name)] = profile
                    
        except Exception as e:
            logger.warning(f"Could not profile table {table_name}: {e}")
    
    def _profile_column(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        column_name: str,
        data_type: str,
        is_pk: bool,
        is_nullable: bool
    ) -> Optional[ColumnProfile]:
        """Profile a single column"""
        try:
            # Get distinct count
            distinct_query = f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"'
            distinct_count = conn.execute(distinct_query).fetchone()[0]
            
            # Get total count
            total_query = f'SELECT COUNT(*) FROM "{table_name}"'
            total_count = conn.execute(total_query).fetchone()[0]
            
            # Get sample values (limited)
            sample_query = f'''
                SELECT DISTINCT "{column_name}" 
                FROM "{table_name}" 
                WHERE "{column_name}" IS NOT NULL 
                LIMIT {min(self.MAX_SAMPLE_SIZE, distinct_count)}
            '''
            sample_result = conn.execute(sample_query).fetchall()
            sample_values = [row[0] for row in sample_result]
            value_set = set(sample_values)
            
            return ColumnProfile(
                table_name=table_name,
                column_name=column_name,
                data_type=data_type,
                is_primary_key=is_pk,
                is_nullable=is_nullable,
                distinct_count=distinct_count,
                total_count=total_count,
                sample_values=sample_values[:100],  # Keep only first 100 for memory
                value_set=value_set
            )
            
        except Exception as e:
            logger.debug(f"Could not profile column {table_name}.{column_name}: {e}")
            return None
    
    def detect_relationships(self, confidence_threshold: float = None) -> List[RelationshipCandidate]:
        """
        Detect all potential relationships between profiled tables.
        Returns candidates above the confidence threshold.
        """
        if confidence_threshold is None:
            confidence_threshold = self.MIN_CONFIDENCE
            
        candidates = []
        
        # Get list of tables
        tables = list(set(profile.table_name for profile in self.profiles.values()))
        
        # Compare columns across different tables
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:  # Only compare each pair once
                table_candidates = self._detect_relationships_between_tables(table1, table2)
                candidates.extend(table_candidates)
        
        # Filter by confidence and sort
        candidates = [c for c in candidates if c.confidence >= confidence_threshold]
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        logger.info(f"Found {len(candidates)} relationship candidates above confidence {confidence_threshold}")
        return candidates
    
    def _detect_relationships_between_tables(
        self,
        table1: str,
        table2: str
    ) -> List[RelationshipCandidate]:
        """Detect potential relationships between two specific tables"""
        candidates = []
        
        # Get columns for each table
        table1_cols = [(t, c, p) for (t, c), p in self.profiles.items() if t == table1]
        table2_cols = [(t, c, p) for (t, c), p in self.profiles.items() if t == table2]
        
        # Compare each column pair
        for t1, c1, profile1 in table1_cols:
            for t2, c2, profile2 in table2_cols:
                candidate = self._evaluate_column_pair(profile1, profile2)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _evaluate_column_pair(
        self,
        profile1: ColumnProfile,
        profile2: ColumnProfile
    ) -> Optional[RelationshipCandidate]:
        """Evaluate if two columns might be related"""
        
        # Skip if different data types (with some flexibility)
        if not self._are_types_compatible(profile1.data_type, profile2.data_type):
            return None
        
        # Calculate various similarity scores
        name_similarity = self._calculate_name_similarity(profile1.column_name, profile2.column_name)
        value_overlap = self._calculate_value_overlap(profile1, profile2)
        
        # Determine direction: which is the foreign key and which is the key
        # Generally, the column with higher cardinality is the key
        if profile1.cardinality_ratio > profile2.cardinality_ratio:
            source_profile, target_profile = profile2, profile1
        else:
            source_profile, target_profile = profile1, profile2
        
        # Build confidence score
        confidence = 0.0
        reasons = []
        
        # Name similarity contributes to confidence
        if name_similarity > self.NAME_SIMILARITY_THRESHOLD:
            confidence += 0.3 * name_similarity
            reasons.append(f"name_match_{int(name_similarity*100)}%")
        
        # Value overlap is crucial
        if value_overlap > self.VALUE_MATCH_THRESHOLD:
            confidence += 0.5 * value_overlap
            reasons.append(f"value_overlap_{int(value_overlap*100)}%")
        elif value_overlap > 0.3:  # Partial overlap still counts
            confidence += 0.2 * value_overlap
            reasons.append(f"partial_overlap_{int(value_overlap*100)}%")
        
        # Key indicators
        if target_profile.is_likely_key:
            confidence += 0.15
            reasons.append("target_is_key")
        
        if target_profile.is_primary_key:
            confidence += 0.05
            reasons.append("target_is_pk")
        
        # Common naming patterns
        if self._has_fk_naming_pattern(source_profile.column_name, target_profile.column_name):
            confidence += 0.1
            reasons.append("fk_naming_pattern")
        
        # Don't create relationships with very low confidence
        if confidence < 0.3:
            return None
        
        return RelationshipCandidate(
            source_table=source_profile.table_name,
            source_column=source_profile.column_name,
            target_table=target_profile.table_name,
            target_column=target_profile.column_name,
            confidence=min(confidence, 1.0),
            reason=" | ".join(reasons),
            match_ratio=value_overlap
        )
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two data types are compatible for a relationship"""
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        # Exact match
        if type1_lower == type2_lower:
            return True
        
        # Numeric types
        numeric_types = {'integer', 'bigint', 'int', 'smallint', 'tinyint', 'double', 'float', 'decimal', 'numeric'}
        if any(t in type1_lower for t in numeric_types) and any(t in type2_lower for t in numeric_types):
            return True
        
        # String types
        string_types = {'varchar', 'text', 'char', 'string'}
        if any(t in type1_lower for t in string_types) and any(t in type2_lower for t in string_types):
            return True
        
        return False
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two column names"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return 1.0
        
        # Remove common suffixes/prefixes
        name1_clean = re.sub(r'^(fk_|ref_)', '', name1_lower)
        name1_clean = re.sub(r'(_id|_key|_code)$', '', name1_clean)
        name2_clean = re.sub(r'^(fk_|ref_)', '', name2_lower)
        name2_clean = re.sub(r'(_id|_key|_code)$', '', name2_clean)
        
        if name1_clean == name2_clean:
            return 0.95
        
        # Use difflib for similarity
        return difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
    
    def _calculate_value_overlap(self, profile1: ColumnProfile, profile2: ColumnProfile) -> float:
        """Calculate what percentage of values overlap between columns"""
        if not profile1.value_set or not profile2.value_set:
            return 0.0
        
        # Find intersection
        intersection = profile1.value_set & profile2.value_set
        
        if not intersection:
            return 0.0
        
        # Calculate overlap ratio relative to the smaller set (likely the foreign key)
        smaller_set_size = min(len(profile1.value_set), len(profile2.value_set))
        if smaller_set_size == 0:
            return 0.0
        
        return len(intersection) / smaller_set_size
    
    def _has_fk_naming_pattern(self, fk_name: str, pk_name: str) -> bool:
        """Check if column names follow foreign key naming patterns"""
        fk_lower = fk_name.lower()
        pk_lower = pk_name.lower()
        
        # Common patterns:
        # 1. customer_id -> id (table is customer)
        # 2. customer_id -> customer_id
        # 3. cust_id -> customer_id
        # 4. fk_customer -> id
        
        patterns = [
            fk_lower.startswith('fk_'),
            fk_lower.startswith('ref_'),
            fk_lower.endswith('_id') and pk_lower in ['id', 'code', 'key'],
            fk_lower.endswith('_code') and pk_lower in ['code', 'id'],
            fk_lower.endswith('_key') and pk_lower in ['key', 'id'],
        ]
        
        return any(patterns)
    
    def get_recommended_relationships(
        self,
        max_per_table_pair: int = 1,
        min_confidence: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Get recommended relationships for the schema board.
        Filters to most likely relationships only.
        
        Args:
            max_per_table_pair: Maximum relationships to return between any two tables
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of relationship dictionaries ready for use
        """
        all_candidates = self.detect_relationships(confidence_threshold=min_confidence)
        
        # Group by table pair
        table_pairs: Dict[Tuple[str, str], List[RelationshipCandidate]] = {}
        for candidate in all_candidates:
            # Create canonical table pair (sorted alphabetically)
            tables = tuple(sorted([candidate.source_table, candidate.target_table]))
            if tables not in table_pairs:
                table_pairs[tables] = []
            table_pairs[tables].append(candidate)
        
        # Select top N for each table pair
        recommendations = []
        for table_pair, candidates in table_pairs.items():
            # Sort by confidence
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            # Take top N
            top_candidates = candidates[:max_per_table_pair]
            recommendations.extend(top_candidates)
        
        # Convert to dictionaries
        return [c.to_dict() for c in recommendations]


def detect_relationships_for_tables(
    db_path: str,
    table_names: List[str],
    min_confidence: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Convenience function to detect relationships for a set of tables.
    
    Args:
        db_path: Path to DuckDB database
        table_names: List of table names to analyze
        min_confidence: Minimum confidence for relationships
    
    Returns:
        List of recommended relationships
    """
    detector = RelationshipDetector(db_path)
    detector.profile_tables(table_names)
    return detector.get_recommended_relationships(min_confidence=min_confidence)

