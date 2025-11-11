"""
Find alternative column matches for ambiguous queries.
Shows top alternatives when multiple similar columns exist.
"""

import difflib
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ColumnAlternative:
    """Alternative column that could match user's intent"""
    table_alias: str
    column_name: str
    similarity_score: float
    reason: str


def find_column_alternatives(
    query_token: str,
    table_metadata: Dict[str, Dict[str, Any]],
    top_k: int = 3,
    min_similarity: float = 0.6
) -> List[ColumnAlternative]:
    """
    Find alternative columns that match a query token.
    
    Args:
        query_token: Word from user query (e.g., "charge", "amount")
        table_metadata: Dictionary of table metadata
        top_k: Maximum alternatives to return
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of ColumnAlternative objects, sorted by similarity
    """
    alternatives = []
    query_lower = query_token.lower()
    
    for table_alias, metadata in table_metadata.items():
        columns = metadata.get('columns', [])
        
        for col in columns:
            col_lower = col.lower()
            
            # Calculate various similarity metrics
            scores = []
            reasons = []
            
            # Exact match
            if col_lower == query_lower:
                scores.append(1.0)
                reasons.append("exact_match")
            
            # Substring match
            if query_lower in col_lower or col_lower in query_lower:
                overlap = min(len(query_lower), len(col_lower)) / max(len(query_lower), len(col_lower))
                scores.append(0.85 * overlap)
                reasons.append("substring")
            
            # Fuzzy match using difflib
            fuzzy_score = difflib.SequenceMatcher(None, query_lower, col_lower).ratio()
            if fuzzy_score >= min_similarity:
                scores.append(fuzzy_score * 0.9)
                reasons.append("fuzzy")
            
            # Word-level match (for multi-word columns like "charge_amount")
            col_words = col_lower.replace('_', ' ').split()
            query_words = query_lower.replace('_', ' ').split()
            common_words = set(col_words) & set(query_words)
            if common_words:
                word_score = len(common_words) / max(len(col_words), len(query_words))
                scores.append(word_score * 0.8)
                reasons.append("word_match")
            
            # Take best score
            if scores:
                best_score = max(scores)
                best_reason_idx = scores.index(best_score)
                best_reason = reasons[best_reason_idx]
                
                if best_score >= min_similarity:
                    alternatives.append(ColumnAlternative(
                        table_alias=table_alias,
                        column_name=col,
                        similarity_score=best_score,
                        reason=best_reason
                    ))
    
    # Sort by similarity and take top k
    alternatives.sort(key=lambda x: x.similarity_score, reverse=True)
    return alternatives[:top_k]


def group_alternatives_by_query_term(
    query: str,
    table_metadata: Dict[str, Dict[str, Any]],
    detected_columns: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each term in the query, find alternative column matches.
    Only return alternatives where there are multiple good options.
    
    Args:
        query: User's natural language query
        table_metadata: Table metadata
        detected_columns: Already detected columns from query parser
        top_k: Max alternatives per term
    
    Returns:
        Dict mapping query term -> list of alternatives with metadata
    """
    import re
    
    # Extract meaningful tokens from query
    tokens = re.findall(r'\b\w{3,}\b', query.lower())  # Words with 3+ chars
    
    alternatives_map = {}
    
    # Track which columns were already detected
    detected_col_names = {
        f"{det.get('table_alias', '')}.{det.get('column_name', '')}"
        for det in detected_columns
    }
    
    for token in tokens:
        # Skip common SQL/query words
        if token in {'select', 'from', 'where', 'group', 'order', 'table', 'show', 'list', 'get'}:
            continue
        
        alternatives = find_column_alternatives(token, table_metadata, top_k=top_k + 2)
        
        if len(alternatives) > 1:  # Only show if there are multiple options
            # Format alternatives
            alt_list = []
            for alt in alternatives:
                col_key = f"{alt.table_alias}.{alt.column_name}"
                is_selected = col_key in detected_col_names
                
                alt_list.append({
                    'table_alias': alt.table_alias,
                    'column_name': alt.column_name,
                    'similarity': round(alt.similarity_score * 100, 1),
                    'reason': alt.reason,
                    'is_selected': is_selected
                })
            
            if alt_list:
                alternatives_map[token] = alt_list[:top_k]
    
    return alternatives_map

