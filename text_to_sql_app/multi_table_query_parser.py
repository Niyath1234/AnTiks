"""
Enhanced query parser for multi-table SQL generation.
Identifies which columns, tables, and joins are needed for a natural language query.
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import difflib

logger = logging.getLogger(__name__)


@dataclass
class TableColumnMatch:
    """Match between query tokens and table columns"""
    table_name: str
    table_alias: str
    column_name: str
    query_token: str
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'value_match', 'inferred'


@dataclass
class QueryIntent:
    """Parsed query intent with identified tables, columns, and operations"""
    original_query: str
    requested_columns: List[TableColumnMatch]
    required_tables: List[str]  # Table aliases
    required_joins: List[Dict[str, Any]]  # Join specifications
    operations: List[str]  # SQL operations (SELECT, SUM, GROUP BY, etc.)
    filters: List[Dict[str, Any]]  # WHERE conditions
    aggregations: List[str]  # Aggregate functions needed
    grouping: bool  # Whether GROUP BY is needed
    ordering: Optional[str]  # ORDER BY specification
    
    def get_tables_set(self) -> Set[str]:
        """Get unique set of required table aliases"""
        return set(self.required_tables)
    
    def get_column_names_for_table(self, table_alias: str) -> List[str]:
        """Get all columns needed from a specific table"""
        return [m.column_name for m in self.requested_columns if m.table_alias == table_alias]


class MultiTableQueryParser:
    """Parse natural language queries to identify required tables and joins"""
    
    def __init__(
        self,
        table_metadata: Dict[str, Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ):
        """
        Initialize parser with table metadata and relationships.
        
        Args:
            table_metadata: Dict mapping table_alias -> {
                'name': str,
                'columns': List[str],
                'column_types': Dict[str, str],
                'column_values': Dict[str, List[str]]
            }
            relationships: List of relationship dicts with source_alias, target_alias, columns, etc.
        """
        self.table_metadata = table_metadata
        self.relationships = relationships
        
        # Build indices for fast lookups
        self._build_indices()
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from NLTK with a fallback set."""
        fallback = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
            "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
            "such", "that", "the", "their", "then", "there", "these",
            "they", "this", "to", "was", "will", "with", "each", "per",
            "across", "within", "between", "from", "using"
        }
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                words = stopwords.words("english")
            except LookupError:
                nltk.download("stopwords", quiet=True)
                words = stopwords.words("english")
            return set(word.lower() for word in words) | fallback
        except Exception:
            return fallback

    def _is_stopword(self, token: str) -> bool:
        """Check if a token is a stopword; handles small typos via fuzzy match."""
        lower = token.lower()
        if lower in self.stopwords:
            return True
        close = difflib.get_close_matches(lower, list(self.stopwords), n=1, cutoff=0.85)
        return bool(close)
    
    def _build_indices(self):
        """Build search indices for columns and values"""
        # Column name index: column_name (lowercase) -> [(table_alias, actual_column_name)]
        self.column_index: Dict[str, List[Tuple[str, str]]] = {}
        
        # Value index: value (lowercase) -> [(table_alias, column_name)]
        self.value_index: Dict[str, List[Tuple[str, str]]] = {}
        
        for table_alias, metadata in self.table_metadata.items():
            columns = metadata.get('columns', [])
            column_values = metadata.get('column_values', {})
            
            # Index columns
            for col in columns:
                col_lower = col.lower()
                if col_lower not in self.column_index:
                    self.column_index[col_lower] = []
                self.column_index[col_lower].append((table_alias, col))
                
                # Also index column without underscores for flexibility
                col_no_underscore = col_lower.replace('_', '')
                if col_no_underscore != col_lower:
                    if col_no_underscore not in self.column_index:
                        self.column_index[col_no_underscore] = []
                    self.column_index[col_no_underscore].append((table_alias, col))
            
            # Index values
            for col, values in column_values.items():
                for val in values:
                    val_lower = str(val).lower()
                    if len(val_lower) >= 3:  # Only index meaningful values
                        if val_lower not in self.value_index:
                            self.value_index[val_lower] = []
                        self.value_index[val_lower].append((table_alias, col))
    
    def parse_query(self, query: str) -> QueryIntent:
        """
        Parse a natural language query into structured intent.
        
        Args:
            query: Natural language query
        
        Returns:
            QueryIntent object with identified tables, columns, and operations
        """
        query_lower = query.lower()
        tokens = self._tokenize_query(query_lower)
        
        # Step 1: Identify operations
        operations = self._identify_operations(query_lower)
        aggregations = self._identify_aggregations(query_lower)
        grouping = self._needs_grouping(query_lower, aggregations)
        ordering = self._identify_ordering(query_lower)
        
        # Step 2: Find columns mentioned in query
        column_matches = self._find_column_matches(query, tokens)
        
        # Step 3: Find values that might indicate columns
        value_matches = self._find_value_matches(query, tokens)
        column_matches.extend(value_matches)
        
        # Step 4: Identify which tables are needed
        required_tables = self._identify_required_tables(column_matches)
        
        # Step 5: Determine necessary joins
        required_joins = self._determine_joins(required_tables)
        
        # Step 6: Parse filters
        filters = self._parse_filters(query, column_matches)
        
        return QueryIntent(
            original_query=query,
            requested_columns=column_matches,
            required_tables=required_tables,
            required_joins=required_joins,
            operations=operations,
            filters=filters,
            aggregations=aggregations,
            grouping=grouping,
            ordering=ordering
        )
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query into words"""
        raw_tokens = re.findall(r'\b\w+\b', query)
        tokens: List[str] = []
        for token in raw_tokens:
            if len(token) <= 1:
                continue
            if self._is_stopword(token):
                continue
            tokens.append(token)
        return tokens
    
    def _identify_operations(self, query_lower: str) -> List[str]:
        """Identify SQL operations requested in the query"""
        operations = ['SELECT']  # Always need SELECT
        
        operation_keywords = {
            'SUM': ['total', 'sum', 'add up', 'sum of'],
            'AVG': ['average', 'avg', 'mean'],
            'COUNT': ['count', 'how many', 'number of'],
            'MAX': ['max', 'maximum', 'highest', 'largest', 'top'],
            'MIN': ['min', 'minimum', 'lowest', 'smallest'],
            'DISTINCT': ['distinct', 'unique', 'different'],
            'GROUP BY': [' per ', ' by ', ' for each ', 'group by', 'grouped by', 'split by'],
            'ORDER BY': ['sort', 'order by', 'sorted by', 'ascending', 'descending'],
            'LIMIT': ['top ', 'first ', 'limit'],
        }
        
        for op, keywords in operation_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if op not in operations:
                    operations.append(op)
        
        return operations
    
    def _identify_aggregations(self, query_lower: str) -> List[str]:
        """Identify aggregate functions needed"""
        aggs = []
        
        if any(kw in query_lower for kw in ['total', 'sum']):
            aggs.append('SUM')
        if any(kw in query_lower for kw in ['average', 'avg', 'mean']):
            aggs.append('AVG')
        if any(kw in query_lower for kw in ['count', 'how many', 'number of']):
            aggs.append('COUNT')
        if any(kw in query_lower for kw in ['max', 'maximum', 'highest']):
            aggs.append('MAX')
        if any(kw in query_lower for kw in ['min', 'minimum', 'lowest']):
            aggs.append('MIN')
        
        return list(set(aggs))
    
    def _needs_grouping(self, query_lower: str, aggregations: List[str]) -> bool:
        """Determine if GROUP BY is needed"""
        has_agg = len(aggregations) > 0
        has_by = any(kw in query_lower for kw in [' per ', ' by ', ' for each ', 'group'])
        
        return has_agg and has_by
    
    def _identify_ordering(self, query_lower: str) -> Optional[str]:
        """Identify ORDER BY requirements"""
        if 'ascending' in query_lower or 'asc' in query_lower:
            return 'ASC'
        if 'descending' in query_lower or 'desc' in query_lower:
            return 'DESC'
        if any(kw in query_lower for kw in ['top', 'highest', 'largest']):
            return 'DESC'
        if any(kw in query_lower for kw in ['lowest', 'smallest']):
            return 'ASC'
        return None
    
    def _find_column_matches(self, query: str, tokens: List[str]) -> List[TableColumnMatch]:
        """Find columns mentioned in the query"""
        matches = []
        query_lower = query.lower()
        
        # Direct column name matching
        for table_alias, metadata in self.table_metadata.items():
            columns = metadata.get('columns', [])
            table_name = metadata.get('name', table_alias)
            
            for col in columns:
                col_lower = col.lower()
                
                # Exact match
                if col_lower in query_lower:
                    matches.append(TableColumnMatch(
                        table_name=table_name,
                        table_alias=table_alias,
                        column_name=col,
                        query_token=col_lower,
                        confidence=1.0,
                        match_type='exact'
                    ))
                    continue
                
                # Fuzzy match on tokens
                col_words = col_lower.split('_')
                matched_words = sum(1 for word in col_words if word in tokens)
                
                if matched_words > 0 and matched_words >= len(col_words) / 2:
                    confidence = matched_words / len(col_words)
                    matches.append(TableColumnMatch(
                        table_name=table_name,
                        table_alias=table_alias,
                        column_name=col,
                        query_token='_'.join([w for w in col_words if w in tokens]),
                        confidence=confidence,
                        match_type='fuzzy'
                    ))
        
        return matches
    
    def _expand_value_synonyms(self, value: str) -> Set[str]:
        """Return a set of synonyms for a literal value (case-insensitive)."""
        base = value.strip().lower()
        synonyms = {base}
        synonym_map = {
            "y": {"y", "yes", "true", "1", "on"},
            "yes": {"y", "yes", "true", "1", "on"},
            "n": {"n", "no", "false", "0", "off"},
            "no": {"n", "no", "false", "0", "off"},
        }
        if base in synonym_map:
            synonyms.update(synonym_map[base])
        return synonyms

    def _find_value_matches(self, query: str, tokens: List[str]) -> List[TableColumnMatch]:
        """Find columns by matching values in the query"""
        matches = []
        query_lower = query.lower()
        token_set = set(tokens)
        
        for table_alias, metadata in self.table_metadata.items():
            table_name = metadata.get('name', table_alias)
            column_values = metadata.get('column_values', {})
            
            for col, values in column_values.items():
                for val in values:
                    val_str = str(val).lower()
                    candidates = self._expand_value_synonyms(val_str)
                    if any(candidate in query_lower for candidate in candidates if len(candidate) >= 1):
                        matches.append(TableColumnMatch(
                            table_name=table_name,
                            table_alias=table_alias,
                            column_name=col,
                            query_token=next(iter(candidates)),
                            confidence=0.9,
                            match_type='value_match'
                        ))
                        break  # Only one match per column
                    if any(candidate in token_set for candidate in candidates):
                        matches.append(TableColumnMatch(
                            table_name=table_name,
                            table_alias=table_alias,
                            column_name=col,
                            query_token=next(iter(candidates)),
                            confidence=0.85,
                            match_type='value_match'
                        ))
                        break
        
        return matches
    
    def _identify_required_tables(self, column_matches: List[TableColumnMatch]) -> List[str]:
        """Identify which tables are needed based on column matches"""
        tables = set()
        
        for match in column_matches:
            tables.add(match.table_alias)
        
        return list(tables)
    
    def _determine_joins(self, required_tables: List[str]) -> List[Dict[str, Any]]:
        """
        Determine necessary joins to connect required tables.
        Uses relationships to build a minimal spanning tree of joins.
        """
        if len(required_tables) <= 1:
            return []
        
        # Build graph of table relationships
        table_set = set(required_tables)
        relevant_relationships = [
            rel for rel in self.relationships
            if rel.get('source_alias') in table_set and rel.get('target_alias') in table_set
        ]
        
        if not relevant_relationships:
            logger.warning(f"No relationships found between tables: {required_tables}")
            return []
        
        # Use a simple approach: connect tables using available relationships
        # Start with first table and progressively add others
        connected = {required_tables[0]}
        joins = []
        
        while len(connected) < len(required_tables):
            # Find a relationship that connects a connected table to an unconnected one
            found = False
            for rel in relevant_relationships:
                src = rel.get('source_alias')
                tgt = rel.get('target_alias')
                
                if src in connected and tgt not in connected:
                    joins.append(rel)
                    connected.add(tgt)
                    found = True
                    break
                elif tgt in connected and src not in connected:
                    # Reverse the relationship
                    joins.append({
                        'source_alias': tgt,
                        'source_column': rel.get('target_column'),
                        'target_alias': src,
                        'target_column': rel.get('source_column'),
                        'join_type': rel.get('join_type', 'inner')
                    })
                    connected.add(src)
                    found = True
                    break
            
            if not found:
                # Can't connect all tables
                logger.warning(f"Could not connect all tables. Connected: {connected}, Required: {table_set}")
                break
        
        return joins
    
    def _parse_filters(self, query: str, column_matches: List[TableColumnMatch]) -> List[Dict[str, Any]]:
        """Parse WHERE clause conditions from query"""
        filters = []
        query_lower = query.lower()
        
        # Look for comparison operators
        comparison_patterns = [
            (r'(\w+)\s+(?:is\s+)?(?:greater\s+than|>)\s+(\d+)', 'GT'),
            (r'(\w+)\s+(?:is\s+)?(?:less\s+than|<)\s+(\d+)', 'LT'),
            (r'(\w+)\s+(?:is\s+)?(?:equal\s+to|=|equals)\s+(.+?)(?:\s|$)', 'EQ'),
            (r'after\s+(\d{4})', 'GT_DATE'),
            (r'before\s+(\d{4})', 'LT_DATE'),
        ]
        
        # This is a simplified filter parser - can be enhanced
        return filters


def parse_multi_table_query(
    query: str,
    table_metadata: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> QueryIntent:
    """
    Convenience function to parse a multi-table query.
    
    Args:
        query: Natural language query
        table_metadata: Table metadata dictionary
        relationships: List of relationships between tables
    
    Returns:
        QueryIntent with identified tables, columns, and joins
    """
    parser = MultiTableQueryParser(table_metadata, relationships)
    return parser.parse_query(query)

