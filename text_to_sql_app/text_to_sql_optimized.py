"""
Optimized Text-to-SQL Solution for M4 Mac
Uses GGUF quantization for efficient inference on Apple Silicon
"""

import os
import json
import logging
import sqlite3
import re
import difflib
import difflib
import duckdb
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(name)s,%(levelname)s,%(message)s'
)
logger = logging.getLogger(__name__)

GENERIC_COLUMN_TOKENS = {
    "total",
    "amount",
    "value",
    "values",
    "count",
    "counts",
    "number",
    "numbers",
    "sum",
    "avg",
    "average",
    "score",
}

# Try to import llama-cpp-python, fallback to transformers if not available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclass
class TextToSQLConfig:
    """Configuration for Text-to-SQL system"""
    # LLM Configuration
    llm_model: str = "defog/sqlcoder-7b-2"  # HuggingFace model name
    use_gguf: bool = True  # Use GGUF quantization if available (recommended for M4 Mac)
    gguf_model_path: Optional[str] = None  # Path to GGUF file, auto-detected if None
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    max_new_tokens: int = 512
    temperature: float = 0.1
    
    # GGUF-specific settings
    n_ctx: int = 2048  # Context window for GGUF
    n_threads: int = 0  # 0 = auto-detect (use all cores)
    n_gpu_layers: int = 0  # 0 = CPU only, -1 = use all GPU layers (Metal)
    
    # Vector Store
    vector_store_path: str = "./vector_store"
    collection_name: str = "sql_metadata"
    
    # Database
    database_path: str = "./sample_database.db"
    
    # Correction Loop
    max_correction_attempts: int = 3
    
    # RAG
    top_k_metadata: int = 5


def find_gguf_file(model_name: str) -> Optional[str]:
    """Find GGUF file in HuggingFace cache"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir_name)
    
    if not os.path.exists(model_path):
        return None
    
    # Search for GGUF files
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.gguf'):
                full_path = os.path.join(root, file)
                # Prefer Q5_K_M or Q4_K_M quantization
                if 'q5_k_m' in file.lower() or 'q4_k_m' in file.lower():
                    return full_path
                # Fallback to any GGUF
                return full_path
    
    return None


# Import shared classes from original file
from .text_to_sql_architecture import MetadataExtractor, VectorStore, SQLExecutor
from .multi_table_query_parser import parse_multi_table_query, QueryIntent
from .column_alternatives import group_alternatives_by_query_term


class DuckDBMetadataExtractor(MetadataExtractor):
    """Metadata extractor for DuckDB databases"""

    def connect(self):
        import duckdb

        if not hasattr(self, "duck_conn") or self.duck_conn is None:
            self.duck_conn = duckdb.connect(self.db_path, read_only=True)
            try:
                self.duck_conn.execute("PRAGMA threads=1")
            except Exception:
                pass

    def get_schema_metadata(self) -> List[Dict[str, Any]]:
        self.connect()
        cursor = self.duck_conn.cursor()
        tables = cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
        metadata = []
        for (table_name,) in tables:
            columns = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            column_info = []
            for cid, name, dtype, notnull, default_val, pk in columns:
                column_info.append({
                    "column_name": name,
                    "data_type": dtype,
                    "is_nullable": not bool(notnull),
                    "is_primary_key": bool(pk),
                    "default_value": default_val,
                })
            metadata.append({
                "table_name": table_name,
                "columns": column_info,
                "foreign_keys": [],
                "description": f"Table {table_name} with {len(column_info)} columns",
            })
        return metadata

    def close(self):
        if hasattr(self, "duck_conn") and self.duck_conn:
            self.duck_conn.close()


class DuckDBSQLExecutor:
    """Execute SQL queries in DuckDB"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            conn.execute("PRAGMA threads=1")
        except Exception:
            pass
        return conn

    def execute_query(self, sql: str, **kwargs) -> Tuple[bool, Any, Optional[str]]:
        try:
            with self._connect() as conn:
                result = conn.execute(sql)
                if result.description:
                    try:
                        df = result.fetch_df()
                    except Exception as fetch_error:
                        logger.warning(f"fetch_df() failed with {fetch_error}; falling back to fetchall().")
                        result = conn.execute(sql)
                        rows = result.fetchall()
                        columns = [desc[0] for desc in result.description]
                        df = pd.DataFrame(rows, columns=columns)
                    return True, df, None
                return True, "Query executed successfully.", None
        except Exception as e:
            logger.error(f"DuckDB SQL Error: {e}")
            return False, None, str(e)

    def validate_syntax(self, sql: str, **kwargs) -> Tuple[bool, Optional[str]]:
        try:
            with self._connect() as conn:
                conn.execute(f"EXPLAIN {sql}")
                return True, None
        except Exception as e:
            return False, str(e)

    def close(self):
        pass


class SQLLLM_GGUF:
    """Optimized LLM using GGUF with llama.cpp for Apple Silicon"""
    
    _MODEL_CACHE: Dict[str, "Llama"] = {}
    _TOKENIZER_CACHE: Dict[str, Any] = {}
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load GGUF model"""
        logger.info("Loading GGUF model with llama.cpp...")
        
        # Find GGUF file
        if self.config.gguf_model_path:
            gguf_path = self.config.gguf_model_path
        else:
            gguf_path = find_gguf_file(self.config.llm_model)
        
        if not gguf_path or not os.path.exists(gguf_path):
            raise FileNotFoundError(
                f"GGUF file not found. Please download it or set gguf_model_path.\n"
                f"Searched in: ~/.cache/huggingface/hub/models--{self.config.llm_model.replace('/', '--')}"
            )
        
        logger.info(f"Found GGUF file: {gguf_path}")
        
        cache_key = f"{gguf_path}|ctx={self.config.n_ctx}|gpu={self.config.n_gpu_layers}"
        if cache_key in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[cache_key]
            self.tokenizer = self._TOKENIZER_CACHE.get(self.config.llm_model)
            logger.info("Reusing cached GGUF model instance.")
            return
        
        # Load tokenizer from HuggingFace (needed for prompt formatting)
        try:
            from transformers import AutoTokenizer
            if self.config.llm_model in self._TOKENIZER_CACHE:
                self.tokenizer = self._TOKENIZER_CACHE[self.config.llm_model]
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.llm_model,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self._TOKENIZER_CACHE[self.config.llm_model] = self.tokenizer
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using basic tokenization.")
            self.tokenizer = None
        
        # Load GGUF model with Metal support for M4 Mac
        try:
            self.model = Llama(
                model_path=gguf_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads or None,
                use_mmap=True,
                use_mlock=False,
            )
            logger.info("✅ GGUF model loaded successfully with Metal acceleration")
            self._MODEL_CACHE[cache_key] = self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {e}")
    
    def generate_sql(self, prompt: str) -> str:
        """Generate SQL from prompt using GGUF"""
        if not self.model:
            self.load_model()
        
        try:
            # Generate with GGUF
            output = self.model(
                prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                stop=["###", "\n\n\n"],  # Stop tokens for SQLCoder
                echo=False,
            )
            
            generated_text = output['choices'][0]['text'].strip()
            
            # Extract SQL query (remove any markdown formatting)
            sql = generated_text
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            # Clean up common artifacts
            sql = sql.split('\n\n')[0]  # Take first SQL statement
            sql = sql.strip()
            
            return sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise


class SQLLLM_Transformers:
    """Original transformers-based implementation with quantization support"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def _get_device(self) -> str:
        """Determine device to use"""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """Load model with quantization if available"""
        logger.info(f"Loading model {self.config.llm_model} on {self.device}...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            # Try to use 4-bit quantization for memory efficiency
            quantization_config = None
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization for memory efficiency")
            except Exception as e:
                logger.warning(f"4-bit quantization not available: {e}. Using full precision.")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            elif self.device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
                model_kwargs["device_map"] = "auto"
            elif self.device == "mps":
                model_kwargs["torch_dtype"] = torch.float32
                # Don't use device_map for MPS
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                **model_kwargs
            )
            
            if self.device == "mps" and not quantization_config:
                self.model = self.model.to("mps")
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" and not quantization_config else torch.float32
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_sql(self, prompt: str) -> str:
        """Generate SQL from prompt"""
        if not self.pipeline:
            self.load_model()
        
        try:
            output = self.pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = output[0]['generated_text'].strip()
            
            # Extract SQL query
            sql = generated_text
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            return sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise


class SQLLLM:
    """Unified LLM interface - automatically selects best backend"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        
        # Choose backend based on availability and config
        if config.use_gguf and LLAMA_CPP_AVAILABLE:
            try:
                # Check if GGUF file exists
                gguf_path = config.gguf_model_path or find_gguf_file(config.llm_model)
                if gguf_path and os.path.exists(gguf_path):
                    self.backend = SQLLLM_GGUF(config)
                    logger.info("Using GGUF backend (optimized for Apple Silicon)")
                else:
                    logger.warning("GGUF file not found, falling back to transformers")
                    self.backend = SQLLLM_Transformers(config)
            except Exception as e:
                logger.warning(f"Could not use GGUF backend: {e}. Falling back to transformers.")
                self.backend = SQLLLM_Transformers(config)
        else:
            self.backend = SQLLLM_Transformers(config)
            if config.use_gguf:
                logger.warning("GGUF requested but llama-cpp-python not available. Install with: pip install llama-cpp-python")
    
    def load_model(self):
        """Load model using selected backend"""
        self.backend.load_model()
    
    def generate_sql(self, prompt: str) -> str:
        """Generate SQL using selected backend"""
        return self.backend.generate_sql(prompt)


class TextToSQLSystem:
    """Optimized Text-to-SQL system with GGUF support"""

    COLUMN_VALUE_LIMIT = 40
    SQL_KEYWORDS = {
        "select", "from", "where", "join", "inner", "left", "right", "full", "outer",
        "on", "group", "by", "order", "asc", "desc", "limit", "distinct", "count",
        "sum", "avg", "min", "max", "case", "when", "then", "else", "end", "and",
        "or", "not", "as", "having"
    }
    
    def __init__(self, config: TextToSQLConfig):
        # Override to use optimized SQLLLM
        self.config = config
        if config.database_path.endswith('.duckdb'):
            self.metadata_extractor = DuckDBMetadataExtractor(config.database_path)
            self.sql_executor = DuckDBSQLExecutor(config.database_path)
        else:
            self.metadata_extractor = MetadataExtractor(config.database_path)
            self.sql_executor = SQLExecutor(config.database_path)
        self.vector_store = VectorStore(config)
        self.llm = SQLLLM(config)  # Use optimized LLM
        self.table_columns_map: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self._metadata_loaded = False
        
        # Initialize components
        self._initialize_system()

    def _filter_relationships(self) -> None:
        """Remove relationships that reference tables not currently loaded."""
        if not self.relationships:
            return
        valid_aliases = set(self.table_columns_map.keys())
        filtered = []
        for rel in self.relationships:
            src = rel.get("source_alias")
            tgt = rel.get("target_alias")
            if src in valid_aliases and tgt in valid_aliases:
                filtered.append(rel)
        if len(filtered) != len(self.relationships):
            logger.info(
                "Filtered %d stale relationships referencing unloaded tables",
                len(self.relationships) - len(filtered),
            )
        self.relationships = filtered

    @staticmethod
    def _is_table_inventory_query(normalized_query: str) -> bool:
        patterns = [
            r"\bwhat\s+tables\b",
            r"\bwhat\s+all\s+tables\b",
            r"\bwhat\s+tables\s+do\s+i\s+have\b",
            r"\blist\s+(all\s+)?tables\b",
            r"\bshow\s+(all\s+)?tables\b",
            r"\bavailable\s+tables\b",
            r"\bdatasets?\s+available\b",
        ]
        return any(re.search(pattern, normalized_query) for pattern in patterns)

    @staticmethod
    def _slugify_value(value: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
        slug = re.sub(r"_+", "_", slug)
        return slug.strip("_").lower() or "value"

    def _build_single_table_pivot_sql(
        self,
        table_name: str,
        schema_info: Dict[str, Any],
        user_query: str,
        unit_factor: Optional[int] = None,
        unit_label: Optional[str] = None,
    ) -> Optional[str]:
        columns = schema_info.get("columns", []) or []
        column_types = schema_info.get("column_types_map", {}) or {}
        values_map = (schema_info.get("column_values_map") or {})
        value_matches = schema_info.get("value_matches") or []
        pivot_col = None
        if value_matches:
            pivot_col = value_matches[0].get("column")
        if not pivot_col:
            query_lower = user_query.lower()
            for col in columns:
                normalized = col.lower().replace("_", " ")
                if (re.search(rf"\b{re.escape(col.lower())}\b", query_lower) or normalized in query_lower) and values_map.get(col):
                    pivot_col = col
                    break
        if not pivot_col:
            return None
        pivot_values = values_map.get(pivot_col)
        if not pivot_values:
            return None

        # Choose measure column (numeric focus)
        numeric_keywords = ["final_outstanding", "outstanding", "amount", "total", "sum", "balance", "value"]
        query_lower = user_query.lower()
        numeric_cols = []
        for col in columns:
            if col == pivot_col:
                continue
            dtype = (column_types.get(col) or "").lower()
            if any(keyword in col.lower() for keyword in numeric_keywords) or any(
                token in dtype for token in ["double", "decimal", "numeric", "int", "float"]
            ):
                numeric_cols.append(col)
        measure_col = None
        for col in numeric_cols:
            normalized = col.lower().replace("_", " ")
            if re.search(rf"\b{re.escape(col.lower())}\b", query_lower) or normalized in query_lower:
                measure_col = col
                break
        if not measure_col:
            # Prioritize columns containing numeric keywords explicitly
            prioritized = sorted(
                numeric_cols,
                key=lambda c: (0 if any(k in c.lower() for k in ["final_outstanding", "outstanding"]) else 1,
                               0 if any(k in c.lower() for k in ["amount", "balance", "value", "total"]) else 1,
                               c)
            )
            if prioritized:
                measure_col = prioritized[0]
        if not measure_col and numeric_cols:
            measure_col = numeric_cols[0]
        if not measure_col:
            return None

        # Choose dimension column (non-numeric)
        dimension_col = None
        text_cols = []
        for col in columns:
            if col in (pivot_col, measure_col):
                continue
            dtype = (column_types.get(col) or "").lower()
            if not any(token in dtype for token in ["double", "decimal", "numeric", "int", "float"]):
                text_cols.append(col)
        for col in text_cols:
            normalized = col.lower().replace("_", " ")
            if re.search(rf"\b{re.escape(col.lower())}\b", query_lower) or normalized in query_lower:
                dimension_col = col
                break
        if not dimension_col and text_cols:
            dimension_col = text_cols[0]
        if not dimension_col:
            # default to first column not pivot or measure
            for col in columns:
                if col not in (pivot_col, measure_col):
                    dimension_col = col
                    break
        if not dimension_col:
            return None

        select_parts = [dimension_col]
        for value in pivot_values[:10]:
            literal = value.replace("'", "''")
            slug = self._slugify_value(str(value))
            alias = f"{measure_col}_{slug}"
            if unit_label:
                alias = f"{alias}_in_{unit_label}"
            expression = f"SUM(CASE WHEN {pivot_col} = '{literal}' THEN {measure_col} ELSE 0 END)"
            if unit_factor:
                expression = f"{expression} / {unit_factor}"
            select_parts.append(f"{expression} AS {alias}")

        sql = (
            f"SELECT {', '.join(select_parts)}\n"
            f"FROM {table_name}\n"
            f"GROUP BY {dimension_col}\n"
            f"ORDER BY {dimension_col}"
        )
        return sql

    def _rewrite_single_table_sql(
        self,
        sql: str,
        table_name: str,
        schema_info: Dict[str, Any],
        unit_factor: Optional[int] = None,
        unit_label: Optional[str] = None,
    ) -> Optional[str]:
        columns = schema_info.get("columns", []) or []
        if not columns:
            return None

        # Remove JOIN clauses
        join_pattern = re.compile(r"\bjoin\b[\s\S]+?(?=\bwhere\b|\bgroup\b|\border\b|\bhaving\b|\blimit\b|$)", re.IGNORECASE)
        sql_no_join = join_pattern.sub(" ", sql)

        # Normalize FROM clause to remove aliases for main table
        from_pattern = re.compile(
            rf"(from\s+){table_name}\s+(?:as\s+)?([a-zA-Z_][\w]*)",
            re.IGNORECASE,
        )
        alias_match = from_pattern.search(sql_no_join)
        allowed_aliases = {table_name.lower()}
        cleaned_sql = sql_no_join
        if alias_match:
            alias = alias_match.group(2)
            allowed_aliases.add(alias.lower())
            cleaned_sql = from_pattern.sub(rf"\1{table_name}", cleaned_sql)

        column_lookup = {col.lower(): col for col in columns}

        def replace_alias(match):
            alias = match.group(1).lower()
            column = match.group(2)
            if alias not in allowed_aliases:
                if column.lower() in column_lookup:
                    return column_lookup[column.lower()]
                return None  # Unknown column from disallowed table
            return column_lookup.get(column.lower(), column)

        alias_pattern = re.compile(r"([a-zA-Z_][\w]*)\.(\w+)")
        start = 0
        result_parts = []
        for match in alias_pattern.finditer(cleaned_sql):
            replacement = replace_alias(match)
            if replacement is None:
                return None
            result_parts.append(cleaned_sql[start:match.start()])
            result_parts.append(replacement)
            start = match.end()
        result_parts.append(cleaned_sql[start:])
        rewritten = "".join(result_parts)

        # Cleanup redundant whitespace and dangling commas left by join removal
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        if unit_factor and f"/ {unit_factor}" not in rewritten:
            def apply_unit(match):
                expr = match.group(1)
                return f"{expr} / {unit_factor}"

            rewritten = re.sub(r"(?i)(SUM\([^)]*\))", lambda m: apply_unit(m), rewritten)

            if unit_label:
                unit_suffix = f"_in_{unit_label}"

                def adjust_alias(match):
                    alias = match.group(1)
                    if alias.lower().endswith(unit_suffix.lower()):
                        return match.group(0)
                    return f"AS {alias}{unit_suffix}"

                rewritten = re.sub(r"(?i)AS\s+([a-zA-Z_][\w]*)", adjust_alias, rewritten)
 
        return rewritten

    @staticmethod
    def _refine_user_query(user_query: str) -> str:
        """Normalize user phrasing to a single, crisp sentence."""
        query = user_query.strip()
        # Collapse whitespace
        query = re.sub(r"\s+", " ", query)
        # Merge trailing unit instructions into the main sentence
        unit_phrase_pattern = r"\.\s*(in\s+(?:crores|lakhs|millions|billions))"
        query = re.sub(unit_phrase_pattern, r" \1", query, flags=re.IGNORECASE)
        # Remove stray spaces before punctuation
        query = re.sub(r"\s+([,;:.?!])", r"\1", query)
        # Ensure sentence ends with a single period if none present
        if query and query[-1] not in ".?!":
            query = query + "."
        # Prefer lowercase "in" for unit phrases consistency
        query = re.sub(r"\bIn\b", "in", query)
        return query

    @staticmethod
    def _apply_value_corrections_to_sql(sql: str, schema_info: Dict[str, Any]) -> str:
        if not sql or not schema_info:
            return sql

        replacements = []
        for correction in schema_info.get("value_corrections", []) or []:
            original = str(correction.get("original") or "").strip()
            normalized = str(correction.get("normalized") or "").strip()
            if original and normalized and original.lower() != normalized.lower():
                replacements.append((original, normalized))
        for match in schema_info.get("value_matches", []) or []:
            value = str(match.get("value") or "").strip()
            if value:
                replacements.append((value, value))

        if not replacements:
            return sql

        seen = set()
        for original, normalized in replacements:
            key = (original.lower(), normalized)
            if key in seen:
                continue
            seen.add(key)
            pattern = re.compile(rf"'\s*{re.escape(original)}\s*'", re.IGNORECASE)
            sql = pattern.sub(lambda m: f"'{normalized}'", sql)
        return sql

    def _suggest_column_corrections(self, user_query: str, allowed_aliases: Optional[List[str]]) -> List[Dict[str, str]]:
        if not allowed_aliases:
            return []

        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", user_query.lower())
        if not tokens:
            return []

        column_lookup: Dict[str, Dict[str, str]] = {}
        actual_tokens = set()
        for alias in allowed_aliases:
            meta = self.table_columns_map.get(alias)
            if not meta:
                continue
            lookup = {col.lower(): col for col in meta.get("columns", [])}
            column_lookup[alias] = lookup
            actual_tokens.update(lookup.keys())

        suggestions: List[Dict[str, str]] = []
        seen = set()
        for token in tokens:
            if len(token) <= 2 or token in seen or token in actual_tokens:
                continue
            seen.add(token)
            best_alias = None
            best_column = None
            best_ratio = 0.0
            for alias, lookup in column_lookup.items():
                for lowered, original in lookup.items():
                    if lowered == token:
                        best_alias = alias
                        best_column = original
                        best_ratio = 1.0
                        break
                    if lowered in token or token in lowered:
                        # Encourage substring matches (e.g., charge_amount -> charge)
                        ratio = len(lowered) / max(len(token), len(lowered))
                        if ratio > best_ratio:
                            best_alias = alias
                            best_column = original
                            best_ratio = ratio
                if best_ratio == 1.0:
                    break
                matches = difflib.get_close_matches(token, list(lookup.keys()), n=1, cutoff=0.75)
                if matches:
                    match = matches[0]
                    ratio = difflib.SequenceMatcher(None, token, match).ratio()
                    if ratio > best_ratio:
                        best_alias = alias
                        best_column = lookup[match]
            if best_alias and best_column and best_ratio >= 0.75:
                suggestions.append(
                    {
                        "requested": token,
                        "normalized": best_column,
                        "table_alias": best_alias,
                        "table_name": self.table_columns_map[best_alias]["name"],
                    }
                )
        
        return suggestions

    def _validate_sql_columns(self, sql: str, allowed_tables: List[str]) -> None:
        if not sql:
            return

        try:
            parsed = duckdb.parse(sql)
        except Exception as exc:
            logger.warning(f"Could not parse SQL for validation: {exc}")
            return

        allowed_aliases = {
            alias: meta
            for alias, meta in self.table_columns_map.items()
            if not allowed_tables or meta["name"] in allowed_tables
        }
        allowed_column_map = {}
        for alias, meta in allowed_aliases.items():
            for column in meta.get("columns", []):
                allowed_column_map.setdefault(meta["name"].lower(), set()).add(column.lower())
                allowed_column_map.setdefault(alias.lower(), set()).add(column.lower())

        # Fallback: if no allowed tables specified, include all tables we know
        if not allowed_tables:
            for alias, meta in self.table_columns_map.items():
                for column in meta.get("columns", []):
                    allowed_column_map.setdefault(meta["name"].lower(), set()).add(column.lower())
                    allowed_column_map.setdefault(alias.lower(), set()).add(column.lower())

        def recurse(node) -> None:
            if isinstance(node, dict):
                node_type = node.get("type")
                if node_type == "table":
                    table_name = node.get("name", "").lower()
                    if table_name and table_name not in allowed_column_map:
                        raise ValueError(f"Table `{table_name}` is not in allowed schema.")
                if node_type == "column":
                    column_name = node.get("name", "").lower()
                    table_ref = node.get("table", "")
                    if table_ref:
                        table_ref_lower = table_ref.lower()
                        known_cols = allowed_column_map.get(table_ref_lower)
                        if known_cols and column_name not in known_cols:
                            raise ValueError(f"Column `{column_name}` not found in table `{table_ref}`.")
                    else:
                        # Column without explicit table reference: ensure present in at least one allowed table
                        if not any(column_name in cols for cols in allowed_column_map.values() if cols):
                            raise ValueError(f"Column `{column_name}` not found in allowed schema.")

                for value in node.values():
                    recurse(value)
            elif isinstance(node, list):
                for item in node:
                    recurse(item)

        recurse(parsed)

    def _build_table_document(self, table_name: str) -> Optional[str]:
        if not table_name:
            return None
        meta = self.table_columns_map.get(table_name.lower())
        if not meta:
            return None
        columns = meta.get("columns", [])
        column_types = meta.get("column_types", {})
        column_values = meta.get("column_values", {})

        descriptions = []
        for col in columns:
            dtype = column_types.get(col) or self._infer_type_from_name(col)
            desc = f"{col} ({dtype})"
            values = column_values.get(col)
            if values:
                preview = ", ".join(values[:4])
                if len(values) > 4:
                    preview += ", ..."
                desc += f" [Values: {preview}]"
            descriptions.append(desc)
        if not descriptions:
            descriptions.append("(no columns detected)")
        return f"Table: {table_name}\nColumns: {', '.join(descriptions)}"

    def _initialize_system(self):
        """Initialize the system with metadata"""
        if self._metadata_loaded:
            return

        logger.info("Initializing Text-to-SQL system...")

        # Load relationships if present
        relationships_path = Path(self.config.vector_store_path).resolve().parent.parent / "data" / "datasets.json"
        if relationships_path.exists():
            try:
                data = json.loads(relationships_path.read_text())
                self.relationships = data.get("relationships", [])
            except json.JSONDecodeError:
                logger.warning("Could not parse relationships file; ignoring joins")

        # Extract metadata
        try:
            metadata = self.metadata_extractor.get_schema_metadata()
            
            # Format for RAG
            metadata_texts = self.metadata_extractor.format_metadata_for_rag(metadata)
            
            # Add to vector store
            self.vector_store.initialize_collection()
            try:
                existing = self.vector_store.collection.count() if self.vector_store.collection else 0
            except Exception:
                existing = 0
            if existing == 0:
                enriched_texts = []
                for entry, base_text in zip(metadata, metadata_texts):
                    table_name = entry.get("table_name")
                    table_entry = self.table_columns_map.get(table_name.lower())
                    column_values = table_entry.get("column_values", {}) if table_entry else {}
                    columns = entry.get("columns", [])
                    descriptions = []
                    for col in columns:
                        name = col.get("column_name")
                        dtype = col.get("data_type")
                        desc = f"{name} ({dtype})"
                        values = column_values.get(name)
                        if values:
                            preview = ", ".join(values[:10])
                            if len(values) > 10:
                                preview += ", ..."
                            desc += f" [Values: {preview}]"
                        descriptions.append(desc)
                    if table_name:
                        enriched_texts.append(
                            f"Table: {table_name}\nColumns: {', '.join(descriptions)}"
                        )
                    else:
                        enriched_texts.append(base_text)
                self.vector_store.add_metadata(enriched_texts, metadata)
            else:
                logger.info("Vector store already populated; skipping metadata insertion")
            
            # Build table/column lookup for corrections
            self.table_columns_map = {}
            for table_meta in metadata:
                table_name = table_meta.get("table_name")
                columns = table_meta.get("columns", [])
                if not table_name:
                    continue
                logger.info("Profiling table %s", table_name)
                column_names = [col.get("column_name") for col in columns if col.get("column_name")]
                column_lookup = {name.lower(): name for name in column_names}
                column_types = {
                    col.get("column_name"): (col.get("data_type") or "")
                    for col in columns if col.get("column_name")
                }
                column_values = self._collect_column_values(table_name, column_names)
                logger.info("Profiled %s columns for %s", len(column_values), table_name)
                self.table_columns_map[table_name.lower()] = {
                    "name": table_name,
                    "columns": column_names,
                    "column_lookup": column_lookup,
                    "column_types": column_types,
                    "column_values": column_values,
                }
            logger.info("Loaded table metadata: %s", list(self.table_columns_map.keys()))
            if self.relationships:
                logger.info("Loaded %d relationships", len(self.relationships))
                self._filter_relationships()
            
            logger.info("System initialized successfully")
            self._metadata_loaded = True
        except Exception as e:
            logger.warning(f"Could not initialize with database schema: {e}. System will use query-provided schema.")
    
    def _extract_schema_from_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Extract table and column information from user query.
        Looks for patterns like:
        - "table is X" or "table: X" or "table=X"
        - "columns are X, Y, Z" or "column: X, Y, Z" or "columns: X, Y, Z"
        - "given columns are X, Y, Z and table is Y"
        """
        schema_info = {
            "table_name": None,
            "columns": []
        }
        
        # Extract table name - more flexible patterns
        table_patterns = [
            r"table\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # "table is disbursement_register"
            r"table\s*[:=]\s*([a-zA-Z_][a-zA-Z0-9_]*)",  # "table: X" or "table=X"
            r"table\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # "table X"
            r"and\s+table\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # "and table is X"
        ]
        
        for pattern in table_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                schema_info["table_name"] = match.group(1)
                logger.info(f"Extracted table name: {schema_info['table_name']}")
                break
        
        # Extract columns - look for patterns like "given columns are X, Y, Z"
        # Be more flexible with patterns
        column_patterns = [
            r"columns?\s*[:=]\s*([^\n\r]+)",
            r"given\s+columns?\s+are\s*([^\n\r]+)",
            r"columns?\s+are\s*([^\n\r]+)",
        ]
 
        for pattern in column_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                columns_str = match.group(1).strip()
                columns_str = re.sub(r"\s+table\s*[:=]?.*", "", columns_str, flags=re.IGNORECASE)
                # Split by comma and clean up
                columns = [col.strip() for col in re.split(r'[;,]', columns_str)]
                columns = [col for col in columns if col and col.lower() not in ['and', 'table']]
                # Strip trailing punctuation such as '.'
                columns = [col.rstrip('.').strip() for col in columns]
                schema_info["columns"] = columns
                logger.info(f"Extracted columns: {schema_info['columns']}")
                break
        
        # If we found both table and columns, definitely use query schema
        # If we found at least one, also use it
        if schema_info["table_name"] or schema_info["columns"]:
            schema_info = self._apply_schema_corrections(schema_info, user_query)
            logger.info(f"Schema extracted from query (after normalization): {schema_info}")
            return schema_info
        
        return None

    def _apply_schema_corrections(self, schema_info: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        corrections = {"table": None, "columns": []}
        table_entry = None
        table_name = schema_info.get("table_name")
        user_query_lower = user_query.lower()
        
        if table_name:
            key = table_name.lower()
            if key in self.table_columns_map:
                table_entry = self.table_columns_map[key]
                schema_info["table_name"] = table_entry["name"]
            else:
                matches = difflib.get_close_matches(key, list(self.table_columns_map.keys()), n=1, cutoff=0.6)
                if matches:
                    table_entry = self.table_columns_map[matches[0]]
                    corrections["table"] = {
                        "original": table_name,
                        "normalized": table_entry["name"],
                    }
                    schema_info["table_name"] = table_entry["name"]
        elif len(self.table_columns_map) == 1:
            table_entry = next(iter(self.table_columns_map.values()))
            corrections["table"] = {
                "original": None,
                "normalized": table_entry["name"],
            }
            schema_info["table_name"] = table_entry["name"]
        
        columns = schema_info.get("columns") or []
        if not columns:
            detected_cols = self._detect_columns_from_query(user_query, [schema_info.get("table_name")])
            if detected_cols:
                columns = detected_cols
                schema_info["columns"] = columns
        values_map = {}
        if table_entry:
            values_map = table_entry.get("column_values", {})
        if not columns and values_map:
            value_detected_cols = self._detect_columns_from_values(user_query, values_map)
            if value_detected_cols:
                for col in value_detected_cols:
                    if col not in columns:
                        columns.append(col)
                schema_info["columns"] = columns
        normalized_columns = []
        if table_entry:
            lookup = table_entry["column_lookup"]
            for col in columns:
                col_low = col.lower()
                if col_low in lookup:
                    normalized = lookup[col_low]
                else:
                    match = difflib.get_close_matches(col_low, list(lookup.keys()), n=1, cutoff=0.6)
                    if match:
                        normalized = lookup[match[0]]
                        corrections["columns"].append({
                            "original": col,
                            "normalized": normalized,
                            "reason": "fuzzy_match"
                        })
                    else:
                        normalized = col
                normalized_columns.append(normalized)
            if not normalized_columns:
                # Detect columns mentioned in the text if none explicitly provided
                for lookup_key, actual in lookup.items():
                    if lookup_key in user_query_lower:
                        normalized_columns.append(actual)
                        corrections["columns"].append({
                            "original": None,
                            "normalized": actual,
                            "reason": "detected_in_text"
                        })
            schema_info["columns"] = list(dict.fromkeys(normalized_columns))
            schema_info["column_types_map"] = table_entry["column_types"]
            filtered_values = {
                col: values_map.get(col)
                for col in schema_info["columns"]
                if values_map.get(col)
            }
            schema_info["column_values_map"] = filtered_values
            matches, value_corrections, value_instruction = self._detect_value_context(user_query, table_entry["name"], schema_info["columns"], values_map)
            if matches:
                for match in matches:
                    col = match.get("column")
                    if col and col not in schema_info["columns"]:
                        schema_info["columns"].append(col)
            filtered_values = {}
            for col in schema_info["columns"]:
                vals = values_map.get(col)
                if vals:
                    filtered_values[col] = vals
            schema_info["column_values_map"] = filtered_values
            schema_info["value_matches"] = matches
            schema_info["value_corrections"] = value_corrections
            schema_info["value_instruction"] = value_instruction
        else:
            schema_info["column_types_map"] = {}
            schema_info["column_values_map"] = {}
            schema_info["value_matches"] = []
            schema_info["value_corrections"] = []
            schema_info["value_instruction"] = None
        schema_info["corrections"] = corrections
        return schema_info
    
    def _build_schema_text_from_query(self, schema_info: Dict[str, Any]) -> str:
        """Build schema text format from extracted schema info"""
        table_name = schema_info.get("table_name", "unknown_table")
        columns = schema_info.get("columns", [])
        column_types_map = schema_info.get("column_types_map") or {}
        table_entry = self.table_columns_map.get(table_name.lower())
        values_map_full = schema_info.get("column_values_map") or (table_entry.get("column_values", {}) if table_entry else {})
        column_values_map = {
            col: values_map_full.get(col)
            for col in columns
            if values_map_full.get(col)
        }
        
        if columns:
            col_descriptions = []
            for col in columns:
                dtype = column_types_map.get(col)
                if not dtype and table_entry:
                    dtype = table_entry["column_types"].get(col)
                dtype = dtype or self._infer_type_from_name(col)
                desc = f"{col} ({dtype})"
                values = column_values_map.get(col)
                if not values and table_entry:
                    values = table_entry.get("column_values", {}).get(col)
                if values:
                    preview = ", ".join(values[:8])
                    if len(values) > 8:
                        preview += ", ..."
                    desc += f" [Values: {preview}]"
                col_descriptions.append(desc)
            schema_text = f"Table: {table_name}\nColumns: {', '.join(col_descriptions)}"
        else:
            schema_text = f"Table: {table_name}\nColumns: (not specified)"
        
        return schema_text
    
    def _infer_type_from_name(self, column_name: str) -> str:
        lower = column_name.lower()
        if any(keyword in lower for keyword in ["date", "time"]):
            return "DATE"
        if any(keyword in lower for keyword in ["amount", "total", "count", "price", "cost", "balance", "score", "qty"]):
            return "NUMERIC"
        if lower.endswith("_id") or lower == "id":
            return "INTEGER"
        return "TEXT"

    def _detect_columns_from_query(self, user_query: str, table_names: List[str]) -> List[str]:
        query_lower = user_query.lower()
        raw_tokens = re.findall(r"[a-z0-9_]+", query_lower)
        token_set = set()
        for token in raw_tokens:
            token_set.add(token)
            if token.endswith("s") and len(token) > 2:
                token_set.add(token[:-1])
        detected = []

        for table in table_names:
            entry = self.table_columns_map.get(table.lower())
            if not entry:
                continue
            for col in entry.get("columns", []):
                col_lower = col.lower()
                if col_lower in query_lower:
                    detected.append(col)
                    continue
                words = [w for w in col_lower.split("_") if w]
                normalized_words = []
                for w in words:
                    normalized_words.append(w)
                    if w.endswith("s") and len(w) > 3:
                        normalized_words.append(w[:-1])
                normalized_unique = list(dict.fromkeys(normalized_words))
                match_count = sum(1 for w in normalized_unique if w in token_set)
                signal_matches = sum(1 for w in normalized_unique if w in token_set and w not in GENERIC_COLUMN_TOKENS)
                threshold = max(1, math.ceil(len(words) / 2))
                if match_count >= threshold and (signal_matches > 0 or len(words) == 1):
                    detected.append(col)
                    continue
                combined = "".join(words)
                if difflib.get_close_matches(combined, raw_tokens, n=1, cutoff=0.82):
                    detected.append(col)
            if detected:
                break

        unique = []
        seen = set()
        for col in detected:
            if col not in seen:
                unique.append(col)
                seen.add(col)
        return unique

    def _detect_value_context(
        self,
        user_query: str,
        table_name: Optional[str],
        columns: List[str],
        column_values_map: Dict[str, List[str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Optional[str]]:
        if not table_name or not columns:
            return [], [], None
        query_lower = user_query.lower()
        tokens = re.findall(r"[a-z0-9_]+", query_lower)
        matches: List[Dict[str, str]] = []
        corrections: List[Dict[str, str]] = []
        used = set()

        for col in columns:
            values = column_values_map.get(col)
            if not values:
                continue
            lowered_values = [str(v).lower() for v in values]
            # direct substring matches
            for actual, val_lower in zip(values, lowered_values):
                if val_lower and len(val_lower) >= 3 and val_lower in query_lower:
                    if actual not in used:
                        start = query_lower.index(val_lower)
                        raw_fragment = user_query[start:start + len(val_lower)]
                        matches.append({"column": col, "value": actual})
                        if raw_fragment != actual:
                            corrections.append({
                                "column": col,
                                "original": raw_fragment,
                                "normalized": actual
                            })
                        used.add(actual)
            # fuzzy token matches
            for token in tokens:
                if len(token) < 3:
                    continue
                candidates = [token]
                if token.endswith('s'):
                    candidates.append(token[:-1])
                for cand in candidates:
                    match = difflib.get_close_matches(cand, lowered_values, n=1, cutoff=0.82)
                    if match:
                        actual = values[lowered_values.index(match[0])]
                        if actual not in used:
                            corrections.append({"column": col, "original": token, "normalized": actual})
                            used.add(actual)
                        break

        instruction = None
        if matches or corrections:
            parts = []
            for c in corrections:
                parts.append(f"{c['column']}: treat '{c['original']}' as '{c['normalized']}'")
            if matches:
                for m in matches:
                    parts.append(f"{m['column']}: use literal '{m['value']}'")
            instruction = "Use these literal column values -> " + "; ".join(parts)
        elif matches:
            parts = [
                f"column {m['column']}: use value '{m['value']}'"
                for m in matches
            ]
            instruction = "Use detected values exactly as written -> " + "; ".join(parts)
        return matches, corrections, instruction

    def _detect_columns_from_values(self, user_query: str, values_map: Dict[str, List[str]]) -> List[str]:
        query_lower = user_query.lower()
        tokens = re.findall(r"[a-z0-9_]+", query_lower)
        detected = []
        for col, values in values_map.items():
            if not values:
                continue
            lowered = [str(v).lower() for v in values]
            if any(val in query_lower for val in lowered if len(val) >= 3):
                detected.append(col)
                continue
            for token in tokens:
                if len(token) < 3:
                    continue
                candidates = [token]
                if token.endswith('s'):
                    candidates.append(token[:-1])
                match = difflib.get_close_matches(candidates[0], lowered, n=1, cutoff=0.82)
                if not match and len(candidates) > 1:
                    match = difflib.get_close_matches(candidates[1], lowered, n=1, cutoff=0.82)
                if match:
                    detected.append(col)
                    break
        unique = []
        seen = set()
        for col in detected:
            if col not in seen:
                unique.append(col)
                seen.add(col)
        return unique

    def _detect_unit_instruction(self, query_lower: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        unit_map = {
            "crore": (10000000, "crores"),
            "crores": (10000000, "crores"),
            "lakh": (100000, "lakhs"),
            "lakhs": (100000, "lakhs"),
            "million": (1000000, "millions"),
            "millions": (1000000, "millions"),
            "billion": (1000000000, "billions"),
            "billions": (1000000000, "billions"),
        }
        for keyword, (factor, label) in unit_map.items():
            if keyword in query_lower:
                instruction = (
                    f"If a metric is requested in {label}, divide monetary aggregates by {factor} and alias the result columns with '_in_{label}'."
                )
                return instruction, factor, label
        return None, None, None

    def _sanitize_sql_columns(
        self,
        sql: str,
        allowed_aliases: List[str],
        schema_info: Dict[str, Any],
    ) -> str:
        """
        Ensure generated SQL references only real columns from loaded tables.
        Rewrites hallucinated column names (e.g., loan_id -> loan_account_id).
        """
        if not sql:
            return sql

        # Collect allowed columns from current aliases
        allowed_columns: Dict[str, str] = {}
        aliases = allowed_aliases or list(self.table_columns_map.keys())
        for alias in aliases:
            table_meta = self.table_columns_map.get(alias)
            if not table_meta:
                continue
            for col in table_meta.get("columns", []):
                allowed_columns.setdefault(col.lower(), col)

        # Include explicitly selected columns from schema info
        for col in schema_info.get("columns", []) or []:
            allowed_columns[col.lower()] = col

        if not allowed_columns:
            return sql

        # Map query tokens to preferred columns (from detection step)
        token_map: Dict[str, str] = {}
        for detection in schema_info.get("column_detections", []) or []:
            token_raw = detection.get("query_token") or ""
            column_name = detection.get("column_name")
            if not column_name:
                continue
            normalized_token = token_raw.strip().lower().replace(" ", "_")
            if normalized_token:
                token_map.setdefault(normalized_token, column_name)
            token_map.setdefault(column_name.lower(), column_name)

        # Helper to choose replacement
        def choose_column(token: str) -> str:
            lower = token.lower()
            # Priority 1: Exact match
            if lower in allowed_columns:
                return allowed_columns[lower]
            # Priority 2: Normalized match (spaces to underscores)
            norm = lower.replace(" ", "_")
            if norm in allowed_columns:
                return allowed_columns[norm]
            # Priority 3: Token map from detected columns
            if lower in token_map and token_map[lower].lower() in allowed_columns:
                return allowed_columns[token_map[lower].lower()]
            if norm in token_map and token_map[norm].lower() in allowed_columns:
                return allowed_columns[token_map[norm].lower()]
            if lower in token_map:
                return token_map[lower]
            if norm in token_map:
                return token_map[norm]
            # Priority 4: High-confidence fuzzy match (cutoff=0.75, stricter)
            matches = difflib.get_close_matches(lower, list(allowed_columns.keys()), n=1, cutoff=0.75)
            if matches:
                logger.warning(f"Column '{token}' fuzzy-matched to '{allowed_columns[matches[0]]}' (strict)")
                return allowed_columns[matches[0]]
            # Priority 5: Lower-confidence fuzzy match (cutoff=0.65)
            matches = difflib.get_close_matches(lower, list(allowed_columns.keys()), n=1, cutoff=0.65)
            if matches:
                logger.warning(f"Column '{token}' fuzzy-matched to '{allowed_columns[matches[0]]}' (medium)")
                return allowed_columns[matches[0]]
            # Priority 6: Sequence ratio fallback
            best_match = None
            best_score = 0.0
            for candidate in allowed_columns.keys():
                score = difflib.SequenceMatcher(None, lower, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_match = candidate
            if best_match and best_score >= 0.55:
                logger.warning(f"Column '{token}' sequence-matched to '{allowed_columns[best_match]}' (score={best_score:.2f})")
                return allowed_columns[best_match]
            # No match found - raise explicit error so correction loop can handle it
            available_list = list(allowed_columns.values())
            preview = ", ".join(available_list[:10]) if available_list else "(no columns)"
            message = (
                f"Column '{token}' is not available in the current schema. "
                f"Choose from: {preview}."
            )
            logger.error(message)
            raise ValueError(message)

        sql = re.sub(
            r"(\.)([A-Za-z_][\w]*)",
            lambda m: m.group(1) + choose_column(m.group(2)),
            sql,
        )

        def standalone_replacer(match):
            token = match.group(0)
            lower = token.lower()
            if lower in self.SQL_KEYWORDS:
                return token
            if lower in allowed_columns or lower in token_map:
                return choose_column(token)
            return token

        sql = re.sub(r"\b([A-Za-z_][\w]*)\b", standalone_replacer, sql)
        return sql

    def _coerce_numeric_operations(self, sql: str) -> str:
        """Wrap subtraction expressions so both sides are cast to DOUBLE."""
        if not sql:
            return sql

        def replacer(match):
            left = match.group(1)
            right = match.group(2)
            return f"(TRY_CAST({left} AS DOUBLE) - TRY_CAST({right} AS DOUBLE))"

        pattern = re.compile(
            r"(?<!CASE\s)([A-Za-z_][\w\.]*)\s*-\s*([A-Za-z_][\w\.]*)",
            flags=re.IGNORECASE,
        )
        return pattern.sub(replacer, sql)

    def _sanitize_sql_tables(
        self,
        sql: str,
        allowed_aliases: List[str],
        schema_info: Dict[str, Any],
    ) -> str:
        """
        Remove JOIN clauses that reference tables not currently loaded.
        If FROM references an unavailable table, replace it with the first allowed table.
        """
        if not sql:
            return sql

        allowed_aliases_set = set(allowed_aliases or [])
        if not allowed_aliases_set:
            allowed_aliases_set = set(self.table_columns_map.keys())

        allowed_table_names = {
            meta["name"]
            for alias, meta in self.table_columns_map.items()
            if alias in allowed_aliases_set or not allowed_aliases
        }
        allowed_table_names = {name.lower() for name in allowed_table_names if name}

        # Sanitize FROM clause
        def from_replacer(match):
            table_name = match.group(1)
            lower = table_name.lower()
            if lower in allowed_table_names:
                return match.group(0)
            if allowed_table_names:
                replacement = next(iter(allowed_table_names))
                return f"FROM {replacement}"
            return match.group(0)

        sql = re.sub(r"\bFROM\s+([A-Za-z_][\w]*)", from_replacer, sql, flags=re.IGNORECASE)

        # Remove disallowed joins
        join_pattern = re.compile(
            r"\bJOIN\s+([A-Za-z_][\w]*)\s+(?:AS\s+)?[A-Za-z_][\w]*\s+ON\b"
            r"[^;]*?(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def join_replacer(match):
            table_name = match.group(1)
            if table_name.lower() in allowed_table_names:
                return match.group(0)
            logger.info("Removing join on unavailable table %s", table_name)
            return ""

        sql = join_pattern.sub(join_replacer, sql)
        return sql

    def _build_prompt(
        self,
        user_query: str,
        metadata_context: List[Dict],
        correction_notes: Optional[str] = None,
        unit_instruction: Optional[str] = None,
        value_instruction: Optional[str] = None,
        allowed_tables: Optional[List[str]] = None,
        pivot_instruction: Optional[str] = None,
        fallback_instruction: Optional[str] = None,
        column_corrections: Optional[List[Dict[str, str]]] = None,
        columns_context: Optional[List[str]] = None,
    ) -> str:
        """Build prompt with user query and metadata context"""
        # Clean the query to remove schema info for the actual question
        clean_query = user_query
        # Remove schema information from query to avoid confusion
        clean_query = re.sub(r'given\s+columns?\s+are\s+[^.\n]+(?:and\s+table\s+is\s+[a-zA-Z_][a-zA-Z0-9_]*)?', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'and\s+table\s+is\s+[a-zA-Z_][a-zA-Z0-9_]*', '', clean_query, flags=re.IGNORECASE)
        clean_query = clean_query.strip()
        if not clean_query:
            clean_query = user_query  # Fallback if all removed
        
        context_text = "\n\n".join([result["document"] for result in metadata_context])
        
        refined_summary = self._refine_user_query(user_query)
        analysis_tables = allowed_tables or []
        analysis = self._analyze_request(refined_summary, analysis_tables, columns_context)
        
        # Build exhaustive column list from allowed tables
        available_columns_by_table = []
        tables_to_list = allowed_tables or list(self.table_columns_map.keys())
        for table_name in tables_to_list:
            table_meta = self.table_columns_map.get(table_name)
            if table_meta:
                cols = table_meta.get("columns", [])
                available_columns_by_table.append(f"  {table_name}: {', '.join(cols)}")
        
        columns_list = "\n".join(available_columns_by_table) if available_columns_by_table else "No columns available"
        
        instruction_lines = [
            "- CRITICAL: Use ONLY columns listed below under 'AVAILABLE COLUMNS'. DO NOT invent, guess, or use similar column names.",
            "- If a concept is not in the available columns, you CANNOT include it in the SQL.",
            "- When joining another table, aggregate it first in a CTE grouped by the join keys, then join the summary so each base row appears once.",
            "- Do NOT use functions like TO_DATE or TO_CHAR; instead use TRY_CAST(column AS DATE) or CAST(column AS VARCHAR).",
            "- When the request asks for totals, averages, yearly or grouped results, include the appropriate aggregate functions and GROUP BY clauses.",
            "- Only add filters or date conditions if they are explicitly mentioned in the question. Do not infer additional constraints.",
            "- When returning identifiers, names, or list-style answers without aggregates, use DISTINCT so each item appears once unless duplicates are explicitly requested.",
            "- Do not introduce new tables or JOIN clauses unless the user explicitly references another table. If only one table appears in the schema, any JOIN is invalid.",
            "- When the user asks for a pivot or for column headers by a category (e.g., secured vs unsecured), implement it with conditional aggregation (SUM(CASE WHEN ...)) on the same table.",
            "- Assume the columns already have suitable data types (e.g., numeric amounts, date fields). Use simple expressions and avoid verbose conversions such as to_number or to_date unless explicitly required.",
            "- Return only the final SQL query without explanations.",
            f"\n### AVAILABLE COLUMNS (use ONLY these):\n{columns_list}\n"
        ]
        if allowed_tables:
            unique_tables = list(dict.fromkeys(allowed_tables))
            if len(unique_tables) == 1:
                table_name = unique_tables[0]
                instruction_lines.append(
                    f"- Restrict the SQL to the table `{table_name}`. Do not reference or JOIN any other tables in the query."
                )
                instruction_lines.append(
                    "- Avoid writing JOIN clauses; use a single-table query based solely on the provided schema."
                )
            else:
                formatted = ", ".join(f"`{t}`" for t in unique_tables)
                instruction_lines.append(
                    f"- Only use these tables: {formatted}. Do not introduce or JOIN any other tables besides this list."
                )
        if pivot_instruction:
            instruction_lines.append(f"- {pivot_instruction}")
        if fallback_instruction:
            instruction_lines.append(f"- {fallback_instruction}")
        if correction_notes:
            instruction_lines.append(
                f"- Apply the following schema normalizations before writing SQL: {correction_notes}."
            )
        if unit_instruction:
            instruction_lines.append(f"- {unit_instruction}")
        if value_instruction:
            instruction_lines.append(f"- {value_instruction}")
        if column_corrections:
            corrections_text = "; ".join(
                f"{corr['requested']} -> {corr['table_alias']}.{corr['normalized']}"
                for corr in column_corrections
            )
            instruction_lines.append(
                f"- Normalize column references using these mappings: {corrections_text}."
            )
        if analysis.get("operations"):
            ops_text = ", ".join(analysis["operations"])
            instruction_lines.append(f"- Ensure the SQL applies these operations: {ops_text}.")

        instructions = "Follow these rules when writing SQL:\n" + "\n".join(instruction_lines)

        analysis_lines = [
            "### Analysis",
            f"Task: {analysis['task']}",
        ]
        if analysis["tables"]:
            analysis_lines.append("Tables: " + ", ".join(analysis["tables"]))
        if analysis["columns"]:
            analysis_lines.append("Columns: " + ", ".join(analysis["columns"]))
        if analysis["operations"]:
            analysis_lines.append("Operations: " + ", ".join(analysis["operations"]))
        analysis_lines.append(f"Expected result: {analysis['result_format']}")
        analysis_block = "\n".join(analysis_lines)

        # Build explicit column list
        column_reminder = []
        for result in metadata_context:
            doc = result.get('document', '')
            if 'Table: ' in doc and 'Columns: ' in doc:
                table_line = doc.split('\n')[0]  # Get "Table: X"
                columns_line = doc.split('Columns: ')[1] if 'Columns: ' in doc else ''
                # Extract just column names (before the data type)
                if columns_line:
                    column_reminder.append(f"{table_line}:")
                    parts = columns_line.split(', ')
                    for part in parts[:10]:  # Limit to avoid huge lists
                        col_name = part.split(' (')[0].strip() if ' (' in part else part.split('[')[0].strip()
                        if col_name:
                            column_reminder.append(f"  • {col_name}")
        
        columns_reminder_text = '\n'.join(column_reminder) if column_reminder else '(see schema above)'
        
        if "sqlcoder" in self.config.llm_model.lower():
            prompt = f"""### Task
Generate a SQL query to answer the following question: {clean_query}

### Database Schema
{context_text}

### EXACT Column Names (use these EXACTLY as written)
{columns_reminder_text}

{analysis_block}

### Instructions
{instructions}

### SQL Query
"""
        else:
            prompt = f"""You are a SQL expert. Generate a SQL query to answer the following question.

Question: {user_query}

Database Schema:
{context_text}

+{analysis_block}

Instructions:
{instructions}

SQL Query:
"""
        return prompt
    
    def _build_multi_table_prompt(
        self,
        user_query: str,
        query_intent: QueryIntent,
        metadata_context: List[Dict],
        unit_instruction: Optional[str] = None,
        value_instruction: Optional[str] = None,
    ) -> str:
        """Build prompt for multi-table query with explicit join instructions"""
        
        context_text = "\n\n".join([result["document"] for result in metadata_context])
        
        # Build join instructions
        join_instructions = []
        if query_intent.required_joins:
            join_descriptions = []
            for join in query_intent.required_joins:
                src_alias = join.get('source_alias')
                src_col = join.get('source_column')
                tgt_alias = join.get('target_alias')
                tgt_col = join.get('target_column')
                join_type = join.get('join_type', 'inner').upper()
                
                join_desc = f"{join_type} JOIN {tgt_alias} ON {src_alias}.{src_col} = {tgt_alias}.{tgt_col}"
                join_descriptions.append(join_desc)
            
            join_instructions.append(
                f"- Use these joins to connect tables: {'; '.join(join_descriptions)}"
            )
        
        # Build column selection hints
        column_hints = []
        for match in query_intent.requested_columns:
            column_hints.append(f"{match.table_alias}.{match.column_name}")
        
        if column_hints:
            join_instructions.append(
                f"- Include these columns in the query: {', '.join(column_hints[:10])}"
            )
        
        # Build operation hints
        if query_intent.aggregations:
            agg_desc = ', '.join(query_intent.aggregations)
            join_instructions.append(f"- Apply these aggregations: {agg_desc}")
        
        if query_intent.grouping:
            join_instructions.append("- Use GROUP BY for aggregated results")
        
        if query_intent.ordering:
            join_instructions.append(f"- Order results {query_intent.ordering}")
        
        # Build exhaustive column list
        available_columns_by_table = []
        for alias in query_intent.required_tables:
            table_meta = self.table_columns_map.get(alias)
            if table_meta:
                cols = table_meta.get("columns", [])
                available_columns_by_table.append(f"  {alias}: {', '.join(cols)}")
        
        columns_list = "\n".join(available_columns_by_table) if available_columns_by_table else "No columns available"
        
        instruction_lines = [
            "- CRITICAL: Use ONLY columns listed below. DO NOT invent, guess, or use similar column names.",
            "- If a concept is not in the available columns, you CANNOT include it in the SQL.",
            "- When joining another table, aggregate it first in a CTE grouped by the join keys, then join the summary so each base row appears once.",
            "- Do NOT use functions like TO_DATE or TO_CHAR; instead use TRY_CAST(column AS DATE) or CAST(column AS VARCHAR).",
            "- Follow the join specifications exactly as provided.",
            "- When aggregating, ensure all non-aggregated columns are in GROUP BY.",
            "- Return only the final SQL query without explanations.",
        ]
        
        instruction_lines.append(f"\n### AVAILABLE COLUMNS (use ONLY these):\n{columns_list}\n")
        instruction_lines.extend(join_instructions)
        
        if unit_instruction:
            instruction_lines.append(f"- {unit_instruction}")
        if value_instruction:
            instruction_lines.append(f"- {value_instruction}")
        
        instructions = "Follow these rules when writing SQL:\n" + "\n".join(instruction_lines)
        
        # Build analysis
        analysis_lines = [
            "### Analysis",
            f"Task: {query_intent.original_query}",
            f"Tables needed: {', '.join(query_intent.required_tables)}",
            f"Operations: {', '.join(query_intent.operations)}",
        ]
        
        if query_intent.requested_columns:
            col_summary = [f"{m.table_alias}.{m.column_name}" for m in query_intent.requested_columns[:5]]
            analysis_lines.append(f"Key columns: {', '.join(col_summary)}")
        
        analysis_block = "\n".join(analysis_lines)
        
        if "sqlcoder" in self.config.llm_model.lower():
            prompt = f"""### Task
Generate a SQL query to answer the following question: {user_query}

### Database Schema
{context_text}

{analysis_block}

### Instructions
{instructions}

### SQL Query
"""
        else:
            prompt = f"""You are a SQL expert. Generate a SQL query to answer the following question.

Question: {user_query}

Database Schema:
{context_text}

{analysis_block}

Instructions:
{instructions}

SQL Query:
"""
        return prompt
    
    def _correct_sql(
        self,
        sql: str,
        error_message: str,
        user_query: str,
        metadata_context: List[Dict],
        allowed_tables: Optional[List[str]] = None,
    ) -> str:
        """Correct SQL using error message"""
        schema_context = "\n\n".join([result['document'] for result in metadata_context])
        
        table_instruction = ""
        if allowed_tables:
            unique_tables = list(dict.fromkeys(allowed_tables))
            if len(unique_tables) == 1:
                table_instruction = (
                    f"\n\nUse only the table `{unique_tables[0]}` when fixing the SQL. Do not add JOIN clauses, aliases from other tables, or references to tables beyond `{unique_tables[0]}`."
                )
            else:
                formatted = ", ".join(f"`{t}`" for t in unique_tables)
                table_instruction = (
                    f"\n\nRestrict the corrected SQL to these tables only: {formatted}. Do not introduce additional tables or JOINs."
                )

        correction_prompt = f"""The following SQL query has an error. Please correct it.

Original Question: {user_query}

Database Schema:
{schema_context}

Original SQL Query:
{sql}

Error Message:
{error_message}

Corrected SQL Query:
{table_instruction}
"""
        
        corrected_sql = self.llm.generate_sql(correction_prompt)
        return corrected_sql
    
    def query(self, user_query: str, skip_execution: bool = False, sandbox: bool = False) -> Dict[str, Any]:
        """
        Main query method with RAG and self-correction.
        Supports dynamic schema extraction from query.
        
        Args:
            user_query: Natural language query (can include schema info)
            skip_execution: If True, skip SQL execution (for queries with external schemas)
        """
        logger.info(f"Processing query: {user_query}")
 
        alias_lookup_global = {meta["name"]: alias for alias, meta in self.table_columns_map.items()}
        value_instruction: Optional[str] = None
        base_query_line = next((line.strip() for line in user_query.splitlines() if line.strip()), user_query.strip())
        unit_instruction, unit_factor, unit_label = self._detect_unit_instruction(base_query_line.lower())
        metadata_context_docs: List[Dict[str, Any]] = []

        # Step 1: Check if schema is provided in query (PRIORITY - use this if found)
        query_schema = self._extract_schema_from_query(base_query_line)
        use_query_schema = query_schema is not None
        
        if use_query_schema:
            schema_info_used = {
                "source": "query",
                "table": query_schema.get("table_name"),
                "columns": query_schema.get("columns", []),
                "corrections": query_schema.get("corrections"),
                "column_types_map": query_schema.get("column_types_map"),
                "column_values_map": query_schema.get("column_values_map"),
                "value_matches": query_schema.get("value_matches", []),
                "value_corrections": query_schema.get("value_corrections", []),
                "column_corrections": query_schema.get("column_corrections", []),
            }
            value_instruction = query_schema.get("value_instruction")
            allowed_tables = [schema_info_used.get("table")] if schema_info_used.get("table") else []
            allowed_aliases = []
            table_name = query_schema.get("table_name")
            if table_name:
                alias = alias_lookup_global.get(table_name)
                if alias:
                    allowed_aliases.append(alias)
        else:
            allowed_aliases = []
            schema_info_used = {
                "source": "database",
                "tables": [],
                "context": [],
                "corrections": None,
                "column_types_map": None,
                "column_values_map": None,
                "columns": [],
                "value_matches": [],
                "value_corrections": [],
                "column_corrections": [],
            }
            user_lower = base_query_line.lower()
            detected_aliases = []
            for alias, meta in self.table_columns_map.items():
                table_name = meta["name"]
                if alias.lower() in user_lower or table_name.lower() in user_lower:
                    detected_aliases.append(alias)

            allowed_alias_set = set(detected_aliases)
            relationships = self.relationships or []
            expanded = True
            while expanded:
                expanded = False
                for rel in relationships:
                    src = rel.get("source_alias")
                    tgt = rel.get("target_alias")
                    if not src or not tgt:
                        continue
                    if src in allowed_alias_set and tgt not in allowed_alias_set:
                        allowed_alias_set.add(tgt)
                        expanded = True
                    elif tgt in allowed_alias_set and src not in allowed_alias_set:
                        allowed_alias_set.add(src)

            ordered_aliases: List[str] = []
            for alias in detected_aliases:
                if alias in allowed_alias_set and alias not in ordered_aliases:
                    ordered_aliases.append(alias)
            for alias in allowed_alias_set:
                if alias not in ordered_aliases:
                    ordered_aliases.append(alias)
            allowed_aliases = ordered_aliases
            allowed_tables = []
            for alias in allowed_aliases:
                meta = self.table_columns_map.get(alias)
                if meta:
                    allowed_tables.append(meta["name"])
            schema_info_used["tables"] = allowed_tables

        # Ensure we only keep aliases that actually exist right now
        valid_aliases = set(self.table_columns_map.keys())
        allowed_aliases = [alias for alias in (allowed_aliases or []) if alias in valid_aliases]
        # Recompute allowed tables to match filtered aliases
        if allowed_aliases:
            allowed_tables = []
            for alias in allowed_aliases:
                table_meta = self.table_columns_map.get(alias)
                if table_meta:
                    allowed_tables.append(table_meta["name"])
            schema_info_used["tables"] = allowed_tables
 
        # Build context documents explicitly from allowed tables
        context_documents: List[str] = []
        final_tables = allowed_tables or schema_info_used.get("tables", [])
        seen_tables = []
        for table_name in final_tables:
            if not table_name or table_name in seen_tables:
                continue
            seen_tables.append(table_name)
            doc = self._build_table_document(table_name)
            if doc:
                context_documents.append(doc)
                metadata_context_docs.append({"document": doc, "metadata": {"table_name": table_name}})
        if context_documents:
            schema_info_used["context"] = context_documents
            schema_info_used.setdefault("tables", seen_tables)

        schema_info_used["value_instruction"] = schema_info_used.get("value_instruction") or value_instruction

        correction_notes = None
        corrections = schema_info_used.get("corrections", {})
        if corrections:
            table_fix = corrections.get("table")
            col_fixes = corrections.get("columns")
            notes = []
            if table_fix and table_fix.get("original") and table_fix.get("normalized"):
                notes.append(f"table '{table_fix['original']}' -> '{table_fix['normalized']}'")
            if col_fixes:
                for fix in col_fixes:
                    original = fix.get("original")
                    normalized = fix.get("normalized")
                    if original and normalized and original.lower() != normalized.lower():
                        notes.append(f"column '{original}' -> '{normalized}'")
            if notes:
                correction_notes = ", ".join(notes)

        if not use_query_schema:
            detected_columns = self._detect_columns_from_query(user_query, schema_info_used.get("tables", []))
            if detected_columns:
                schema_info_used["columns"] = detected_columns
                # Record detections for UI/alternatives
                table_alias_primary = None
                tables_list = schema_info_used.get("tables", [])
                if tables_list:
                    table_alias_primary = alias_lookup_global.get(tables_list[0])
                detections = schema_info_used.get("column_detections", [])
                for col in detected_columns:
                    detections.append(
                        {
                            "table_name": tables_list[0] if tables_list else None,
                            "table_alias": table_alias_primary,
                            "column_name": col,
                            "query_token": col.replace("_", " "),
                            "confidence": 0.9,
                            "match_type": "detected",
                        }
                    )
                schema_info_used["column_detections"] = detections
            
            # Get primary table if available
            tables_list = schema_info_used.get("tables", [])
            table_name_primary = tables_list[0] if tables_list else None
            table_alias_primary = None
            if table_name_primary:
                table_alias_primary = alias_lookup_global.get(table_name_primary)
            primary_entry = self.table_columns_map.get(table_alias_primary) if table_alias_primary else None
            if primary_entry:
                schema_info_used["column_types_map"] = primary_entry.get("column_types")
                matches, value_corrections, val_instruction = self._detect_value_context(
                    user_query,
                    primary_entry["name"],
                    schema_info_used.get("columns", []) or primary_entry.get("columns", []),
                    primary_entry.get("column_values", {}),
                )
                if matches:
                    for match in matches:
                        col = match.get("column")
                        if col and col not in schema_info_used["columns"]:
                            schema_info_used.setdefault("columns", []).append(col)
                schema_info_used["value_matches"] = matches
                schema_info_used["value_corrections"] = value_corrections
                if val_instruction:
                    schema_info_used["value_instruction"] = val_instruction

        allowed_aliases_list = allowed_aliases if isinstance(allowed_aliases, list) else list(allowed_aliases) if allowed_aliases else []
        column_corrections = self._suggest_column_corrections(user_query, allowed_aliases_list)
        if column_corrections:
            schema_info_used["column_corrections"] = column_corrections
            logger.info("Column corrections suggested: %s", column_corrections)
        else:
            schema_info_used.setdefault("column_corrections", [])

        normalized_query = base_query_line.strip().lower()
        if self._is_table_inventory_query(normalized_query):
            all_tables = sorted({meta.get("name") for meta in self.table_columns_map.values() if meta.get("name")})
            schema_snapshot = {
                "source": "database",
                "tables": all_tables,
                "columns": [],
                "context": [],
            }
            status = "No datasets loaded." if not all_tables else "Available tables: " + ", ".join(all_tables)
            return {
                "success": True,
                "sql": "-- No SQL generated (inventory request)",
                "schema_info": schema_snapshot,
                "status_message": status,
                "data_columns": None,
                "data_rows": None,
                "row_count": None,
                "sandbox_validated": False,
                "attempts": 0,
            }

        # Step 2: Build prompt with context
        fallback_instruction = None
        pivot_instruction = None
        if allowed_tables and len(allowed_tables) == 1:
            fallback_instruction = (
                f"If the model attempts to JOIN, immediately rewrite the SQL using only `{allowed_tables[0]}`."
            )
            pivot_keywords = [
                r"columns?\s+split\s+by",
                r"as\s+columns",
                r"column\s+headers",
                r"pivot",
            ]
            if any(re.search(pattern, user_query, flags=re.IGNORECASE) for pattern in pivot_keywords):
                pivot_instruction = (
                    "Use conditional aggregation within table `{}`: compute SUM(CASE WHEN <pivot_column> = 'Value' THEN <measure> ELSE 0 END) for each distinct pivot value, group by the row dimension, and avoid JOIN clauses.".format(allowed_tables[0])
                )

        normalized_query = base_query_line.strip().lower()
        allowed_aliases_list = allowed_aliases if isinstance(allowed_aliases, list) else list(allowed_aliases) if allowed_aliases else []

        def resolve_primary_table() -> Tuple[Optional[str], Optional[str]]:
            alias = None
            table = None
            if allowed_aliases_list:
                alias = allowed_aliases_list[0]
            elif allowed_tables:
                alias = alias_lookup_global.get(allowed_tables[0])
            if alias:
                table_entry = self.table_columns_map.get(alias)
                if table_entry:
                    table = table_entry.get("name")
            elif allowed_tables:
                table = allowed_tables[0]
                alias = alias_lookup_global.get(table)
            return alias, table

        alias_for_columns, table_for_columns = resolve_primary_table()
        column_request_keywords = ["column", "columns", "fields", "schema"]
        wants_columns_only = any(keyword in normalized_query for keyword in column_request_keywords)

        if wants_columns_only and alias_for_columns:
            table_entry = self.table_columns_map.get(alias_for_columns)
            if table_entry:
                column_list = table_entry.get("columns", [])
                schema_info_used["columns"] = column_list
                schema_info_used["tables"] = [table_entry.get("name") or table_for_columns or alias_for_columns]
                doc = None
                if table_entry.get("name"):
                    doc = self._build_table_document(table_entry.get("name"))
                if doc:
                    schema_info_used["context"] = [doc]
                rows = [{"column_name": col} for col in column_list]
                return {
                    "success": True,
                    "sql": f"-- Columns for table {table_entry.get('name') or table_for_columns or alias_for_columns}",
                    "status_message": "Returned schema columns from metadata.",
                    "attempts": 0,
                    "metadata_used": schema_info_used.get("tables", []),
                    "schema_info": schema_info_used,
                    "column_detections": schema_info_used.get("column_detections", []),
                    "column_alternatives": schema_info_used.get("column_alternatives", {}),
                    "sandbox_validated": False,
                    "data_columns": ["column_name"],
                    "data_rows": rows,
                    "row_count": len(rows),
                }

        allowed_tables_for_prompt = allowed_tables or schema_info_used.get("tables", [])

        # Determine if this is a multi-table query
        is_multi_table = len(allowed_tables_for_prompt) > 1
        query_intent = None
        numeric_lit_normalizations = []

        if is_multi_table and self.relationships:
            # Use multi-table query parser
            try:
                query_intent = parse_multi_table_query(
                    user_query,
                    self.table_columns_map,
                    self.relationships
                )
                
                # Update schema_info with parsed intent
                if query_intent.requested_columns:
                    # Add any columns found by the parser
                    for match in query_intent.requested_columns:
                        if match.column_name not in schema_info_used.get("columns", []):
                            schema_info_used.setdefault("columns", []).append(match.column_name)
                
                logger.info(f"Multi-table query detected. Tables: {query_intent.required_tables}, Joins: {len(query_intent.required_joins)}")
                
                # Store column detections for response
                schema_info_used["column_detections"] = [
                    {
                        "table_name": match.table_name,
                        "table_alias": match.table_alias,
                        "column_name": match.column_name,
                        "query_token": match.query_token,
                        "confidence": match.confidence,
                        "match_type": match.match_type
                    }
                    for match in query_intent.requested_columns
                ]
                
            except Exception as e:
                logger.warning(f"Could not parse multi-table query: {e}. Falling back to standard prompt.")
                is_multi_table = False
        
        # Find alternative columns (for both single and multi table cases)
        try:
            alternatives = group_alternatives_by_query_term(
                user_query,
                self.table_columns_map,
                schema_info_used.get("column_detections", []),
                top_k=3
            )
            if alternatives:
                schema_info_used["column_alternatives"] = alternatives
                logger.info(f"Found alternatives for: {list(alternatives.keys())}")
        except Exception as alt_error:
            logger.warning(f"Could not find column alternatives: {alt_error}")
        
        # Build appropriate prompt
        if is_multi_table and query_intent and query_intent.required_joins:
            prompt = self._build_multi_table_prompt(
                user_query,
                query_intent,
                metadata_context_docs,
                unit_instruction,
                value_instruction,
            )
        else:
            prompt = self._build_prompt(
                user_query,
                metadata_context_docs,
                correction_notes,
                unit_instruction,
                value_instruction,
                allowed_tables=allowed_tables_for_prompt,
                pivot_instruction=pivot_instruction,
                fallback_instruction=fallback_instruction,
                column_corrections=schema_info_used.get("column_corrections"),
                columns_context=schema_info_used.get("columns", []),
            )
        
        # Step 3: Generate SQL with self-correction loop
        sql = None
        attempt = 0
        last_error = None
        
        while attempt < self.config.max_correction_attempts:
            attempt += 1
            logger.info(f"Attempt {attempt}/{self.config.max_correction_attempts}")
            
            try:
                if attempt == 1:
                    sql = self.llm.generate_sql(prompt)
                else:
                    sql = self._correct_sql(sql, last_error, user_query, metadata_context_docs, allowed_tables)
                
                logger.info(f"Generated SQL: {sql}")

                try:
                    sql = self._rewrite_unsupported_functions(sql)
                    sql = self._rewrite_joined_aggregates(sql)
                    sql = self._sanitize_sql_tables(
                        sql,
                        allowed_aliases_list,
                        schema_info_used,
                    )

                    sql = self._sanitize_sql_columns(
                        sql,
                        allowed_aliases_list,
                        schema_info_used,
                    )

                    sql = self._normalize_sql_literals(sql, schema_info_used)
                    sql = self._coerce_numeric_comparisons(sql, schema_info_used)
                    sql = self._coerce_numeric_operations(sql)
                except ValueError as sanitize_error:
                    last_error = str(sanitize_error)
                    logger.warning(f"Sanitization error: {sanitize_error}")
                    continue

                try:
                    sql = self._cast_numeric_aggregates(sql)
                    self._validate_sql_columns(sql, allowed_tables)
                except ValueError as validation_error:
                    last_error = str(validation_error)
                    logger.warning(f"Schema validation error: {validation_error}")
                    continue
                
                # Guardrail: prevent JOINs when only a single table is allowed
                if allowed_tables and len(allowed_tables) == 1:
                    if re.search(r"\bjoin\b", sql, flags=re.IGNORECASE):
                        rewritten_sql = self._rewrite_single_table_sql(
                            sql,
                            allowed_tables[0],
                            schema_info_used,
                            unit_factor,
                            unit_label,
                        )
                        if rewritten_sql:
                            logger.warning("Detected disallowed JOIN; rewriting SQL to single-table form.")
                            sql = rewritten_sql
                        else:
                            fallback_sql = self._build_single_table_pivot_sql(
                                allowed_tables[0],
                                schema_info_used,
                                user_query,
                                unit_factor,
                                unit_label,
                            )
                            if fallback_sql:
                                logger.warning("Detected disallowed JOIN; using single-table pivot fallback.")
                                sql = fallback_sql
                            else:
                                last_error = (
                                    f"JOIN clause detected, but only table `{allowed_tables[0]}` is available. "
                                    "Rewrite the SQL using that table alone without any JOINs."
                                )
                                logger.warning("Detected disallowed JOIN; requesting regeneration without joins.")
                                continue
                
                sql = self._apply_value_corrections_to_sql(sql, schema_info_used)

                skip_due_to_schema = use_query_schema and not sandbox
                if skip_execution or skip_due_to_schema:
                    logger.info("Skipping SQL execution (either requested skip or schema provided without sandbox)")
                    message = "SQL generated successfully. (Execution skipped - schema provided in query)"
                    if sandbox:
                        message = "SQL generated, but sandbox validation was skipped (table not available)."
                    return {
                        "success": True,
                        "sql": sql,
                        "status_message": message,
                        "attempts": attempt,
                        "metadata_used": [r.get("metadata", {}).get("table_name", "unknown") for r in metadata_context_docs if r.get("metadata")],
                        "schema_info": schema_info_used,
                        "column_detections": schema_info_used.get("column_detections", []),
                        "column_alternatives": schema_info_used.get("column_alternatives", {}),
                        "sandbox_validated": False,
                        "data_columns": None,
                        "data_rows": None,
                        "row_count": None
                    }

                try:
                    is_valid, error = self.sql_executor.validate_syntax(sql, use_duckdb=sandbox)
                except Exception as e:
                    logger.warning(f"Could not validate syntax: {e}. Skipping execution.")
                    return {
                        "success": True,
                        "sql": sql,
                        "status_message": "SQL generated successfully. (Execution skipped - table may not exist in database)",
                        "attempts": attempt,
                        "metadata_used": [r.get("metadata", {}).get("table_name", "unknown") for r in metadata_context_docs if r.get("metadata")],
                        "schema_info": schema_info_used,
                        "column_detections": schema_info_used.get("column_detections", []),
                        "column_alternatives": schema_info_used.get("column_alternatives", {}),
                        "sandbox_validated": False,
                        "data_columns": None,
                        "data_rows": None,
                        "row_count": None
                    }
                
                if is_valid:
                    try:
                        success, result, error = self.sql_executor.execute_query(sql, use_duckdb=sandbox)
                        
                        if success:
                            status_message = "SQL validated successfully against sandbox database." if sandbox else "SQL executed successfully."
                            data_columns = None
                            data_rows = None
                            row_count = None
                            if isinstance(result, pd.DataFrame):
                                data_columns = list(result.columns)
                                data_rows = result.to_dict(orient="records")
                                row_count = len(result)
                                if sandbox:
                                    status_message = f"SQL validated successfully against sandbox (rows: {row_count})"
                            elif isinstance(result, str):
                                status_message = result
                            return {
                                "success": True,
                                "sql": sql,
                                "status_message": status_message,
                                "attempts": attempt,
                                "metadata_used": [r.get("metadata", {}).get("table_name", "unknown") for r in metadata_context_docs if r.get("metadata")],
                                "schema_info": schema_info_used,
                                "column_detections": schema_info_used.get("column_detections", []),
                                "column_alternatives": schema_info_used.get("column_alternatives", {}),
                                "sandbox_validated": sandbox,
                                "data_columns": data_columns,
                                "data_rows": data_rows,
                                "row_count": row_count
                            }
                        else:
                            last_error = error
                            logger.warning(f"Execution error: {error}")
                    except Exception as e:
                        logger.warning(f"Could not execute query: {e}. Table may not exist.")
                        message = f"SQL generated successfully. (Execution skipped - {str(e)})"
                    return {
                        "success": True,
                        "sql": sql,
                        "status_message": message,
                        "attempts": attempt,
                            "metadata_used": [r.get("metadata", {}).get("table_name", "unknown") for r in metadata_context_docs if r.get("metadata")],
                            "schema_info": schema_info_used,
                            "column_detections": schema_info_used.get("column_detections", []),
                            "column_alternatives": schema_info_used.get("column_alternatives", {}),
                            "sandbox_validated": False,
                            "data_columns": None,
                            "data_rows": None,
                            "row_count": None
                        }
                else:
                    last_error = error
                    logger.warning(f"Syntax error: {error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error in attempt {attempt}: {e}")
        
        # All attempts failed
        return {
            "success": False,
            "sql": sql,
            "error": last_error,
            "attempts": attempt,
            "metadata_used": [r.get("metadata", {}).get("table_name", "unknown") for r in metadata_context_docs if r.get("metadata")],
            "schema_info": schema_info_used,
            "column_detections": schema_info_used.get("column_detections", []),
            "column_alternatives": schema_info_used.get("column_alternatives", {}),
            "sandbox_validated": False,
            "status_message": f"SQL generation failed after {attempt} attempts: {last_error}",
            "data_columns": None,
            "data_rows": None,
            "row_count": None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.metadata_extractor.close()
        self.sql_executor.close()

    def _collect_column_values(self, table_name: str, column_names: List[str]) -> Dict[str, List[str]]:
        column_values: Dict[str, List[str]] = {}
        limit = self.COLUMN_VALUE_LIMIT
        for index, col in enumerate(column_names):
            try:
                query = (
                    f'SELECT DISTINCT "{col}" AS value '
                    f'FROM "{table_name}" WHERE "{col}" IS NOT NULL '
                    f'LIMIT {limit + 1}'
                )
                success, result, _ = self.sql_executor.execute_query(query, use_duckdb=True)
                if success and isinstance(result, pd.DataFrame):
                    values = [str(v) for v in result["value"].dropna().tolist()]
                    if len(values) <= limit:
                        column_values[col] = values
            except Exception as exc:
                logger.debug("Value sampling failed for %s.%s: %s", table_name, col, exc)
                continue
        return column_values

    @staticmethod
    def _cast_numeric_aggregates(sql: str) -> str:
        pattern = re.compile(r"\b(SUM|AVG)\s*\(\s*(DISTINCT\s+)?([^()]+?)\s*\)", re.IGNORECASE)

        def replacer(match):
            func = match.group(1)
            distinct = match.group(2) or ""
            expr = match.group(3).strip()
            expr_upper = expr.upper()
            if any(keyword in expr_upper for keyword in ("TRY_CAST", "CAST", "CASE", "WHEN")):
                return match.group(0)
            simple_column = re.fullmatch(r"[A-Za-z_][\w]*", expr)
            dotted_column = re.fullmatch(r"[A-Za-z_][\w]*\.[A-Za-z_][\w]*", expr)
            if simple_column or dotted_column:
                return f"{func}({distinct}TRY_CAST({expr} AS DOUBLE))"
            return match.group(0)

        return pattern.sub(replacer, sql)

    @staticmethod
    def _analyze_request(refined_query: str, tables: Optional[List[str]], columns: Optional[List[str]]) -> Dict[str, Any]:
        tables = tables or []
        columns = columns or []
        unique_tables = [t for i, t in enumerate(tables) if t and t not in tables[:i]]
        unique_columns = [c for i, c in enumerate(columns) if c and c not in columns[:i]]

        lower = refined_query.lower()
        operations: List[str] = []
        def add_op(op):
            if op not in operations:
                operations.append(op)

        if any(keyword in lower for keyword in ["total", "sum", "overall"]):
            add_op("SUM")
        if any(keyword in lower for keyword in ["average", "avg", "mean"]):
            add_op("AVG")
        if any(keyword in lower for keyword in ["count", "how many", "number of"]):
            add_op("COUNT")
        if any(keyword in lower for keyword in ["max", "highest", "largest", "top"]):
            add_op("MAX")
        if any(keyword in lower for keyword in ["min", "lowest", "smallest"]):
            add_op("MIN")
        if any(keyword in lower for keyword in ["distinct", "unique"]):
            add_op("DISTINCT")
        if any(keyword in lower for keyword in [" per ", " by ", "group", "split "]):
            add_op("GROUP BY")
        if any(keyword in lower for keyword in ["top", "highest", "lowest", "ascending", "descending"]):
            add_op("ORDER BY")

        if "pivot" in lower or "column headers" in lower or "as columns" in lower:
            add_op("PIVOT/CASE")

        if "limit" in lower or "top" in lower:
            add_op("LIMIT")

        if not operations:
            add_op("SELECT")

        if "group by" in lower or "per " in lower or " by " in lower or "each" in lower:
            result_format = "Grouped result set"
        elif any(op in operations for op in ["SUM", "AVG", "COUNT", "MAX", "MIN"]):
            result_format = "Aggregated metrics"
        elif "DISTINCT" in operations:
            result_format = "Distinct list"
        else:
            result_format = "Detail rows"

        return {
            "task": refined_query.rstrip("."),
            "tables": unique_tables,
            "columns": unique_columns,
            "operations": operations,
            "result_format": result_format,
        }

    def _normalize_sql_literals(
        self,
        sql: str,
        schema_info: Dict[str, Any],
    ) -> str:
        """Snap string literals to the closest known column values."""
        if not sql:
            return sql

        # Build map of column -> known values
        column_values_map: Dict[str, Set[str]] = {}
        schema_map = schema_info.get("column_values_map") or {}
        for col, values in schema_map.items():
            if values:
                column_values_map.setdefault(col.lower(), set()).update(str(v) for v in values)

        literal_normalizations: List[Dict[str, str]] = schema_info.setdefault("literal_normalizations", [])

        for alias, meta in self.table_columns_map.items():
            for col, values in meta.get("column_values", {}).items():
                if values:
                    column_values_map.setdefault(col.lower(), set()).update(str(v) for v in values)

        if not column_values_map:
            return sql

        def match_literal(column: str, literal: str) -> Optional[str]:
            choices = column_values_map.get(column.lower())
            if not choices:
                return None
            literal_clean = literal.strip().lower()
            synonyms = {
                "yes": "y",
                "no": "n",
                "true": "y",
                "false": "n",
                "on": "y",
                "off": "n",
            }
            if literal_clean in synonyms:
                target = synonyms[literal_clean]
                for choice in choices:
                    if choice.lower() == target:
                        return choice
            for choice in choices:
                if choice.lower() == literal_clean:
                    return choice
            best = difflib.get_close_matches(
                literal_clean,
                [choice.lower() for choice in choices],
                n=1,
                cutoff=0.6,
            )
            if best:
                target = best[0]
                for choice in choices:
                    if choice.lower() == target:
                        return choice
            return None

        def replace_equality(match: re.Match) -> str:
            left = match.group(1)
            literal = match.group(2)
            column = left.split(".")[-1]
            normalized = match_literal(column, literal)
            if normalized is None or normalized == literal:
                return match.group(0)
            literal_normalizations.append({
                "column": column,
                "original": literal,
                "normalized": normalized,
            })
            return f"{left} = '{normalized}'"

        def replace_in(match: re.Match) -> str:
            left = match.group(1)
            values = match.group(2)
            column = left.split(".")[-1]
            parts = [p.strip() for p in values.split(",") if p.strip()]
            normalized_parts = []
            updated = False
            for part in parts:
                if len(part) >= 2 and part[0] == part[-1] == "'":
                    literal = part[1:-1]
                    normalized = match_literal(column, literal)
                    if normalized is not None and normalized != literal:
                        normalized_parts.append(f"'{normalized}'")
                        literal_normalizations.append({
                            "column": column,
                            "original": literal,
                            "normalized": normalized,
                        })
                        updated = True
                        continue
                normalized_parts.append(part)
            if not updated:
                return match.group(0)
            joined = ", ".join(normalized_parts)
            return f"{left} IN ({joined})"

        sql = re.sub(
            r"([A-Za-z_][\w\.]*?)\s*=\s*'([^']*)'",
            replace_equality,
            sql,
        )
        sql = re.sub(
            r"([A-Za-z_][\w\.]*?)\s+IN\s*\(([^)]*)\)",
            replace_in,
            sql,
            flags=re.IGNORECASE,
        )
        return sql

    def _coerce_numeric_comparisons(
        self,
        sql: str,
        schema_info: Dict[str, Any],
    ) -> str:
        """Force numeric comparisons to cast string columns to DOUBLE."""
        if not sql:
            return sql

        allowed_columns: Set[str] = set()
        for alias, meta in self.table_columns_map.items():
            for col in meta.get("columns", []):
                allowed_columns.add(col.lower())
        for col in schema_info.get("columns", []) or []:
            allowed_columns.add(col.lower())

        numeric_pattern = re.compile(
            r"([A-Za-z_][\w\.]*)\s*(>=|<=|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)"
        )

        quoted_numeric_pattern = re.compile(
            r"([A-Za-z_][\w\.]*)\s*(>=|<=|>|<|=)\s*'([0-9]+(?:\.[0-9]+)?)'"
        )

        def numeric_replacer(match: re.Match) -> str:
            left = match.group(1)
            op = match.group(2)
            literal = match.group(3)
            column = left.split(".")[-1].lower()
            if column not in allowed_columns:
                return match.group(0)
            upper = left.upper()
            if "TRY_CAST" in upper or "CAST" in upper:
                return match.group(0)
            return f"TRY_CAST({left} AS DOUBLE) {op} {literal}"

        sql = numeric_pattern.sub(numeric_replacer, sql)

        def quoted_numeric_replacer(match: re.Match) -> str:
            left = match.group(1)
            op = match.group(2)
            literal = match.group(3)
            column = left.split(".")[-1].lower()
            if column not in allowed_columns:
                return match.group(0)
            upper = left.upper()
            if "TRY_CAST" in upper or "CAST" in upper:
                return match.group(0)
            return f"TRY_CAST({left} AS DOUBLE) {op} {literal}"

        sql = quoted_numeric_pattern.sub(quoted_numeric_replacer, sql)

        # Handle date/time literals
        date_literal_pattern = re.compile(
            r"([A-Za-z_][\w\.]*)\s*(>=|<=|>|<|=)\s*'([^']+)'"
        )

        def date_replacer(match: re.Match) -> str:
            left = match.group(1)
            op = match.group(2)
            literal = match.group(3)
            column = left.split(".")[-1]
            column_type = self._lookup_column_type(column) or self._infer_type_from_name(column)
            if not column_type:
                return match.group(0)
            column_type_lower = column_type.lower()
            if "try_cast" in left.lower() or "cast" in left.lower():
                return match.group(0)
            if "timestamp" in column_type_lower or "datetime" in column_type_lower:
                target = "TIMESTAMP"
            elif "date" in column_type_lower:
                target = "DATE"
            else:
                return match.group(0)
            return (
                f"TRY_CAST({left} AS {target}) {op} TRY_CAST('{literal}' AS {target})"
            )

        sql = date_literal_pattern.sub(date_replacer, sql)

        coalesce_numeric_pattern = re.compile(
            r"COALESCE\s*\(([^)]+)\)\s*(>=|<=|>|<|=)\s*([0-9]+(?:\.[0-9]+)?)"
        )

        def coalesce_numeric_replacer(match: re.Match) -> str:
            expr = match.group(1)
            op = match.group(2)
            literal = match.group(3)
            return f"COALESCE({expr}) {op} {literal}"

        sql = numeric_pattern.sub(numeric_replacer, sql)
        sql = quoted_numeric_pattern.sub(quoted_numeric_replacer, sql)
        sql = coalesce_numeric_pattern.sub(coalesce_numeric_replacer, sql)

        return sql

    def _lookup_column_type(self, column_name: str) -> Optional[str]:
        """Return the stored data type for a column if known."""
        column_lower = column_name.lower()
        for meta in self.table_columns_map.values():
            column_types = meta.get("column_types", {})
            for name, dtype in column_types.items():
                if name.lower() == column_lower:
                    return dtype
        return None

    def _rewrite_unsupported_functions(self, sql: str) -> str:
        """Rewrite non-DuckDB functions (TO_DATE/TO_CHAR) into safe expressions."""
        if not sql:
            return sql

        def to_date_replacer(match: re.Match) -> str:
            expr = match.group(1).strip()
            return f"TRY_CAST({expr} AS DATE)"

        def to_char_replacer(match: re.Match) -> str:
            expr = match.group(1).strip()
            return f"CAST({expr} AS VARCHAR)"

        sql = re.sub(r"TO_DATE\s*\(([^,]+?),\s*'[^']*'\)", to_date_replacer, sql, flags=re.IGNORECASE)
        sql = re.sub(r"TO_CHAR\s*\(([^,]+?),\s*'[^']*'\)", to_char_replacer, sql, flags=re.IGNORECASE)
        return sql

    def _requires_dedup_join(self, sql: str) -> bool:
        return bool(sql and re.search(r"\bJOIN\b", sql, re.IGNORECASE))

    def _rewrite_joined_aggregates(self, sql: str) -> str:
        """Detect aggregation on primary table while joining another table that may multiply rows."""
        if not self._requires_dedup_join(sql):
            return sql

        from_match = re.search(r"FROM\s+([A-Za-z_][\w]*)\s+(?:AS\s+)?([A-Za-z_][\w]*)", sql, re.IGNORECASE)
        if not from_match:
            return sql
        base_table, base_alias = from_match.group(1), from_match.group(2)
        base_alias_lower = base_alias.lower()

        join_matches = re.finditer(
            r"JOIN\s+([A-Za-z_][\w]*)\s+(?:AS\s+)?([A-Za-z_][\w]*)\s+ON\s+([^\n]+?)\b(?:JOIN|WHERE|GROUP|HAVING|ORDER|LIMIT|$)",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )

        agg_patterns = [
            rf"sum\s*\((?:\s*try_cast\()?(?:{base_alias_lower}\.\w+)",
            rf"count\s*\((?:\s*distinct\s*)?(?:{base_alias_lower}\.\w+)",
            rf"avg\s*\((?:\s*try_cast\()?(?:{base_alias_lower}\.\w+)",
        ]

        if not any(re.search(pattern, sql, re.IGNORECASE) for pattern in agg_patterns):
            return sql

        for match in join_matches:
            join_table, join_alias, on_clause = match.group(1), match.group(2), match.group(3)
            join_alias_lower = join_alias.lower()
            pairs = []  # (base_key, join_key)
            for key_match in re.finditer(
                r"([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*=\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)",
                on_clause,
                flags=re.IGNORECASE,
            ):
                left_alias, left_col, right_alias, right_col = key_match.groups()
                if left_alias.lower() == base_alias_lower and right_alias.lower() == join_alias_lower:
                    pairs.append((left_col, right_col))
                elif right_alias.lower() == base_alias_lower and left_alias.lower() == join_alias_lower:
                    pairs.append((right_col, left_col))

            join_keys = [p[1] for p in pairs]
            base_keys = [p[0] for p in pairs]

            keys_text = ", ".join(join_keys) if join_keys else "<join_key>"
            base_key_text = ", ".join(base_keys) if base_keys else "<base_key>"

            guidance = (
                f"Aggregations on `{base_alias}` detected while joining `{join_alias}`, which can duplicate rows from `{base_table}`. "
                "Rewrite the SQL so the joined table is pre-aggregated. For example:\n\n"
                f"WITH {join_alias}_summary AS (\n"
                f"  SELECT {keys_text}, /* aggregated metrics for {join_alias} */\n"
                f"  FROM {join_table}\n"
                f"  GROUP BY {keys_text}\n"
                ")\n"
                f"SELECT /* aggregated columns */\n"
                f"FROM {base_table} {base_alias}\n"
                f"LEFT JOIN {join_alias}_summary {join_alias} ON /* re-use the original join condition using {base_alias}.{base_key_text} */\n"
                "\nThen reference the aggregated columns from the summary instead of raw values from the joined table."
            )
            logger.warning(guidance)
            raise ValueError(guidance)

        return sql


def main():
    """Example usage with optimized GGUF model"""
    print("\n" + "="*70)
    print("Optimized Text-to-SQL System for M4 Mac")
    print("="*70)
    
    # Configuration - optimized for M4 Mac
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        use_gguf=True,  # Use GGUF for better performance
        embedding_model="all-MiniLM-L6-v2",
        device="auto",
        database_path="./sample_database.db",
        max_correction_attempts=3,
        # GGUF optimizations
        n_ctx=2048,  # Context window
        n_threads=0,  # Auto-detect CPU cores
        n_gpu_layers=-1,  # Use all Metal GPU layers (M4 Mac)
    )
    
    print(f"Model: {config.llm_model}")
    print(f"Backend: {'GGUF (optimized)' if config.use_gguf else 'Transformers'}")
    print("="*70 + "\n")
    
    # Initialize system
    system = TextToSQLSystem(config)
    
    try:
        # Example queries
        queries = [
            "Show me all employees in the Engineering department",
            "What is the total salary for each department?",
        ]
        
        for query in queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            result = system.query(query)
            
            if result["success"]:
                print(f"\n✅ SQL: {result['sql']}")
                print(f"\n📊 Result:")
                print(result['result'])
                print(f"\n🔄 Attempts: {result['attempts']}")
            else:
                print(f"\n❌ Failed after {result['attempts']} attempts")
                print(f"Error: {result.get('error', 'Unknown error')}")
                if result.get('sql'):
                    print(f"Last SQL: {result['sql']}")
            
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()

