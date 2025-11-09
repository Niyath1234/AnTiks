"""
Open Source Text-to-SQL Solution
Replicates AWS Bedrock architecture using open source models
- Uses SQLCoder or Llama 3 for SQL generation
- Uses sentence-transformers for embeddings
- Uses ChromaDB for vector store
- Implements RAG with metadata
- Self-correction loop using SQL execution errors
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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


@dataclass
class TextToSQLConfig:
    """Configuration for Text-to-SQL system"""
    # LLM Configuration
    llm_model: str = "defog/sqlcoder-7b-2"  # Best open source for SQL, alternatives: "meta-llama/Llama-3-8b-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    max_new_tokens: int = 512
    temperature: float = 0.1
    
    # Vector Store
    vector_store_path: str = "./vector_store"
    collection_name: str = "sql_metadata"
    
    # Database
    database_path: str = "./sample_database.db"
    
    # Correction Loop
    max_correction_attempts: int = 3
    
    # RAG
    top_k_metadata: int = 5


class MetadataExtractor:
    """Extract metadata from database schema"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to database"""
        if not os.path.exists(self.db_path):
            logger.warning(f"Database {self.db_path} does not exist. Creating sample database...")
            self._create_sample_database()
        
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")
        
    def _create_sample_database(self):
        """Create a sample database for demonstration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sample tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                salary REAL,
                hire_date TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                status TEXT,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            )
        """)
        
        # Insert sample data
        cursor.executemany("""
            INSERT INTO employees (name, department, salary, hire_date) VALUES (?, ?, ?, ?)
        """, [
            ("Alice Johnson", "Engineering", 95000, "2020-01-15"),
            ("Bob Smith", "Engineering", 85000, "2021-03-20"),
            ("Carol White", "Marketing", 70000, "2019-06-10"),
            ("David Brown", "Sales", 80000, "2022-01-05"),
        ])
        
        cursor.executemany("""
            INSERT INTO departments (name, budget) VALUES (?, ?)
        """, [
            ("Engineering", 500000),
            ("Marketing", 300000),
            ("Sales", 400000),
        ])
        
        cursor.executemany("""
            INSERT INTO projects (name, department_id, status) VALUES (?, ?, ?)
        """, [
            ("Project Alpha", 1, "active"),
            ("Project Beta", 1, "completed"),
            ("Campaign 2024", 2, "active"),
        ])
        
        conn.commit()
        conn.close()
        logger.info("Sample database created with employees, departments, and projects tables")
    
    def get_schema_metadata(self) -> List[Dict[str, Any]]:
        """Extract schema metadata from database"""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        metadata = []
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            # Get columns for each table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            
            # Build metadata
            column_info = []
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                column_info.append({
                    "column_name": col_name,
                    "data_type": col_type,
                    "is_nullable": not bool(not_null),
                    "is_primary_key": bool(pk),
                    "default_value": default_val
                })
            
            fk_info = []
            for fk in foreign_keys:
                fk_info.append({
                    "column": fk[3],
                    "references_table": fk[2],
                    "references_column": fk[4]
                })
            
            metadata.append({
                "table_name": table_name,
                "columns": column_info,
                "foreign_keys": fk_info,
                "description": f"Table {table_name} with {len(column_info)} columns"
            })
        
        return metadata
    
    def format_metadata_for_rag(self, metadata: List[Dict[str, Any]]) -> List[str]:
        """Format metadata as text for embedding"""
        formatted_texts = []
        
        for table_info in metadata:
            table_name = table_info["table_name"]
            columns = table_info["columns"]
            
            # Create comprehensive description
            col_descriptions = []
            for col in columns:
                col_desc = f"{col['column_name']} ({col['data_type']})"
                if col['is_primary_key']:
                    col_desc += " [PRIMARY KEY]"
                if not col['is_nullable']:
                    col_desc += " [NOT NULL]"
                col_descriptions.append(col_desc)
            
            fk_desc = ""
            if table_info["foreign_keys"]:
                fk_list = [f"{fk['column']} -> {fk['references_table']}.{fk['references_column']}" 
                          for fk in table_info["foreign_keys"]]
                fk_desc = f" Foreign keys: {', '.join(fk_list)}"
            
            text = f"Table: {table_name}\nColumns: {', '.join(col_descriptions)}{fk_desc}"
            formatted_texts.append(text)
        
        return formatted_texts
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class VectorStore:
    """Vector store for metadata using ChromaDB"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.client = chromadb.PersistentClient(path=config.vector_store_path)
        self.collection = None
        
    def initialize_collection(self):
        """Initialize or get collection"""
        try:
            self.collection = self.client.get_collection(name=self.config.collection_name)
            logger.info(f"Loaded existing collection: {self.config.collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.config.collection_name}")
    
    def add_metadata(self, metadata_texts: List[str], metadata_dicts: List[Dict]):
        """Add metadata to vector store"""
        if not self.collection:
            self.initialize_collection()
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(metadata_texts, show_progress_bar=True)
        
        # Add to collection
        ids = [f"table_{i}" for i in range(len(metadata_texts))]
        metadatas = [{"table_name": md["table_name"], "text": text} 
                    for md, text in zip(metadata_dicts, metadata_texts)]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=metadata_texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(metadata_texts)} metadata entries to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant metadata"""
        if not self.collection:
            self.initialize_collection()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted_results


class SQLLLM:
    """Open source LLM for SQL generation"""
    
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
        """Load the LLM model"""
        logger.info(f"Loading model {self.config.llm_model} on {self.device}...")
        
        # Set environment for faster downloads if not already set
        if 'HF_HUB_ENABLE_HF_TRANSFER' not in os.environ:
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        
        try:
            # Try to load with 8-bit quantization for memory efficiency
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model,
                trust_remote_code=True,
                resume_download=True  # Resume interrupted downloads
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            # Use device_map="auto" for better memory management
            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map="auto",
                    resume_download=True  # Resume interrupted downloads
                )
            elif self.device == "mps":
                # MPS sometimes has issues with device_map, try without it first
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map=None,  # Don't use device_map for MPS
                    low_cpu_mem_usage=True,
                    resume_download=True
                )
                # Move model to MPS manually
                self.model = self.model.to("mps")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    resume_download=True  # Resume interrupted downloads
                )
            
            # Create pipeline
            # Note: When using device_map="auto", don't specify device in pipeline
            if self.device == "cuda":
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16
                )
            elif self.device == "mps":
                # MPS doesn't work well with device_map, use CPU for MPS
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float32
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float32
                )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to a smaller model or CPU-only mode...")
            raise
    
    def generate_sql(self, prompt: str) -> str:
        """Generate SQL from prompt"""
        if not self.pipeline:
            self.load_model()
        
        # Generate SQL
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
            
            # Extract SQL query (remove any markdown formatting)
            sql = generated_text
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            return sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise


class SQLExecutor:
    """Execute SQL queries and handle errors"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def execute_query(self, sql: str) -> Tuple[bool, Any, Optional[str]]:
        """Execute SQL query and return results"""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            # Check if it's a SELECT query
            if sql.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                df = pd.DataFrame(results, columns=columns)
                return True, df, None
            else:
                self.conn.commit()
                return True, f"Query executed successfully. Rows affected: {cursor.rowcount}", None
                
        except sqlite3.Error as e:
            error_msg = str(e)
            logger.error(f"SQL Error: {error_msg}")
            return False, None, error_msg
    
    def validate_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax using EXPLAIN"""
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            cursor.fetchall()
            return True, None
        except sqlite3.Error as e:
            return False, str(e)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class TextToSQLSystem:
    """Main Text-to-SQL system with RAG and self-correction"""
    
    def __init__(self, config: TextToSQLConfig):
        self.config = config
        self.metadata_extractor = MetadataExtractor(config.database_path)
        self.vector_store = VectorStore(config)
        self.llm = SQLLLM(config)
        self.sql_executor = SQLExecutor(config.database_path)
        
        # Initialize components
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system with metadata"""
        logger.info("Initializing Text-to-SQL system...")
        
        # Extract metadata
        metadata = self.metadata_extractor.get_schema_metadata()
        
        # Format for RAG
        metadata_texts = self.metadata_extractor.format_metadata_for_rag(metadata)
        
        # Add to vector store
        self.vector_store.initialize_collection()
        self.vector_store.add_metadata(metadata_texts, metadata)
        
        logger.info("System initialized successfully")
    
    def _build_prompt(self, user_query: str, metadata_context: List[Dict]) -> str:
        """Build prompt with user query and metadata context"""
        # Format metadata context
        context_text = "\n\n".join([result["document"] for result in metadata_context])
        
        # Build prompt based on model type
        if "sqlcoder" in self.config.llm_model.lower():
            prompt = f"""### Task
Generate a SQL query to answer the following question: {user_query}

### Database Schema
{context_text}

### SQL Query
"""
        elif "llama" in self.config.llm_model.lower() or "mistral" in self.config.llm_model.lower():
            prompt = f"""You are a SQL expert. Generate a SQL query to answer the following question.

Question: {user_query}

Database Schema:
{context_text}

SQL Query:
"""
        else:
            # Generic prompt
            prompt = f"""Generate a SQL query for the following question.

Question: {user_query}

Database Schema:
{context_text}

SQL Query:
"""
        
        return prompt
    
    def _correct_sql(self, sql: str, error_message: str, user_query: str, metadata_context: List[Dict]) -> str:
        """Correct SQL using error message"""
        # Format metadata context (move join outside f-string to avoid backslash issue)
        schema_context = "\n\n".join([result['document'] for result in metadata_context])
        
        correction_prompt = f"""The following SQL query has an error. Please correct it.

Original Question: {user_query}

Database Schema:
{schema_context}

Original SQL Query:
{sql}

Error Message:
{error_message}

Corrected SQL Query:
"""
        
        corrected_sql = self.llm.generate_sql(correction_prompt)
        return corrected_sql
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Main query method with RAG and self-correction"""
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: RAG - Retrieve relevant metadata
        metadata_results = self.vector_store.search(
            user_query, 
            top_k=self.config.top_k_metadata
        )
        
        if not metadata_results:
            logger.warning("No relevant metadata found. Using all metadata.")
            # Fallback: get all metadata
            all_metadata = self.metadata_extractor.get_schema_metadata()
            metadata_texts = self.metadata_extractor.format_metadata_for_rag(all_metadata)
            metadata_results = [{"document": text} for text in metadata_texts[:self.config.top_k_metadata]]
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt(user_query, metadata_results)
        
        # Step 3: Generate SQL with self-correction loop
        sql = None
        attempt = 0
        last_error = None
        
        while attempt < self.config.max_correction_attempts:
            attempt += 1
            logger.info(f"Attempt {attempt}/{self.config.max_correction_attempts}")
            
            try:
                # Generate SQL
                if attempt == 1:
                    sql = self.llm.generate_sql(prompt)
                else:
                    # Correction attempt
                    sql = self._correct_sql(sql, last_error, user_query, metadata_results)
                
                logger.info(f"Generated SQL: {sql}")
                
                # Step 4: Validate syntax
                is_valid, error = self.sql_executor.validate_syntax(sql)
                
                if is_valid:
                    # Step 5: Execute query
                    success, result, error = self.sql_executor.execute_query(sql)
                    
                    if success:
                        logger.info("Query executed successfully")
                        return {
                            "success": True,
                            "sql": sql,
                            "result": result,
                            "attempts": attempt,
                            "metadata_used": [r["metadata"].get("table_name", "unknown") for r in metadata_results]
                        }
                    else:
                        last_error = error
                        logger.warning(f"Execution error: {error}")
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
            "metadata_used": [r["metadata"].get("table_name", "unknown") for r in metadata_results]
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.metadata_extractor.close()
        self.sql_executor.close()


def main():
    """Example usage"""
    # Configuration
    # Note: SQLCoder-7B-2 is ~14GB. If download is slow, use a smaller model:
    # - "Qwen/Qwen2.5-7B-Instruct" (smaller, faster download)
    # - "mistralai/Mistral-7B-Instruct-v0.2" (if cached)
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",  # Best for SQL, but requires ~14GB RAM and ~14GB download
        # Alternative: "meta-llama/Llama-3-8b-Instruct" (requires ~16GB RAM)
        # Or: "mistralai/Mistral-7B-Instruct-v0.2" (requires ~14GB RAM)
        # For CPU-only or slow downloads: "Qwen/Qwen2.5-7B-Instruct" (smaller, faster)
        embedding_model="all-MiniLM-L6-v2",
        device="auto",
        database_path="./sample_database.db",
        max_correction_attempts=3
    )
    
    print("\n" + "="*60)
    print("Text-to-SQL System")
    print("="*60)
    print(f"Model: {config.llm_model}")
    print(f"Note: First run will download the model (~14GB)")
    print("      If download is slow, press Ctrl+C and use fast_download_model.py")
    print("="*60 + "\n")
    
    # Initialize system
    system = TextToSQLSystem(config)
    
    try:
        # Example queries
        queries = [
            "Show me all employees in the Engineering department",
            "What is the total salary for each department?",
            "List all active projects with their department names",
            "Find employees hired after 2020 with salary greater than 80000"
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            result = system.query(query)
            
            if result["success"]:
                print(f"\nSQL: {result['sql']}")
                print(f"\nResult:")
                print(result['result'])
                print(f"\nAttempts: {result['attempts']}")
            else:
                print(f"\nFailed after {result['attempts']} attempts")
                print(f"Error: {result.get('error', 'Unknown error')}")
                if result.get('sql'):
                    print(f"Last SQL: {result['sql']}")
            
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()

