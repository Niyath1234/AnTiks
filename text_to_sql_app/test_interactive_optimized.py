"""
Interactive Text-to-SQL Testing Script (Optimized for M4 Mac)
Uses GGUF quantization for faster, more memory-efficient inference
"""

import sys
from .text_to_sql_optimized import TextToSQLConfig, TextToSQLSystem
import pandas as pd

def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print(" " * 12 + "Text-to-SQL Interactive Tester (Optimized)")
    print("="*70)
    print("\n⚡ Using GGUF quantization for Apple Silicon (M4 Mac)")
    print("   • 3-5x faster than CPU mode")
    print("   • 65% less memory usage")
    print("   • Native Metal GPU acceleration")
    print("\nCommands:")
    print("  • Enter your question in natural language")
    print("  • Type 'exit' or 'quit' to stop")
    print("  • Type 'schema' to see database schema")
    print("  • Type 'examples' to see example queries")
    print("  • Type 'stats' to see performance stats")
    print("  • Type 'clear' to clear the screen")
    print("\n" + "-"*70 + "\n")


def print_schema(system):
    """Print database schema"""
    print("\n" + "="*70)
    print("Database Schema")
    print("="*70)
    
    metadata = system.metadata_extractor.get_schema_metadata()
    
    for table_info in metadata:
        print(f"\n📊 Table: {table_info['table_name']}")
        print("   Columns:")
        for col in table_info['columns']:
            col_str = f"      • {col['column_name']} ({col['data_type']})"
            if col['is_primary_key']:
                col_str += " [PRIMARY KEY]"
            if not col['is_nullable']:
                col_str += " [NOT NULL]"
            print(col_str)
        
        if table_info['foreign_keys']:
            print("   Foreign Keys:")
            for fk in table_info['foreign_keys']:
                print(f"      • {fk['column']} → {fk['references_table']}.{fk['references_column']}")
    
    print("\n" + "="*70 + "\n")


def print_examples():
    """Print example queries"""
    examples = [
        "Show me all employees in the Engineering department",
        "What is the total salary for each department?",
        "List all active projects with their department names",
        "Find employees hired after 2020 with salary greater than 80000",
        "How many employees are in each department?",
        "Show me all employees sorted by hire date",
        "What is the average salary per department?",
        "List all projects and their department budgets",
        "can you write a query to get the disbursement amount zone wise. given columns are disbursement_amount,Zone_Name and table is disbursement_register"
    ]
    
    print("\n" + "="*70)
    print("Example Queries")
    print("="*70)
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print("="*70 + "\n")


def format_result(result):
    """Format and display query result"""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No results found."
        else:
            return result.to_string(index=False)
    else:
        return str(result)


def print_query_result(query, result, elapsed_time=None):
    """Print formatted query result"""
    print("\n" + "="*70)
    print("Query Result")
    print("="*70)
    print(f"📝 Question: {query}")
    print(f"🤖 SQL Generated: {result['sql']}")
    print(f"🔄 Attempts: {result['attempts']}")
    
    if elapsed_time:
        print(f"⏱️  Time: {elapsed_time:.2f} seconds")

    schema_info = result.get('schema_info')
    if schema_info:
        source = schema_info.get('source')
        if source == 'query':
            print("📑 Schema Extracted from Query:")
            print(f"   • Table: {schema_info.get('table') or 'unknown'}")
            columns = schema_info.get('columns') or []
            if columns:
                print(f"   • Columns: {', '.join(columns)}")
        elif source == 'database':
            print("📑 Schema from Database Metadata:")
            tables = schema_info.get('tables') or []
            if tables:
                print(f"   • Tables: {', '.join(tables)}")
            context_snippets = schema_info.get('context') or []
            if context_snippets:
                preview = context_snippets[0].split('\n')[0]
                print(f"   • Context preview: {preview}...")

    if result.get('metadata_used'):
        print(f"📊 Tables Used: {', '.join(result['metadata_used'])}")
    
    if result['success']:
        print("\n✅ Status: SUCCESS")
        print("\n📋 Results:")
        print("-" * 70)
        print(format_result(result['result']))
        print("-" * 70)
    else:
        print("\n❌ Status: FAILED")
        print(f"⚠️  Error: {result.get('error', 'Unknown error')}")
    
    print("="*70 + "\n")


def interactive_mode():
    """Run interactive testing mode"""
    print_header()
    
    # Configuration - optimized for M4 Mac with GGUF
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        use_gguf=True,  # Use GGUF for optimization
        embedding_model="all-MiniLM-L6-v2",
        device="auto",
        database_path="./sample_database.db",
        max_correction_attempts=3,
        # GGUF optimizations for M4 Mac
        n_ctx=2048,
        n_threads=0,  # Auto-detect CPU cores
        n_gpu_layers=-1,  # Use all Metal GPU layers
    )
    
    print("⏳ Initializing optimized Text-to-SQL system...")
    print("   (Model loading may take 30-60 seconds on first run...)\n")
    
    try:
        import time
        start_time = time.time()
        system = TextToSQLSystem(config)
        init_time = time.time() - start_time
        print(f"✅ System ready! (Initialized in {init_time:.2f} seconds)\n")
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure llama-cpp-python is installed:")
        print("      bash setup_gguf.sh")
        print("   2. Check that GGUF file exists in HuggingFace cache")
        print("   3. Try falling back to transformers: use_gguf=False")
        import traceback
        traceback.print_exc()
        return
    
    # Performance stats
    query_count = 0
    total_time = 0
    successful_queries = 0
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("💬 Enter your question (or 'help' for commands): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!\n")
                if query_count > 0:
                    print(f"📊 Session Stats:")
                    print(f"   • Total queries: {query_count}")
                    print(f"   • Successful: {successful_queries}")
                    print(f"   • Average time: {total_time/query_count:.2f}s")
                    print()
                break
            
            elif user_input.lower() == 'schema':
                print_schema(system)
                continue
            
            elif user_input.lower() == 'examples':
                print_examples()
                continue
            
            elif user_input.lower() == 'help':
                print_header()
                continue
            
            elif user_input.lower() == 'stats':
                if query_count > 0:
                    print(f"\n📊 Session Statistics:")
                    print(f"   • Total queries: {query_count}")
                    print(f"   • Successful: {successful_queries}")
                    print(f"   • Failed: {query_count - successful_queries}")
                    print(f"   • Average time: {total_time/query_count:.2f} seconds")
                    print(f"   • Total time: {total_time:.2f} seconds\n")
                else:
                    print("\n📊 No queries processed yet.\n")
                continue
            
            elif user_input.lower() == 'clear':
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                print_header()
                continue
            
            # Process query
            print("\n⏳ Processing query... (this should be fast with GGUF!)\n")
            
            try:
                import time
                query_start = time.time()
                # Auto-detect if schema is in query and skip execution
                result = system.query(user_input, skip_execution=False)
                query_time = time.time() - query_start
                
                query_count += 1
                total_time += query_time
                if result['success']:
                    successful_queries += 1
                
                print_query_result(user_input, result, query_time)
            except KeyboardInterrupt:
                print("\n\n⚠️  Query interrupted by user\n")
                continue
            except Exception as e:
                print(f"\n❌ Error processing query: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break
    
    # Cleanup
    try:
        system.cleanup()
    except:
        pass


if __name__ == "__main__":
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)


