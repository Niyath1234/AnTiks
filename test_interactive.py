"""
Interactive Text-to-SQL Testing Script
Allows dynamic input and output for testing the model
"""

import sys
from text_to_sql_architecture import TextToSQLConfig, TextToSQLSystem
import pandas as pd

def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print(" " * 15 + "Text-to-SQL Interactive Tester")
    print("="*70)
    print("\nCommands:")
    print("  • Enter your question in natural language")
    print("  • Type 'exit' or 'quit' to stop")
    print("  • Type 'schema' to see database schema")
    print("  • Type 'examples' to see example queries")
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
        "List all projects and their department budgets"
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
            # Format DataFrame nicely
            return result.to_string(index=False)
    else:
        return str(result)


def print_query_result(query, result):
    """Print formatted query result"""
    print("\n" + "="*70)
    print("Query Result")
    print("="*70)
    print(f"📝 Question: {query}")
    print(f"🤖 SQL Generated: {result['sql']}")
    print(f"🔄 Attempts: {result['attempts']}")
    
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
    
    # Configuration - use CPU to avoid MPS memory issues
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        embedding_model="all-MiniLM-L6-v2",
        device="cpu",  # Use CPU for reliability
        database_path="./sample_database.db",
        max_correction_attempts=3
    )
    
    print("⏳ Initializing Text-to-SQL system...")
    print("   (This may take a minute on first run - loading model...)\n")
    
    try:
        system = TextToSQLSystem(config)
        print("✅ System ready!\n")
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        print("\n💡 Tips:")
        print("   - Make sure the model is downloaded")
        print("   - Check that you have enough RAM (model needs ~14GB)")
        print("   - Try using device='cpu' in the config")
        return
    
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
            
            elif user_input.lower() == 'clear':
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                print_header()
                continue
            
            # Process query
            print("\n⏳ Processing query... (this may take 30-60 seconds)\n")
            
            try:
                result = system.query(user_input)
                print_query_result(user_input, result)
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


