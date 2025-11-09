"""
Single Query Tester - Test the model with a single query via command line
Usage: python test_single_query.py "your question here"
"""

import sys
import json
import pandas as pd

from .text_to_sql_optimized import TextToSQLConfig, TextToSQLSystem

def format_result(result):
    """Format query result for display"""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No results found."
        else:
            return result.to_string(index=False)
    else:
        return str(result)


def test_query(query: str):
    """Test a single query"""
    print("\n" + "="*70)
    print("Text-to-SQL Query Test")
    print("="*70)
    print(f"📝 Question: {query}")
    print("="*70 + "\n")
    
    # Configuration
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        embedding_model="all-MiniLM-L6-v2",
        device="cpu",  # Use CPU for reliability
        database_path="./sample_database.db",
        max_correction_attempts=3
    )
    
    print("⏳ Initializing system...")
    try:
        system = TextToSQLSystem(config)
        print("✅ System ready!\n")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    print("⏳ Processing query... (this may take 30-60 seconds)\n")
    
    try:
        result = system.query(query)
        
        print("="*70)
        print("Results")
        print("="*70)
        print(f"🤖 Generated SQL:")
        print(f"   {result['sql']}\n")
        
        print(f"🔄 Attempts: {result['attempts']}")
        
        if result.get('metadata_used'):
            print(f"📊 Tables Used: {', '.join(result['metadata_used'])}")
        
        if result['success']:
            print(f"\n✅ Status: SUCCESS\n")
            print(f"📋 Results:")
            print("-" * 70)
            print(format_result(result['result']))
            print("-" * 70)
        else:
            print(f"\n❌ Status: FAILED")
            print(f"⚠️  Error: {result.get('error', 'Unknown error')}\n")
        
        print("="*70 + "\n")
        
        # Also output as JSON for programmatic use
        if '--json' in sys.argv:
            output = {
                'query': query,
                'success': result['success'],
                'sql': result['sql'],
                'attempts': result['attempts'],
                'metadata_used': result.get('metadata_used', []),
            }
            if result['success']:
                if isinstance(result['result'], pd.DataFrame):
                    output['result'] = result['result'].to_dict('records')
                else:
                    output['result'] = str(result['result'])
            else:
                output['error'] = result.get('error', 'Unknown error')
            
            print("\nJSON Output:")
            print(json.dumps(output, indent=2))
        
        system.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        system.cleanup()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_query.py 'your question here' [--json]")
        print("\nExamples:")
        print('  python test_single_query.py "Show me all employees"')
        print('  python test_single_query.py "What is the total salary per department?" --json')
        sys.exit(1)
    
    query = sys.argv[1]
    test_query(query)


