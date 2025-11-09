"""
Test script to verify model files are loaded correctly
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from text_to_sql_architecture import TextToSQLConfig, SQLLLM

def check_model_files(model_name: str):
    """Check if model files exist in cache"""
    from huggingface_hub import snapshot_download
    
    print("\n" + "="*60)
    print("Checking Model Files")
    print("="*60)
    
    try:
        # Get cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        
        if os.path.exists(model_path):
            print(f"✅ Model cache directory found: {model_path}")
            
            # Check for key files
            key_files = [
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
            ]
            
            # Find all safetensors files
            safetensors_files = []
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.safetensors'):
                        safetensors_files.append(file)
                        print(f"  ✅ Found: {file}")
            
            print(f"\n📊 Found {len(safetensors_files)} model shard(s)")
            
            # Check for GGUF file (if using quantization)
            gguf_files = []
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.gguf'):
                        gguf_files.append(file)
                        print(f"  ✅ Found: {file}")
            
            if gguf_files:
                print(f"\n📊 Found {len(gguf_files)} GGUF file(s)")
            
            # Check config files
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file in key_files:
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        print(f"  ✅ Found: {file} ({size:,} bytes)")
            
            return True
        else:
            print(f"❌ Model cache directory not found: {model_path}")
            print("   Model may not be downloaded yet")
            return False
            
    except Exception as e:
        print(f"❌ Error checking files: {e}")
        return False


def test_model_loading():
    """Test loading the model"""
    print("\n" + "="*60)
    print("Testing Model Loading")
    print("="*60)
    
    # Use CPU for testing to avoid MPS memory issues
    # The model is ~14GB which is too large for most MPS devices
    config = TextToSQLConfig(
        llm_model="defog/sqlcoder-7b-2",
        device="cpu"  # Use CPU to avoid MPS memory issues
    )
    
    llm = SQLLLM(config)
    
    print(f"📱 Device: {llm.device}")
    print(f"🤖 Model: {config.llm_model}")
    print(f"\n⏳ Loading model (this may take a minute)...\n")
    
    try:
        llm.load_model()
        print("\n✅ Model loaded successfully!")
        
        # Check model attributes
        if llm.model is not None:
            print(f"  ✅ Model object: {type(llm.model).__name__}")
        if llm.tokenizer is not None:
            print(f"  ✅ Tokenizer: {type(llm.tokenizer).__name__}")
        if llm.pipeline is not None:
            print(f"  ✅ Pipeline: {type(llm.pipeline).__name__}")
        
        return llm, True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_model_generation(llm: SQLLLM):
    """Test if model can generate text"""
    print("\n" + "="*60)
    print("Testing Model Generation")
    print("="*60)
    
    # Simple test prompt
    test_prompt = """### Task
Generate a SQL query to answer the following question: Show me all employees

### Database Schema
Table: employees
Columns: id (INTEGER) [PRIMARY KEY], name (TEXT) [NOT NULL], department (TEXT), salary (REAL), hire_date (TEXT)

### SQL Query
"""
    
    print("📝 Test prompt:")
    print("-" * 60)
    print(test_prompt.strip())
    print("-" * 60)
    print("\n⏳ Generating SQL (this may take 30-60 seconds)...\n")
    
    try:
        sql = llm.generate_sql(test_prompt)
        print("✅ Generation successful!")
        print(f"\n📤 Generated SQL:")
        print("-" * 60)
        print(sql)
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    model_name = "defog/sqlcoder-7b-2"
    
    print("\n" + "="*60)
    print("Model Loading Test Suite")
    print("="*60)
    print(f"Model: {model_name}")
    print("="*60)
    
    # Step 1: Check if files exist
    files_exist = check_model_files(model_name)
    
    if not files_exist:
        print("\n⚠️  Model files not found in cache.")
        print("   Run the main script first to download the model.")
        return
    
    # Step 2: Test loading
    llm, loaded = test_model_loading()
    
    if not loaded:
        print("\n❌ Model loading failed. Check the error above.")
        return
    
    # Step 3: Test generation
    generation_works = test_model_generation(llm)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✅ Files exist: {files_exist}")
    print(f"✅ Model loaded: {loaded}")
    print(f"✅ Generation works: {generation_works}")
    
    if files_exist and loaded and generation_works:
        print("\n🎉 All tests passed! Model is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

