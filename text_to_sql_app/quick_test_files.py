"""
Quick test to verify model files are downloaded correctly
Without loading the full model into memory
"""

import os
from pathlib import Path

def check_model_files():
    """Check if all model files are present"""
    model_name = "defog/sqlcoder-7b-2"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    
    print("\n" + "="*60)
    print("Model File Verification")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Cache path: {model_path}")
    print("="*60)
    
    if not os.path.exists(model_path):
        print("❌ Model cache directory not found!")
        return False
    
    print("✅ Model cache directory exists\n")
    
    # Check for key files
    required_files = {
        "config.json": False,
        "generation_config.json": False,
        "tokenizer_config.json": False,
        "tokenizer.json": False,
        "model.safetensors.index.json": False,
    }
    
    safetensors_files = []
    total_size = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            if file.endswith('.safetensors'):
                safetensors_files.append((file, file_size))
                total_size += file_size
            elif file in required_files:
                required_files[file] = True
                print(f"  ✅ {file} ({file_size:,} bytes)")
    
    # Check results
    print(f"\n📊 Model Files Summary:")
    print(f"  • Safetensors shards: {len(safetensors_files)}")
    
    for file, size in sorted(safetensors_files):
        size_gb = size / (1024**3)
        print(f"    ✅ {file} ({size_gb:.2f} GB)")
    
    total_gb = total_size / (1024**3)
    print(f"  • Total model size: {total_gb:.2f} GB")
    
    # Check for GGUF file
    gguf_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.gguf'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                gguf_files.append((file, file_size))
    
    if gguf_files:
        print(f"\n📊 Quantized Model Files:")
        for file, size in gguf_files:
            size_gb = size / (1024**3)
            print(f"    ✅ {file} ({size_gb:.2f} GB)")
    
    # Check required files
    missing_files = [f for f, found in required_files.items() if not found]
    if missing_files:
        print(f"\n⚠️  Missing required files: {missing_files}")
        return False
    
    print(f"\n✅ All required files present!")
    print(f"✅ Model files are ready to use")
    
    return True


def test_tokenizer_only():
    """Test loading just the tokenizer (lightweight test)"""
    print("\n" + "="*60)
    print("Testing Tokenizer Loading")
    print("="*60)
    
    try:
        from transformers import AutoTokenizer
        
        print("⏳ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "defog/sqlcoder-7b-2",
            trust_remote_code=True
        )
        
        print("✅ Tokenizer loaded successfully!")
        print(f"  • Vocab size: {len(tokenizer)}")
        print(f"  • Model max length: {tokenizer.model_max_length}")
        
        # Test encoding
        test_text = "SELECT * FROM employees"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\n✅ Tokenizer test:")
        print(f"  • Input: '{test_text}'")
        print(f"  • Encoded: {len(tokens)} tokens")
        print(f"  • Decoded: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Quick Model File Test")
    print("="*60)
    
    # Test 1: Check files exist
    files_ok = check_model_files()
    
    # Test 2: Test tokenizer (lightweight)
    if files_ok:
        tokenizer_ok = test_tokenizer_only()
    else:
        tokenizer_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✅ Files exist: {files_ok}")
    print(f"✅ Tokenizer works: {tokenizer_ok}")
    
    if files_ok and tokenizer_ok:
        print("\n🎉 Model files are downloaded correctly!")
        print("\n💡 To test the full model:")
        print("   - Use device='cpu' in your config (MPS runs out of memory)")
        print("   - Or use quantization with a library like llama.cpp")
        print("   - Run: python text_to_sql_architecture.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print("="*60 + "\n")


