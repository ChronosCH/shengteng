print("Starting simple test...")

try:
    import os
    print("✓ os imported")
    
    import sys
    print("✓ sys imported")
    
    import json
    print("✓ json imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    # Test file existence
    files_to_check = [
        "training/config_manager.py",
        "training/tfnet_model.py", 
        "training/modules.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
    
    print("Simple test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
