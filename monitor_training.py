#!/usr/bin/env python3
"""
Simple script to monitor training progress
"""

import os
import time
import glob

def monitor_training():
    """Monitor training progress by checking log files"""
    print("=== TFNet Training Monitor ===")
    
    log_dir = "training/logs"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
    
    # Find the latest log file
    log_files = glob.glob(os.path.join(log_dir, "training_*.log"))
    if not log_files:
        print("No training log files found")
        return
    
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Monitoring: {latest_log}")
    print("-" * 50)
    
    # Monitor the log file
    last_size = 0
    while True:
        try:
            current_size = os.path.getsize(latest_log)
            if current_size > last_size:
                # Read new content
                with open(latest_log, 'r', encoding='utf-8') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content.strip():
                        print(new_content.strip())
                last_size = current_size
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error monitoring log: {e}")
            break

if __name__ == "__main__":
    monitor_training()
