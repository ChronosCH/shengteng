#!/usr/bin/env python3
"""
TFNet训练系统实用功能
跨平台兼容性和错误处理
"""

import os
import sys
import platform
import traceback
from pathlib import Path
from typing import Union, Optional, List

def get_platform_info():
    """获取平台信息用于调试"""
    return {
        'system': platform.system(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'current_directory': os.getcwd()
    }

def normalize_path(path: Union[str, Path]) -> str:
    """规范化路径以实现跨平台兼容性"""
    if path is None:
        return ""
    
    # Convert to string if Path object
    if isinstance(path, Path):
        path = str(path)
    
    # Normalize path separators and resolve relative paths
    normalized = os.path.normpath(os.path.abspath(path))
    return normalized

def ensure_directory_exists(directory: Union[str, Path], create: bool = True) -> bool:
    """
    Ensure directory exists, optionally create it
    
    Args:
        directory: Directory path
        create: Whether to create directory if it doesn't exist
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    if not directory:
        return False
    
    try:
        dir_path = normalize_path(directory)
        
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                return True
            else:
                print(f"Error: Path exists but is not a directory: {dir_path}")
                return False
        
        if create:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
            return True
        else:
            print(f"✗ Directory does not exist: {dir_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error with directory {directory}: {e}")
        return False

def safe_file_path(file_path: Union[str, Path], create_parent: bool = True) -> Optional[str]:
    """
    Get safe file path with optional parent directory creation
    
    Args:
        file_path: File path
        create_parent: Whether to create parent directory
        
    Returns:
        str: Normalized file path or None if invalid
    """
    if not file_path:
        return None
    
    try:
        normalized_path = normalize_path(file_path)
        
        if create_parent:
            parent_dir = os.path.dirname(normalized_path)
            if parent_dir and not ensure_directory_exists(parent_dir, create=True):
                return None
        
        return normalized_path
        
    except Exception as e:
        print(f"Error processing file path {file_path}: {e}")
        return None

def check_file_exists(file_path: Union[str, Path], description: str = "File") -> bool:
    """
    检查文件是否存在，提供描述性错误信息
    
    Args:
        file_path: 要检查的文件路径
        description: 错误信息的描述
        
    Returns:
        bool: 如果文件存在返回True
    """
    if not file_path:
        print(f"✗ {description}: Path is empty")
        return False
    
    try:
        normalized_path = normalize_path(file_path)
        
        if os.path.exists(normalized_path):
            if os.path.isfile(normalized_path):
                print(f"✓ {description}: {normalized_path}")
                return True
            else:
                print(f"✗ {description}: Path exists but is not a file: {normalized_path}")
                return False
        else:
            print(f"✗ {description}: File not found: {normalized_path}")
            return False
            
    except Exception as e:
        print(f"✗ {description}: Error checking file {file_path}: {e}")
        return False

def check_directory_exists(dir_path: Union[str, Path], description: str = "Directory") -> bool:
    """
    检查目录是否存在，提供描述性错误信息
    
    Args:
        dir_path: 要检查的目录路径
        description: 错误信息的描述
        
    Returns:
        bool: 如果目录存在返回True
    """
    if not dir_path:
        print(f"✗ {description}: Path is empty")
        return False
    
    try:
        normalized_path = normalize_path(dir_path)
        
        if os.path.exists(normalized_path):
            if os.path.isdir(normalized_path):
                print(f"✓ {description}: {normalized_path}")
                return True
            else:
                print(f"✗ {description}: Path exists but is not a directory: {normalized_path}")
                return False
        else:
            print(f"✗ {description}: Directory not found: {normalized_path}")
            return False
            
    except Exception as e:
        print(f"✗ {description}: Error checking directory {dir_path}: {e}")
        return False

def safe_import(module_name: str, package: str = None) -> tuple:
    """
    安全导入模块，带错误处理
    
    Args:
        module_name: 要导入的模块名称
        package: 相对导入的包名
        
    Returns:
        tuple: (模块对象, 成功标志, 错误信息)
    """
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
        else:
            module = __import__(module_name)
        return module, True, None
    except ImportError as e:
        return None, False, f"Import error: {e}"
    except Exception as e:
        return None, False, f"Unexpected error: {e}"

def print_error_details(error: Exception, context: str = ""):
    """
    打印详细错误信息用于调试
    
    Args:
        error: 异常对象
        context: 上下文描述
    """
    print(f"{'='*60}")
    print(f"ERROR DETAILS")
    print(f"{'='*60}")
    
    if context:
        print(f"Context: {context}")
    
    print(f"Error Type: {type(error).__name__}")
    print(f"Error Message: {str(error)}")
    
    # Platform info
    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']} ({platform_info['platform']})")
    print(f"Python: {platform_info['python_version']}")
    print(f"Working Directory: {platform_info['current_directory']}")
    
    # Traceback
    print(f"\nTraceback:")
    traceback.print_exc()
    print(f"{'='*60}")

def validate_dataset_structure(base_path: Union[str, Path]) -> dict:
    """
    Validate CE-CSL dataset structure
    
    Args:
        base_path: Base path to CE-CSL dataset
        
    Returns:
        dict: Validation results
    """
    results = {
        'valid': True,
        'missing_paths': [],
        'found_paths': [],
        'errors': []
    }
    
    if not base_path:
        results['valid'] = False
        results['errors'].append("Base path is empty")
        return results
    
    # Expected paths
    expected_paths = [
        'video/train',
        'video/dev', 
        'video/test',
        'label/train.csv',
        'label/dev.csv',
        'label/test.csv'
    ]
    
    try:
        base_normalized = normalize_path(base_path)
        
        for rel_path in expected_paths:
            full_path = os.path.join(base_normalized, rel_path)
            
            if os.path.exists(full_path):
                results['found_paths'].append(full_path)
            else:
                results['missing_paths'].append(full_path)
                results['valid'] = False
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error validating dataset: {e}")
    
    return results

def print_dataset_validation(validation_results: dict):
    """Print dataset validation results"""
    print("Dataset Validation Results:")
    print("-" * 40)
    
    if validation_results['valid']:
        print("✓ Dataset structure is valid")
    else:
        print("✗ Dataset structure is invalid")
    
    if validation_results['found_paths']:
        print(f"\nFound paths ({len(validation_results['found_paths'])}):")
        for path in validation_results['found_paths']:
            print(f"  ✓ {path}")
    
    if validation_results['missing_paths']:
        print(f"\nMissing paths ({len(validation_results['missing_paths'])}):")
        for path in validation_results['missing_paths']:
            print(f"  ✗ {path}")
    
    if validation_results['errors']:
        print(f"\nErrors ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"  ✗ {error}")

def create_safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create safe filename for cross-platform compatibility
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        str: Safe filename
    """
    if not filename:
        return "unnamed_file"
    
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(' .')
    
    # Limit length
    if len(safe_name) > max_length:
        name, ext = os.path.splitext(safe_name)
        max_name_length = max_length - len(ext)
        safe_name = name[:max_name_length] + ext
    
    # Ensure not empty
    if not safe_name:
        safe_name = "unnamed_file"
    
    return safe_name
