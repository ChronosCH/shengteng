"""
文件处理和上传管理模块
提供文件上传、存储、处理、安全检查等功能
"""

import os
import hashlib
import mimetypes
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO
from datetime import datetime, timedelta
import aiofiles
import uuid

from fastapi import UploadFile, HTTPException, status
from PIL import Image
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import settings
from utils.database import db_manager
from utils.cache import cache_manager

logger = setup_logger(__name__)


class FileManager:
    """文件管理器"""
    
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.temp_dir = Path("temp")
        self.allowed_extensions = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
            'video': {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'},
            'audio': {'.mp3', '.wav', '.aac', '.ogg', '.m4a'},
            'document': {'.pdf', '.doc', '.docx', '.txt', '.rtf'},
            'data': {'.json', '.csv', '.xlsx', '.xml'}
        }
        self.max_file_sizes = {
            'image': 10 * 1024 * 1024,  # 10MB
            'video': 100 * 1024 * 1024,  # 100MB
            'audio': 50 * 1024 * 1024,   # 50MB
            'document': 20 * 1024 * 1024,  # 20MB
            'data': 10 * 1024 * 1024    # 10MB
        }
        
        self._ensure_directories()
        logger.info("文件管理器初始化完成")
    
    def _ensure_directories(self):
        """确保目录存在"""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建按类型分类的子目录
        for file_type in self.allowed_extensions.keys():
            (self.upload_dir / file_type).mkdir(exist_ok=True)
    
    def _get_file_type(self, filename: str) -> Optional[str]:
        """根据文件扩展名确定文件类型"""
        ext = Path(filename).suffix.lower()
        
        for file_type, extensions in self.allowed_extensions.items():
            if ext in extensions:
                return file_type
        
        return None
    
    def _generate_filename(self, original_filename: str, user_id: int = None) -> str:
        """生成唯一文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        ext = Path(original_filename).suffix.lower()
        
        if user_id:
            return f"{user_id}_{timestamp}_{unique_id}{ext}"
        else:
            return f"{timestamp}_{unique_id}{ext}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """验证上传文件"""
        # 检查文件名
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件名不能为空"
            )
        
        # 检查文件类型
        file_type = self._get_file_type(file.filename)
        if not file_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {Path(file.filename).suffix}"
            )
        
        # 检查文件大小
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        # 重置文件指针
        await file.seek(0)
        
        max_size = self.max_file_sizes.get(file_type, 10 * 1024 * 1024)
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件过大，最大允许 {max_size / 1024 / 1024:.1f}MB"
            )
        
        # 检查MIME类型
        detected_mime = mimetypes.guess_type(file.filename)[0]
        if file.content_type and detected_mime:
            if not file.content_type.startswith(detected_mime.split('/')[0]):
                logger.warning(f"MIME类型不匹配: {file.content_type} vs {detected_mime}")
        
        return {
            "file_type": file_type,
            "file_size": file_size,
            "mime_type": file.content_type,
            "detected_mime": detected_mime
        }
    
    async def save_file(self, file: UploadFile, user_id: int = None, 
                       metadata: Dict = None) -> Dict[str, Any]:
        """保存上传文件"""
        # 验证文件
        validation_info = await self.validate_file(file)
        file_type = validation_info["file_type"]
        
        # 生成文件名和路径
        filename = self._generate_filename(file.filename, user_id)
        file_path = self.upload_dir / file_type / filename
        
        try:
            # 保存文件
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            
            # 处理文件（如图像压缩、视频预处理等）
            processed_info = await self._process_file(file_path, file_type)
            
            # 构建文件信息
            file_info = {
                "original_filename": file.filename,
                "filename": filename,
                "file_path": str(file_path),
                "file_type": file_type,
                "file_size": validation_info["file_size"],
                "file_hash": file_hash,
                "mime_type": validation_info["mime_type"],
                "user_id": user_id,
                "upload_time": datetime.now().isoformat(),
                "metadata": metadata or {},
                "processed_info": processed_info
            }
            
            # 缓存文件信息
            await cache_manager.set("file_info", file_hash, file_info, ttl=3600)
            
            logger.info(f"文件保存成功: {filename}, 用户: {user_id}")
            return file_info
            
        except Exception as e:
            # 删除部分上传的文件
            if file_path.exists():
                file_path.unlink()
            
            logger.error(f"保存文件失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"保存文件失败: {str(e)}"
            )
    
    async def _process_file(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """处理上传的文件"""
        processed_info = {}
        
        try:
            if file_type == "image":
                processed_info = await self._process_image(file_path)
            elif file_type == "video":
                processed_info = await self._process_video(file_path)
            elif file_type == "audio":
                processed_info = await self._process_audio(file_path)
            
        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            processed_info = {"error": str(e)}
        
        return processed_info
    
    async def _process_image(self, file_path: Path) -> Dict[str, Any]:
        """处理图像文件"""
        try:
            with Image.open(file_path) as img:
                # 获取图像信息
                width, height = img.size
                format_info = img.format
                mode = img.mode
                
                # 生成缩略图
                thumbnail_path = file_path.parent / f"thumb_{file_path.name}"
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, format=format_info)
                
                # 如果图像过大，创建压缩版本
                compressed_path = None
                if width > 1920 or height > 1080:
                    compressed_path = file_path.parent / f"compressed_{file_path.name}"
                    img_resized = img.resize((min(1920, width), min(1080, height)), Image.Resampling.LANCZOS)
                    img_resized.save(compressed_path, format=format_info, quality=85)
                
                return {
                    "width": width,
                    "height": height,
                    "format": format_info,
                    "mode": mode,
                    "thumbnail_path": str(thumbnail_path),
                    "compressed_path": str(compressed_path) if compressed_path else None
                }
                
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return {"error": str(e)}
    
    async def _process_video(self, file_path: Path) -> Dict[str, Any]:
        """处理视频文件"""
        try:
            cap = cv2.VideoCapture(str(file_path))
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 提取第一帧作为封面
            ret, frame = cap.read()
            if ret:
                thumbnail_path = file_path.parent / f"thumb_{file_path.stem}.jpg"
                cv2.imwrite(str(thumbnail_path), frame)
            
            cap.release()
            
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "thumbnail_path": str(thumbnail_path) if ret else None
            }
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            return {"error": str(e)}
    
    async def _process_audio(self, file_path: Path) -> Dict[str, Any]:
        """处理音频文件"""
        try:
            # 这里可以集成librosa或其他音频处理库
            # 现在先返回基本信息
            file_size = file_path.stat().st_size
            
            return {
                "file_size": file_size,
                "processed": True
            }
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            return {"error": str(e)}
    
    async def get_file_info(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        # 先从缓存获取
        file_info = await cache_manager.get("file_info", file_hash)
        if file_info:
            return file_info
        
        # 从数据库获取（如果有文件表的话）
        # 这里可以添加数据库查询逻辑
        
        return None
    
    async def delete_file(self, file_hash: str, user_id: int = None) -> bool:
        """删除文件"""
        try:
            file_info = await self.get_file_info(file_hash)
            if not file_info:
                return False
            
            # 检查权限
            if user_id and file_info.get("user_id") != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="无权删除此文件"
                )
            
            # 删除主文件
            file_path = Path(file_info["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            # 删除相关文件（缩略图、压缩版本等）
            processed_info = file_info.get("processed_info", {})
            for key in ["thumbnail_path", "compressed_path"]:
                if key in processed_info and processed_info[key]:
                    thumb_path = Path(processed_info[key])
                    if thumb_path.exists():
                        thumb_path.unlink()
            
            # 从缓存删除
            await cache_manager.delete("file_info", file_hash)
            
            logger.info(f"文件删除成功: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """清理临时文件"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            deleted_count = 0
            
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            
            logger.info(f"清理了 {deleted_count} 个临时文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            "total_files": 0,
            "total_size": 0,
            "files_by_type": {},
            "size_by_type": {}
        }
        
        try:
            for file_type in self.allowed_extensions.keys():
                type_dir = self.upload_dir / file_type
                if type_dir.exists():
                    files = list(type_dir.glob("*"))
                    file_count = len(files)
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    
                    stats["files_by_type"][file_type] = file_count
                    stats["size_by_type"][file_type] = total_size
                    stats["total_files"] += file_count
                    stats["total_size"] += total_size
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
        
        return stats


# 全局文件管理器实例
file_manager = FileManager()