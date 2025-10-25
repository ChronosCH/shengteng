"""
通义千问API集成模块
用于将手语识别的零散词汇转换为完整的中英对译句子
"""
import os
import json
import requests
from typing import Optional, Dict, Any


class QwenAPI:
    """通义千问API封装类"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-plus"):
        """
        初始化通义千问API客户端
        
        Args:
            api_key: API密钥,如果为None则从环境变量DASHSCOPE_API_KEY读取
            model: 使用的模型名称,默认为qwen-plus
        """
        self.api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "未找到API密钥。请通过参数传入或设置环境变量DASHSCOPE_API_KEY"
            )
        
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
    def _build_prompt(self, gloss_words: str) -> str:
        """
        构建提示词
        
        Args:
            gloss_words: 手语识别的词汇序列
        Returns:
            完整的提示词
        """
        prompt = f"""你是一个专业的手语翻译助手。你的任务是将手语识别系统输出的零散词汇(gloss)转换为流畅、自然的完整句子。

手语识别的特点:
1. 词汇可能是简化的、不完整的,或者是手语特有的符号
2. 可能包含特殊标记,如"__ON__"表示开始,"loc-"表示位置等
3. 语法顺序可能与正常语言不同
4. 某些词汇可能是德语或其他语言(因为数据集来自多语言环境)

你的任务:
1. 理解这些零散词汇的含义
2. 组织成语法正确、语义连贯的完整句子
3. 同时提供中文和英文两个版本
4. 确保翻译自然流畅,符合两种语言的表达习惯

输出格式要求(严格按照以下JSON格式):
{{
    "chinese": "中文完整句子",
    "english": "English complete sentence",
    "confidence": "high/medium/low",
    "explanation": "简短说明识别到的关键信息"
}}

手语识别词汇序列:
{gloss_words}

请直接输出JSON格式的结果,不要包含其他文字:"""
        
        return prompt
    
    def translate_gloss_to_sentence(
        self, 
        gloss_words: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        将手语词汇转换为完整句子
        
        Args:
            gloss_words: 手语识别的词汇序列,空格分隔
            temperature: 生成温度,控制随机性
            max_tokens: 最大生成token数
            
        Returns:
            包含中英文翻译的字典:
            {
                'chinese': str,
                'english': str,
                'confidence': str,
                'explanation': str,
                'raw_response': str  # 原始API响应
            }
        """
        prompt = self._build_prompt(gloss_words)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": "message"
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取生成的文本
            if 'output' in result and 'choices' in result['output']:
                generated_text = result['output']['choices'][0]['message']['content']
                
                # 尝试解析JSON响应
                try:
                    # 清理可能的markdown代码块标记
                    generated_text = generated_text.strip()
                    if generated_text.startswith('```json'):
                        generated_text = generated_text[7:]
                    if generated_text.startswith('```'):
                        generated_text = generated_text[3:]
                    if generated_text.endswith('```'):
                        generated_text = generated_text[:-3]
                    generated_text = generated_text.strip()
                    
                    parsed_result = json.loads(generated_text)
                    
                    return {
                        'chinese': parsed_result.get('chinese', ''),
                        'english': parsed_result.get('english', ''),
                        'confidence': parsed_result.get('confidence', 'unknown'),
                        'explanation': parsed_result.get('explanation', ''),
                        'raw_response': generated_text,
                        'success': True
                    }
                except json.JSONDecodeError:
                    # 如果无法解析JSON,返回原始文本
                    return {
                        'chinese': '',
                        'english': '',
                        'confidence': 'unknown',
                        'explanation': '无法解析API响应',
                        'raw_response': generated_text,
                        'success': False,
                        'error': 'JSON解析失败'
                    }
            else:
                return {
                    'chinese': '',
                    'english': '',
                    'confidence': 'unknown',
                    'explanation': 'API响应格式异常',
                    'raw_response': str(result),
                    'success': False,
                    'error': 'API响应格式异常'
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'chinese': '',
                'english': '',
                'confidence': 'unknown',
                'explanation': f'API请求失败: {str(e)}',
                'raw_response': '',
                'success': False,
                'error': str(e)
            }
    
    def batch_translate(self, gloss_list: list) -> list:
        """
        批量翻译多个手语词汇序列
        
        Args:
            gloss_list: 手语词汇序列列表
            
        Returns:
            翻译结果列表
        """
        results = []
        for gloss_words in gloss_list:
            result = self.translate_gloss_to_sentence(gloss_words)
            results.append(result)
        return results


def test_qwen_api():
    """测试函数"""
    # 从环境变量读取API密钥
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("请设置环境变量DASHSCOPE_API_KEY")
        return
    
    # 创建API客户端
    client = QwenAPI(api_key=api_key)
    
    # 测试示例
    test_gloss = "__ON__ LIEB ZUSCHAUER ABEND WINTER NULL loc-REGION UEBERSCHWEMMUNG AMERIKA"
    
    print("测试手语词汇:", test_gloss)
    print("\n正在调用通义千问API...")
    
    result = client.translate_gloss_to_sentence(test_gloss)
    
    print("\n" + "="*80)
    print("翻译结果:")
    print("="*80)
    if result['success']:
        print(f"中文: {result['chinese']}")
        print(f"英文: {result['english']}")
        print(f"置信度: {result['confidence']}")
        print(f"说明: {result['explanation']}")
    else:
        print(f"翻译失败: {result.get('error', '未知错误')}")
        print(f"原始响应: {result['raw_response']}")
    print("="*80)


if __name__ == '__main__':
    test_qwen_api()
