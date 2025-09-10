/**
 * 输入验证和清理工具
 * 防止XSS攻击和数据注入
 */

// 输入验证规则
export interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  customValidator?: (value: string) => boolean;
  errorMessage?: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  sanitizedValue: string;
}

/**
 * HTML实体编码
 */
export function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * 移除HTML标签
 */
export function stripHtml(html: string): string {
  const div = document.createElement('div');
  div.innerHTML = html;
  return div.textContent || div.innerText || '';
}

/**
 * 清理输入数据
 */
export function sanitizeInput(input: string, maxLength: number = 1000): string {
  if (typeof input !== 'string') {
    throw new Error('输入必须是字符串类型');
  }

  // 长度限制
  if (input.length > maxLength) {
    input = input.substring(0, maxLength);
  }

  // HTML转义
  let sanitized = escapeHtml(input);

  // 移除潜在的危险字符序列
  const dangerousPatterns = [
    /javascript:/gi,
    /vbscript:/gi,
    /data:text\/html/gi,
    /on\w+\s*=/gi, // 事件处理器
    /<script[^>]*>.*?<\/script>/gi,
    /<style[^>]*>.*?<\/style>/gi,
    /<iframe[^>]*>.*?<\/iframe>/gi,
    /<object[^>]*>.*?<\/object>/gi,
    /<embed[^>]*>.*?<\/embed>/gi,
  ];

  dangerousPatterns.forEach(pattern => {
    sanitized = sanitized.replace(pattern, '');
  });

  // 移除多余的空白字符
  sanitized = sanitized.replace(/\s+/g, ' ').trim();

  return sanitized;
}

/**
 * 验证用户名
 */
export function validateUsername(username: string): ValidationResult {
  const errors: string[] = [];
  const sanitized = sanitizeInput(username, 50);

  if (!sanitized) {
    errors.push('用户名不能为空');
  } else {
    if (sanitized.length < 3) {
      errors.push('用户名至少需要3个字符');
    }
    if (sanitized.length > 50) {
      errors.push('用户名不能超过50个字符');
    }
    if (!/^[a-zA-Z0-9_\u4e00-\u9fa5]+$/.test(sanitized)) {
      errors.push('用户名只能包含字母、数字、下划线和中文字符');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedValue: sanitized
  };
}

/**
 * 验证邮箱
 */
export function validateEmail(email: string): ValidationResult {
  const errors: string[] = [];
  const sanitized = sanitizeInput(email, 254);

  if (!sanitized) {
    errors.push('邮箱地址不能为空');
  } else {
    const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!emailPattern.test(sanitized)) {
      errors.push('请输入有效的邮箱地址');
    }
    if (sanitized.length > 254) {
      errors.push('邮箱地址过长');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedValue: sanitized
  };
}

/**
 * 验证密码强度
 */
export function validatePassword(password: string): ValidationResult {
  const errors: string[] = [];
  
  if (!password) {
    errors.push('密码不能为空');
  } else {
    if (password.length < 8) {
      errors.push('密码至少需要8个字符');
    }
    if (password.length > 128) {
      errors.push('密码不能超过128个字符');
    }
    if (!/(?=.*[a-z])/.test(password)) {
      errors.push('密码必须包含至少一个小写字母');
    }
    if (!/(?=.*[A-Z])/.test(password)) {
      errors.push('密码必须包含至少一个大写字母');
    }
    if (!/(?=.*\d)/.test(password)) {
      errors.push('密码必须包含至少一个数字');
    }
    if (!/(?=.*[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?])/.test(password)) {
      errors.push('密码必须包含至少一个特殊字符');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedValue: password // 密码不进行清理，保持原样
  };
}

/**
 * 通用输入验证
 */
export function validateInput(value: string, rules: ValidationRule): ValidationResult {
  const errors: string[] = [];
  let sanitized = sanitizeInput(value, rules.maxLength || 1000);

  // 必填验证
  if (rules.required && !sanitized) {
    errors.push(rules.errorMessage || '此字段为必填项');
    return { isValid: false, errors, sanitizedValue: sanitized };
  }

  // 如果不是必填且为空，则跳过其他验证
  if (!rules.required && !sanitized) {
    return { isValid: true, errors: [], sanitizedValue: sanitized };
  }

  // 长度验证
  if (rules.minLength && sanitized.length < rules.minLength) {
    errors.push(rules.errorMessage || `至少需要${rules.minLength}个字符`);
  }

  if (rules.maxLength && sanitized.length > rules.maxLength) {
    errors.push(rules.errorMessage || `不能超过${rules.maxLength}个字符`);
  }

  // 模式验证
  if (rules.pattern && !rules.pattern.test(sanitized)) {
    errors.push(rules.errorMessage || '输入格式不正确');
  }

  // 自定义验证
  if (rules.customValidator && !rules.customValidator(sanitized)) {
    errors.push(rules.errorMessage || '输入不符合要求');
  }

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedValue: sanitized
  };
}

/**
 * 验证文件上传
 */
export function validateFile(file: File, options: {
  maxSize?: number; // 字节
  allowedTypes?: string[];
  allowedExtensions?: string[];
}): ValidationResult {
  const errors: string[] = [];

  if (!file) {
    errors.push('请选择文件');
    return { isValid: false, errors, sanitizedValue: '' };
  }

  // 文件大小验证
  if (options.maxSize && file.size > options.maxSize) {
    const maxSizeMB = Math.round(options.maxSize / (1024 * 1024));
    errors.push(`文件大小不能超过${maxSizeMB}MB`);
  }

  // 文件类型验证
  if (options.allowedTypes && !options.allowedTypes.includes(file.type)) {
    errors.push(`不支持的文件类型: ${file.type}`);
  }

  // 文件扩展名验证
  if (options.allowedExtensions) {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!options.allowedExtensions.includes(extension)) {
      errors.push(`不支持的文件扩展名: ${extension}`);
    }
  }

  // 文件名验证（防止路径遍历攻击）
  const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
  if (safeName !== file.name) {
    errors.push('文件名包含不安全字符');
  }

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedValue: safeName
  };
}

/**
 * 防止XSS的URL验证
 */
export function validateUrl(url: string): ValidationResult {
  const errors: string[] = [];
  const sanitized = sanitizeInput(url, 2048);

  if (!sanitized) {
    errors.push('URL不能为空');
  } else {
    try {
      const urlObj = new URL(sanitized);
      
      // 只允许HTTP和HTTPS协议
      if (!['http:', 'https:'].includes(urlObj.protocol)) {
        errors.push('只支持HTTP和HTTPS协议');
      }
      
      // 检查是否包含危险字符
      if (/[<>'"]/g.test(sanitized)) {
        errors.push('URL包含不安全字符');
      }
      
    } catch (e) {
      errors.push('无效的URL格式');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    sanitizedValue: sanitized
  };
}

/**
 * 批量验证表单数据
 */
export function validateForm(data: Record<string, any>, rules: Record<string, ValidationRule>): {
  isValid: boolean;
  errors: Record<string, string[]>;
  sanitizedData: Record<string, any>;
} {
  const errors: Record<string, string[]> = {};
  const sanitizedData: Record<string, any> = {};
  let isValid = true;

  for (const [field, value] of Object.entries(data)) {
    if (rules[field]) {
      const result = validateInput(String(value || ''), rules[field]);
      if (!result.isValid) {
        errors[field] = result.errors;
        isValid = false;
      }
      sanitizedData[field] = result.sanitizedValue;
    } else {
      // 对于没有规则的字段，仍然进行基本清理
      sanitizedData[field] = typeof value === 'string' ? sanitizeInput(value) : value;
    }
  }

  return { isValid, errors, sanitizedData };
}
