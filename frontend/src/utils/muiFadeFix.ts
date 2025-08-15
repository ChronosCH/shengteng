/**
 * MUI Fade组件scrollTop错误的全局修复
 * 这个修复确保所有可能被Fade组件引用的DOM元素都有scrollTop属性
 */

// 修复MUI Fade组件的scrollTop错误
const originalQuerySelector = Document.prototype.querySelector;
const originalGetElementById = Document.prototype.getElementById;

// 重写querySelector，确保返回的元素有scrollTop属性
Document.prototype.querySelector = function(selector: string) {
  const element = originalQuerySelector.call(this, selector);
  if (element && typeof (element as any).scrollTop === 'undefined') {
    Object.defineProperty(element, 'scrollTop', {
      value: 0,
      writable: true,
      configurable: true
    });
  }
  return element;
};

// 重写getElementById，确保返回的元素有scrollTop属性
Document.prototype.getElementById = function(id: string) {
  const element = originalGetElementById.call(this, id);
  if (element && typeof (element as any).scrollTop === 'undefined') {
    Object.defineProperty(element, 'scrollTop', {
      value: 0,
      writable: true,
      configurable: true
    });
  }
  return element;
};

// 为已存在的元素添加scrollTop属性
function patchExistingElements() {
  const allElements = document.querySelectorAll('*');
  allElements.forEach(element => {
    if (typeof (element as any).scrollTop === 'undefined') {
      Object.defineProperty(element, 'scrollTop', {
        value: 0,
        writable: true,
        configurable: true
      });
    }
  });
}

// 在DOM加载完成后修复现有元素
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', patchExistingElements);
} else {
  patchExistingElements();
}

export {};
