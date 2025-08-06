/**
 * WebXR Service for AR Sign Language Overlay
 * WebXR 增强现实手语叠加服务
 */

export interface XRSessionConfig {
  mode: 'immersive-ar' | 'immersive-vr' | 'inline';
  requiredFeatures?: string[];
  optionalFeatures?: string[];
}

export interface AROverlayConfig {
  avatarScale: number;
  avatarPosition: { x: number; y: number; z: number };
  opacity: number;
  followCamera: boolean;
  showBackground: boolean;
}

export interface XRSessionInfo {
  isSupported: boolean;
  isActive: boolean;
  mode: string | null;
  inputSources: XRInputSource[];
  environmentBlendMode: string | null;
}

class WebXRService {
  private xrSession: XRSession | null = null;
  private xrReferenceSpace: XRReferenceSpace | null = null;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.Camera | null = null;
  private avatarGroup: THREE.Group | null = null;
  private isInitialized = false;
  private animationFrameId: number | null = null;

  constructor() {
    this.checkWebXRSupport();
  }

  /**
   * 检查 WebXR 支持
   */
  async checkWebXRSupport(): Promise<boolean> {
    if (!navigator.xr) {
      console.warn('WebXR not supported in this browser');
      return false;
    }

    try {
      const isARSupported = await navigator.xr.isSessionSupported('immersive-ar');
      const isVRSupported = await navigator.xr.isSessionSupported('immersive-vr');
      
      console.log('WebXR Support:', { AR: isARSupported, VR: isVRSupported });
      return isARSupported || isVRSupported;
    } catch (error) {
      console.error('Error checking WebXR support:', error);
      return false;
    }
  }

  /**
   * 初始化 WebXR 会话
   */
  async initializeSession(config: XRSessionConfig = { mode: 'immersive-ar' }): Promise<boolean> {
    if (!navigator.xr) {
      throw new Error('WebXR not supported');
    }

    try {
      // 请求 XR 会话
      this.xrSession = await navigator.xr.requestSession(config.mode, {
        requiredFeatures: config.requiredFeatures || ['local'],
        optionalFeatures: config.optionalFeatures || ['dom-overlay', 'hit-test', 'anchors']
      });

      // 设置参考空间
      this.xrReferenceSpace = await this.xrSession.requestReferenceSpace('local');

      // 初始化渲染器
      await this.initializeRenderer();

      // 设置事件监听器
      this.setupEventListeners();

      this.isInitialized = true;
      console.log('WebXR session initialized successfully');
      return true;

    } catch (error) {
      console.error('Failed to initialize WebXR session:', error);
      throw error;
    }
  }

  /**
   * 初始化 Three.js 渲染器
   */
  private async initializeRenderer(): Promise<void> {
    if (!this.xrSession) {
      throw new Error('XR session not available');
    }

    // 创建 WebGL 上下文
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', { xrCompatible: true });
    
    if (!gl) {
      throw new Error('WebGL2 not supported');
    }

    // 创建 Three.js 渲染器
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      context: gl,
      alpha: true,
      antialias: true
    });

    if (this.renderer) {
      this.renderer.setPixelRatio(window.devicePixelRatio);
      this.renderer.xr.enabled = true;
      this.renderer.xr.setSession(this.xrSession);
    }

    // 创建场景
    this.scene = new THREE.Scene();

    // 创建相机
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

    // 添加光照
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    this.scene.add(directionalLight);

    // 创建 Avatar 组
    this.avatarGroup = new THREE.Group();
    this.scene.add(this.avatarGroup);
  }

  /**
   * 设置事件监听器
   */
  private setupEventListeners(): void {
    if (!this.xrSession) return;

    this.xrSession.addEventListener('end', () => {
      this.cleanup();
    });

    this.xrSession.addEventListener('inputsourceschange', (event) => {
      console.log('Input sources changed:', event);
    });

    this.xrSession.addEventListener('select', (event) => {
      this.handleSelect(event);
    });
  }

  /**
   * 处理选择事件
   */
  private handleSelect(event: XRInputSourceEvent): void {
    console.log('XR Select event:', event);
    // 可以在这里处理用户交互，比如移动 Avatar 位置
  }

  /**
   * 开始渲染循环
   */
  startRenderLoop(): void {
    if (!this.renderer || !this.scene || !this.camera) {
      throw new Error('Renderer not initialized');
    }

    const animate = (timestamp: number, frame?: XRFrame) => {
      if (!this.xrSession || !this.renderer || !this.scene || !this.camera) {
        return;
      }

      if (frame) {
        // XR 渲染
        const pose = frame.getViewerPose(this.xrReferenceSpace!);
        
        if (pose) {
          // 更新相机矩阵
          this.camera.matrix.fromArray(pose.transform.matrix);
          this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);

          // 更新 Avatar 位置（可选：让 Avatar 跟随用户视角）
          this.updateAvatarPosition(pose);
        }

        this.renderer.render(this.scene, this.camera);
      }

      this.animationFrameId = this.xrSession.requestAnimationFrame(animate);
    };

    this.xrSession!.requestAnimationFrame(animate);
  }

  /**
   * 更新 Avatar 位置
   */
  private updateAvatarPosition(pose: XRViewerPose): void {
    if (!this.avatarGroup) return;

    // 将 Avatar 放置在用户前方 2 米处
    const transform = pose.transform;
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    
    // 从变换矩阵中提取位置和旋转
    const matrix = new THREE.Matrix4().fromArray(transform.matrix);
    matrix.decompose(position, quaternion, new THREE.Vector3());

    // 计算前方位置
    const forward = new THREE.Vector3(0, 0, -2);
    forward.applyQuaternion(quaternion);
    
    this.avatarGroup.position.copy(position.add(forward));
    this.avatarGroup.quaternion.copy(quaternion);
  }

  /**
   * 添加手语 Avatar 到 AR 场景
   */
  addSignLanguageAvatar(avatarMesh: THREE.Object3D, config: AROverlayConfig): void {
    if (!this.avatarGroup) {
      throw new Error('Avatar group not initialized');
    }

    // 清除现有 Avatar
    this.avatarGroup.clear();

    // 应用配置
    avatarMesh.scale.setScalar(config.avatarScale);
    avatarMesh.position.set(config.avatarPosition.x, config.avatarPosition.y, config.avatarPosition.z);

    // 设置透明度
    avatarMesh.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach(mat => {
            mat.transparent = true;
            mat.opacity = config.opacity;
          });
        } else {
          child.material.transparent = true;
          child.material.opacity = config.opacity;
        }
      }
    });

    this.avatarGroup.add(avatarMesh);
    console.log('Sign language avatar added to AR scene');
  }

  /**
   * 更新 Avatar 动画
   */
  updateAvatarAnimation(keyframes: number[][][]): void {
    if (!this.avatarGroup || keyframes.length === 0) return;

    // 这里可以实现关键点到 3D 模型的映射
    // 简化实现：只更新整体位置和旋转
    const firstFrame = keyframes[0];
    if (firstFrame && firstFrame.length > 0) {
      // 使用第一个关键点作为参考
      const referencePoint = firstFrame[0];
      
      // 更新 Avatar 位置（这里需要根据实际的关键点映射来实现）
      this.avatarGroup.position.x += (referencePoint[0] - 0.5) * 0.1;
      this.avatarGroup.position.y += (0.5 - referencePoint[1]) * 0.1;
    }
  }

  /**
   * 设置 AR 叠加配置
   */
  updateOverlayConfig(config: Partial<AROverlayConfig>): void {
    if (!this.avatarGroup) return;

    if (config.avatarScale !== undefined) {
      this.avatarGroup.scale.setScalar(config.avatarScale);
    }

    if (config.avatarPosition) {
      this.avatarGroup.position.set(
        config.avatarPosition.x,
        config.avatarPosition.y,
        config.avatarPosition.z
      );
    }

    if (config.opacity !== undefined) {
      this.avatarGroup.traverse((child) => {
        if (child instanceof THREE.Mesh && child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(mat => {
              mat.opacity = config.opacity!;
            });
          } else {
            child.material.opacity = config.opacity!;
          }
        }
      });
    }
  }

  /**
   * 获取会话信息
   */
  getSessionInfo(): XRSessionInfo {
    return {
      isSupported: !!navigator.xr,
      isActive: !!this.xrSession,
      mode: this.xrSession?.mode || null,
      inputSources: this.xrSession ? Array.from(this.xrSession.inputSources) : [],
      environmentBlendMode: this.xrSession?.environmentBlendMode || null,
    };
  }

  /**
   * 结束 XR 会话
   */
  async endSession(): Promise<void> {
    if (this.xrSession) {
      await this.xrSession.end();
    }
  }

  /**
   * 清理资源
   */
  private cleanup(): void {
    if (this.animationFrameId) {
      if (this.xrSession) {
        this.xrSession.cancelAnimationFrame(this.animationFrameId);
      }
      this.animationFrameId = null;
    }

    if (this.renderer) {
      this.renderer.dispose();
      this.renderer = null;
    }

    this.scene = null;
    this.camera = null;
    this.avatarGroup = null;
    this.xrSession = null;
    this.xrReferenceSpace = null;
    this.isInitialized = false;

    console.log('WebXR session cleaned up');
  }

  /**
   * 检查是否已初始化
   */
  isReady(): boolean {
    return this.isInitialized && !!this.xrSession;
  }
}

// 导出单例实例
export const webxrService = new WebXRService();
export default webxrService;
