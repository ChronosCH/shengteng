/**
 * WebXR Type Definitions
 * WebXR API 类型定义
 */

declare global {
  interface Navigator {
    xr?: XRSystem;
  }

  interface XRSystem {
    isSessionSupported(mode: XRSessionMode): Promise<boolean>;
    requestSession(mode: XRSessionMode, options?: XRSessionInit): Promise<XRSession>;
  }

  type XRSessionMode = 'inline' | 'immersive-vr' | 'immersive-ar';

  interface XRSessionInit {
    requiredFeatures?: string[];
    optionalFeatures?: string[];
    domOverlay?: {
      root: Element;
    };
  }

  interface XRSession extends EventTarget {
    mode: XRSessionMode;
    inputSources: XRInputSourceArray;
    environmentBlendMode: XREnvironmentBlendMode;
    
    requestReferenceSpace(type: XRReferenceSpaceType): Promise<XRReferenceSpace>;
    requestAnimationFrame(callback: XRFrameRequestCallback): number;
    cancelAnimationFrame(handle: number): void;
    end(): Promise<void>;

    addEventListener(type: 'end', listener: (event: XRSessionEvent) => void): void;
    addEventListener(type: 'inputsourceschange', listener: (event: XRInputSourceChangeEvent) => void): void;
    addEventListener(type: 'select', listener: (event: XRInputSourceEvent) => void): void;
    addEventListener(type: 'selectstart', listener: (event: XRInputSourceEvent) => void): void;
    addEventListener(type: 'selectend', listener: (event: XRInputSourceEvent) => void): void;
  }

  type XREnvironmentBlendMode = 'opaque' | 'additive' | 'alpha-blend';
  type XRReferenceSpaceType = 'viewer' | 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';

  interface XRReferenceSpace extends XRSpace {
    getOffsetReferenceSpace(originOffset: XRRigidTransform): XRReferenceSpace;
  }

  interface XRSpace extends EventTarget {}

  interface XRRigidTransform {
    position: DOMPointReadOnly;
    orientation: DOMPointReadOnly;
    matrix: Float32Array;
    inverse: XRRigidTransform;
  }

  type XRFrameRequestCallback = (time: DOMHighResTimeStamp, frame: XRFrame) => void;

  interface XRFrame {
    session: XRSession;
    getViewerPose(referenceSpace: XRReferenceSpace): XRViewerPose | null;
    getPose(space: XRSpace, baseSpace: XRSpace): XRPose | null;
  }

  interface XRViewerPose extends XRPose {
    views: XRView[];
  }

  interface XRPose {
    transform: XRRigidTransform;
    emulatedPosition: boolean;
  }

  interface XRView {
    eye: XREye;
    projectionMatrix: Float32Array;
    transform: XRRigidTransform;
  }

  type XREye = 'left' | 'right' | 'none';

  interface XRInputSourceArray extends Array<XRInputSource> {
    [Symbol.iterator](): IterableIterator<XRInputSource>;
  }

  interface XRInputSource {
    handedness: XRHandedness;
    targetRayMode: XRTargetRayMode;
    targetRaySpace: XRSpace;
    gripSpace?: XRSpace;
    gamepad?: Gamepad;
    profiles: string[];
  }

  type XRHandedness = 'none' | 'left' | 'right';
  type XRTargetRayMode = 'gaze' | 'tracked-pointer' | 'screen';

  interface XRSessionEvent extends Event {
    session: XRSession;
  }

  interface XRInputSourceEvent extends Event {
    frame: XRFrame;
    inputSource: XRInputSource;
  }

  interface XRInputSourceChangeEvent extends Event {
    session: XRSession;
    added: XRInputSource[];
    removed: XRInputSource[];
  }

  // WebGL Extensions for WebXR
  interface WebGLRenderingContext {
    makeXRCompatible(): Promise<void>;
  }

  interface WebGL2RenderingContext {
    makeXRCompatible(): Promise<void>;
  }

  // Three.js WebXR Extensions
  namespace THREE {
    interface WebGLRenderer {
      xr: WebXRManager;
    }

    interface WebXRManager {
      enabled: boolean;
      isPresenting: boolean;
      setSession(session: XRSession | null): Promise<void>;
      getSession(): XRSession | null;
      setReferenceSpaceType(type: XRReferenceSpaceType): void;
      getReferenceSpace(): XRReferenceSpace | null;
      setFramebufferScaleFactor(scale: number): void;
      getFramebufferScaleFactor(): number;
      addEventListener(type: 'sessionstart', listener: () => void): void;
      addEventListener(type: 'sessionend', listener: () => void): void;
      removeEventListener(type: 'sessionstart', listener: () => void): void;
      removeEventListener(type: 'sessionend', listener: () => void): void;
    }
  }
}

export {};
