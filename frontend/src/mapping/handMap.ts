// 手部关键点到骨骼名称映射（解剖学命名）
// Mediapipe 21点：0 wrist; 拇指 1-4；食指 5-8；中指 9-12；无名指 13-16；小指 17-20
export interface FingerBoneMap {
  joints: number[]; // mediapipe indices
  bones: string[];  // bone segments between those joints (tip无骨骼)
}

export interface HandBoneMapping {
  wrist: string;
  fingers: Record<string, FingerBoneMap>;
}

// 说明：大多数标准人形骨架命名采用 Proximal/Middle/Distal Phalanx；
// 拇指只有 Proximal/Distal（无 Middle/Intermediate）。
// 若你的模型骨骼名不同（如 "Index1/2/3"），只需把 bones 数组里的字符串替换为模型里的真实名称即可。

export const LEFT_HAND_BONE_MAPPING: HandBoneMapping = {
  // 常见命名：Hand.L / Wrist.L / hand_l 等，请按你的模型实际名称修改
  wrist: 'Hand.L',
  fingers: {
    thumb:  { joints: [1, 2, 3, 4], bones: ['ThumbMetacarpal.L', 'ThumbProximalPhalanx.L', 'ThumbDistalPhalanx.L'] },
    index:  { joints: [5, 6, 7, 8], bones: ['IndexProximalPhalanx.L', 'IndexMiddlePhalanx.L', 'IndexDistalPhalanx.L'] },
    middle: { joints: [9, 10, 11, 12], bones: ['MiddleProximalPhalanx.L', 'MiddleMiddlePhalanx.L', 'MiddleDistalPhalanx.L'] },
    ring:   { joints: [13, 14, 15, 16], bones: ['RingProximalPhalanx.L', 'RingMiddlePhalanx.L', 'RingDistalPhalanx.L'] },
    pinky:  { joints: [17, 18, 19, 20], bones: ['PinkyProximalPhalanx.L', 'PinkyMiddlePhalanx.L', 'PinkyDistalPhalanx.L'] },
  },
};

export const RIGHT_HAND_BONE_MAPPING: HandBoneMapping = {
  wrist: 'Hand.R',
  fingers: {
    thumb:  { joints: [1, 2, 3, 4], bones: ['ThumbMetacarpal.R', 'ThumbProximalPhalanx.R', 'ThumbDistalPhalanx.R'] },
    index:  { joints: [5, 6, 7, 8], bones: ['IndexProximalPhalanx.R', 'IndexMiddlePhalanx.R', 'IndexDistalPhalanx.R'] },
    middle: { joints: [9, 10, 11, 12], bones: ['MiddleProximalPhalanx.R', 'MiddleMiddlePhalanx.R', 'MiddleDistalPhalanx.R'] },
    ring:   { joints: [13, 14, 15, 16], bones: ['RingProximalPhalanx.R', 'RingMiddlePhalanx.R', 'RingDistalPhalanx.R'] },
    pinky:  { joints: [17, 18, 19, 20], bones: ['PinkyProximalPhalanx.R', 'PinkyMiddlePhalanx.R', 'PinkyDistalPhalanx.R'] },
  },
};

// 关节屈曲角度范围（近似值，按需要微调）；key 使用去掉左右后缀的通用段名
export const JOINT_LIMITS: Record<string, { flex: [number, number] }> = {
  // 四指（食/中/无名/小）
  ProximalPhalanx:  { flex: [0, 95] },   // MCP→PIP 段
  MiddlePhalanx:    { flex: [0, 105] },  // PIP→DIP 段
  DistalPhalanx:    { flex: [0, 80] },   // DIP→Tip 段

  // 拇指（无 Middle）
  ThumbMetacarpal:      { flex: [-15, 45] }, // CMC屈伸（仿射近似，常用于拇指张合）
  ThumbProximalPhalanx: { flex: [0, 80] },   // MCP→IP
  ThumbDistalPhalanx:   { flex: [0, 80] },   // IP→Tip
};
