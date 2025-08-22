from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "knowledge-map-auto.md"
IGNORE_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".mypy_cache", ".ruff_cache"}

CATEGORIES = {
    "配置与入口": [
        r'if __name__ == ["\']__main__["\']',
        r"\bargparse\b|\btyper\b",
        r"\bBaseSettings\b|\bpydantic\b",
        r"yaml\.safe_load|\b\.env\b|\bConfigLoader\b",
    ],
    "数据与预处理": [
        r"class\s+\w*Dataset\b",
        r"\b__getitem__\b|\bGeneratorDataset\b",
        r"\bcv2\b|\bmediapipe\b|\bpandas\b",
        r"\bnormalize\b|\bstandardize\b|\bpad\b|\bmask\b",
    ],
    "模型定义": [
        r"class\s+\w+\((?:nn\.Cell|nn\.Module)\)",
        r"\bdef\s+construct\b|\bmindspore\.nn\b|\bops\.",
        r"\bLSTM\b|\bTransformer\b|\bAttention\b|\bTCN\b",
    ],
    "训练循环": [
        r"\bfor\s+epoch\b|\btrain_step\b|\bfit\(",
        r"\boptimizer\b|\bAdamW?\b|\bMomentum\b",
        r"\bset_context\b|\bamp_level\b|\bloss[_ ]?scale\b",
    ],
    "损失与评估": [
        r"\bFocal\b|\bLabelSmooth|\bCrossEntropy",
        r"\bconfusion\b|\bprecision\b|\brecall\b|\bf1\b",
        r"\bAUC\b|\bROC\b|\bsklearn\.metrics\b",
    ],
    "学习率与训练策略": [
        r"\bcosine\b|\bWarmup\b|\bOneCycle\b|\bReduceLROnPlateau\b",
        r"\bEarlyStop\b|\bearly stop\b|\bEMA\b|\bgrad(?:ient)? clip",
    ],
    "昇腾与分布式": [
        r'device_target\s*=\s*["\']Ascend["\']',
        r"\bset_auto_parallel_context\b|\binit\(",
        r"\bHCCL\b|\bhccl\b",
    ],
    "推理与服务": [
        r"\bFastAPI\b|\bAPIRouter\b|\bWebSocket\b|\buvicorn\b",
        r"\bMindIR\b|\bom\b|\bacl\b|\bload_model\b|\bpredict\b|\binference\b",
    ],
    "联邦学习": [
        r"\bFedAvg\b|\bFederated\b|\bModelUpdate\b|\bFederatedRound\b",
    ],
    "部署与运维": [
        r"\bDockerfile\b|\bdocker-compose\.ya?ml\b",
        r"\brequirements\.txt\b|\bpyproject\.toml\b|\bsetup\.py\b",
        r"\bdeploy\b|\bstart\.(?:sh|ps1)\b",
    ],
    "文档与指南": [
        r"#\s|^---\s*$|\bREADME\.md\b|\bdevelopment[-_]guide\b",
    ],
}

TEXT_EXT = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".ini", ".txt"}
SPECIAL_FILENAMES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml"}

def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in IGNORE_DIRS:
                # Skip entire subtree
                for _ in p.rglob("*"):
                    pass
                continue
            continue
        if p.name in SPECIAL_FILENAMES or p.suffix in TEXT_EXT:
            # heuristically skip large files
            if p.stat().st_size > 2_000_000:
                continue
            yield p

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def main():
    results = {k: [] for k in CATEGORIES}
    for f in iter_files(ROOT):
        txt = read_text(f)
        if not txt:
            continue
        for cat, patterns in CATEGORIES.items():
            for pat in patterns:
                if re.search(pat, txt, re.IGNORECASE | re.MULTILINE):
                    results[cat].append(str(f.relative_to(ROOT)))
                    break  # file matched this category; avoid duplicates

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as w:
        w.write("# 仓库知识点自动映射（草稿）\n\n")
        w.write(f"根目录：{ROOT}\n\n")
        for cat, files in results.items():
            w.write(f"## {cat}\n\n")
            if files:
                for path in sorted(set(files)):
                    w.write(f"- {path}\n")
            else:
                w.write("- （未扫描到，可能在非文本文件或命名不一致）\n")
            w.write("\n")
    print(f"Wrote: {OUT}")

if __name__ == "__main__":
    main()