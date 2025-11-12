# deepct/deepct/adaptor.py
# -*- coding: utf-8 -*-
"""
OrangePi / Ascend / 本地缓存 / ModelScope 一体化适配器。

Notebook / 脚本用法示例:

    from deepct import DeepCT, orangepiai_adaptor

    tokenizer, model, device = orangepiai_adaptor()
    dc = DeepCT(model, metrics=[
        "selfattn_cov_spectrum",
        "selfattn_repr_correlation",
        "intrinsic_dim",
    ])

所有和硬件、下载路径、ModelScope 等相关的逻辑都封装在这里。
"""

import os
import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===== 工具函数 =====

def _pick_device() -> str:
    """尝试选择 NPU -> CUDA -> CPU（仅作为优先级，不保证一定成功）。"""
    # Ascend / torch_npu
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    except Exception:
        pass

    # CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Fallback: CPU
    return "cpu"


def _ensure_modelscope_installed():
    """如果没有 modelscope，则自动 pip 安装一次。"""
    try:
        import modelscope  # noqa: F401
        return
    except ImportError:
        print("[Adaptor] modelscope 未安装，正在自动安装 modelscope ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "modelscope"],
        )
        print("[Adaptor] modelscope 安装完成")


def _download_with_modelscope(ms_model_id: str, cache_root: str) -> str:
    """
    使用 ModelScope 下载模型到 cache_root.
    返回实际模型目录路径。
    """
    _ensure_modelscope_installed()
    from modelscope import snapshot_download

    os.makedirs(cache_root, exist_ok=True)
    print(f"[Adaptor] 使用 ModelScope 下载模型: {ms_model_id}")
    model_dir = snapshot_download(
        ms_model_id,
        cache_dir=cache_root,
        local_files_only=False,
    )
    print(f"[Adaptor] ModelScope 下载完成，路径: {model_dir}")
    return model_dir


def _find_local_model_dir(candidates):
    """在候选列表中找第一个存在 config.json 的目录。"""
    for path in candidates:
        config_path = os.path.join(path, "config.json")
        if os.path.isfile(config_path):
            return path
    return None


# ===== 对外主函数 =====

def orangepiai_adaptor(
    ms_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    cache_root: str = "./model_cache/qwen",
    prefer_modelscope: bool = True,
):
    """
    OrangePi / Ascend 开发板适配入口（完全自动化）：

    1. 自动选 device (NPU > CUDA > CPU)
    2. 在 cache_root 下查找本地模型：
         - {cache_root}/Qwen2.5-0.5B-Instruct
         - {cache_root}/Qwen2___5-0___5B-Instruct
    3. 若未找到且 prefer_modelscope=True:
         - 自动安装 modelscope（若缺）
         - 用 ModelScope 下载 ms_model_id 到 cache_root
    4. 若仍失败，则直接用 transformers 的远程名（需外网）
    5. 从确定的路径加载 tokenizer / model
    6. 尝试将模型移动到选中设备；如失败，自动回退到 CPU
    7. 返回 (tokenizer, model, device)
    """
    # 先选一个“理想设备”
    device = _pick_device()

    # 常见的本地目录候选（按你现有路径调整）
    candidates = [
        os.path.join(cache_root, "Qwen2.5-0.5B-Instruct"),
        os.path.join(cache_root, "Qwen2___5-0___5B-Instruct"),
    ]

    local_dir = _find_local_model_dir(candidates)

    # 1) 本地已有，直接用
    if local_dir is not None:
        model_src = local_dir
        local_files_only = True
        print(f"[Adaptor] 使用本地模型目录: {model_src}")

    # 2) 尝试用 ModelScope 下载
    elif prefer_modelscope:
        try:
            model_src = _download_with_modelscope(ms_model_id, cache_root)
            local_files_only = True
        except Exception as e:
            print(f"[Adaptor] ModelScope 下载失败: {e}")
            print(f"[Adaptor] 将尝试直接使用远程模型标识: {ms_model_id}")
            model_src = ms_model_id
            local_files_only = False

    # 3) 直接远程（例如有公网 / 同事电脑）
    else:
        model_src = ms_model_id
        local_files_only = False

    print(f"[Adaptor] 准备从 {model_src} 加载模型 (local_files_only={local_files_only})")

    # 加载 tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(
        model_src,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_src,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype="auto",
    )

    # 尝试把模型搬到“理想设备”，失败则回退到 CPU
    target_device = device
    try:
        model.to(target_device)
    except Exception as e:
        print(f"[Adaptor] 模型迁移到 {target_device} 失败: {e}")
        print("[Adaptor] 自动回退到 CPU 执行（用于分析/画图足够）")
        target_device = "cpu"
        model.to(target_device)

    print(f"[Adaptor] 使用设备: {target_device}")
    return tokenizer, model, target_device
