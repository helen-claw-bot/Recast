#!/usr/bin/env python3
"""
Gemini Nano Banana 批量生图工具

用法:
  # 纯文生图，生成5张
  python scripts/gemini-gen.py --prompt "两个人站在天台上" --count 5

  # 带参考图生成
  python scripts/gemini-gen.py --prompt "基于参考图生成..." --ref photo1.png photo2.png --count 3

  # 图片编辑（基于一张底图）
  python scripts/gemini-gen.py --prompt "把两个人靠近一点" --edit base.png --count 3

  # 指定输出目录
  python scripts/gemini-gen.py --prompt "..." --count 5 --outdir output/shot04

环境变量:
  GEMINI_API_KEY  — Google AI Studio API key
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("需要 requests 库，安装中...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests


def encode_image(path: str) -> tuple[str, str]:
    """读取图片并返回 (base64, mime_type)"""
    p = Path(path)
    ext = p.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/png")
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def generate_image(
    api_key: str,
    prompt: str,
    ref_images: list[str] | None = None,
    edit_image: str | None = None,
    model: str = "gemini-2.0-flash-exp",
) -> bytes | None:
    """调用 Gemini API 生成图片，返回 PNG bytes 或 None"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    # 构建 parts
    parts = []

    # 参考图（多张）
    if ref_images:
        for i, img_path in enumerate(ref_images):
            data, mime = encode_image(img_path)
            parts.append({"text": f"Reference Photo {i+1}:"})
            parts.append({"inline_data": {"mime_type": mime, "data": data}})

    # 编辑底图
    if edit_image:
        data, mime = encode_image(edit_image)
        parts.append({"text": "Base image to edit:"})
        parts.append({"inline_data": {"mime_type": mime, "data": data}})

    # prompt
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 1.0,
        },
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        print(f"  请求失败: {e}")
        return None

    if resp.status_code == 429:
        print("  触发速率限制，等待 30 秒...")
        time.sleep(30)
        return generate_image(api_key, prompt, ref_images, edit_image, model)

    if resp.status_code != 200:
        print(f"  API 错误 {resp.status_code}: {resp.text[:500]}")
        return None

    result = resp.json()

    # 从响应中提取图片
    try:
        candidates = result["candidates"]
        for candidate in candidates:
            for part in candidate.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    img_data = part["inlineData"]["data"]
                    return base64.b64decode(img_data)
                if "inline_data" in part:
                    img_data = part["inline_data"]["data"]
                    return base64.b64decode(img_data)
    except (KeyError, IndexError):
        pass

    # 没有图片，打印文字响应
    try:
        for candidate in result.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    print(f"  模型回复（无图片）: {part['text'][:200]}")
    except Exception:
        pass

    print(f"  未返回图片。完整响应: {json.dumps(result, ensure_ascii=False)[:500]}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Gemini Nano Banana 批量生图")
    parser.add_argument("--prompt", "-p", required=True, help="生成 prompt")
    parser.add_argument("--ref", nargs="+", help="参考图片路径（可多张）")
    parser.add_argument("--edit", help="编辑模式：底图路径")
    parser.add_argument("--count", "-n", type=int, default=5, help="生成数量（默认5）")
    parser.add_argument("--outdir", "-o", default="output/gen", help="输出目录")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash-exp", help="模型名")
    parser.add_argument("--delay", type=float, default=5, help="每次请求间隔秒数（默认5）")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 GEMINI_API_KEY")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 生成时间戳前缀
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(f"📸 Gemini 批量生图")
    print(f"  模型: {args.model}")
    print(f"  数量: {args.count}")
    print(f"  输出: {outdir}/")
    if args.ref:
        print(f"  参考图: {', '.join(args.ref)}")
    if args.edit:
        print(f"  编辑底图: {args.edit}")
    print(f"  Prompt: {args.prompt[:100]}...")
    print()

    success = 0
    for i in range(args.count):
        print(f"[{i+1}/{args.count}] 生成中...")
        img_bytes = generate_image(
            api_key=api_key,
            prompt=args.prompt,
            ref_images=args.ref,
            edit_image=args.edit,
            model=args.model,
        )

        if img_bytes:
            filename = f"{timestamp}_{i+1:02d}.png"
            filepath = outdir / filename
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            size_kb = len(img_bytes) / 1024
            print(f"  ✅ 保存: {filepath} ({size_kb:.0f} KB)")
            success += 1
        else:
            print(f"  ❌ 失败")

        # 间隔避免速率限制
        if i < args.count - 1:
            time.sleep(args.delay)

    print(f"\n完成! {success}/{args.count} 张成功，保存在 {outdir}/")


if __name__ == "__main__":
    main()
