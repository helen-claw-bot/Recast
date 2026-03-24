# Recast 🎬

> 小说改编 AI 短剧 — 从文字到画面的重铸工坊

## 项目结构

```
Recast/
├── screenplay/        # 📜 剧本（对白、分集大纲、场景描述）
├── storyboard/        # 🎨 分镜设计（镜头语言、画面构图）
├── shots/             # 📸 剧照（AI 生成 prompt + 成品图）
├── characters/        # 👤 角色设计（人设文档、定妆照 prompt）
├── prompts/           # 🤖 AI Prompt 模板（通用/可复用）
├── audio/             # 🎙️ 音频素材
│   ├── raw/           #    原始录音 & 转写文本
│   └── vocals/        #    人声分离结果
├── scripts/           # 🔧 工具脚本（demucs, whisperx 等）
├── assets/            # 📦 素材（字体、参考图、风格板）
└── docs/              # 📝 项目文档 & 制作笔记
```

## 工具链

| 工具 | 用途 |
|------|------|
| demucs (htdemucs) | 人声/背景音分离 |
| whisperx | 语音转写 + 时间戳对齐 |
| resemblyzer | 说话人声纹聚类 |

## 技术备忘

- CPU-only 环境 (torch 2.8.0+cpu)
- demucs `--two-stems vocals` 模式
- whisperx `small` 模型, `int8` 量化
