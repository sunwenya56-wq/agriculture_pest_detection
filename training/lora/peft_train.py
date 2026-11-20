"""Qwen2.5-7B 农业病虫害防治 LoRA 微调脚本（纯 transformers + peft + datasets）。

用法示例（路径按实际替换）：
    python training/lora/peft_train.py \
        --model_path models/Qwen2.5-7B-Instruct \
        --data_path training/lora/agri_pest_sft.jsonl \
        --output_dir models/qwen_agri_lora \
        --epochs 3 --batch_size 2 --grad_accum 8 --lr 2e-4 \
        --lora_rank 16 --max_len 1024 --quant 4bit

数据格式：UTF-8 JSONL，每行 {"messages": [{"role": ..., "content": ...}, ...]}，
最后一条必须是 assistant，训练损失只计算 assistant 回复部分。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
VALID_ROLES = {"system", "user", "assistant"}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要整数，收到 {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"必须 >= 1，收到 {parsed}")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要数字，收到 {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"必须 > 0，收到 {parsed}")
    return parsed


def _dropout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要数字，收到 {value!r}") from exc
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError(f"必须在 [0, 1) 内，收到 {parsed}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B 农业领域 LoRA 微调（纯 PEFT，不依赖 LLaMA-Factory）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="基座模型本地目录，或 Hugging Face 仓库 ID")
    parser.add_argument("--data_path", required=True, help="训练数据 JSONL 路径")
    parser.add_argument("--output_dir", required=True, help="LoRA adapter 输出目录")
    parser.add_argument("--epochs", type=_positive_int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=_positive_int, default=2, help="单卡 batch size")
    parser.add_argument("--grad_accum", type=_positive_int, default=8, help="梯度累积步数")
    parser.add_argument("--lr", type=_positive_float, default=2e-4, help="学习率")
    parser.add_argument("--lora_rank", type=_positive_int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=_positive_int, default=32, help="LoRA alpha（通常为 2×rank）")
    parser.add_argument("--lora_dropout", type=_dropout, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_len", type=_positive_int, default=1024, help="单条样本最大 token 数")
    parser.add_argument(
        "--quant",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="基座量化方式；4bit 为 QLoRA，需要 bitsandbytes 和 CUDA",
    )
    parser.add_argument(
        "--target_modules",
        default=DEFAULT_TARGET_MODULES,
        help="LoRA 注入的线性层，逗号分隔",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args(argv)


def fail(message: str) -> None:
    print(f"\n[错误] {message}\n", file=sys.stderr)
    raise SystemExit(2)


def _looks_like_hub_id(value: str) -> bool:
    return "/" in value and ":" not in value and not value.startswith(".")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = Path(args.model_path).expanduser()
    data_path = Path(args.data_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not model_path.exists() and not _looks_like_hub_id(args.model_path):
        fail(
            f"基座模型路径不存在：{model_path}\n"
            "请下载 Qwen2.5-7B-Instruct 到本地，或用 Hugging Face 仓库 ID（例如 Qwen/Qwen2.5-7B-Instruct）。"
        )
    if not model_path.exists():
        print(f"[提示] 本地未找到 {model_path}，将按仓库 ID 尝试在线加载。")

    if not data_path.is_file():
        fail(f"训练数据不存在：{data_path}\n请先准备 JSONL 数据，可参考 training/lora/dataset.example.jsonl。")
    if data_path.suffix.lower() not in {".jsonl", ".json"}:
        print(f"[提示] 数据文件扩展名为 {data_path.suffix}，仍按 JSONL 逐行解析。")

    if output_dir.exists() and (output_dir / "adapter_config.json").exists():
        fail(
            f"输出目录已存在训练产物：{output_dir}\n"
            "请换一个 --output_dir，或先备份/删除旧 adapter，避免覆盖。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return model_path, data_path, output_dir


def load_records(data_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"第 {lineno} 行 JSON 解析失败：{exc.msg}")
                continue

            messages = obj.get("messages") if isinstance(obj, dict) else None
            if not isinstance(messages, list) or not messages:
                errors.append(f"第 {lineno} 行缺少非空的 messages 数组")
                continue

            bad = False
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    errors.append(f"第 {lineno} 行第 {index + 1} 条消息不是对象")
                    bad = True
                    break
                if message.get("role") not in VALID_ROLES:
                    errors.append(f"第 {lineno} 行第 {index + 1} 条消息 role 非法：{message.get('role')!r}")
                    bad = True
                    break
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    errors.append(f"第 {lineno} 行第 {index + 1} 条消息 content 为空")
                    bad = True
                    break
            if bad:
                continue

            if messages[-1].get("role") != "assistant":
                errors.append(f"第 {lineno} 行最后一条消息必须是 assistant")
                continue
            if not any(item.get("role") == "user" for item in messages):
                errors.append(f"第 {lineno} 行至少需要一条 user 消息")
                continue
            records.append(obj)

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:5])
        more = f"\n  ... 另有 {len(errors) - 5} 条错误" if len(errors) > 5 else ""
        fail(f"数据校验失败，共 {len(errors)} 处问题：\n{preview}{more}")
    if not records:
        fail(f"数据文件没有有效样本：{data_path}")
    return records


def build_features(
    records: list[dict[str, Any]], tokenizer: Any, max_len: int
) -> tuple[list[dict[str, list[int]]], int]:
    features: list[dict[str, list[int]]] = []
    skipped = 0

    for record in records:
        messages = record["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(full_text, add_special_tokens=False, truncation=False)

        input_ids = list(encoded["input_ids"])
        if len(input_ids) > max_len:
            skipped += 1
            continue

        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = list(input_ids)
        labels[:prompt_len] = [-100] * prompt_len
        if all(label == -100 for label in labels):
            skipped += 1
            continue

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )

    if not features:
        fail(
            f"所有样本都超过 --max_len={max_len} 或被完全截断。\n"
            "请增大 --max_len（例如 1536/2048），或缩短 assistant 回复。"
        )
    return features, skipped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_path, data_path, output_dir = validate_args(args)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as exc:
        fail(f"缺少依赖：{exc.name}。请先执行 pip install -r requirements-train.txt")

    if args.quant != "none" and not torch.cuda.is_available():
        fail("--quant 8bit/4bit 需要 CUDA 设备；CPU 环境请使用 --quant none。")
    if args.quant != "none":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            fail(
                "量化加载需要 bitsandbytes，但当前环境未安装。\n"
                "请执行 pip install bitsandbytes；Windows 下如无可用轮子，请改用 --quant none。"
            )

    records = load_records(data_path)
    print(f"[数据] 有效样本 {len(records)} 条：{data_path}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[设备] {device_name}，显存 {total_gb:.1f} GB")
        if args.quant == "none" and total_gb < 20:
            print("[提示] 显存低于 20GB 时，7B bf16 LoRA 可能 OOM，建议 --quant 4bit 或减小 batch/max_len。")
    else:
        print("[提示] 未检测到 CUDA，将在 CPU 上训练（速度很慢，仅建议做小样本冒烟测试）。")

    if cuda_available and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif cuda_available:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        fail(
            f"分词器加载失败：{exc}\n"
            "请确认模型目录完整（含 tokenizer_config.json / tokenizer.json），或网络可访问 Hugging Face。"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": compute_dtype,
    }
    if args.quant == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print("[加载] 正在加载基座模型（首次加载可能需要数分钟）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    except Exception as exc:  # noqa: BLE001
        fail(
            f"基座模型加载失败：{exc}\n"
            "请检查模型文件是否完整、显存是否足够，以及 transformers / accelerate 版本是否匹配。"
        )
    model.config.use_cache = False

    if args.quant != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_names = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    if not target_names:
        fail("--target_modules 不能为空。")
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [name for name in target_names if name not in module_names]
    if missing:
        fail(
            "以下 target_modules 在模型中不存在：" + "、".join(missing) + "\n"
            "请检查模型结构，或改用 --target_modules q_proj,k_proj,v_proj,o_proj。"
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_names,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if args.quant == "none":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    features, skipped = build_features(records, tokenizer, args.max_len)
    if skipped:
        print(f"[数据] 因超过 max_len={args.max_len} 跳过 {skipped} 条；实际参与训练 {len(features)} 条。")
    if len(features) < 2:
        print("[提示] 有效样本少于 2 条，仅适合验证流程，不能代表训练效果。")

    dataset = Dataset.from_list(features)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    bf16_enabled = cuda_available and compute_dtype == torch.bfloat16
    fp16_enabled = cuda_available and compute_dtype == torch.float16
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.quant != "none" else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_pin_memory=cuda_available,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(
        f"[训练] epochs={args.epochs} batch={args.batch_size} grad_accum={args.grad_accum} "
        f"effective_batch={args.batch_size * args.grad_accum} lr={args.lr} "
        f"rank={args.lora_rank} alpha={args.lora_alpha} max_len={args.max_len} quant={args.quant}"
    )
    try:
        result = trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            fail(
                "显存不足。可依次尝试：--quant 4bit、--batch_size 1、--grad_accum 16、"
                "把 --max_len 降到 768，或缩短 assistant 回复。"
            )
        raise
    except KeyboardInterrupt:
        fail("训练被中断，未保存 adapter。")

    loss = float(result.training_loss)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n[完成] 训练 loss = {loss:.6f}")
    print(f"[完成] adapter 已保存到：{output_dir}")
    print("[下一步] 推理部署：AGRI_LLM_BACKEND=local，AGRI_LOCAL_ADAPTER_PATH 指向该目录。")
    print("[提示] 训练 loss 不等于验证集效果，请按 README 的评测流程做人工与 A/B 对比。")


if __name__ == "__main__":
    main()
