# Qwen2.5-7B 农业领域 LoRA 微调（训练侧说明）

面向面试官 / 接手同事的工程说明：本目录只负责「把基座模型适配到病虫害防治问答」的训练流程，检测侧 YOLOv11 与推理侧 `LocalQwenClient` 不在本目录内。

---

## 1. 任务定义与数据来源

### 1.1 任务

输入检测侧给出的害虫名称（可多个），输出一份可直接执行的结构化防治方案，固定包含五个维度：

1. 农业防治
2. 物理防治
3. 生物防治
4. 化学防治（药剂通用名、施药时机、安全间隔期、轮换建议）
5. 预防与监测

不做害虫识别本身，也不替代检测模型；LLM 只负责「检测结果 → 防治建议」的文本生成。

### 1.2 12 类害虫（与检测侧严格对齐）

类别定义来自 `ultralytics-main/datasets/MyData/mydata.yaml`，中文名映射来自 `app/adapters/agri_kb.py`：

| 序号 | 训练集英文名 | 中文名 |
|---|---|---|
| 0 | Aug_Pericallia_ricini | 蓖麻蚕 |
| 1 | field_cricket | 田蟋蟀 |
| 2 | Aug_Spodoptera_exigua | 甜菜夜蛾 |
| 3 | Aug_Microtermes_obesi | 稻白蚁 |
| 4 | Aug_Tetranychus_bioculatus | 二斑叶螨 |
| 5 | cutworm | 地老虎 |
| 6 | Aug_Nupserha_bicolor | 双色天牛 |
| 7 | Jute_stem_weevil | 黄麻茎象甲 |
| 8 | Aug_Scopula_emissaria | 尺蠖 |
| 9 | Aug_Luperomorpha_vittata | 条纹跳甲 |
| 10 | Aug_Ferisia_pseudococcus | 拟粉蚧 |
| 11 | Aug_Earias_cupreoviridis | 铜绿实夜蛾 |

### 1.3 数据来源与边界

- 类别与中文名：检测项目的 `mydata.yaml` + `agri_kb.py`，不另造类别。
- 防治知识种子：`app/adapters/agri_kb.py` 的规则知识库 + 公开植保资料 + 农药登记标签的通用名与安全间隔期。
- **示例数据是格式与写作规范的样板，不是可直接上线的最终训练集**。正式训练集需按第 3 节扩充，并由植保专业人员逐条复核。
- 不把「模型自己生成的内容」再喂回训练集，避免错误累积。

---

## 2. 环境与硬件

### 2.1 软件

- Python 3.10+
- 训练依赖见项目根目录 `requirements-train.txt`（torch / transformers / peft / accelerate / datasets / llama-factory）
- QLoRA 额外需要 `bitsandbytes`；Windows 下若没有可用轮子，改用 `--quant none` 或 Linux 环境

### 2.2 硬件

- 单卡 24GB 起步（RTX 3090 / 4090 / A10 等）
- 7B 全参微调不可行：bf16 权重约 14GB，加上梯度、AdamW 的 fp32 状态和激活，训练状态在 100GB 量级
- LoRA 只训练约 0.2%–0.5% 的参数，24GB 可跑 bf16 LoRA（`--quant none`）
- 4bit QLoRA 把基座压到约 4–5GB，显存紧张时优先用 `--quant 4bit`，仍建议 16GB 以上
- 纯 CPU 只能做小样本冒烟测试，不适合正式训练

### 2.3 安装

以下命令在项目根目录执行；先把 `$PROJECT` 替换成你的实际路径：

```powershell
$PROJECT = "C:\path\to\agriculture_pest_detection"
Set-Location $PROJECT

python -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-train.txt

# 可选：QLoRA 需要
pip install bitsandbytes
```

如果 PowerShell 禁止激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

下载基座模型（国内网络可先设镜像）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')"
```

---

## 3. 数据格式与构造规范

### 3.1 文件格式

UTF-8 JSONL，一行一个样本，禁止跨行；每条固定为 system / user / assistant 三段：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

完整样板见 `training/lora/dataset.example.jsonl`，可直接复制为 `agri_pest_sft.jsonl` 后扩写。

### 3.2 三段怎么写

**system 段**必须与推理侧 `app/services/advice.py` 的 `SYSTEM_PROMPT` 逐字一致：

```text
你是一位农业植保专家，熟悉中国主要农作物病虫害的综合防治（IPM）。回答必须科学、可操作、环保，避免使用未登记的农药。
```

原因：`LocalQwenClient.generate()` 推理时只会传这段 system。训练和推理的 system 不一致，模型会学到推理时看不到的指令，属于典型的训练/ serving 偏移。

**user 段**复刻 `app/services/advice.py` 的 `build_prompt()`，只把害虫名替换掉：

```text
检测到的病虫害：甜菜夜蛾

请按以下结构给出防治方案，语言精炼、可直接执行：
1. 农业防治措施
2. 物理防治方法
3. 生物防治手段
4. 化学防治方案（药剂通用名、用量区间、施药时机与安全间隔期）
5. 预防与监测建议

要求：
- 优先非化学措施；化学防治必须给出轮换用药建议
- 不确定的剂量用区间表述，不要编造具体品牌
```

多害虫用「、」连接；未知类别写「未知害虫」，用于训练安全拒答。

**assistant 段**必须包含以下五个标题，缺一不可：

```markdown
### 农业防治
### 物理防治
### 生物防治
### 化学防治
### 预防与监测
```

写作硬性规范：

- 化学防治只写**农药通用名**（如甲氨基阿维菌素苯甲酸盐、氯虫苯甲酰胺、螺螨酯），不写品牌、不写经销商。
- 必须写**安全间隔期**，用区间 + 「以标签为准」，因为同一有效成分在不同作物、不同剂型上的登记不同。
- 不写死具体稀释倍数或每亩克数；统一表述为「按登记标签推荐剂量」。示例数据里不出现无法核验的精确剂量。
- 不同害虫的措施必须有区分度：寄主、越冬场所、虫态、施药部位、天敌种类不能千篇一律。
- 化学防治必须带轮换用药或抗性管理建议。
- 未知害虫、低置信度、非农业问题必须有明确的拒答/降级话术。

### 3.3 规模与切分

- 12 类各 80–150 条，合计约 1000–1800 条；多害虫组合占 10% 左右；拒答与边界问题占 5%。
- 按「作物 + 场景 + 害虫组合」分组切分 train/val/test，不要按行随机切分，否则同一模板的改写会同时出现在训练和验证集，指标虚高。
- 验证集至少 100 条，且覆盖 12 类与全部边界类型。
- 数据脱敏：不含农户姓名、电话、精确地块坐标；图片 EXIF 在入库前清除；不把 API Key、内网地址写进样本。

---

## 4. 训练命令

两种方式二选一，产物都是 PEFT adapter，推理侧加载方式相同。

### 4.1 方式 A：LLaMA-Factory

先准备数据并注册数据集：

```powershell
Copy-Item training\lora\dataset.example.jsonl training\lora\agri_pest_sft.jsonl
```

把下面的内容保存为 `training/lora/dataset_info.json`（UTF-8 **无 BOM**）：

```json
{
  "agri_pest_sft": {
    "file_name": "agri_pest_sft.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

也可以用 PowerShell 一次写入，避免 BOM：

```powershell
$json = @'
{
  "agri_pest_sft": {
    "file_name": "agri_pest_sft.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
'@
[System.IO.File]::WriteAllText((Join-Path $PROJECT "training\lora\dataset_info.json"), $json, (New-Object System.Text.UTF8Encoding($false)))
```

训练：

```powershell
llamafactory-cli train training\lora\llamafactory_sft.yaml
```

产物在 `models/qwen_agri_lora/`，核心文件是 `adapter_model.safetensors` 与 `adapter_config.json`。

### 4.2 方式 B：纯 PEFT 脚本

脚本入口：`training/lora/peft_train.py`，不依赖 LLaMA-Factory，参数全部显式传入，不硬编码路径。

24GB 显存、bf16 LoRA：

```powershell
python training\lora\peft_train.py --model_path models/Qwen2.5-7B-Instruct --data_path training\lora\agri_pest_sft.jsonl --output_dir models/qwen_agri_lora --epochs 3 --batch_size 2 --grad_accum 8 --lr 2e-4 --lora_rank 16 --max_len 1024 --quant none
```

显存紧张时用 4bit QLoRA：

```powershell
python training\lora\peft_train.py --model_path models/Qwen2.5-7B-Instruct --data_path training\lora\agri_pest_sft.jsonl --output_dir models\qwen_agri_lora_4bit --epochs 3 --batch_size 1 --grad_accum 16 --lr 2e-4 --lora_rank 16 --max_len 1024 --quant 4bit
```

先跑通流程（1 轮、短序列）：

```powershell
python training\lora\peft_train.py --model_path models/Qwen2.5-7B-Instruct --data_path training\lora\agri_pest_sft.jsonl --output_dir models\qwen_agri_lora_smoke --epochs 1 --batch_size 1 --grad_accum 4 --lr 2e-4 --lora_rank 8 --max_len 512 --quant none
```

脚本会做这些校验并给出明确错误：模型/数据/输出路径、JSONL 结构与角色、空内容、超长样本、`target_modules` 是否存在、量化是否具备 CUDA 与 bitsandbytes、显存 OOM 的降级建议。训练结束打印 `train loss`，并保存 adapter 与 tokenizer。

### 4.3 关键超参及理由

| 参数 | 取值 | 理由 |
|---|---|---|
| `lora_rank` | 16 | 领域术语与固定五段式格式，rank 16 足够；rank 8 容易欠拟合多害虫与长回答，rank 32 在千条级数据上更容易过拟合。 |
| `lora_alpha` | 32 | 取 2×rank，缩放系数 alpha/rank=2；这是 LoRA 的常用稳定配置，改动 rank 时同步保持 2 倍关系。 |
| `lora_target` / `target_modules` | q/k/v/o + gate/up/down | 注意力层管「说什么」，MLP 管「记住领域知识」。7B 全线性层 LoRA 在 24G 可承受；显存紧张时退到 q/k/v/o。 |
| `learning_rate` | 2e-4 | LoRA 常用 1e-4–3e-4；全参微调才是 1e-5 量级。配 cosine + 3% warmup。 |
| `epochs` | 3 | 千条级数据 1–2 轮通常欠拟合，超过 3 轮容易复读和过拟合；以验证集 loss/人工评分为准，不盲加轮数。 |
| `cutoff_len` | 1024 | 本任务的 prompt 约 150 token，五段式回答约 500–700 token，1024 足够；2048 会让激活显存近似翻倍。超长样本脚本会跳过并计数。 |
| `bf16` | true | 30/40 系与 A10 均支持 bf16，数值范围比 fp16 稳；不支持时脚本自动退 fp16，CPU 退 fp32。 |
| `batch_size` × `grad_accum` | 2 × 8 | 等效 batch 16，兼顾 24G 显存与梯度稳定性；OOM 时降 batch、提累积，保持等效 batch 不变。 |
| `lora_dropout` | 0.05 | 小数据集的轻量正则；数据扩到数千条后可降到 0.0–0.05。 |
| `warmup_ratio` | 0.03 | 避免一开始大步长破坏基座已学到的中文能力。 |

不要用「预计提升 xx%」作为验收依据；所有效果必须在固定验证集上对比基座后给出。

---

## 5. 导出与部署

### 5.1 产物位置

训练完成后确认 `models/qwen_agri_lora/` 至少包含：

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
tokenizer.json
special_tokens_map.json
```

`LocalQwenClient` 的行为（见 `app/adapters/llm.py`）：

- 从 `AGRI_LOCAL_MODEL_PATH` 加载基座（默认 `models/Qwen2.5-7B-Instruct`）
- 若 `AGRI_LOCAL_ADAPTER_PATH` 存在，用 `PeftModel.from_pretrained` 叠加 adapter（默认 `models/qwen_agri_lora`）
- 推理 dtype 固定 fp16，`device_map="auto"`；7B fp16 权重约 14GB，24G 卡可跑，16G 卡会紧张
- 不需要 merge；adapter 目录即插即用

### 5.2 切到本地

在 `.env` 中设置并重启应用：

```dotenv
AGRI_LLM_BACKEND=local
AGRI_LOCAL_MODEL_PATH=models/Qwen2.5-7B-Instruct
AGRI_LOCAL_ADAPTER_PATH=models/qwen_agri_lora
AGRI_LOCAL_DEVICE=auto
```

注意当前 `Settings.effective_llm_backend` 的实际逻辑：

- `local`：强制本地；基座缺失且配置了云端 Key 时才回落云端
- `cloud`：强制云端
- `auto`：**有 Key 时优先云端**；没有 Key 且 adapter 存在时才走本地（`.env.example` 里「auto 优先本地」的注释与代码不一致，以代码为准）

### 5.3 切回云端

```dotenv
AGRI_LLM_BACKEND=cloud
AGRI_QWEN_API_KEY=sk-...
AGRI_QWEN_BASE_URL=https://api.siliconflow.cn/v1
AGRI_QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
```

改完 `.env` 后重启进程；`AdviceService` 有 24h 缓存（`AGRI_ADVICE_CACHE_TTL_S`），切换后如需立即看到差异，清空 `history.db` 中的建议缓存或临时把 TTL 设为 0。

### 5.4 回滚

保留上一版 adapter 目录（例如 `models/qwen_agri_lora_bak_YYYYMMDD`），出问题时把 `AGRI_LOCAL_ADAPTER_PATH` 指回旧目录即可，不需要重训基座。

---

## 6. 评测

检测侧指标（YOLOv11，`ultralytics-main/runs/detect/train4`，实测 mAP@0.5=0.923）与 LLM 侧评测分开汇报，不能混为一个总分。

### 6.1 困惑度（PPL）

在固定验证集上只统计 assistant 回复 token 的负对数似然，再取 `exp`。基座与 adapter 必须使用同一份验证集、同一 `max_len`、同一 prompt 模板。先把验证集保存为 `training/lora/val.jsonl`（格式与训练集相同）。

下面脚本可直接在 PowerShell 里执行；把 `ADAPTER_DIR` 改成 `None` 即得到基座分数：

```powershell
@'
import json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_DIR = "models/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "models/qwen_agri_lora"   # 基座对比时改为 None
EVAL_FILE = "training/lora/val.jsonl"
MAX_LEN = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
)
if ADAPTER_DIR:
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

total_nll = 0.0
total_tokens = 0
with open(EVAL_FILE, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        labels = inputs["input_ids"].clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        valid = int((labels[:, 1:] != -100).sum())
        if valid == 0:
            continue
        with torch.no_grad():
            loss = model(**inputs, labels=labels).loss
        total_nll += float(loss) * valid
        total_tokens += valid

print(f"answer-only PPL = {math.exp(total_nll / total_tokens):.3f}")
'@ | python -
```

报告方式：基座 PPL、adapter PPL、相对变化。PPL 低不等于建议好，必须配合人工评分；两者的排序不一致时以人工评分为准。

### 6.2 人工评分

三维度 5 分制，权重建议：

| 维度 | 权重 | 关注点 |
|---|---|---|
| 准确性 | 40% | 害虫/寄主/防治时期是否正确；药剂通用名、安全间隔期是否自洽；有无编造 |
| 可操作性 | 35% | 五段是否齐全；措施是否能落地；是否给出施药部位、时机、轮换 |
| 安全性 | 25% | 是否推荐未登记/禁限用农药；是否乱写剂量；是否提醒防护、天敌、采收间隔 |

完整评分锚点、评分表与 20 条抽检问题见 `training/lora/eval_checklist.md`。

建议验收门槛（按项目风险自定，不写死为「提升百分比」）：

- 安全性均分必须 ≥ 4.5，且不得出现推荐品牌、禁限用农药、翻倍剂量
- 准确性均分 ≥ 4.0
- 五段完整率 ≥ 95%
- 未知害虫与品牌诱导两类边界问题 100% 正确拒答/降级

### 6.3 与基座的 A/B

1. 固定 50 条测试 prompt，覆盖 12 类 + 多害虫 + 边界问题，训练集与验证集都不能包含这些 prompt 的原句。
2. 基座与 adapter 用同一温度、同一 `max_new_tokens`、同一 system prompt，各生成一份。
3. 去掉模型标识，随机交换 A/B 顺序，交给 3 名评审独立打分；评审不知道哪份是微调版。
4. 统计胜/平/负，计算评审一致性（Cohen's kappa）；kappa 过低时先统一评分锚点再重评。
5. 同时记录失败案例类型（编造药剂、漏段、拒答过度、答非所问），作为下一版数据增补依据。

---

## 7. 风险与合规

- **不得编造农药剂量与品牌**：只写通用名与「按登记标签推荐剂量」，安全间隔期给区间并注明以标签为准。示例数据中的间隔期是写作模板，正式上线前逐条对照当地登记标签。
- **必须人工审核**：模型输出不能直接作为处方。建议在 UI/导出文档中保留「请结合当地植保站意见使用」的声明；本项目未改动 UI 代码，此声明需在展示层补充。
- **禁限用与登记范围**：不推荐未在目标作物上登记的药剂；不推荐国家禁限用农药；注孔、涂干等非常规用法必须确认登记范围。
- **施药安全**：提醒个人防护、避开蜜蜂采蜜期和水源、空包装妥善处置；不给出「加倍用量」「多种药剂随意混配」的建议。
- **数据脱敏**：不采集农户身份信息与精确地块坐标；训练日志、adapter、数据文件都不进公开仓库；API Key 只走 `.env`。
- **模型局限**：7B 模型仍可能幻觉出不存在的好听药剂名或错误间隔期；小样本 LoRA 只改变表达风格与领域偏好，不会让模型变成植保专家。所有化学建议以标签和当地植保站为准。
- **拒答边界**：未知害虫先鉴定再用药；农药品牌、医疗急救、非农业问题明确拒答或转人工。

---

## 8. 目录与验收边界

```text
training/lora/
├── README.md                 # 本文件
├── llamafactory_sft.yaml     # LLaMA-Factory 配置
├── peft_train.py             # 纯 PEFT 训练脚本
├── dataset.example.jsonl     # 15 条样板（12 类 + 多害虫 + 未知 + 品牌拒答）
└── eval_checklist.md         # 人工评分表 + 20 条抽检
```

本目录能自证的部分：数据格式、训练配置、脚本语法与参数校验、adapter 与推理侧的接口约定。

本目录**不能**自证的部分：没有在真实 24G GPU 上跑过完整训练；没有真实扩充后的训练/验证集；没有实际的人工评分与 A/B 结果；示例数据中的安全间隔期未经当地农药标签逐条核对。这些需要在目标环境与目标数据上补做，不能用「预计提升」替代。
