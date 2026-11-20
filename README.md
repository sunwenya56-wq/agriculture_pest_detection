# 智慧农业病虫害识别与防治系统

面向农业从业者的桌面端 + HTTP 服务双形态应用：YOLOv11 完成 12 类害虫检测，Qwen2.5-7B（云端 API 或本地 LoRA adapter）生成分维度防治方案，检测结果可落库、导出与追溯。

## 关键指标

| 指标 | 数值 | 数据来源 |
|---|---|---|
| 检测类别 | 12 类农业害虫 | `models/models_config.json` |
| 验证集 mAP@0.5 | **0.923** | `ultralytics-main/runs/detect/train4/results.csv` |
| 当前权重 | `models/best.pt`（5.17 MB，sha256 已登记） | `models/models_config.json` |
| 单元测试 | 15 项，覆盖配置 / DTO / 建议 / 存储迁移 / 批量 / API 装配 | `tests/` |

> 说明：现有权重训练配置为 `yolo11n`、`imgsz=128`、10 epoch。要冲击更高精度，请用 `scripts/train_yolo.py` 以 `imgsz=640`、50 epoch 重训后再替换权重，不要沿用旧数字对外宣称。

## 核心能力

- 单张检测：预览、检测、害虫列表与置信度、防治建议、置信度滑杆、模型切换
- 批量检测：文件夹/多选、进度条、失败隔离、分布统计、报告导出
- 实时检测：摄像头取帧、可调检测间隔、QPainter 现场画框
- 智能建议：Prompt 模板版本化、指数退避重试、SQLite 缓存、断网规则兜底
- 交付能力：JSON / CSV / Excel / 标注图导出、历史记录管理、语音播报
- 服务能力：FastAPI 暴露检测、建议、历史、健康检查与指标

## 架构

```
app/
├── core/        配置(pydantic-settings)、结构化日志、异常体系、DTO、指标
├── adapters/    YOLO 推理、LLM 双通道、SQLite、导出、TTS、农业规则知识库
├── services/    检测/建议/批量/历史/健康检查（可脱离 Qt 单测）
├── api/         FastAPI：/health /v1/detect /v1/advice /v1/history /v1/metrics
├── ui/          PyQt6：主窗口壳 + 三个标签页 + 历史对话框 + QThread
└── container.py 依赖装配（UI 与 API 共用）
```

依赖方向：`ui/api → services → adapters → core`。适配器对 `ultralytics`、`cv2`、`pyttsx3` 全部懒加载，因此服务层测试不依赖 GPU/桌面环境。

## 快速开始

```powershell
# 1. 环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置密钥（不要把真实 Key 写入代码或提交仓库）
Copy-Item .env.example .env
# 编辑 .env，填写 AGRI_QWEN_API_KEY

# 3. 启动桌面端
python main.py

# 4. 或启动 HTTP 服务
.\scripts\run_api.ps1        # 等价于 python -m uvicorn app.api.server:app --port 8000
```

无云端 Key 也能启动：建议会自动走本地规则知识库（界面标记为“降级”），检测与导出功能不受影响。

## 配置项（`.env`，前缀 `AGRI_`）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AGRI_LLM_BACKEND` | `auto` | `auto` 有云端 Key 优先云端，否则本地；也可强制 `cloud` / `local` |
| `AGRI_QWEN_API_KEY` | 空 | SiliconFlow API Key，缺失时使用规则兜底 |
| `AGRI_QWEN_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容地址 |
| `AGRI_QWEN_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 云端模型名 |
| `AGRI_QWEN_RETRIES` | `3` | 429/5xx/超时指数退避重试次数 |
| `AGRI_LOCAL_MODEL_PATH` | `models/Qwen2.5-7B-Instruct` | 本地模型目录 |
| `AGRI_LOCAL_ADAPTER_PATH` | `models/qwen_agri_lora` | LoRA adapter 目录 |
| `AGRI_CONFIDENCE` | `0.25` | 默认置信度阈值 |
| `AGRI_DETECTION_INTERVAL_S` | `1.0` | 实时检测间隔 |
| `AGRI_TTS_ENABLED` | `false` | 是否启用语音播报 |
| `AGRI_DB_PATH` | `history.db` | SQLite 路径（自动兼容旧表结构） |
| `AGRI_LOG_DIR` | `logs` | 日志与 `metrics.jsonl` 输出目录 |

完整列表见 [.env.example](.env.example)。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health?deep=false` | 配置/模型/存储/LLM/TTS 自检 |
| POST | `/v1/detect` | 上传图片（multipart）检测，可选生成建议 |
| POST | `/v1/advice` | 按害虫名称生成防治建议 |
| GET | `/v1/history?limit=50` | 检测历史与统计 |
| DELETE | `/v1/history/{id}` | 删除单条历史 |
| GET | `/v1/metrics` | 检测 P50/P95、建议成功率、降级次数 |

```bash
curl -X POST http://127.0.0.1:8000/v1/detect -F "file=@leaf.jpg" -G -d "generate_advice=true"
curl -X POST http://127.0.0.1:8000/v1/advice -H "Content-Type: application/json" \
     -d "{\"pest_names\":[\"地老虎\"]}"
```

## 测试与质量

```powershell
pip install -r requirements-dev.txt
python -m pytest          # 15 项单测
ruff check app tests main.py scripts
```

- 日志：`logs/app.log`（JSON 行 + 轮转 10MB×5，trace_id 贯穿检测/建议）
- 指标：`logs/metrics.jsonl`，状态栏每 5 秒刷新 P95 与成功率
- 健康检查：启动时自检模型、存储、LLM、TTS，状态栏显示“正常/降级”

## 打包交付

```powershell
.\scripts\build.ps1       # PyInstaller onedir + 版本号 zip
```

产物：`dist/AgriPestDetection/` 与 `dist/AgriPestDetection-<version>-win64.zip`，模型与 `models_config.json` 已随包携带。

## 模型训练与 LoRA 微调

- 检测重训：`python scripts/train_yolo.py --data ultralytics-main/mydata.yaml --imgsz 640 --epochs 50`
- LLM 微调：见 [training/lora/README.md](training/lora/README.md)，含 LLaMA-Factory 与纯 PEFT 两套可执行命令、数据规范、评测清单
- 部署切换：adapter 放入 `models/qwen_agri_lora`，把 `AGRI_LLM_BACKEND` 设为 `local` 即为私有化推理；设回 `cloud` 即走云端

## 安全与合规

- 密钥只从环境变量 / `.env` 读取；`.env` 已在 `.gitignore` 中
- 若历史版本曾提交过真实 Key，请立即到服务商后台吊销并重新签发
- 农药建议仅作参考，实际用药须遵守当地登记标签与安全间隔期
- 用户图片与历史记录仅存本地 SQLite，不上传第三方（云端建议仅发送害虫名称，不发送原图）

## 已知限制

- 现有权重为低分辨率短周期训练产物，复杂田间场景建议重训后再评估
- 语音播报依赖 `pyttsx3`（Windows SAPI），跨平台音色有限
- 本地 LLM 通道需 24G 显存级别环境；CI 只跑轻量单测，不做 GPU 推理
- 无多用户鉴权，API 默认仅监听 `127.0.0.1`，不要直接暴露公网
- Ultralytics 采用 AGPL-3.0 许可，商用交付前请确认授权方式；`ultralytics-main/` 源码已排除在仓库外
