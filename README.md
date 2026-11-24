# Models at Home Studio

Локальная desktop-студия для обучения LLM, VLM, VLA и небольших world models. Electron + React, Python/FastAPI и собственный ML runtime включены в исходники. Веса моделей, Python, CUDA и ML-зависимости устанавливаются отдельно.

**Версия 0.2.0 — релиз-кандидат.** Проверяемая платформа: Linux x64, Python 3.12, NVIDIA CUDA. Windows/macOS остаются целями сборки; проверки на реальных Windows/macOS в этом выпуске не выполнены. Состояние релизных проверок: [RELEASE.md](RELEASE.md); изменения: [CHANGELOG.md](CHANGELOG.md).

## Что умеет

- **Training Studio:** pretrain и SFT, Home/Hugging Face/Blueprint, метрики, checkpoints, продолжение и inference. Есть legacy LLM GRPO; набор его возможностей отличается от VLM/VLA.
- **VLM Studio:** современные Qwen3-VL/Qwen3.5 и пресет Gemma 3. Full/LoRA/QLoRA, обычный Transformers и optional Unsloth. Собственные GRPO, Dr.GRPO, DAPO и SDPO; TRL trainer для этих objectives не используется. Gemma 3 требует отдельной GPU-проверки.
- **VLA Studio:** изображение + инструкция + optional state → continuous/discrete/hybrid action chunks. Native VLA с нуля либо VLM + Action Head. SFT, GRPO, Dr.GRPO и DAPO loss; экспорт и action preview. RL reward сейчас сравнивает действия с демонстрациями. Робот, симулятор и ОС автоматически не управляются.
- **Never Give Up (2026):** отдельный режим в VLM/VLA — ограниченный повторный sampling сложных примеров поверх собственного GRPO. Синхронная одна GPU; для VLA рекомендуем SFT → NGU. Это не curiosity agent NGU 2020 и не полная distributed-система авторов.
- **World Models:** native Latent GRU / Gaussian RSSM обучаются на эпизодах с действиями, состояниями, rewards/dones; доступны validation и свободный rollout. Встроенные DIAMOND CS:GO/Atari используют настоящие локальные веса и реагируют на действия. Их pretrained-тюнинг пока не подключён.
- **Ресурсы World Models:** свободная VRAM выбранной GPU, RAM и диск; блокировка явно неподходящих запусков. Для native GRU/RSSM доступна настоящая проба 5 шагов и приблизительная ETA. Это не гарантия отсутствия OOM и не оценка произвольной модели.
- **Model Builder, Study, Notebooks, Agent Studio:** конструктор архитектур, материалы, управляемый JupyterLab и агент через отдельный `llama-server`/GGUF.

Cosmos 3 доступен как дополнительный local Diffusers pipeline. Штатный guardrail требует доступных весов/HF access. Полная guarded video generation не сертифицирована этим релизом. Pretrained DIAMOND/Cosmos не выдаются за обучаемые через Studio модели.

## Быстрый старт: Native

Рекомендуемый профиль для проверенного VLM/VLA пути — Native backend. Docker опционален; первый ML image build тяжёлый и сам по себе не устанавливает современный VLM Python.

Нужны Node.js 22.14+ из ветки 22 LTS либо совместимая 24 LTS, Python 3.12 и достаточно RAM/диска. Для VLM + Action Head и проверенных GPU-рецептов нужна NVIDIA CUDA. Native VLA и native world models могут обучаться на CPU. Размер нужной GPU определяется моделью, precision, batch и контекстом.

```bash
npm ci
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements-text.txt
python -m pip install --no-deps -e ./models-at-home
```

Перед установкой профиля можно установить подходящие своей машине Torch/torchvision wheels. Для CPU-проверки:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

Используется включённый `./models-at-home`, а не обязательный соседний репозиторий. Другой совместимый runtime задаётся через `MAH_RUNTIME_ROOT`. На Windows активируйте `.venv\Scripts\Activate.ps1` и используйте Python 3.12; CUDA/Unsloth там отдельно не подтверждены.

Для современных VLM/VLA создайте **второе** окружение:

```bash
python3.12 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -r backend/requirements-vlm-modern.txt
# Только если нужен Unsloth:
.venv-vlm/bin/python -m pip install -r backend/requirements-vlm-unsloth.txt
```

Не устанавливайте Transformers 5/Unsloth поверх проверенного text/API Python с Transformers 4. У modern профиля есть bitsandbytes/CUDA требования; это не универсальная инструкция для Mac/CPU.

```bash
MAH_PYTHON="$PWD/.venv/bin/python" MAH_VLM_PYTHON="$PWD/.venv-vlm/bin/python" npm run dev
```

В Settings выберите **Native**. По умолчанию `Auto` сначала пробует Docker, затем Native. `MAH_PYTHON` — API/text Python; `MAH_VLM_PYTHON` — modern VLM/VLA worker. Изоляция сохраняется при training, preflight и inference. В установленном приложении эти интерпретаторы тоже должны быть доступны: ML Python не вшивается в установщик.

Опционально:

```bash
.venv/bin/python -m pip install -r backend/requirements-notebooks.txt
.venv-vlm/bin/python -m pip install -r requirements-world-local.txt
# Только для local Cosmos:
.venv-vlm/bin/python -m pip install -r requirements-world-foundation.txt
```

`MAH_WORLD_PYTHON` позволяет выбрать отдельный world-model Python. Настройте HF-доступ для gated-моделей; веса не входят в установщик. Unsloth может зависеть от TRL как пакета, но наши VLM/VLA loops не вызывают TRL trainers.

## Первый эксперимент

1. Откройте VLA либо World Models и выберите встроенное **синтетическое демо**.
2. Выберите CPU для небольшой native policy/model либо доступную GPU.
3. Выполните preflight. После изменения настроек проверку нужно повторить.
4. Запустите короткое обучение, дождитесь экспорта и проверьте Action preview / Rollout.
5. Для дообучения VLA выберите SFT export, затем GRPO либо NGU с тем же контрактом действий.

Синтетический smoke проверяет механику обучения. Для заявлений о качестве нужны независимые реальные данные и environment success evaluation.

## Данные и обновление

В установленном приложении рабочие данные находятся в `userData/workspace`, отдельно от resources. Корень можно задать через `MODELS_AT_HOME_ROOT`; приложение и workers используют его совместно. Python/ML-код располагается в resources или `MAH_RUNTIME_ROOT`.

Перед обновлением остановите training/play/Jupyter и сохраните копию выбранного workspace: `datasets`, `models`, `out`, `.runs`, `notebooks`, `configs`, `blueprints`, `study_materials`. Путь `output_dir` может быть внешним — такие каталоги тоже сохраняются отдельно. Установка не должна заменять workspace исходниками runtime.

Resume поддерживается только для совместимых полных checkpoints. Legacy LLM GRPO resume не реализован. Старые TRL VLM checkpoints не возобновляются как native GRPO; изменившийся VLA fingerprint/данные требуют нового запуска. Для warm start используйте подходящий final export. Инструкция обновления и допустимые формулировки анонса — [RELEASE.md](RELEASE.md).

## Docker

```bash
docker build -t models-at-home-base:latest ./models-at-home
docker compose up --build
```

Нужны Docker и NVIDIA Container Toolkit для GPU. В Compose пути по умолчанию указывают на `./models-at-home`; `MAH_RUNTIME_ROOT` меняет их. Код API/runtime и пользовательские volumes разделены. Автосборка Electron сравнивает fingerprint образа с текущими исходниками; устаревший image больше не считается подходящим.

Базовый Docker runtime — отдельный legacy ML профиль. Для современных VLM/VLA нужен совместимый modern Python **внутри контейнера**: `MAH_DOCKER_VLM_PYTHON`; для отдельного world Python — `MAH_DOCKER_WORLD_PYTHON`. Эти переменные не создают окружение и не монтируют host venv. Полная пересборка тяжёлого базового CUDA image и все современные Docker training-комбинации не подтверждены данным выпуском.

Jupyter использует случайный сохранённый токен. Управляемый API сервер занимает container-port 8888; host-port подбирается Electron. Compose notebooks profile публикуется на 8889. Не используйте исторический токен `mah-local` из старых инструкций.

## Проверки и сборка

```bash
npm run typecheck
npm test
.venv/bin/python -m pip install -r backend/requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest backend/tests -ra
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-vlm/bin/python -m pytest backend/tests -ra
npm run build:linux
```

`npm run build:linux` создаёт AppImage и deb в `release/`; `build:win`/`build:mac` — цели других платформ. Сборочные команды не означают тестирования на соответствующей ОС. Номер версии в package/installer/API согласован; ML library имеет собственную версию.

CI отдельно проверяет API-only установку и CPU ML. В API-only окружении ML-тесты явно пропускаются. GPU доказательства выполняются отдельно:

```bash
.venv/bin/python scripts/smoke-training.py --multi-gpu
.venv-vlm/bin/python scripts/smoke-vlm.py --stage vlm_grpo --algorithm ngu --verify-resume
.venv-vlm/bin/python scripts/smoke-vla.py --algorithm ngu --device cpu --verify-resume
```

Desktop smoke: `node scripts/smoke-desktop.cjs /path/to/unpacked/executable`. Опциональные `MAH_SMOKE_VLA_TRAIN=1`, `MAH_SMOKE_VLA_ALGORITHM=ngu`, `MAH_SMOKE_WORLD_TRAIN=1`, `MAH_SMOKE_WORLD_RESOURCES=1` включают настоящие эксперименты. Он использует временный workspace и диагностический `--no-sandbox` на ограниченном тестовом хосте; production renderer по-прежнему sandboxed, isolation включена, Node integration выключена.

## Документация

- [Релизные проверки и анонс](RELEASE.md), [история изменений](CHANGELOG.md).
- [VLM: установка и проверенные рецепты](docs/VLM_VALIDATION.md).
- [VLA: действия, SFT и RL](docs/VLA_STUDIO.md).
- [NGU: источники, sampling и ограничения](docs/NGU.md).
- [Native World Models](docs/WORLD_MODELS_STUDIO.md), [DIAMOND/Cosmos](docs/WORLD_FOUNDATION_MODELS.md).
- [Первоначальная карта](docs/PROJECT_MAP.md) и [история исправлений](docs/FIXES.md) — исторические документы, а не актуальная матрица поддержки.

Лицензии: Studio — MIT, включённый `homellm` — Apache-2.0, DIAMOND-код — MIT. У pretrained weights собственные условия upstream; лицензия кода не означает разрешения на распространение весов.
