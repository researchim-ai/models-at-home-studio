# Проверка и обновление 0.2.0

Статус: релиз-кандидат, проверка 24 ноября 2026. Этот документ относится к текущему исходному дереву; старые JSON-отчёты отмечены своими датами. Локальные успешные проверки не равны выполненному GitHub Actions workflow или опубликованному релизу.

## Подтверждённая область

- Linux x64; Python 3.12; Torch 2.9.1; text Transformers 4.57.3 и modern Transformers 5.17.0; RTX 3090 Ti / RTX 3090.
- Полные backend/ML-тесты в обоих рабочих Python и отдельная чистая установка CPU-профиля.
- TypeScript, frontend tests, обновлённые зависимости, Vite и Linux packaging.
- Реальные LLM multi-GPU pretrain/resume/inference/SFT, VLM native SDPO и VLA pretrained Unsloth QLoRA SFT → NGU → resume → action prediction.
- Собранный Electron: девять страниц, VLA NGU и World training/export/prediction, resource benchmark/ETA, Builder handoff и reload/health.
- JupyterLab: отдельная чистая установка, реальный API start, случайный token, authenticated status, повторный start и managed stop.
- Docker backend: сборка финального API слоя на существующем ML image проверяется отдельно от полной пересборки CUDA base. Детальная область и оставшиеся ограничения фиксируются в машинном отчёте.

Итоговые числа, версии, scope и артефакты: [release-validation-2026-11-24.json](docs/release-validation-2026-11-24.json). По отдельным режимам: [VLM](docs/VLM_VALIDATION.md), [VLA](docs/VLA_STUDIO.md), [NGU](docs/NGU.md), [World Models](docs/WORLD_MODELS_STUDIO.md).

## До публикации

1. Просмотрите текущие изменения и включите весь runtime/новые backend files в релизный commit. `models-at-home/homellm`, `backend/services/ngu.py` и остальные новые исходники нужны при checkout; изменения одного README недостаточны.
2. Зафиксируйте `package.json` **и** `package-lock.json`; версия Studio/API — 0.2.0, библиотека `homellm` имеет собственную версию.
3. Запустите `npm ci`, `npm run typecheck`, `npm test`, backend-тесты и `npm run build:linux`. Проверьте отсутствие новых предупреждений `npm audit`.
4. Проверьте итоговый пакет на целевой машине с её Python/GPU, сначала на коротком синтетическом эксперименте. Диагностический smoke на ограниченном хосте использует `--no-sandbox`; это не доказательство штатной sandbox-конфигурации на каждой Linux-системе.
5. Для Windows/macOS выполните отдельную сборку, установку и smoke на соответствующей ОС, прежде чем заявлять поддержку релиза на ней.
6. Используйте финальные installer files и их SHA-256 из машинного отчёта. Не включайте чужие pretrained weights, пользовательские datasets, `.env`, `.runs` или venv в публикацию.

Готовые Linux файлы находятся в `release/`. Установщики содержат UI/API/ML source и небольшие examples; Python/Torch/CUDA и веса в них не вшиваются. Linux sandbox требует нормальной системной поддержки Chromium sandbox/user namespaces; в production renderer сохранены sandbox, context isolation и запрет Node integration.

## Как обновить существующую установку

1. Остановите training, local world play и Jupyter. Дождитесь остановки принадлежащих Studio workers.
2. Определите свой workspace: установленная программа использует `userData/workspace`, либо `MODELS_AT_HOME_ROOT`; кастомные output directories могут быть вне него.
3. Сделайте резервную копию `.runs`, `out`, `models`, `datasets`, `notebooks`, `configs`, `blueprints`, `study_materials` и внешних output directories. Не удаляйте старые final exports после установки.
4. Установите новый пакет либо обновите исходники и выполните `npm ci`. Для source install используйте включённый `./models-at-home`.
5. Обновляйте text/API и modern VLM Python раздельно по README. В установленном приложении задайте `MAH_PYTHON` / `MAH_VLM_PYTHON`; пути venv не должны исчезнуть при замене старого installer directory.
6. В Docker новая версия может потребовать пересборки образа. Если auto-build выключен, устаревший образ отклоняется с понятной ошибкой; текущие пользовательские volumes не должны удаляться. Modern VLM Python задаётся контейнерным путём, не host venv.
7. Проверьте preflight на выбранной модели и короткий train/export/preview. Для resume используйте только полный совместимый checkpoint; historical TRL/legacy states и старые VLA fingerprints могут не подойти. Warm start из совместимого final policy — отдельный новый запуск.

## Что можно писать в анонсе

> Models at Home Studio 0.2.0 готовится к выпуску: локальная студия LLM/VLM/VLA и world models. В обновлении — обучение action policies, собственные GRPO-варианты, режим Never Give Up для VLM/VLA, native GRU/RSSM и локальная проба DIAMOND. Есть preflight, метрики, checkpoints/resume и preview экспорта; для native world training — замер ресурсов и приблизительного времени. Проверяемый профиль релиз-кандидата — Linux/NVIDIA CUDA; ML Python и веса устанавливаются отдельно.

В длинном анонсе добавьте границы: VLA RL проверяет соответствие демонстрациям; реальный robot/environment adapter нужен отдельно. NGU — single-GPU синхронная реализация. Pretrained DIAMOND/Cosmos пока доступны для inference, а не Studio-тюнинга. Gemma 3, full guarded Cosmos generation, Windows/macOS и modern Docker training matrix не сертифицированы этим прогоном.

Не обещайте «любая модель на любой GPU», «всё обучается из коробки», «универсальный робот» или воспроизведённые исследовательские benchmark-результаты. Synthetic/tiny smoke доказывает работу механики, а не качество реального управления.
