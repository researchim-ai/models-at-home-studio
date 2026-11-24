# Changelog

## 0.2.0 — релиз-кандидат, 24 ноября 2026

### Добавлено

- VLA Studio: Native policy и VLM + Action Head; continuous/discrete/hybrid chunks, SFT, собственные GRPO / Dr.GRPO / DAPO loss, validation, checkpoints/resume и action preview.
- Never Give Up (2026) в VLM/VLA: bounded adaptive sampling, объединение попыток, verifier rewards, метрики, проверенный SFT → NGU путь.
- World Models Studio: собственные Latent GRU/RSSM training и свободный rollout; встроенная локальная проба pretrained DIAMOND CS:GO/Atari.
- Карточка ресурсов World Models: выбранная GPU/RAM/диск, ограничения запуска, реальная native training-проба и приблизительная ETA.
- Современный отдельный VLM/VLA Python и собственный VLM RL transport поверх `homellm`, optional Unsloth loader/acceleration.

### Исправлено перед релизом

- ML runtime включён в проект и ищется отдельно от изменяемого workspace, в том числе isolated workers.
- VLM image loading/assistant masks и совместимость processor; VLA FP16 overflow retry и точный optimizer/sampler/RNG resume.
- Неполные checkpoints, изменённые данные и несовместимые legacy formats отклоняются явно.
- Docker image сверяется с fingerprint backend/runtime; устаревший `latest` не принимается автоматически.
- Исправлены Docker build context, наличие `uv` в финальном слое и пути Compose к включённому runtime. Auto-build не отправляет весь runtime workspace с весами/данными.
- В Docker пробрасывается путь к отдельному modern VLM Python. Его окружение пользователь создаёт внутри образа самостоятельно.
- API-only тесты больше не требуют Torch; добавлен отдельный CPU ML job с проверкой наличия runtime.
- Исправлены инструкции установки, обновления, пользовательского workspace и случайного Jupyter token.

### Зависимости

- Electron 45.4.1, Vitest 5, обновлённые React Router, i18next и syntax highlighter; точечные overrides транзитивных зависимостей.
- В момент релизной проверки `npm audit` не сообщает известных уязвимостей для полного JS dependency tree. Это результат проверки на конкретную дату, не гарантия отсутствия будущих находок.

### Границы

- Основная проверенная платформа — Linux x64/NVIDIA CUDA. Windows/macOS installers и настоящие пользователи на этих ОС ещё не проверены.
- RL reward VLA основан на демонстрациях; environment adapters и real-robot success evaluation не входят в выпуск.
- NGU — синхронный single-GPU sampling поверх собственного GRPO. Legacy text-only LLM NGU и distributed система статьи не добавлены.
- Pretrained DIAMOND/Cosmos тюнинг через Studio не подключён. Полная guarded Cosmos generation требует отдельной проверки и HF access.
- Полная пересборка базового CUDA image и вся modern training matrix внутри Docker не подтверждены; для modern VLM/VLA рекомендуем проверенный Native профиль.

Подтверждения, артефакты и текст для анонса: [RELEASE.md](RELEASE.md).

## 0.1.0

Исходная desktop-студия LLM/VLM, Model Builder, Study Center, Agent Studio и Notebooks. История поэтапных исправлений сохранена в [docs/FIXES.md](docs/FIXES.md).
