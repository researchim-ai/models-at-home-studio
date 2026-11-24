# VLA Studio

Отдельная страница `/vla` для обучения action policies. Использует привычные run directories, process supervision, метрики, логи, checkpoints и общий continue API. LLM/VLM страницы сохранены.

## Поддержанные сценарии

- **Робототехника:** изображение камеры, инструкция и optional proprioceptive state → непрерывный action vector/chunk. Размерность состояния/действий, названия осей и горизонт настраиваются.
- **Игры и симуляторы:** дискретные команды, аналоговые оси либо смешанная схема.
- **Desktop:** screenshot/instruction → mouse deltas/scroll и отдельные категориальные головы для mouse button/key. Пресет содержит пример словаря команд; числа классов и семантика контрактов изменяемы.

Доступны SFT/behavior cloning и GRPO / Dr.GRPO / DAPO loss с reward по демонстрациям. Политика предсказывает данные действий; эта страница не подключает конкретного робота, игру или ОС и не выполняет выданные команды. Для смешанных размеров viewport используйте нормализованные координаты либо передавайте размеры viewport в state; единицы действий должны оставаться согласованными в демонстрациях.

Для closed-loop RL нужны отдельные environment adapters, наблюдения и environment reward. Текущий action GRPO семплирует новые действия из policy, но оценивает их близость к записанным target actions; успех в реальной среде этим не измеряется.

## Два вида моделей

**Native VLA:** наша небольшая модель с нуля — CNN с сохраняемой пространственной сеткой для изображения (по умолчанию 4×4, чтобы положение объектов влияло на действия), UTF-8 byte embeddings + GRU для инструкции, state vector и MLP action head. Обучается целиком, работает на CPU/CUDA. Это baseline, не импортированная OpenVLA/SmolVLA/ACT policy и не сертификат robot deployment.

**VLM + Action Head:** современная HF VLM извлекает мультимодальные hidden states; последний непустой token feature объединяется с состоянием. Новая обучаемая action-head выдаёт весь action horizon. Основа поддерживает Full, LoRA и 4-bit QLoRA, обычный Transformers либо optional Unsloth. По умолчанию vision tower заморожена; head обучается всегда. Языковая SFT loss для действий не используется.

Для pretrained Qwen3.5 основной проверенный вариант — bf16 LoRA. QLoRA проверена технически, но сохраняются предупреждения upstream о погрешности 4-bit для этого семейства. Остальные семейства требуют отдельной проверки.

## Интерфейс

1. Выберите SFT либо GRPO / Dr.GRPO / DAPO loss, сценарий и архитектуру. Для улучшения готовой policy укажите её `final_model` как начальную VLA policy; в истории есть кнопка «Дообучить GRPO». Контракт действий, архитектура и нормализация должны совпадать.
2. Укажите JSONL train и optional holdout, размерности, action type и horizon. Кнопка **Синтетическое демо** подставляет готовые примеры нужного домена и соответствующую схему.
3. Нажмите **Проверить VLA**. Показываются кадр, инструкция, state, реальные action targets и найденные ошибки. До успешного актуального preflight запуск заблокирован; изменения формы и старые async-ответы сбрасывают разрешение.
4. Запустите обучение. Виден optimizer step, action/validation loss, график, MAE в исходных единицах, classification accuracy, статус, ошибки, логи и export path. Stop завершает управляемый worker.
5. В истории можно выбрать метрики и явно продолжить до показанного числа шагов. В **Action preview** загрузите кадр, state JSON и инструкцию, чтобы получить action chunk из экспорта.

Синтетические изображения/действия из `examples/vla` проверяют формат и механику, а не качество управления. Их можно воспроизвести через `scripts/create-vla-demo.py`.

## Контракт демонстраций

JSONL: одна строка — одно наблюдение и **явный** chunk действий. `image` — локальный путь от каталога JSONL; `instruction` — непустая строка; `state` — вектор длины state_dim (при state_dim=0 поле можно опустить). `episode_id` необязателен и помогает проверять пересечения holdout.

Пример непрерывных действий, action_dim=2, state_dim=1, horizon=2:

```json
{"image":"frames/001.png","instruction":"move to red","state":[0.1],"actions":[[0.2,-0.1],[0.1,0.0]],"episode_id":"episode_01"}
```

Для discrete, action_dim обозначает число классов; каждый timestep — integer class ID:

```json
{"image":"frames/001.png","instruction":"jump","state":[],"actions":[1,0],"episode_id":"game_01"}
```

Hybrid: action_dim — размер непрерывного вектора, discrete_classes — число классов каждой категориальной головы. Для dim=3 и classes=[4,16]:

```json
{"image":"frames/001.png","instruction":"click target","state":[],"actions":[{"continuous":[12.0,-4.0,0.0],"discrete":[1,5]},{"continuous":[0.0,0.0,0.0],"discrete":[0,0]}],"episode_id":"desktop_01"}
```

Порядок координат, единицы, частота timesteps, абсолютные/относительные значения, press/release и таблицы классов задаются вашим контрактом. `action_names`, `discrete_names` и `discrete_labels` сохраняются с policy export. Значения не превращаются автоматически в OS keycodes или actuator commands.

Chunks длиной 1..horizon допустимы: terminal padding маскируется в loss и не входит в статистики. Строки не склеиваются, поэтому обучение не пересекает границы эпизодов. Если во всём train нет полных chunks, preflight/train отклоняет настройку, чтобы не выдавать необученные будущие timesteps за предсказание.

Схема целиком проверяется на правильные размерности, допустимые классы, finite float32 значения, существующие файлы и длину UTF-8 инструкций; ошибочные строки не пропускаются. Preflight декодирует изображения/processor на первых 200 строках каждого split, а не доказывает целостность всех кадров большого датасета. Во время обучения чтение остальных кадров строгое.

## Loss, normalization и validation

Непрерывные actions и state нормализуются по **train-only** mean/std; короткий padding исключён. Константные координаты используют scale=1. Native loop оптимизирует SmoothL1 по непрерывным координатам и cross entropy по классам. Hybrid суммирует continuous loss и среднюю categorical loss по головам. Padding не получает градиенты.

Validation использует train statistics. Отдельно считаются normalized action loss, raw-unit continuous MAE и categorical accuracy. Preflight предупреждает о пересечении image paths/episode IDs между train и validation. Это не гарантирует независимости семантически дублированных кадров; реальные splits нужны по отдельным эпизодам/сессиям.

## Checkpoint и экспорт

Формат `mah_vla_v1`: `policy.json` содержит архитектуру, action contract, конфигурацию и normalization statistics. Native policy сохраняется в `policy.safetensors`. Для VLM — отдельные `action_head.safetensors` и каталог `backbone` с полной моделью или PEFT adapter + processor. LoRA export требует доступной исходной базы.

Checkpoint дополнительно хранит optimizer, scheduler, fp16 scaler, RNG, shuffle generator, data order/cursor и train statistics. `complete.json` пишется атомарно после всех файлов; continue выбирает полные checkpoints. Проверяется fingerprint конфигурации и JSONL; изменение изображений по прежнему пути в fingerprint не входит. Для продления max_steps scheduler пересчитывается под новый план.

## Runtime и API

VLA workers используют ту же изолированную `.venv-vlm`/`MAH_VLM_PYTHON`, что современная VLM. Основной API/text Python не обновляется. Native policy может работать на CPU (float32); VLM backbone требует CUDA. Указывается одна GPU. Predict — отдельный ограниченный по времени процесс: он загружает экспорт, выдаёт JSON и освобождает модель/GPU после запроса.

- `GET /api/vla/examples` — явно синтетические пресеты.
- `POST /api/vla/preflight` — проверка без загрузки весов backbone и без создания run.
- `POST /api/vla/start`, `POST /api/vla/stop/{run_id}`, `GET /api/vla/runs`.
- Общие `/api/training/runs/{id}/metrics`, `/logs`, `/continue` работают с VLA.
- `POST /api/vla/predict` — `model_path`, image_base64, instruction, state, optional device/gpu_ids; результат actions в исходных единицах или class IDs.

## Подтверждённые проверки

В [машиночитаемом отчёте](vla-validation-2026-11-23.json) сохранены 12 завершённых обучающих экспериментов (CPU/GPU): Native continuous/discrete/hybrid, robotics/simulation/desktop, pretrained Qwen3.5-0.8B LoRA/QLoRA через Transformers и Unsloth, а также Full на tiny Qwen3-VL через оба backend. Проверены loss/gradients, validation, изменение head/adapters, checkpoint resume и reload/prediction. Для Native CPU дополнительно получено побитовое совпадение resumed и непрерывного обучения.

Готовый Electron-пакет проверен через настоящий UI: desktop demo → preflight → action training → chart/validation metrics → export → загрузка кадра → hybrid action prediction. Отдельно исправлен ранний показ export path и потеря выбранного активного run после его завершения.

Итоговые проверки: 183 backend/ML-теста, 32 клиентских теста, TypeScript, Vite build и полный smoke готового Electron-приложения.

## Воспроизведение

```bash
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --device cpu
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --action-type discrete
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --domain desktop --action-type hybrid
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --domain simulation --action-type hybrid
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --model-path /path/to/Qwen3.5-0.8B --method lora
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --model-path /path/to/Qwen3.5-0.8B --method qlora --action-type hybrid
PYTHONPATH=/path/to/models-at-home .venv/bin/python scripts/smoke-vla.py --model-path /path/to/Qwen3.5-0.8B --backend unsloth --domain desktop --action-type hybrid
PYTHONPATH=/path/to/models-at-home .venv-vlm/bin/python scripts/smoke-vla.py --model-path /path/to/Qwen3.5-0.8B --tiny-backbone --method full
```

Успешный smoke проверяет реальные optimizer updates, nonzero gradients, validation, resume, изменение policy/head/LoRA weights и prediction после загрузки экспорта. Tiny и synthetic проверки не доказывают качество реального управления. Closed-loop environment success rate, multiple cameras/video, environment replay, rollout execution, environment rewards и готовые адаптеры LeRobot/OpenVLA/ACT не входят в эту первую реализацию.


## Action GRPO (24 ноября 2026)

`training_algorithm`: `sft`, `grpo`, `dr_grpo`, `dapo`. Stage — `vla_sft` либо `vla_grpo`; общий start/runs/stop/continue API обслуживает оба. Никакой TRL trainer не используется: action likelihood/masks передаются в наш `homellm.training.rl.loss.GRPOLoss`, advantages — в `compute_advantages`.

- Continuous: независимые Normal для нормализованных координат с фиксированным `grpo_action_std`; inference возвращает mean в исходных единицах.
- Discrete: Categorical logits/temperature; hybrid объединяет Gaussian и отдельные категориальные головы. Log-prob — совместная вероятность вектора на timestep.
- Для каждого наблюдения семплируется `grpo_group_size` новых chunks. Reward: отрицательный средний MSE нормализованных continuous actions и/или categorical accuracy по валидным шагам. Padding не влияет на reward, advantages, loss или KL.
- Advantages нормализуются внутри группы наблюдения. GRPO использует clipped ratio к detached old policy, `grpo_updates` optimizer updates на одну группу и KL к начальной policy. Dr.GRPO отключает std-нормализацию и использует фиксированный horizon; DAPO loss использует token-level reduction и асимметричный clip. Dynamic sampling/filtering, остальные компоненты полной DAPO training системы и SDPO teacher для действий здесь не реализованы.
- KL вычисляется аналитически между распределениями текущих и начальных выходов. Reference outputs фиксируются для наблюдений обучающего датасета до первого update и сохраняются в checkpoint; второй экземпляр VLM на GPU не нужен. CPU bank ограничен 256 MB; `grpo_kl_beta=0` отключает bank и KL. Dropout отключается для согласованного sampling/likelihood.
- `grpo_sft_weight` по умолчанию 0; положительное значение добавляет явный supervised anchor. Поля группы, std/temperature, clip, KL и числа updates доступны в интерфейсе. Большой sample buffer отклоняется до старта.
- Начальная policy необязательна: Native может учиться с нуля; для pretrained action head практичнее сначала SFT. Warm start загружает веса и прежнюю train-only нормализацию. Это отдельный новый эксперимент, а не optimizer resume.

Resume сохраняет sampler/order/cursor, Torch/CUDA RNG, AdamW/scaler/scheduler, immutable reference и незавершённую группу с sampled actions/old log-probs/advantages. Optimizer шаг считается только после успешного update; FP16 overflow уменьшает scale и повторяет те же microbatches. Learning rate после линейного warmup постоянный, поэтому изменение max_steps не меняет уже пройденный optimizer путь. Checkpoint на последнем шаге сохраняется даже вне save_every. Fingerprint включает JSONL, содержимое кадров, initial policy weights и reward/algorithm параметры; изменённые данные не продолжаются молча. Старые checkpoints с прежней схемой fingerprint требуют нового запуска.

Тесты `test_vla_training.py` проверяют реальные updates и побитовый CPU resume посреди группы для SFT, GRPO continuous/hybrid, Dr.GRPO discrete и DAPO loss hybrid; отдельно проверяется SFT→GRPO и отклонение изменённого кадра. Сквозные smoke также проверяют pretrained Qwen3.5 LoRA/QLoRA через Transformers/Unsloth и Full на tiny Qwen3-VL. Отчёт текущего прогона: `vla-grpo-validation-2026-11-24.json`. Эти synthetic/tiny проверки подтверждают работу trainer, а не качество управления роботами.


NGU (Never Give Up, 2026) доступен в VLM/VLA как отдельный bounded adaptive sampling режим поверх собственного GRPO. Настройки, источники и границы: [NGU.md](NGU.md).
