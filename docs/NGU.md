# Never Give Up (2026) в Studio

## Источники и выбор

Проверены оригинальные публикации и код авторов:

- [Never Give Up — 11 сентября 2026](https://arxiv.org/abs/2609.13443), [код авторов](https://github.com/mnoukhov/never-give-up): дополнительный sampling для нерешённых prompts, накопление попыток и общий baseline. Это adaptive sampling поверх GRPO objective.
- [OAPL — 22 февраля 2026](https://arxiv.org/abs/2602.19362): off-policy обучение с lagged inference policy; требует отдельной реализации objective и соответствующего rollout режима.
- [SAPO — 25 ноября 2025](https://arxiv.org/abs/2511.20347): мягкое ограничение off-policy updates вместо жёсткого clip; авторы также исследуют Qwen3-VL.
- [GSPO — 24 июля 2025](https://arxiv.org/abs/2507.18071): importance ratio и clip на уровне последовательности.

В этой задаче реализован NGU. Это список релевантных проверенных работ, а не исчерпывающий рейтинг публикаций. Название NGU также использовалось в [работе 2020 года об exploration](https://arxiv.org/abs/2002.06038); её episodic novelty/RND agent здесь не реализован.

## Где доступен

**VLM Studio:** stage `vlm_grpo`, `grpo_algorithm: ngu`. **VLA Studio:** `training_algorithm: ngu`, stage `vla_grpo`. В обоих случаях пункт называется «Never Give Up (2026) + GRPO». Собственный `homellm` GRPO loss сохранён; TRL trainer не используется. Legacy LLM Training Studio пока не имеет этого режима.

Это ограниченная **синхронная реализация на одной GPU**, подходящая нашему trainer. Политика фиксирована во время повторных попыток одного наблюдения. Все попытки относятся к одной версии весов; поэтому staleness filtering и stale-only baseline не нужны. Полная асинхронная distributed система авторов с Ray/vLLM/DeepSpeed не воспроизводится.

## Поведение

1. Семплируется группа ответов/action chunks без эталона во входе policy.
2. Если все ответы успешны, лёгкая группа пропускается. Если есть и успехи, и неудачи — группа принимается.
3. Если все неуспешны, с заданной вероятностью делается следующая попытка того же prompt/наблюдения.
4. Предыдущие неудачи сохраняются: после успеха все попытки объединяются и получают один общий centered baseline. Нельзя считать advantages только по последней попытке.
5. `ngu_max_attempts` и `ngu_max_refill_rounds` ограничивают работу. Невозможный verifier приводит к явной ошибке при исчерпании бюджета, без фиктивного optimizer update.

В VLM используются существующие бинарные `exact_match` / `contains` rewards. В VLA реализована адаптация к demonstration verifier: continuous RMSE в train-normalized единицах не выше `ngu_action_tolerance`, все валидные categorical actions должны совпасть. Padding не оценивается. Это проверка соответствия демонстрации; environment success по-прежнему требует environment adapter.

NGU использует centered advantages без std-нормализации. Policy ratio/clip и KL обслуживаются существующими собственными objective. Сигнал не заменяется SFT: дополнительный SFT anchor включается только явным `grpo_sft_weight`.

## Настройки и расходы

- `ngu_max_attempts` = 4: число sampling-раундов на наблюдение, включая первый; допустимо 1–64.
- `ngu_continue_probability` = 0.9: вероятность продолжать после полностью неудачной группы; допустимо 0–1.
- `ngu_max_refill_rounds` = 8: дополнительные попытки получить информативные prompt-группы для optimizer update; допустимо 0–100.
- Только VLA: `ngu_action_tolerance` = 0.5, в нормализованных единицах. Размерность/горизонт и точность категорий существенно влияют на достижимость успеха.

Пик rollout inference использует исходный group size. Объём сохранённых samples, число последующих gradient forwards и время могут вырасти: максимум `group_size × ngu_max_attempts` samples на наблюдение. У VLM не более 512 completions на prompt и 2M потенциальных rollout tokens на update без учёта отброшенных prompts. VLA ограничивает sample buffer 16M координатами. Дополнительный refill тоже увеличивает вычисления; UI показывает ориентиры, не гарантированную длительность.

Метрики `ngu_stats`: prompts, rounds, samples, retries, mixed, easy, budget, give_up. Они видны в интерфейсе и сохраняются в checkpoint. Изменение NGU настроек меняет fingerprint; resume требует тех же sampler/verifier settings. Новые NGU поля не меняют fingerprint прежнего обычного GRPO. VLA pending groups, old log-probs, reference bank и RNG сохраняются как раньше.

## Рекомендуемый VLA путь

Сначала SFT с нужным контрактом действий, затем история → «Дообучить GRPO» → выбрать NGU. Тот же export можно указать вручную в «Начальная VLA policy». Если случайная action head не достигает verifier threshold даже изредка, увеличение количества попыток не заменит полезную начальную policy.

## Проверки

Unit/optimizer tests проверяют поздний успех после неудач, общий baseline, пропуск лёгких групп, вероятность отказа, bounded sampling, binary verifier contract, padding, сборку настоящих VLM Experience groups, отсутствие optimizer update при недостижимом verifier и побитовый CPU resume NGU для continuous/hybrid/discrete actions.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-vlm/bin/python -m pytest backend/tests -q
.venv-vlm/bin/python scripts/smoke-vla.py --algorithm ngu --device cpu --verify-resume --result /tmp/ngu-vla.json
.venv-vlm/bin/python scripts/smoke-vlm.py --stage vlm_grpo --algorithm ngu --verify-resume --result /tmp/ngu-vlm.json
.venv-vlm/bin/python scripts/smoke-vla.py --algorithm ngu --device cuda --model-path /path/to/Qwen3.5-0.8B --backend unsloth --method qlora --sft-warmup 30 --result /tmp/ngu-warm.json
```

Фактические результаты: [ngu-validation-2026-11-24.json](ngu-validation-2026-11-24.json). Tiny/synthetic прогоны подтверждают работу механики и не доказывают преимущества на сложных реальных задачах.
