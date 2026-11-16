# Контракт avatar-worker, revision 1

Русская версия [переданного исходного контракта](evidence/avatar-contract.md). Решение владельца от 2026-11-17:
собственные Vast-машины, основной пул 2×H100 SXM, резерв 1×H100 SXM; картинка и готовый голос проекта, без синтеза речи.

## Транспорт

Перед каждым вызовом студия получает маршрут через `POST https://run.vast.ai/route/`: cost 100 для start, 1 для state/file.
Затем HTTPS с закреплённым корневым CA Vast, envelope `{"auth_data": <route answer>, "payload": <body>}`.
Endpoints: `studiolab-avatar` и резервный `studiolab-avatar-b`. Транспорт тот же, что у comfy-worker revision 3.

## POST /clip/start

Ровно семь ключей, все обязательны. Неизвестные/повторные ключи, NaN, числовые строки вместо чисел, bool вместо int → 400.

| Поле | Ограничение |
| --- | --- |
| `request` | 32 строчных hex-символа |
| `mode` | Только `avatar` |
| `prompt` | 1–2000 символов, русский или английский |
| `size` | `480p` или `720p` |
| `seed` | Целое 0–2147483647 |
| `image` | Base64 без data:-префикса; один JPEG/PNG, ≤ 12 MiB, ≤ 16 777 216 пикселей, ≤ 8192 по стороне |
| `audio` | Base64 без data:-префикса; WAV/PCM, MP3, M4A/AAC, OGG/Opus/Vorbis; ≤ 24 MiB; одна звуковая дорожка, 2,0–60,0 с |

Изображение уже обрезано студией под нужное соотношение сторон. Воркер применяет EXIF orientation и RGB-преобразование,
не обрезает повторно. Длительность звука измеряет ffprobe в частном каталоге задания.
Выход за границы: 400 `{"error":"invalid","detail":"audio must last 2 to 60 seconds"}`.

Ответы:

- 200 `{"id":"<32 lowercase hex>","state":"queued"}`.
- Тот же request и те же mode/prompt/size/seed/байты image/audio → тот же id и текущее state.
- Тот же request с другим содержимым → 409 `{"error":"conflict"}`.
- Недопустимый запрос → 400 `{"error":"invalid","detail":"<short English>"}`.
- Одно выполняемое и два ожидающих места заняты, backend недоступен или retiring → 429 `{"error":"busy"}`.

## Съёмка

25 fps. Первый сегмент — 93 кадра / 3,72 с; каждое продолжение добавляет 80 кадров / 3,2 с при 13 conditioning frames.
`segments = 1 + ceil(max(0, duration - 3.72) / 3.2)`. Округление финальной длительности — к ближайшему целому кадру,
половина вверх. Примеры: 2 с → 1 сегмент; 3,72 → 1; 3,73 → 2; 10,12 → 3; **10,14 → 4**; 60 → 19.

Размер — собственный bucket модели по соотношению сторон изображения при 480p/720p. Фактические width/height сообщает state.
Наблюдения из испытания владельца: 16:9 при 480p → 832×480; портрет 4:5 → 544×704 при 480p, 864×1056 при 720p.
Выход — H.264/yuv420p, CRF ≤ 16 и AAC в MP4. Исходный голос перекодируется в AAC на протяжении всего клипа;
отделённый вокал никогда не становится звуковой дорожкой результата.

## POST /clip/state и /clip/file

Тело строго `{"id":"<32 lowercase hex>"}`. Неизвестный id → 404 `{"error":"unknown"}`.

State содержит `state` queued/running/done/failed, `progress` 0..1 (завершённые сегменты / все сегменты), `stage`, `fps=25`,
`frames`, `segments`, `gpus`. `width`/`height` появляются, когда известен фактический bucket; после done — `exec_s`,
при ошибке — короткий `error`. `machine` содержит публичные числа экзамена и `boot_s`.
Стадии задания: `queued`, `loading`, `segment N/M`, `muxing`, `done`, `failed`. Веса скачиваются до READY, поэтому принятому
заданию обычно не нужна стадия `downloading_models`.

File возвращает весь MP4, `Content-Type: video/mp4`, точный Content-Length. До done → 409 `{"error":"not_ready"}`.
TTL — 3600 с после done или failed; queued/running не истекают. Прочие правила очереди, повторной передачи и простоя —
в [README](README.md#жизнь-машины).
