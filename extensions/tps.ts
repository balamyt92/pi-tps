import type { AssistantMessage, Usage } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * TPS — честная метрика генерации.
 *
 * ВАЖНО для локального инференса (llama.cpp / MTP / speculative decoding):
 * клиентский `usage` от локального сервера НЕ совпадает с его внутренним
 * счётчиком. Ground truth — только логи самого сервера.
 *
 * Здесь считается то, что клиент действительно может измерить:
 *   TTFT     — отправка запроса → первый токен
 *   eff TPS  — отправка запроса → ПОСЛЕДНИЙ токен ← главная метрика
 *   decode   — первое → последнее генеративное событие (вторая метрика)
 *   gaps     — распределение интервалов между дельтами (видно берсты MTP)
 *   per-call — по КАЖДОМУ вызову отдельно; сбойные (error/aborted) и
 *              незавершённые (pending) — в отдельную корзину, из итогов исключены
 *
 * Почему eff TPS главный (проверено на llama.cpp + MTP, июль 2025):
 * «decode window» (первый→последний токен) завышает на 1.5–3x, потому что
 * первая дельта приходит с буфером ~1.5–2.5s после начала декода — окно
 * сжимается. Eff TPS (запрос→последний токен) совпадает с wall-временем
 * сервера (prefill + decode) с точностью до сети (<5%).
 *
 * Примечание про reasoning: output ВКЛЮЧАЕТ reasoning-токены (согласно
 * типам pi-ai). Они генерируются моделью с той же скоростью, поэтому в
 * decode TPS они учитываются честно; при этом текстовая часть показана
 * отдельно (текст = output − reasoning).
 */

interface CallRecord {
    index: number;
    reqSentMs: number | null;
    respAtMs: number | null;
    firstTokenMs: number | null;
    lastDeltaMs: number | null;
    endMs: number;
    deltaCount: number;
    textChars: number;
    thinkingChars: number;
    toolCallChars: number;
    gaps: number[];
    usage: Usage | null;
    stopReason: string;
}

function isAssistantMessage(message: unknown): message is AssistantMessage {
    return !!message && typeof message === "object" && (message as { role?: unknown }).role === "assistant";
}

function num(v: unknown): number {
    return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

/** Единый формат чисел: точка в дробях, запятые в разрядах (dev-стандарт). */
function fmt(n: number, digits = 0): string {
    return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Unicode-correct подсчёт символов (surrogate pairs = 1 символ). */
function charLen(s: string): number {
    return Array.from(s).length;
}

function pct(xs: number[], p: number): number {
    if (xs.length === 0) return 0;
    const s = [...xs].sort((a, b) => a - b);
    // nearest-rank: ранг ceil(p*n) (1-based), минус 1 под 0-based индекс.
    // floor(p*n) давал перекос вверх на p50: для чётного n брал верхний элемент.
    const rank = Math.ceil(p * s.length) - 1;
    return s[Math.max(0, Math.min(s.length - 1, rank))];
}

function avg(xs: number[]): number {
    if (xs.length === 0) return 0;
    return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function maxOf(xs: number[]): number {
    if (xs.length === 0) return 0;
    return xs.reduce((a, b) => (b > a ? b : a), 0);
}

function fmtSecOrDash(x: number | null, digits = 2): string {
    return x === null ? "—" : `${fmt(x, digits)}s`;
}

/** Русские множественные формы: 1 сбой / 2 сбоя / 5 сбоев. */
function pluralRu(n: number, one: string, few: string, many: string): string {
    const m10 = n % 10;
    const m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
}

function isFailed(rec: CallRecord): boolean {
    return rec.stopReason === "error" || rec.stopReason === "aborted";
}

/**
 * Порог «токенов на дельту», выше которого ответ похож на спекулятивное
 * декодирование (MTP и родственные).
 *
 * Смысл: при обычном декодировании провайдер отдаёт ~1 токен на дельту.
 * Когда сервер добирает токены черновиком, в один SSE-чанк попадает несколько
 * токенов, и отношение `output / deltaCount` растёт. 1.5 — нижняя граница,
 * с которой отношение уже нельзя объяснить округлением на одно-двухтокенных
 * дельтах. Порог грубый и намеренно консервативный: он для подписи
 * «похоже на спекуляции», а не для измерения.
 */
const SPECULATIVE_TOK_PER_DELTA_THRESHOLD = 1.5;

/** message_end так и не пришёл (обрыв потока) — запись не финализирована. */
function isIncomplete(rec: CallRecord): boolean {
    return rec.stopReason === "pending";
}

/** В итоговые агрегаты идут только финализированные успешные вызовы. */
function isOk(rec: CallRecord): boolean {
    return !isFailed(rec) && !isIncomplete(rec);
}

export default function (pi: ExtensionAPI) {
    let runStartMs: number | null = null;
    let calls: CallRecord[] = [];
    let pendingReqSentMs: number | null = null;
    let pendingRespAtMs: number | null = null;
    // Наполнение разговора по итогам ПРЕДЫДУЩЕГО хода: promptOf + output =
    // input + cacheRead + cacheWrite + output последнего отчитавшегося вызова.
    // Переживает resetRun: снимается в начале нового хода, до очистки `calls`.
    // База для процента прироста. Сбрасывается безусловно, включая null:
    // неизмеримый ход обязан обнулить базу, иначе следующий ход припишет себе
    // накопленный рост нескольких ходов (см. resetRun).
    let prevFill: number | null = null;
    // Оценка сообщения пользователя в токенах (~4 символа на токен) из события
    // before_agent_start. Вычитается из entryFill на ходе, где prevFill === null,
    // чтобы база первого хода совпадала по смыслу с prevFill (без нового сообщения).
    let userMsgTokens: number | null = null;

    /**
     * Запись, открытая текущим потоком: поставлена на `message_start`,
     * закрыта и обнулена на `message_end`; дополнительно обнуляется на
     * `agent_end` (см. ниже).
     *
     * Почему нельзя ключевать по id сообщения или по идентичности объекта.
     * Проверено по исходникам `pi-agent-core/dist/agent-loop.js` и типам
     * `pi-ai`:
     *   - `AssistantMessage` не имеет стабильного `id`
     *     (`pi-ai/dist/types.d.ts:307-329`). `responseId` опционален и у
     *     части провайдеров отсутствует вовсе; `timestamp` имеет
     *     миллисекундную грануляцию. Оба ненадёжны как ключ в пределах хода.
     *   - `message_start` и каждый `message_update` эмиттируют
     *     `{ ...partialMessage }` — НОВЫЙ экземпляр на каждое событие
     *     (`agent-loop.js:248` и `:265`; там же `:279` и `:292` — тот же
     *     spread для `{ ...finalMessage }`). Идентичность объекта между
     *     событиями не выполняется.
     *
     * Явный `openRecord` строго лучше чтения `calls[calls.length - 1]`:
     * он не даёт финализировать уже закрытую запись. Если `message_end`
     * придёт без предшествующего `message_start` (расширение подключилось
     * посреди потока после `/reload`, либо провайдер с непарным потоком),
     * раньше он перезаписал `usage`/`stopReason` предыдущей корректной
     * записи. Теперь такой вызов просто игнорируется, а предыдущая запись
     * остаётся нетронутой.
     */
    let openRecord: CallRecord | null = null;
    // Семантика pi (проверено по исходникам agent-loop.js / agent-session.js):
    //  - before_agent_start — ровно ОДИН call site (agent-session.js:1025), в
    //    preflight-блоке prompt() до _runAgentPrompt. Ретраи, авто-компакция и
    //    agent.continue() его НЕ перевзводят.
    //    ВАЖНО: сообщения, доставленные через steer()/followUp() или
    //    sendCustomMessage({ triggerTurn }), идут в обход before_agent_start
    //    вовсе. Окно статистики = один ход пользователя ВКЛЮЧАЯ стиринг и
    //    follow-up'ы внутри него; они накопление не сбрасывают.
    //  - agent_start/agent_end эмитятся на КАЖДЫЙ low-level run: ретраи,
    //    авто-компакция и continuation'ы (agent.continue()) перевзводят эту пару.
    //  - agent_settled эмитится ОДИН раз за ход пользователя (finally в
    //    _runAgentPrompt, agent-session.js:877).
    //  - каждое continue() — это новый provider request, поэтому before_provider_request
    //    перевзводится и reqSentMs каждой попытки свежий; проваленная попытка —
    //    отдельная error-запись, не портит eff TPS успешных.
    // Сброс накопителя привязан к before_agent_start, а не к agent_start и не к
    // agent_settled: так свежий ход гарантированно стартует с чистого листа
    // независимо от того, отработал ли предыдущий agent_settled (например, после
    // аборта), а ретрай внутри хода статистику не стирает, потому что
    // before_agent_start не повторяется.
    function resetRun(userPrompt: string | null): void {
        // Снять наполнение завершённого хода ДО очистки — это база для % прироста.
        // Сброс безусловный: null тоже записывается, чтобы просроченная база
        // не переживала неизмеримый ход и не растягивала окно прироста.
        prevFill = turnFill(calls);
        userMsgTokens = userPrompt === null ? null : Math.round(charLen(userPrompt) / 4);
        runStartMs = Date.now();
        calls = [];
        pendingReqSentMs = null;
        pendingRespAtMs = null;
        openRecord = null;
    }

    pi.on("before_agent_start", (event) => {
        const prompt = (event as { prompt?: unknown }).prompt;
        resetRun(typeof prompt === "string" ? prompt : null);
    });

    pi.on("before_provider_request", () => {
        pendingReqSentMs = Date.now();
        pendingRespAtMs = null;
    });

    pi.on("after_provider_response", (event) => {
        if (event.status >= 400) return;
        pendingRespAtMs = Date.now();
    });

    pi.on("message_start", (event) => {
        if (!isAssistantMessage(event.message)) return;
        const record: CallRecord = {
            index: calls.length + 1,
            reqSentMs: pendingReqSentMs,
            respAtMs: pendingRespAtMs,
            firstTokenMs: null,
            lastDeltaMs: null,
            endMs: Date.now(),
            deltaCount: 0,
            textChars: 0,
            thinkingChars: 0,
            toolCallChars: 0,
            gaps: [],
            usage: null,
            stopReason: "pending",
        };
        calls.push(record);
        openRecord = record;
    });

    pi.on("message_update", (event) => {
        if (!isAssistantMessage(event.message)) return;
        const ev = event.assistantMessageEvent as { type?: string; delta?: string };
        const kind = ev?.type ?? "";
        if (!kind.endsWith("_delta")) return;
        const delta = ev?.delta ?? "";
        if (!delta) return;

        const rec = openRecord;
        if (!rec) return;

        const now = Date.now();
        if (rec.firstTokenMs === null) {
            rec.firstTokenMs = now;
        } else if (rec.lastDeltaMs !== null) {
            rec.gaps.push(now - rec.lastDeltaMs);
        }
        rec.lastDeltaMs = now;
        rec.deltaCount += 1;

        // Разделяем типы контента: только text_delta — «видимый» текст.
        if (kind === "text_delta") rec.textChars += charLen(delta);
        else if (kind === "thinking_delta") rec.thinkingChars += charLen(delta);
        else if (kind === "toolcall_delta") rec.toolCallChars += charLen(delta);
    });

    pi.on("message_end", (event) => {
        if (!isAssistantMessage(event.message)) return;
        const rec = openRecord;
        // Нет открытой записи → `message_end` без пары `message_start`.
        // Не финализируем предыдущую запись: она уже закрыта и корректна.
        if (!rec) return;
        rec.usage = event.message.usage ?? null;
        rec.stopReason = event.message.stopReason ?? "unknown";
        rec.endMs = Date.now();
        openRecord = null;
    });

    /**
     * Конец low-level run'а закрывает незакрытую запись.
     *
     * При обрыве потока `message_end` может не прийти, и `openRecord`
     * останется висеть на оборванной записи. Без этого сброса блуждающий
     * `message_end`, пришедший после `agent_end` но до следующего
     * `message_start`, финализировал бы оборванную запись чужими
     * `usage`/`stopReason` — и она выбыла бы из корзины незавершённых прямо
     * в итоги.
     *
     * Легитимную финализацию сброс не «съедает»: `message_end` всегда
     * предшествует `agent_end` внутри того же run'а, поэтому к моменту
     * `agent_end` открытой записи уже нет.
     */
    pi.on("agent_end", () => {
        openRecord = null;
    });

    function decodeTps(rec: CallRecord): number {
        if (rec.firstTokenMs === null || rec.lastDeltaMs === null) return 0;
        const sec = (rec.lastDeltaMs - rec.firstTokenMs) / 1000;
        if (sec <= 0) return 0;
        return num(rec.usage?.output) / sec;
    }

    function ttft(rec: CallRecord): number | null {
        if (rec.firstTokenMs === null || rec.reqSentMs === null) return null;
        return (rec.firstTokenMs - rec.reqSentMs) / 1000;
    }

    /**
     * Главная метрика: output / (запрос → последний токен).
     * Окно закрывается на ПОСЛЕДНЕМ ТОКЕНЕ (lastDeltaMs), а не на message_end:
     * в endMs попадает post-stream оверхед (финализация сообщения), который на
     * коротких ответах систематически занижал TPS.
     */
    function effTps(rec: CallRecord): number | null {
        if (rec.reqSentMs === null) return null;
        const lastTokenMs = rec.lastDeltaMs ?? rec.endMs;
        const sec = (lastTokenMs - rec.reqSentMs) / 1000;
        if (sec <= 0) return null;
        return num(rec.usage?.output) / sec;
    }

    function okCalls(): CallRecord[] {
        return calls.filter(isOk);
    }

    function aggregate(records: CallRecord[]) {
        let out = 0;
        let reasoning = 0;
        let chars = 0;
        let genSec = 0;
        let effSec = 0;
        const ttfts: number[] = [];

        for (const c of records) {
            const cOut = num(c.usage?.output);
            out += cOut;
            reasoning += num(c.usage?.reasoning);
            chars += c.textChars;
            const t = ttft(c);
            if (t !== null) ttfts.push(t);
            // Время в знаменатель — только у записей с токенами. Вызов без usage
            // (провайдер не отчитался) раздувал бы знаменатель и уронил
            // средние TPS до бессмысленных значений.
            if (cOut <= 0) continue;
            if (c.firstTokenMs !== null && c.lastDeltaMs !== null && c.lastDeltaMs > c.firstTokenMs) {
                genSec += (c.lastDeltaMs - c.firstTokenMs) / 1000;
            }
            if (c.reqSentMs !== null) {
                effSec += ((c.lastDeltaMs ?? c.endMs) - c.reqSentMs) / 1000;
            }
        }

        return { out, reasoning, chars, genSec, effSec, ttfts };
    }

    /** Промпт вызова = весь вход: input + cacheRead + cacheWrite. */
    function promptOf(rec: CallRecord): number {
        const u = rec.usage;
        if (!u) return 0;
        return num(u.input) + num(u.cacheRead) + num(u.cacheWrite);
    }

    /**
     * Наполнение разговора по итогам вызова: промпт + ответ.
     * cacheRead обязателен: при промпт-кэшировании основной объём диалога
     * лежит в cacheRead, а `input` — только некэшированный остаток; без
     * cacheRead база занижалась в разы и процент раздувался (+84% вместо ~1%).
     */
    function fillOf(rec: CallRecord): number | null {
        const p = promptOf(rec);
        if (p <= 0) return null;
        return p + num(rec.usage?.output);
    }

    /**
     * Наполнение хода: fillOf последнего вызова, отчитавшегося usage.
     * Если у такого вызова промпт нулевой (отчитан только output — типично
     * для OpenAI-совместимых прокси без prompt_tokens), наполнение считается
     * неизмеренным (null). Прежний откат к более раннему вызову тихо
     * подставлял устаревшее число и переносил ошибку через prevFill на
     * следующий ход в противоположную сторону.
     */
    function turnFill(records: CallRecord[]): number | null {
        const last = [...records].reverse().find((c) => c.usage !== null);
        return last ? fillOf(last) : null;
    }

    /**
     * «Старая» часть разговора на входе в ход = промпт первого вызова.
     * База, когда prevFill нет (первый ход после /reload или неизмеримый
     * прошлый ход): старый контекст уже виден во входе первого запроса,
     * поэтому процент считается сразу. Из базы вычитается userMsgTokens —
     * оценка сообщения пользователя этого хода, — иначе оно попадало в
     * знаменатель первого хода и в числитель последующих, и одинаковый
     * прирост давал разные проценты.
     */
    function entryFill(records: CallRecord[]): number | null {
        const first = records.find((c) => promptOf(c) > 0);
        return first ? promptOf(first) : null;
    }

    /**
     * Заполнение контекстного окна в процентах.
     */
    function fillPct(fill: number, ctxWindow: number): number {
        return ctxWindow > 0 ? (fill / ctxWindow) * 100 : 0;
    }

    /**
     * Прирост заполнения контекстного окна в процентных пунктах.
     *
     * Формула: (cur − prev) / ctxWindow × 100.
     * Показывает только дельту, без текущего процента — он уже виден в футере.
     * `—` если база или наполнение не измерены.
     */
    function contextGrowthLabel(base: number | null, cur: number | null, ctxWindow: number): string {
        if (cur === null || base === null || base <= 0 || ctxWindow <= 0) return "заполнение —";
        const delta = fillPct(cur, ctxWindow) - fillPct(base, ctxWindow);
        const sign = delta >= 0 ? "+" : "";
        return `заполнение ${sign}${fmt(delta, 1)}%`;
    }

    pi.on("agent_settled", (_event, ctx) => {
        if (!ctx.hasUI) return;
        if (runStartMs === null) return;

        const ok = okCalls();
        const agg = aggregate(ok);
        if (agg.out <= 0) return;

        const eff = agg.effSec > 0 ? agg.out / agg.effSec : 0;
        const wallSec = (Date.now() - runStartMs) / 1000;
        // База: наполнение прошлого хода (prevFill). Если его нет (первый ход
        // после /reload или прошлый ход не измерился) — «старая» часть
        // разговора на входе: промпт первого вызова минус оценка сообщения
        // пользователя, чтобы база совпадала по смыслу с prevFill.
        const entry = entryFill(calls);
        const base = prevFill ?? (entry === null ? null : Math.max(0, entry - (userMsgTokens ?? 0)));
        const contextWindow =
            ctx.model?.contextWindow ??
            (Number(process.env.PI_TPS_CONTEXT_WINDOW) || 0);
        const growth = contextGrowthLabel(base, turnFill(calls), contextWindow);
        const failedCount = calls.length - ok.length;

        const incompleteCount = calls.filter(isIncomplete).length;
        const badTail =
            (failedCount > 0
                ? ` · ❌ ${fmt(failedCount)} ${pluralRu(failedCount, "сбой", "сбоя", "сбоев")}`
                : "") +
            (incompleteCount > 0
                ? ` · ⚠️ ${fmt(incompleteCount)} ${pluralRu(incompleteCount, "вызов не завершён", "вызова не завершено", "вызовов не завершено")}`
                : "");

        const parts = [
            `⚡ ${fmt(eff, 1)} ток/с`,
            `TTFT ${fmt(avg(agg.ttfts), 2)}s`,
            `${fmt(agg.out)} out / ${fmt(agg.reasoning)} think`,
            growth,
            `${ok.length} выз. · ход ${fmt(wallSec, 1)}s${badTail}`,
        ];

        ctx.ui.notify(parts.join(" · "), "info");
    });

    pi.registerCommand("tps", {
        description: "Развёрнутая статистика генерации по каждому LLM-вызову",
        handler: async (_args, ctx) => {
            if (!ctx.hasUI) return;
            if (calls.length === 0) {
                ctx.ui.notify("pi-tps: нет данных — отправь сообщение агенту.", "info");
                return;
            }

            const lines: string[] = [];

            // ── Карточки вызовов: статус, токены, скорость, диагностика ──
            for (const c of calls) {
                const out = num(c.usage?.output);
                const hasOut = out > 0;
                const e = effTps(c);
                const d = decodeTps(c);
                const t = ttft(c);
                const reasoning = num(c.usage?.reasoning);
                const tokPerDelta = c.deltaCount > 0 ? out / c.deltaCount : 0;
                const status = isFailed(c)
                    ? `❌ ${c.stopReason}`
                    : isIncomplete(c)
                      ? "⚠️ не завершён"
                      : `✅ ${c.stopReason}`;

                lines.push(
                    `${status}  #${c.index} — ${fmt(out)} ток. out` +
                    (reasoning > 0 ? ` (reasoning ${fmt(reasoning)})` : "") +
                    ` · in ${fmt(num(c.usage?.input))}`
                );
                lines.push(
                    `   скорость ${e === null || !hasOut ? "—" : fmt(e, 1) + " ток/с"} · ` +
                    `decode ${hasOut ? fmt(d, 1) : "—"} ток/с · 1-й токен ${fmtSecOrDash(t)}`
                );
                const charParts: string[] = [];
                if (c.thinkingChars > 0) charParts.push(`thinking ${fmt(c.thinkingChars)}`);
                if (c.toolCallChars > 0) charParts.push(`toolcall ${fmt(c.toolCallChars)}`);
                lines.push(
                    `   ${fmt(c.textChars)} симв.` +
                    (charParts.length > 0 ? ` (+${charParts.join(" · ")})` : "") +
                    ` · ${c.deltaCount} дельт · ~${hasOut ? fmt(tokPerDelta, 2) : "—"} ток/дельта · ` +
                    `паузы p50 ${fmt(pct(c.gaps, 0.5))}ms / p95 ${fmt(pct(c.gaps, 0.95))}ms`
                );
                lines.push("");
            }

            // ── Итоги по успешным вызовам ──
            const ok = okCalls();
            const agg = aggregate(ok);
            const failed = calls.filter(isFailed);
            const incomplete = calls.filter(isIncomplete);
            // Полный промпт (input + cacheRead + cacheWrite), как в метрике
            // «контекст +N%» уведомления; сырой input под тем же словом
            // «контекст» давал два разных определения в одном отчёте.
            const maxPrompt = Math.max(0, ...calls.map((c) => promptOf(c)));
            const totalCacheRead = ok.reduce((a, c) => a + num(c.usage?.cacheRead), 0);
            const totalCost = ok.reduce((a, c) => a + num(c.usage?.cost?.total), 0);
            const eff = agg.effSec > 0 ? agg.out / agg.effSec : 0;
            const decode = agg.genSec > 0 ? agg.out / agg.genSec : 0;
            const charsPerSec = agg.genSec > 0 ? agg.chars / agg.genSec : 0;

            const okLabel = pluralRu(ok.length, "успешный вызов", "успешных вызова", "успешных вызовов");
            if (ok.length === 0) {
                lines.push("── ИТОГИ ─ нет успешных вызовов, считать нечего ─");
            } else {
                lines.push(`── ИТОГИ · ${fmt(ok.length)} ${okLabel} ──`);
                lines.push(`⚡ Скорость:    ${fmt(eff, 2)} ток/с  — твоя реальная: запрос → последний токен`);
                lines.push(`⏳ Decode:      ${fmt(decode, 2)} ток/с  — 1-й → последний токен, завышен буфером`);
                lines.push(
                    `⏱  1-й токен:   ${fmt(avg(agg.ttfts), 2)}s avg` +
                    (agg.ttfts.length > 0 ? ` · ${fmt(maxOf(agg.ttfts), 2)}s max` : "")
                );
                lines.push(`🔤 Токены:      ${fmt(agg.out)} out = текст ${fmt(agg.out - agg.reasoning)} + reasoning ${fmt(agg.reasoning)}`);
                lines.push(`🗒  Контекст:    max ${fmt(maxPrompt)} · cache read ${fmt(totalCacheRead)}`);
                lines.push(`💬 Символы:     ${fmt(agg.chars)} текст · ${fmt(charsPerSec, 1)} зн/с`);
                if (totalCost > 0) {
                    lines.push(`💰 Cost:        $${totalCost.toFixed(4)}`);
                }
            }

            // Эвристика на спекулятивное декодирование.
            const totalDeltas = ok.reduce((a, c) => a + c.deltaCount, 0);
            const tokPerDelta = totalDeltas > 0 ? agg.out / totalDeltas : 0;
            if (tokPerDelta > SPECULATIVE_TOK_PER_DELTA_THRESHOLD) {
                lines.push(`🔮 ~${fmt(tokPerDelta, 2)} ток/дельта — похоже на speculative decoding (MTP)`);
            }
            if (totalCacheRead === 0 && maxPrompt > 10_000) {
                lines.push(`ℹ️  cache read 0 — сервер не отдаёт кэш-статистику в клиент`);
            }

            const excluded: string[] = [];
            if (failed.length > 0) {
                excluded.push(`❌ ${fmt(failed.length)} ${pluralRu(failed.length, "сбойный вызов", "сбойных вызова", "сбойных вызовов")}`);
            }
            if (incomplete.length > 0) {
                excluded.push(`⚠️ ${fmt(incomplete.length)} ${pluralRu(incomplete.length, "незавершённый вызов", "незавершённых вызова", "незавершённых вызовов")}`);
            }
            if (excluded.length > 0) {
                lines.push(`Исключено из итогов: ${excluded.join(" · ")}`);
            }

            lines.push("");
            lines.push("Скорость (eff) = запрос → последний токен; совпадает с wall сервера (<5%).");
            lines.push("Decode = 1-й → последний токен; завышен буфером 1-й дельты — только для диагностики.");

            ctx.ui.notify(lines.join("\n"), "info");
        },
    });
}
