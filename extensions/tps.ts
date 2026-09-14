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

function fmt(n: number, digits = 0): string {
    return n.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Unicode-correct подсчёт символов (surrogate pairs = 1 символ). */
function charLen(s: string): number {
    return Array.from(s).length;
}

function pct(xs: number[], p: number): number {
    if (xs.length === 0) return 0;
    const s = [...xs].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))];
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

function isFailed(rec: CallRecord): boolean {
    return rec.stopReason === "error" || rec.stopReason === "aborted";
}

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
    // Семантика pi (проверено по исходникам agent-loop.js / agent-session.js):
    //  - agent_start/agent_end эмитятся на КАЖДЫЙ low-level run: ретраи,
    //    авто-компакция и continuation'ы (agent.continue()) перевзводят эту пару.
    //  - agent_settled эмитится ОДИН раз за ход пользователя.
    //  - каждое continue() — это новый provider request, поэтому before_provider_request
    //    перевзводится и reqSentMs каждой попытки свежий; проваленная попытка —
    //    отдельная error-запись, не портит eff TPS успешных.
    // Значит сбрасывать накопленные данные можно только когда предыдущий ход
    // уже устаканился, иначе ретрай посреди хода сотрёт статистику.
    let runActive = false;

    pi.on("agent_start", () => {
        if (!runActive) {
            runStartMs = Date.now();
            calls = [];
            pendingReqSentMs = null;
            pendingRespAtMs = null;
        }
        runActive = true;
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
        calls.push({
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
        });
    });

    pi.on("message_update", (event) => {
        if (!isAssistantMessage(event.message)) return;
        const ev = event.assistantMessageEvent as { type?: string; delta?: string };
        const kind = ev?.type ?? "";
        if (!kind.endsWith("_delta")) return;
        const delta = ev?.delta ?? "";
        if (!delta) return;

        const rec = calls[calls.length - 1];
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
        const rec = calls[calls.length - 1];
        if (!rec) return;
        rec.usage = event.message.usage ?? null;
        rec.stopReason = event.message.stopReason ?? "unknown";
        rec.endMs = Date.now();
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

    pi.on("agent_settled", (_event, ctx) => {
        runActive = false;
        if (!ctx.hasUI) return;
        if (runStartMs === null) return;

        const ok = okCalls();
        const agg = aggregate(ok);
        if (agg.out <= 0) return;

        const wallSec = (Date.now() - runStartMs) / 1000;
        const eff = agg.effSec > 0 ? agg.out / agg.effSec : 0;
        // Последний НЕНУЛЕВОЙ usage: если финальный вызов провалился без usage,
        // «контекст 0» вводил бы в заблуждение.
        const lastUsage = [...calls].reverse().find((c) => c.usage !== null)?.usage ?? null;
        const lastInput = num(lastUsage?.input);
        const failedCount = calls.length - ok.length;

        const parts = [
            `⚡ ${fmt(eff, 1)} tok/s`,
            `TTFT ${fmt(avg(agg.ttfts), 2)}s`,
            `out ${fmt(agg.out)} (текст ${fmt(agg.out - agg.reasoning)})`,
            `${fmt(agg.chars)} симв.`,
            `контекст ${fmt(lastInput)}`,
            `${fmt(wallSec, 1)}s · ${ok.length} выз.` + (failedCount > 0 ? ` (+${failedCount} сбойных)` : ""),
        ];

        ctx.ui.notify(parts.join(" · "), "info");
    });

    pi.registerCommand("tps", {
        description: "Развёрнутая статистика генерации по каждому LLM-вызову",
        handler: async (_args, ctx) => {
            if (!ctx.hasUI) return;
            if (calls.length === 0) {
                ctx.ui.notify("Нет данных — отправь сообщение агенту.", "info");
                return;
            }

            const lines: string[] = [];

            for (const c of calls) {
                const out = num(c.usage?.output);
                const d = decodeTps(c);
                const e = effTps(c);
                const t = ttft(c);
                const reasoning = num(c.usage?.reasoning);
                const tokPerDelta = c.deltaCount > 0 ? out / c.deltaCount : 0;
                const flag = isFailed(c) ? " ❌" : "";

                lines.push(
                    `#${c.index}${flag}  out ${fmt(out)}  in ${fmt(num(c.usage?.input))}  ` +
                    `eff ${e === null ? "—" : fmt(e, 1) + " tok/s"}  ` +
                    `(decode ${fmt(d, 1)} tok/s)  ` +
                    `TTFT ${fmtSecOrDash(t)}  ` +
                    `${c.deltaCount} дельт (~${fmt(tokPerDelta, 2)} ток/дельта)  ` +
                    `gaps p50 ${fmt(pct(c.gaps, 0.5))}ms p95 ${fmt(pct(c.gaps, 0.95))}ms  ` +
                    `stop=${c.stopReason}` +
                    (reasoning > 0 ? `  reasoning ${fmt(reasoning)}` : "")
                );
            }

            const ok = okCalls();
            const agg = aggregate(ok);
            const failedCount = calls.length - ok.length;
            const maxInput = Math.max(0, ...calls.map((c) => num(c.usage?.input)));
            const totalCacheRead = ok.reduce((a, c) => a + num(c.usage?.cacheRead), 0);
            const totalCost = ok.reduce((a, c) => a + num(c.usage?.cost?.total), 0);
            const eff = agg.effSec > 0 ? agg.out / agg.effSec : 0;
            const decode = agg.genSec > 0 ? agg.out / agg.genSec : 0;
            const charsPerSec = agg.genSec > 0 ? agg.chars / agg.genSec : 0;

            lines.push("─".repeat(60));
            lines.push(
                `Итого: out ${fmt(agg.out)} (текст ${fmt(agg.out - agg.reasoning)} · reasoning ${fmt(agg.reasoning)}) · ` +
                `контекст (max) ${fmt(maxInput)} · ${fmt(agg.genSec, 1)}s генерации`
            );
            lines.push(`Символы: ${fmt(agg.chars)} · ${fmt(charsPerSec, 1)} зн/с`);
            lines.push(
                `TTFT: avg ${fmt(avg(agg.ttfts), 2)}s · max ${agg.ttfts.length > 0 ? fmt(maxOf(agg.ttfts), 2) + "s" : "—"}`
            );
            lines.push(`Eff TPS: ${fmt(eff, 2)} tok/s  ← запрос→последний токен (совпадает со стеной сервера)`);
            lines.push(`Decode:  ${fmt(decode, 2)} tok/s  (первый→последний дельта; завышен из-за буфера первой дельты)`);
            if (failedCount > 0) {
                lines.push(`Сбойных вызовов (error/aborted): ${failedCount} — исключены из итогов`);
            }

            if (totalCacheRead > 0) {
                lines.push(`Cache read: ${fmt(totalCacheRead)}`);
            } else if (maxInput > 10_000) {
                lines.push(`Cache read: 0 — сервер не отдаёт кэш-статистику в клиент`);
            }
            if (totalCost > 0) {
                lines.push(`Cost: $${totalCost.toFixed(4)}`);
            }

            // Эвристика на спекулятивное декодирование.
            const totalDeltas = ok.reduce((a, c) => a + c.deltaCount, 0);
            const tokPerDelta = totalDeltas > 0 ? agg.out / totalDeltas : 0;
            if (tokPerDelta > 1.5) {
                lines.push(`~${fmt(tokPerDelta, 2)} ток/дельта — похоже на speculative decoding (MTP)`);
            }

            ctx.ui.notify(lines.join("\n"), "info");
        },
    });
}
