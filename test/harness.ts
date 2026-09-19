// Smoke-харнес pi-tps: мок ExtensionAPI + виртуальные часы.
// Запуск: node test/harness.ts
// Сценарии: 1) обычный ход из двух вызовов 2) ретрай посреди хода 3) pending + /tps

let clock = 1_000_000;
Date.now = () => clock;
const advance = (ms: number) => {
    clock += ms;
};

type Handler = (event: any, ctx: any) => unknown;

const notifications: string[] = [];
const ctx = {
    hasUI: true,
    ui: {
        notify(msg: string, type?: string) {
            notifications.push(msg);
            console.log(`\n┌── NOTIFY (${type}) ${"─".repeat(50)}`);
            console.log(msg.split("\n").map((l) => "│ " + l).join("\n"));
            console.log("└" + "─".repeat(62));
        },
    },
};

const handlers = new Map<string, Handler[]>();
const commands = new Map<string, { description: string; handler: (args: string, ctx: any) => Promise<unknown> }>();

const pi: any = {
    on(event: string, h: Handler) {
        if (!handlers.has(event)) handlers.set(event, []);
        handlers.get(event)!.push(h);
    },
    registerCommand(name: string, def: any) {
        commands.set(name, def);
    },
};

import tps from "../extensions/tps.ts";

tps(pi);

async function emit(type: string, payload: any = {}) {
    for (const h of handlers.get(type) ?? []) await h(payload, ctx);
}

function assistantMsg(extra: Record<string, unknown> = {}) {
    return { role: "assistant", content: [], ...extra };
}

interface CallOpts {
    reqAt?: number;
    respDelay?: number;
    startDelay?: number;
    deltas?: Array<{ kind: string; text: string; gap: number }>;
    endDelay?: number;
    usage?: Record<string, unknown>;
    stopReason?: string;
    emitEnd?: boolean;
    respStatus?: number;
}

async function streamCall(opts: CallOpts) {
    advance(opts.reqAt ?? 0);
    await emit("before_provider_request", {});
    advance(opts.respDelay ?? 1000);
    await emit("after_provider_response", { status: opts.respStatus ?? 200 });
    advance(opts.startDelay ?? 0);
    await emit("message_start", { message: assistantMsg() });
    for (const d of opts.deltas ?? []) {
        advance(d.gap);
        await emit("message_update", {
            message: assistantMsg(),
            assistantMessageEvent: { type: d.kind, delta: d.text },
        });
    }
    if (opts.emitEnd !== false) {
        advance(opts.endDelay ?? 50);
        await emit("message_end", {
            message: assistantMsg({ usage: opts.usage, stopReason: opts.stopReason ?? "stop" }),
        });
    }
}

function textDeltas(n: number, gap: number, prefix = "word") {
    return Array.from({ length: n }, (_, i) => ({ kind: "text_delta", text: `${prefix}${i} `, gap }));
}

let failures = 0;
function check(name: string, cond: boolean) {
    console.log(`${cond ? "✔" : "✘"} ${name}`);
    if (!cond) failures++;
}

async function main() {
    // ───────── Сценарий 1: обычный ход, два успешных вызова ─────────
    console.log("\n═══ Сценарий 1: обычный ход (2 вызова) ═══");
    await emit("before_agent_start"); // новый ход пользователя → сброс накопителя
    await emit("agent_start");

    // Вызов 1: TTFT 1.6s, eff = 100/2.1 = 47.6, decode = 100/0.5 = 200
    await streamCall({
        reqAt: 0,
        respDelay: 1500,
        startDelay: 100,
        deltas: [
            { kind: "thinking_delta", text: "хм, думаю ", gap: 100 },
            { kind: "thinking_delta", text: "о задаче ", gap: 100 },
            ...textDeltas(10, 40),
        ],
        endDelay: 50,
        usage: { input: 1000, output: 100, cacheRead: 500, cacheWrite: 0, reasoning: 20 },
        stopReason: "toolUse",
    });

    advance(500); // тулзы

    // Вызов 2: TTFT 0.83s, eff = 250/2.3 = 108.7
    await streamCall({
        reqAt: 3000,
        respDelay: 800,
        startDelay: 0,
        deltas: textDeltas(50, 30),
        endDelay: 50,
        usage: { input: 1200, output: 250, cacheRead: 600, cacheWrite: 0 },
        stopReason: "stop",
    });

    const n1 = notifications.length;
    await emit("agent_settled");
    check("agent_settled дал ровно одно уведомление", notifications.length === n1 + 1);
    check("уведомление содержит скорость и 1-й токен", /⚡ .* ток\/с/.test(notifications[n1]) && notifications[n1].includes("1-й токен"));
    check("нет упоминания сбоев", !notifications[n1].includes("сбой"));

    // /tps по сценарию 1: корректные множественные формы и суммарный cache read
    notifications.length = 0;
    await commands.get("tps")!.handler("", ctx);
    const r1 = notifications[0] ?? "";
    check("/tps: «2 успешных вызова» (склонение)", r1.includes("2 успешных вызова"));
    check("/tps: cache read 1,100 (500+600)", r1.includes("cache read 1,100"));
    check("/tps: reasoning выделен", r1.includes("+ reasoning 20"));

    // ───────── Сценарий 2: ретрай посреди хода (agent_start повторно) ─────────
    console.log("\n═══ Сценарий 2: ретрай — данные не должны потеряться ═══");
    await emit("before_agent_start"); // новый ход пользователя → сброс
    await emit("agent_start"); // low-level run 1

    // Провальный вызов (500)
    await streamCall({
        reqAt: 0,
        respDelay: 300,
        respStatus: 500,
        startDelay: 0,
        deltas: [],
        usage: undefined,
        stopReason: "error",
    });
    await emit("agent_end", { messages: [] }); // pi эмитит agent_end, потом retry → agent_start
    advance(2000); // backoff
    await emit("agent_start"); // повторный low-level run — НЕ должен стереть вызовы

    // Успешный вызов после ретрая
    await streamCall({
        reqAt: 0,
        respDelay: 700,
        deltas: textDeltas(20, 50),
        usage: { input: 1300, output: 100, cacheRead: 500, cacheWrite: 0 },
        stopReason: "stop",
    });

    const n2 = notifications.length;
    await emit("agent_settled");
    check("после ретрая уведомление есть", notifications.length === n2 + 1);
    check("сбой показан в уведомлении", notifications[n2].includes("❌ 1 сбой"));

    // /tps должен показать оба вызова
    notifications.length = 0;
    await commands.get("tps")!.handler("", ctx);
    const report = notifications[0] ?? "";
    check("/tps показывает #1 error", report.includes("❌ error") && report.includes("#1"));
    check("/tps показывает #2 stop", report.includes("✅ stop") && report.includes("#2"));
    check("/tps показывает исключённые", report.includes("Исключено из итогов"));
    check("/tps показывает cache read", report.includes("cache read 500"));

    // ───────── Сценарий 3: незавершённая запись (message_end не пришёл) ─────────
    console.log("\n═══ Сценарий 3: pending-запись ═══");
    await emit("before_agent_start"); // новый ход → сброс
    await emit("agent_start");
    await streamCall({
        reqAt: 0,
        respDelay: 500,
        deltas: textDeltas(5, 40),
        emitEnd: false, // обрыв
    });
    const n3 = notifications.length;
    await emit("agent_settled");
    check("pending не даёт итогов (out=0 → нет уведомления)", notifications.length === n3);

    notifications.length = 0;
    await commands.get("tps")!.handler("", ctx);
    check("/tps помечает ⚠️ не завершён", (notifications[0] ?? "").includes("⚠️ не завершён"));

    // ───────── Итог ─────────
    console.log(`\n${failures === 0 ? "✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" : `❌ ПРОВАЛЕНО ПРОВЕРОК: ${failures}`}`);
    process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
