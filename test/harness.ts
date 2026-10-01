// Smoke-харнес pi-tps: мок ExtensionAPI + виртуальные часы.
// Запуск: node test/harness.ts
// Сценарии: 1) обычный ход из двух вызовов 2) ретрай посреди хода 3) pending + /tps
// 4) stray message_end 5) обрыв + agent_end
// 6) граничные случаи метрики «контекст»: кэш, компакция, неизмеримый ход,
//    согласованность баз entryFill/prevFill, обрыв без agent_settled

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
function check(name: string, cond: boolean, actual?: unknown) {
    console.log(`${cond ? "✔" : "✘"} ${name}${cond ? "" : ` — факт: ${JSON.stringify(actual)}`}`);
    if (!cond) failures++;
}

// Один полный самодостаточный ход: before_agent_start (+ опциональный prompt
// пользователя) → agent_start → run → agent_settled; возвращает уведомление.
async function notifyOnce(run: () => Promise<void>, userPrompt?: string): Promise<string> {
    await emit("before_agent_start", userPrompt === undefined ? {} : { prompt: userPrompt });
    await emit("agent_start");
    await run();
    notifications.length = 0;
    await emit("agent_settled");
    return notifications[0] ?? "";
}

const growthOf = (s: string): string => s.match(/контекст (\S+)/)?.[1] ?? "(нет)";

// Калибровка без межсценарных зависимостей: первый ход устанавливает базу
// baseFill (prompt = baseFill−100, output = 100), второй ход даёт curUsage;
// возвращается процент из уведомления второго хода (база — prevFill).
async function growthFrom(baseFill: number, curUsage: Record<string, unknown>): Promise<string> {
    await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: baseFill - 100, output: 100, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    });
    return growthOf(await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: curUsage,
            stopReason: "stop",
        });
    }));
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
    check(
        "уведомление: скорость, TTFT, out/think, контекст +37% (первый ход считается сразу), ход 8.1s",
        /⚡ .* ток\/с/.test(notifications[n1]) &&
            notifications[n1].includes("TTFT") &&
            notifications[n1].includes("350 out / 20 think") &&
            notifications[n1].includes("контекст +37%") &&
            notifications[n1].includes("ход 8.1s")
    );
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
    // Явная установка базы: наполнение 2050 (1200+600+0+250), чтобы сценарий 2
    // не зависел от фикстур сценария 1 (prevFill снимается на следующем
    // before_agent_start).
    await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(25, 30),
            usage: { input: 1200, output: 250, cacheRead: 600, cacheWrite: 0 },
            stopReason: "stop",
        });
    });
    await emit("before_agent_start"); // новый ход пользователя → prevFill = 2050
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

    // Успешный вызов после ретрая. Наполнение = промпт+output = (1600+500+0)+100 = 2200.
    // База (прошлый ход) = (1200+600+0)+250 = 2050 → (2200−2050)/2050 = +7%.
    // Формула = (после − до)/до, как в эталоне пользователя 100k→110k = +10%.
    await streamCall({
        reqAt: 0,
        respDelay: 700,
        deltas: textDeltas(20, 50),
        usage: { input: 1600, output: 100, cacheRead: 500, cacheWrite: 0 },
        stopReason: "stop",
    });

    const n2 = notifications.length;
    await emit("agent_settled");
    check("после ретрая уведомление есть", notifications.length === n2 + 1);
    check("сбой показан в уведомлении", notifications[n2].includes("❌ 1 сбой"));
    check("контекст +7% от прошлого наполнения (2050→2200, с cacheRead)", notifications[n2].includes("контекст +7%"));

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

    // ───────── Сценарий 4: message_end без пары message_start ─────────
    console.log("\n═══ Сценарий 4: stray message_end не портит закрытую запись ═══");
    await emit("before_agent_start");
    await emit("agent_start");
    await streamCall({
        reqAt: 0,
        respDelay: 500,
        deltas: textDeltas(10, 40),
        usage: { input: 1000, output: 100, cacheRead: 0, cacheWrite: 0 },
        stopReason: "stop",
    });

    // Блуждающий message_end без предшествующего message_start.
    // До openRecord он перезаписал бы usage/stopReason записи #1.
    advance(100);
    await emit("message_end", {
        message: assistantMsg({ usage: { input: 999, output: 999 }, stopReason: "error" }),
    });

    notifications.length = 0;
    await commands.get("tps")!.handler("", ctx);
    const r4 = notifications[0] ?? "";
    check("#1 остался ✅ stop (не стал error)", r4.includes("✅ stop") && r4.includes("#1"));
    check("stray-событие не создало запись #2", !r4.includes("#2"));
    check("output #1 остался 100 (не 999)", r4.includes("100 ток. out") && !r4.includes("999"));

    // ───────── Сценарий 5: agent_end закрывает оборванную запись ─────────
    console.log("\n═══ Сценарий 5: обрыв потока + agent_end ═══");
    await emit("before_agent_start");
    await emit("agent_start");
    // Вызов оборвался: message_start был, message_end не пришёл.
    await streamCall({
        reqAt: 0,
        respDelay: 500,
        deltas: textDeltas(5, 40),
        emitEnd: false,
    });
    await emit("agent_end", { messages: [] }); // обязан закрыть openRecord

    // Блуждающий message_end ПОСЛЕ agent_end.
    advance(100);
    await emit("message_end", {
        message: assistantMsg({ usage: { input: 777, output: 777 }, stopReason: "stop" }),
    });

    notifications.length = 0;
    await commands.get("tps")!.handler("", ctx);
    const r5 = notifications[0] ?? "";
    check("оборванная запись осталась ⚠️ не завершён", r5.includes("⚠️ не завершён"));
    check("stray message_end не вписал 777", !r5.includes("777"));
    check("успешных итогов нет (все вызовы в корзине)", r5.includes("нет успешных вызовов"));

    // ───────── Сценарий 6: граничные случаи метрики «контекст» ─────────
    console.log("\n═══ Сценарий 6: метрика «контекст» — границы ═══");

    // 6.1 Эталон пользователя: 100k → 110k = +10% (prevFill-ветка)
    const g61 = await growthFrom(100_000, { input: 1000, output: 10_000, cacheRead: 99_000, cacheWrite: 0 });
    check("6.1 эталон 100k→110k = +10%", g61 === "+10%", g61);

    // 6.2 input=0, весь вход в кэше: база считается (ловит мутацию «input<=0 → 0»)
    const g62 = await growthFrom(10_000, { input: 0, output: 100, cacheRead: 10_900, cacheWrite: 0 });
    check("6.2 input=0, весь вход в кэше (10000→11000) = +10%", g62 === "+10%", g62);

    // 6.3 cacheWrite входит в наполнение (ловит удаление слагаемого из promptOf)
    const g63 = await growthFrom(10_000, { input: 0, output: 100, cacheRead: 0, cacheWrite: 10_900 });
    check("6.3 cacheWrite входит в наполнение (10000→11000) = +10%", g63 === "+10%", g63);

    // 6.4 компакция: промпт упал — минус на знаке
    const g64 = await growthFrom(10_000, { input: 500, output: 100, cacheRead: 3_400, cacheWrite: 0 });
    check("6.4 компакция 10000→4000 = −60%", g64 === "-60%", g64);

    // 6.5 наполнение не измерено (prompt=0 во всех вызовах) → «—»
    const g65 = await growthFrom(10_000, { input: 0, output: 100, cacheRead: 0, cacheWrite: 0 });
    check("6.5 prompt=0 → «—»", g65 === "—", g65);

    // 6.6 usage без cache-полей: prompt = input
    const g66 = await growthFrom(10_000, { input: 10_900, output: 100 });
    check("6.6 usage без cache-полей: prompt = input (10000→11000) = +10%", g66 === "+10%", g66);

    // 6.7 последний вызов отчитался только output'ом (prompt=0) → «—»,
    // а не устаревшее наполнение предыдущего вызова
    await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: 10_000, output: 100, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }); // fill = 10100
    const g67 = growthOf(await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: 10_100, output: 100, cacheRead: 0, cacheWrite: 0 },
            stopReason: "toolUse",
        });
        await streamCall({
            deltas: textDeltas(20, 20),
            usage: { input: 0, output: 500, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }));
    check("6.7 вызов с output без промпта → «—» (не откат к прошлому вызову)", g67 === "—", g67);

    // 6.8 согласованность базы: ход 1 (entryFill − оценка сообщения) и ход 2
    // (prevFill) при одинаковом содержательном приросте дают одинаковый процент.
    // Ход 1: старый контекст 10000 + сообщение 500 ток. (2000 симв.) + ответ 200.
    // База = 10500 − 500 = 10000 → рост 700/10000 = +7%.
    const g68a = growthOf(await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: 10_500, output: 200, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }, "x".repeat(2000)));
    // Ход 2: база prevFill = 10700, снова сообщение 500 + ответ 200 → fill 11400
    // → 700/10700 = 6.5% → +7%.
    const g68b = growthOf(await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: 11_200, output: 200, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }, "x".repeat(2000)));
    check("6.8 ход1 (entryFill−msg) и ход2 (prevFill) при равном приросте: оба +7%",
        g68a === "+7%" && g68b === "+7%", [g68a, g68b]);

    // 6.9 неизмеримый ход обнуляет базу: следующий ход показывает свой
    // прирост, а не накопленный с момента последнего измерения
    await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: 10_000, output: 100, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }); // fill = 10100
    await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(20, 20),
            usage: { input: 0, output: 4_000, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }); // неизмерим (prompt=0) → prevFill обязан стать null
    const g69 = growthOf(await notifyOnce(async () => {
        await streamCall({
            deltas: textDeltas(10, 20),
            usage: { input: 14_100, output: 100, cacheRead: 0, cacheWrite: 0 },
            stopReason: "stop",
        });
    }));
    // База = entryFill = 14100 (prevFill null), fill = 14200 → 0.7% → +1%.
    // С протухшей базой 10100 было бы +41%.
    check("6.9 неизмеримый ход не оставляет протухшую базу (+1%, не +41%)", g69 === "+1%", g69);

    // 6.10 ход без agent_settled (обрыв): снимок prevFill в before_agent_start
    // уцелевает — ровно то, ради чего снимок привязан к before_agent_start
    await emit("before_agent_start", {});
    await emit("agent_start");
    await streamCall({
        deltas: textDeltas(10, 20),
        usage: { input: 1_500, output: 100, cacheRead: 0, cacheWrite: 0 },
        stopReason: "stop",
    });
    // agent_settled НЕ эмитим — ход оборвался
    await emit("before_agent_start", {}); // resetRun обязан снять prevFill = 1600
    await emit("agent_start");
    await streamCall({
        deltas: textDeltas(10, 20),
        usage: { input: 1_600, output: 100, cacheRead: 0, cacheWrite: 0 },
        stopReason: "stop",
    });
    notifications.length = 0;
    await emit("agent_settled");
    const g610 = growthOf(notifications[0] ?? "");
    check("6.10 обрыв без agent_settled: база 1600 уцелела → +6%", g610 === "+6%", g610);

    // 6.11 пустой ход (before_agent_start без вызовов) базу не ломает:
    // следующий ход считает от своего entryFill
    await emit("before_agent_start", {}); // пустой ход: prevFill = turnFill([]) = null
    await emit("before_agent_start", {});
    await emit("agent_start");
    await streamCall({
        deltas: textDeltas(10, 20),
        usage: { input: 1_700, output: 100, cacheRead: 0, cacheWrite: 0 },
        stopReason: "stop",
    });
    notifications.length = 0;
    await emit("agent_settled");
    const g611 = growthOf(notifications[0] ?? "");
    check("6.11 пустой ход не ломает базу: следующий от entryFill 1700 → +6%", g611 === "+6%", g611);

    // ───────── Итог ─────────
    console.log(`\n${failures === 0 ? "✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" : `❌ ПРОВАЛЕНО ПРОВЕРОК: ${failures}`}`);
    process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
