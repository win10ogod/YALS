import { PromptTemplate } from "@/common/templating.ts";

import { ToolCall } from "../types/tools.ts";
import { ToolCallProcessor } from "./tools.ts";

export interface OutputParsingOptions {
    promptTemplate?: PromptTemplate;
    allowToolParse: boolean;
    includeReasoning: boolean;
}

export interface ParsedOutput {
    content: string;
    reasoning?: string;
    toolCalls?: ToolCall[];
}

export interface ParsedOutputDelta {
    content?: string;
    reasoning?: string;
    toolCalls?: ToolCall[];
}

export interface StreamingOutputParser {
    process(text: string): ParsedOutputDelta;
    flush(): ParsedOutputDelta;
    toolCalls: ToolCall[];
}

interface ParseHints {
    reasoningStart?: string;
    reasoningEnd?: string;
    toolCallStart?: string;
    toolCallEnd?: string;
    toolCallsStart?: string;
    toolCallsEnd?: string;
    toolCallSep?: string;
}

interface ToolPattern {
    name: string;
    start: string;
    end?: string;
    parse: (block: string) => ToolCall[];
}

const QWEN_TOOL_CALLS_BEGIN = "<\uFF5Ctool\u2581calls\u2581begin\uFF5C>";
const QWEN_TOOL_CALL_BEGIN = "<\uFF5Ctool\u2581call\u2581begin\uFF5C>";
const QWEN_TOOL_SEP = "<\uFF5Ctool\u2581sep\uFF5C>";
const QWEN_TOOL_CALL_END = "<\uFF5Ctool\u2581call\u2581end\uFF5C>";
const QWEN_TOOL_CALLS_END = "<\uFF5Ctool\u2581calls\u2581end\uFF5C>";

const KIMI_TOOL_CALLS_BEGIN = "<|tool_calls_section_begin|>";
const KIMI_TOOL_CALLS_END = "<|tool_calls_section_end|>";
const KIMI_TOOL_CALL_BEGIN = "<|tool_call_begin|>";
const KIMI_TOOL_CALL_END = "<|tool_call_end|>";
const KIMI_TOOL_CALL_ARG = "<|tool_call_argument_begin|>";

const DEFAULT_TOOL_CALL_START = "<tool_call>";
const DEFAULT_TOOL_CALL_END = "</tool_call>";
const DEFAULT_TOOL_CALLS_START = "<tool_calls>";
const DEFAULT_TOOL_CALLS_END = "</tool_calls>";

const TOOL_CALL_PREFIX = "<|tool_call|>";
const TOOLS_PREFIX = "<|tools_prefix|>";
const TOOLS_SUFFIX = "<|tools_suffix|>";

const SEED_TOOL_CALL_START = "<seed:tool_call>";
const SEED_TOOL_CALL_END = "</seed:tool_call>";

const DEFAULT_REASONING_START = "<think>";
const DEFAULT_REASONING_END = "</think>";
const ALT_REASONING_START = "[THINK]";
const ALT_REASONING_END = "[/THINK]";

function toArray<T>(value?: T | T[]): T[] {
    if (value === undefined) {
        return [];
    }
    return Array.isArray(value) ? value : [value];
}

function escapeRegex(text: string) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stripCodeFence(text: string) {
    const trimmed = text.trim();
    const fenceMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)```$/i);
    if (fenceMatch) {
        return fenceMatch[1].trim();
    }
    return trimmed;
}

function parseJsonText(text: string): ToolCall[] {
    try {
        return ToolCallProcessor.fromJson(text);
    } catch {
        return [];
    }
}

function parseToolCallXmlBlock(start: string, block: string, end: string) {
    const wrapped = `${start}${block}${end}`;
    try {
        const extracted = ToolCallProcessor.extractFromText(wrapped);
        return extracted.toolCalls;
    } catch {
        return [];
    }
}

function parseToolCallsTagBlock(block: string) {
    return parseJsonText(stripCodeFence(block));
}

function parseToolCallPrefixBlock(block: string) {
    return parseJsonText(stripCodeFence(block));
}

function parseQwenToolCalls(block: string, sepToken: string) {
    const toolCalls: ToolCall[] = [];
    let cursor = 0;
    while (cursor < block.length) {
        const startIdx = block.indexOf(QWEN_TOOL_CALL_BEGIN, cursor);
        if (startIdx === -1) {
            break;
        }
        const endIdx = block.indexOf(QWEN_TOOL_CALL_END, startIdx + QWEN_TOOL_CALL_BEGIN.length);
        if (endIdx === -1) {
            break;
        }
        const rawCall = block.slice(startIdx + QWEN_TOOL_CALL_BEGIN.length, endIdx);
        const sepIdx = rawCall.indexOf(sepToken);
        if (sepIdx !== -1) {
            const head = rawCall.slice(0, sepIdx).trim();
            let rest = rawCall.slice(sepIdx + sepToken.length).trim();
            let name = head;
            let argsText = rest;
            if (head.toLowerCase() === "function") {
                const newlineIdx = rest.search(/\r?\n/);
                if (newlineIdx !== -1) {
                    name = rest.slice(0, newlineIdx).trim();
                    argsText = rest.slice(newlineIdx + 1).trim();
                } else {
                    name = rest.trim();
                    argsText = "";
                }
            }
            argsText = stripCodeFence(argsText);
            let args: unknown = argsText;
            try {
                args = JSON.parse(argsText);
            } catch {
                // Leave args as string if JSON parsing fails.
            }
            toolCalls.push(
                ToolCall.parse({
                    function: {
                        name,
                        arguments: args ?? "{}",
                    },
                }),
            );
        }
        cursor = endIdx + QWEN_TOOL_CALL_END.length;
    }
    return toolCalls;
}

function parseKimiToolCalls(block: string) {
    const toolCalls: ToolCall[] = [];
    const pattern = new RegExp(
        `${escapeRegex(KIMI_TOOL_CALL_BEGIN)}([\\s\\S]*?)${escapeRegex(KIMI_TOOL_CALL_ARG)}([\\s\\S]*?)${escapeRegex(KIMI_TOOL_CALL_END)}`,
        "g",
    );

    for (const match of block.matchAll(pattern)) {
        const rawName = (match[1] ?? "").trim();
        const rawArgs = stripCodeFence(match[2] ?? "");
        const namePart = rawName.startsWith("functions.")
            ? rawName.slice("functions.".length)
            : rawName;
        const name = namePart.split(":")[0]?.trim() ?? "";
        let args: unknown = rawArgs;
        try {
            args = JSON.parse(rawArgs);
        } catch {
            // Keep raw args text.
        }
        if (name) {
            toolCalls.push(
                ToolCall.parse({
                    function: {
                        name,
                        arguments: args ?? "{}",
                    },
                }),
            );
        }
    }

    return toolCalls;
}

function parseSeedToolCalls(block: string) {
    const toolCalls: ToolCall[] = [];
    const functionMatch = block.match(/<function=([^>]+)>/);
    if (!functionMatch) {
        return toolCalls;
    }
    const name = functionMatch[1]?.trim();
    if (!name) {
        return toolCalls;
    }

    const args: Record<string, unknown> = {};
    const paramRegex = /<parameter=([^>]+)>([\s\S]*?)<\/parameter>/g;
    for (const match of block.matchAll(paramRegex)) {
        const key = match[1]?.trim();
        if (!key) {
            continue;
        }
        const rawValue = (match[2] ?? "").trim();
        let value: unknown = rawValue;
        try {
            value = JSON.parse(rawValue);
        } catch {
            // Leave as string if not JSON.
        }
        args[key] = value;
    }

    toolCalls.push(
        ToolCall.parse({
            function: {
                name,
                arguments: args,
            },
        }),
    );

    return toolCalls;
}

function getParsingHints(promptTemplate?: PromptTemplate): ParseHints {
    if (!promptTemplate) {
        return {};
    }

    const metadata = promptTemplate.metadata;

    return {
        reasoningStart: metadata.reasoning_start,
        reasoningEnd: metadata.reasoning_end,
        toolCallStart: metadata.tool_call_start,
        toolCallEnd: metadata.tool_call_end,
        toolCallsStart: metadata.tool_calls_start,
        toolCallsEnd: metadata.tool_calls_end,
        toolCallSep: metadata.tool_call_sep,
    };
}

function buildToolPatterns(hints: ParseHints): ToolPattern[] {
    const patterns: ToolPattern[] = [];

    const toolCallsStart = hints.toolCallsStart;
    const toolCallsEnd = hints.toolCallsEnd;
    const toolCallStart = hints.toolCallStart;
    const toolCallEnd = hints.toolCallEnd;
    const toolCallSep = hints.toolCallSep;

    if (toolCallsStart && toolCallsEnd) {
        if (toolCallsStart === QWEN_TOOL_CALLS_BEGIN || toolCallsEnd === QWEN_TOOL_CALLS_END) {
            patterns.push({
                name: "qwen_tool_calls",
                start: toolCallsStart,
                end: toolCallsEnd,
                parse: (block) => parseQwenToolCalls(block, toolCallSep ?? QWEN_TOOL_SEP),
            });
        } else if (toolCallsStart === KIMI_TOOL_CALLS_BEGIN || toolCallsEnd === KIMI_TOOL_CALLS_END) {
            patterns.push({
                name: "kimi_tool_calls",
                start: toolCallsStart,
                end: toolCallsEnd,
                parse: parseKimiToolCalls,
            });
        } else {
            patterns.push({
                name: "tool_calls_tag",
                start: toolCallsStart,
                end: toolCallsEnd,
                parse: parseToolCallsTagBlock,
            });
        }
    }

    if (toolCallStart && toolCallEnd) {
        if (toolCallStart === SEED_TOOL_CALL_START || toolCallEnd === SEED_TOOL_CALL_END) {
            patterns.push({
                name: "seed_tool_call",
                start: toolCallStart,
                end: toolCallEnd,
                parse: parseSeedToolCalls,
            });
        } else {
            patterns.push({
                name: "tool_call_tag",
                start: toolCallStart,
                end: toolCallEnd,
                parse: (block) => parseToolCallXmlBlock(toolCallStart, block, toolCallEnd),
            });
        }
    } else if (toolCallStart && toolCallStart === TOOL_CALL_PREFIX) {
        patterns.push({
            name: "tool_call_prefix",
            start: toolCallStart,
            parse: parseToolCallPrefixBlock,
        });
    }

    if (toolCallStart === undefined && toolCallEnd === undefined) {
        patterns.push({
            name: "default_tool_call",
            start: DEFAULT_TOOL_CALL_START,
            end: DEFAULT_TOOL_CALL_END,
            parse: (block) => parseToolCallXmlBlock(DEFAULT_TOOL_CALL_START, block, DEFAULT_TOOL_CALL_END),
        });
    }

    if (toolCallsStart === undefined && toolCallsEnd === undefined) {
        patterns.push({
            name: "default_tool_calls",
            start: DEFAULT_TOOL_CALLS_START,
            end: DEFAULT_TOOL_CALLS_END,
            parse: parseToolCallsTagBlock,
        });
    }

    if (!toolCallsStart && !toolCallsEnd) {
        patterns.push({
            name: "tools_prefix",
            start: TOOLS_PREFIX,
            end: TOOLS_SUFFIX,
            parse: parseToolCallsTagBlock,
        });
    }

    return patterns;
}

function extractDelimitedBlocks(text: string, start: string, end: string) {
    const blocks: string[] = [];
    let content = "";
    let cursor = 0;

    while (cursor < text.length) {
        const startIdx = text.indexOf(start, cursor);
        if (startIdx === -1) {
            content += text.slice(cursor);
            break;
        }
        content += text.slice(cursor, startIdx);
        const endIdx = text.indexOf(end, startIdx + start.length);
        if (endIdx === -1) {
            content += text.slice(startIdx);
            break;
        }
        blocks.push(text.slice(startIdx + start.length, endIdx));
        cursor = endIdx + end.length;
    }

    return { blocks, content };
}

function extractToolCalls(text: string, hints: ParseHints) {
    let content = text;
    const toolCalls: ToolCall[] = [];

    const patterns = buildToolPatterns(hints);

    for (const pattern of patterns) {
        if (!pattern.end) {
            const startIdx = content.indexOf(pattern.start);
            if (startIdx === -1) {
                continue;
            }
            const block = content.slice(startIdx + pattern.start.length);
            const parsed = pattern.parse(block);
            if (parsed.length > 0) {
                toolCalls.push(...parsed);
                content = content.slice(0, startIdx).trimEnd();
                return { toolCalls, content };
            }
            continue;
        }

        if (!content.includes(pattern.start)) {
            continue;
        }

        const extracted = extractDelimitedBlocks(content, pattern.start, pattern.end);
        if (extracted.blocks.length === 0) {
            continue;
        }
        for (const block of extracted.blocks) {
            const parsed = pattern.parse(block);
            if (parsed.length > 0) {
                toolCalls.push(...parsed);
            }
        }
        content = extracted.content.trimEnd();
    }

    if (toolCalls.length === 0) {
        try {
            const extracted = ToolCallProcessor.extractFromText(content);
            if (extracted.toolCalls.length > 0) {
                return { toolCalls: extracted.toolCalls, content: extracted.content };
            }
        } catch {
            // Ignore parse failures.
        }
    }

    return { toolCalls, content };
}

function extractReasoning(
    text: string,
    reasoningStart?: string,
    reasoningEnd?: string,
) {
    if (!reasoningEnd) {
        return { content: text, reasoning: undefined };
    }

    const startToken = reasoningStart;
    const endToken = reasoningEnd;

    const startIdx = startToken ? text.indexOf(startToken) : -1;
    const endIdx = text.indexOf(endToken);

    if (startToken && startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
        const reasoning = text.slice(startIdx + startToken.length, endIdx);
        const content = text.slice(endIdx + endToken.length);
        return { content, reasoning };
    }

    if (endIdx !== -1) {
        const reasoning = text.slice(0, endIdx);
        const content = text.slice(endIdx + endToken.length);
        return { content, reasoning };
    }

    return { content: text, reasoning: undefined };
}

function trailingTokenPrefix(text: string, tokens: string[]) {
    let max = 0;
    for (const token of tokens) {
        const limit = Math.min(text.length, token.length - 1);
        for (let i = limit; i > 0; i--) {
            if (token.startsWith(text.slice(-i))) {
                max = Math.max(max, i);
                break;
            }
        }
    }
    return max;
}

export function parseOutputText(
    text: string,
    options: OutputParsingOptions,
): ParsedOutput {
    const hints = getParsingHints(options.promptTemplate);

    let content = text;
    let toolCalls: ToolCall[] = [];
    let reasoning: string | undefined;

    if (options.allowToolParse) {
        const extracted = extractToolCalls(content, hints);
        toolCalls = extracted.toolCalls;
        content = extracted.content;
    }

    const hasAltReasoning = content.includes(ALT_REASONING_START) ||
        content.includes(ALT_REASONING_END);
    const reasoningStart = hints.reasoningStart ??
        (hasAltReasoning ? ALT_REASONING_START : undefined) ??
        DEFAULT_REASONING_START;
    const reasoningEnd = hints.reasoningEnd ??
        (hasAltReasoning ? ALT_REASONING_END : undefined) ??
        DEFAULT_REASONING_END;

    const reasoningResult = extractReasoning(content, reasoningStart, reasoningEnd);
    content = reasoningResult.content;
    reasoning = reasoningResult.reasoning;

    return {
        content,
        reasoning: options.includeReasoning ? reasoning : undefined,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    };
}

export function createStreamingOutputParser(
    options: OutputParsingOptions,
): StreamingOutputParser {
    const hints = getParsingHints(options.promptTemplate);

    const toolPatterns = options.allowToolParse ? buildToolPatterns(hints) : [];
    const reasoningStart = hints.reasoningStart;
    const reasoningEnd = hints.reasoningEnd;

    const startTokens = [reasoningStart, ...toolPatterns.map((pattern) => pattern.start)]
        .filter((token) => !!token) as string[];
    const endTokens = reasoningEnd ? [reasoningEnd] : [];

    let buffer = "";
    let toolBuffer = "";
    let reasoningBuffer = "";
    let mode: "normal" | "tool" | "reasoning" = "normal";
    let activePattern: ToolPattern | null = null;

    const toolCalls: ToolCall[] = [];

    const processNormal = (output: ParsedOutputDelta) => {
        while (buffer.length > 0) {
            let nextTool: { pattern: ToolPattern; index: number } | null = null;
            for (const pattern of toolPatterns) {
                const idx = buffer.indexOf(pattern.start);
                if (idx === -1) {
                    continue;
                }
                if (!nextTool || idx < nextTool.index) {
                    nextTool = { pattern, index: idx };
                }
            }

            const reasoningIdx = reasoningStart ? buffer.indexOf(reasoningStart) : -1;
            const toolIdx = nextTool ? nextTool.index : -1;

            let nextIdx = -1;
            let nextType: "reasoning" | "tool" | null = null;
            if (reasoningIdx !== -1 && (toolIdx === -1 || reasoningIdx < toolIdx)) {
                nextIdx = reasoningIdx;
                nextType = "reasoning";
            } else if (toolIdx !== -1) {
                nextIdx = toolIdx;
                nextType = "tool";
            }

            if (nextIdx === -1 || !nextType) {
                const keepLen = trailingTokenPrefix(buffer, startTokens);
                const emitLen = buffer.length - keepLen;
                if (emitLen > 0) {
                    output.content = (output.content ?? "") + buffer.slice(0, emitLen);
                    buffer = buffer.slice(emitLen);
                }
                break;
            }

            if (nextIdx > 0) {
                output.content = (output.content ?? "") + buffer.slice(0, nextIdx);
                buffer = buffer.slice(nextIdx);
            }

            if (nextType === "reasoning") {
                if (!reasoningStart) {
                    break;
                }
                buffer = buffer.slice(reasoningStart.length);
                mode = "reasoning";
                reasoningBuffer = "";
                break;
            }

            if (nextType === "tool" && nextTool) {
                buffer = buffer.slice(nextTool.pattern.start.length);
                mode = "tool";
                activePattern = nextTool.pattern;
                toolBuffer = "";
                break;
            }
        }
    };

    const processReasoning = (output: ParsedOutputDelta) => {
        if (!reasoningEnd) {
            reasoningBuffer += buffer;
            buffer = "";
            return;
        }

        const endIdx = buffer.indexOf(reasoningEnd);
        if (endIdx === -1) {
            const keepLen = trailingTokenPrefix(buffer, endTokens);
            const emitLen = buffer.length - keepLen;
            if (emitLen > 0) {
                reasoningBuffer += buffer.slice(0, emitLen);
                buffer = buffer.slice(emitLen);
            }
            return;
        }

        reasoningBuffer += buffer.slice(0, endIdx);
        buffer = buffer.slice(endIdx + reasoningEnd.length);
        if (options.includeReasoning && reasoningBuffer) {
            output.reasoning = (output.reasoning ?? "") + reasoningBuffer;
        }
        reasoningBuffer = "";
        mode = "normal";
    };

    const processTool = (output: ParsedOutputDelta) => {
        if (!activePattern) {
            mode = "normal";
            return;
        }

        if (!activePattern.end) {
            toolBuffer += buffer;
            buffer = "";
            return;
        }

        const endIdx = buffer.indexOf(activePattern.end);
        if (endIdx === -1) {
            const keepLen = trailingTokenPrefix(buffer, [activePattern.end]);
            const emitLen = buffer.length - keepLen;
            if (emitLen > 0) {
                toolBuffer += buffer.slice(0, emitLen);
                buffer = buffer.slice(emitLen);
            }
            return;
        }

        toolBuffer += buffer.slice(0, endIdx);
        buffer = buffer.slice(endIdx + activePattern.end.length);
        const parsed = activePattern.parse(toolBuffer);
        if (parsed.length > 0) {
            toolCalls.push(...parsed);
            output.toolCalls = parsed;
        }
        toolBuffer = "";
        mode = "normal";
        activePattern = null;
    };

    const process = (text: string): ParsedOutputDelta => {
        const output: ParsedOutputDelta = {};
        buffer += text;

        while (buffer.length > 0) {
            if (mode === "normal") {
                processNormal(output);
                if (mode === "normal") {
                    break;
                }
            } else if (mode === "reasoning") {
                processReasoning(output);
                if (mode === "reasoning") {
                    break;
                }
            } else if (mode === "tool") {
                processTool(output);
                if (mode === "tool") {
                    break;
                }
            }
        }

        return output;
    };

    const flush = (): ParsedOutputDelta => {
        const output: ParsedOutputDelta = {};

        if (mode === "reasoning") {
            reasoningBuffer += buffer;
            buffer = "";
            if (options.includeReasoning && reasoningBuffer) {
                output.reasoning = reasoningBuffer;
            }
            reasoningBuffer = "";
            mode = "normal";
        } else if (mode === "tool") {
            toolBuffer += buffer;
            buffer = "";
            if (activePattern) {
                const parsed = activePattern.parse(toolBuffer);
                if (parsed.length > 0) {
                    toolCalls.push(...parsed);
                    output.toolCalls = parsed;
                }
            }
            toolBuffer = "";
            mode = "normal";
            activePattern = null;
        }

        if (buffer.length > 0) {
            output.content = buffer;
            buffer = "";
        }

        return output;
    };

    return {
        process,
        flush,
        toolCalls,
    };
}
