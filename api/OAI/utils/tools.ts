import * as z from "@/common/myZod.ts";

import { ToolCall } from "../types/tools.ts";

const TOOL_CALL_BLOCK_REGEX = /<tool_call>([\s\S]*?)<\/tool_call>/g;
const TOOL_CALL_ARG_REGEX =
    /<arg_key>([\s\S]*?)<\/arg_key>\s*<arg_value>([\s\S]*?)<\/arg_value>/g;
const TOOL_CALL_START_TOKEN = "<tool_call>";
const TOOL_CALL_END_TOKEN = "</tool_call>";

export const TOOL_CALL_SCHEMA = {
    $schema: "http://json-schema.org/draft-07/schema#",
    type: "array",
    items: {
        type: "object",
        properties: {
            function: {
                type: "object",
                properties: {
                    name: { type: "string" },
                    arguments: {
                        // Converted to OAI's string in post process
                        type: "object",
                    },
                },
                required: ["name", "arguments"],
            },
        },
        required: ["function"],
    },
};

function normalizeToolCallArray(raw: unknown): unknown[] | null {
    const normalizeToolCall = (value: unknown): Record<string, unknown> | null => {
        if (!value || typeof value !== "object") {
            return null;
        }

        const rawObject = value as Record<string, unknown>;

        if (rawObject.function && typeof rawObject.function === "object") {
            return rawObject;
        }

        const name = typeof rawObject.name === "string"
            ? rawObject.name
            : typeof rawObject.tool_name === "string"
            ? rawObject.tool_name
            : typeof rawObject.function_name === "string"
            ? rawObject.function_name
            : undefined;

        const args = rawObject.arguments ?? rawObject.parameters ??
            rawObject.args ?? rawObject.input;

        if (name) {
            return {
                function: {
                    name,
                    arguments: args ?? {},
                },
            };
        }

        const entries = Object.entries(rawObject);
        if (entries.length === 1) {
            const [key, value] = entries[0];
            return {
                function: {
                    name: key,
                    arguments: value ?? {},
                },
            };
        }

        return null;
    };

    if (Array.isArray(raw)) {
        const normalized = raw
            .map((item) => normalizeToolCall(item))
            .filter((item) => item !== null);
        return normalized.length > 0 ? normalized : null;
    }

    if (raw && typeof raw === "object") {
        const rawObject = raw as Record<string, unknown>;
        if (Array.isArray(rawObject.tool_calls)) {
            const normalized = rawObject.tool_calls
                .map((item) => normalizeToolCall(item))
                .filter((item) => item !== null);
            return normalized.length > 0 ? normalized : null;
        }
        if ("function" in rawObject) {
            return [rawObject];
        }
        const normalized = normalizeToolCall(rawObject);
        if (normalized) {
            return [normalized];
        }
    }

    return null;
}

function parseToolCallArray(raw: unknown[]): ToolCall[] {
    const ToolListSchema = z.array(ToolCall);
    const toolCalls = ToolListSchema.parse(raw);
    return toolCalls.map((toolCall) => {
        if (typeof toolCall.function.arguments !== "string") {
            toolCall.function.arguments = JSON.stringify(
                toolCall.function.arguments,
            );
        }
        return toolCall;
    });
}

function parseJsonToolCalls(toolCallsString: string): ToolCall[] | null {
    const trimmed = toolCallsString.trim();
    if (!trimmed) {
        return null;
    }

    try {
        const parsed = JSON.parse(trimmed);
        const normalized = normalizeToolCallArray(parsed);
        if (!normalized) {
            return null;
        }
        return parseToolCallArray(normalized);
    } catch {
        return null;
    }
}

function parseXmlToolCallBlock(block: string): ToolCall[] {
    const argMatches = Array.from(block.matchAll(TOOL_CALL_ARG_REGEX));
    if (argMatches.length > 0) {
        const firstIndex = argMatches[0].index ?? 0;
        const name = block.slice(0, firstIndex).trim();
        if (!name) {
            return [];
        }

        const args: Record<string, unknown> = {};
        for (const match of argMatches) {
            const key = match[1]?.trim();
            if (!key) {
                continue;
            }

            const valueText = match[2]?.trim() ?? "";
            let value: unknown = valueText;
            try {
                value = JSON.parse(valueText);
            } catch {
                // Leave as string when it's not valid JSON.
            }
            args[key] = value;
        }

        return parseToolCallArray([
            {
                function: {
                    name,
                    arguments: args,
                },
            },
        ]);
    }

    const trimmed = block.trim();
    if (!trimmed) {
        return [];
    }

    const parsedFromJson = parseJsonToolCalls(trimmed);
    return parsedFromJson ?? [];
}

function parseXmlToolCalls(toolCallsString: string): ToolCall[] {
    const toolCalls: ToolCall[] = [];
    for (const match of toolCallsString.matchAll(TOOL_CALL_BLOCK_REGEX)) {
        const block = match[1] ?? "";
        const parsed = parseXmlToolCallBlock(block);
        if (parsed.length > 0) {
            toolCalls.push(...parsed);
        }
    }
    return toolCalls;
}

export class ToolCallProcessor {
    static fromJson(toolCallsString: string) {
        const parsedJson = parseJsonToolCalls(toolCallsString);
        if (parsedJson) {
            return parsedJson;
        }

        const parsedXml = parseXmlToolCalls(toolCallsString);
        if (parsedXml.length > 0) {
            return parsedXml;
        }

        throw new Error("Unable to parse tool calls.");
    }

    static extractFromText(text: string) {
        const xmlToolCalls = parseXmlToolCalls(text);
        if (xmlToolCalls.length > 0) {
            const startIndex = text.indexOf("<tool_call>");
            const content = startIndex >= 0
                ? text.slice(0, startIndex).trimEnd()
                : text.trimEnd();
            return { toolCalls: xmlToolCalls, content };
        }

        const jsonToolCalls = parseJsonToolCalls(text);
        if (jsonToolCalls) {
            return { toolCalls: jsonToolCalls, content: "" };
        }

        return { toolCalls: [], content: text };
    }
}

export function createInlineToolCallParser() {
    let buffer = "";
    let toolBuffer = "";
    let inToolCall = false;
    const toolCalls: ToolCall[] = [];

    const trailingStartPrefix = (text: string) => {
        const maxLen = Math.min(
            text.length,
            TOOL_CALL_START_TOKEN.length - 1,
        );
        for (let i = maxLen; i > 0; i--) {
            if (TOOL_CALL_START_TOKEN.startsWith(text.slice(-i))) {
                return i;
            }
        }
        return 0;
    };

    const process = (text: string) => {
        buffer += text;
        let output = "";

        while (buffer.length > 0) {
            if (!inToolCall) {
                const startIdx = buffer.indexOf(TOOL_CALL_START_TOKEN);
                if (startIdx === -1) {
                    const keepLen = trailingStartPrefix(buffer);
                    const emitLen = buffer.length - keepLen;
                    if (emitLen > 0) {
                        output += buffer.slice(0, emitLen);
                        buffer = buffer.slice(emitLen);
                    }
                    break;
                }

                output += buffer.slice(0, startIdx);
                buffer = buffer.slice(startIdx);
                inToolCall = true;
                toolBuffer = "";
            } else {
                const endIdx = buffer.indexOf(TOOL_CALL_END_TOKEN);
                if (endIdx === -1) {
                    toolBuffer += buffer;
                    buffer = "";
                    break;
                }

                toolBuffer += buffer.slice(0, endIdx + TOOL_CALL_END_TOKEN.length);
                buffer = buffer.slice(endIdx + TOOL_CALL_END_TOKEN.length);
                inToolCall = false;

                const extracted = ToolCallProcessor.extractFromText(toolBuffer);
                if (extracted.toolCalls.length > 0) {
                    toolCalls.push(...extracted.toolCalls);
                } else {
                    output += toolBuffer;
                }
                toolBuffer = "";
            }
        }

        return output;
    };

    const flush = () => {
        let output = "";

        if (!inToolCall) {
            output += buffer;
        } else {
            const extracted = ToolCallProcessor.extractFromText(
                toolBuffer + buffer,
            );
            if (extracted.toolCalls.length > 0) {
                toolCalls.push(...extracted.toolCalls);
            } else {
                output += toolBuffer + buffer;
            }
        }

        buffer = "";
        toolBuffer = "";
        inToolCall = false;
        return output;
    };

    return {
        process,
        flush,
        toolCalls,
    };
}
