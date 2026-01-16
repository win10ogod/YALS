import * as z from "@/common/myZod.ts";

import { ToolCall } from "../types/tools.ts";
import { ToolCallFormat } from "@/common/templating.ts";

const TOOL_CALL_BLOCK_REGEX = /<tool_call>([\s\S]*?)<\/tool_call>/g;
const TOOL_CALL_ARG_REGEX =
    /<arg_key>([\s\S]*?)<\/arg_key>\s*<arg_value>([\s\S]*?)<\/arg_value>/g;
const TOOL_CALL_START_TOKEN = "<tool_call>";
const TOOL_CALL_END_TOKEN = "</tool_call>";

export interface ToolParserConfig {
    format?: ToolCallFormat;
    startToken?: string;
    endToken?: string;
}

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

/**
 * Normalize a single tool call object to OAI format
 * Supports multiple formats:
 * - OAI format: {"function": {"name": "...", "arguments": "..."}}
 * - Simple format (IQuest): {"name": "...", "arguments": {...}}
 * - Hermes format: {"name": "...", "parameters": {...}}
 */
function normalizeToolCallObject(
    raw: Record<string, unknown>,
): Record<string, unknown> | null {
    // Already in OAI format
    if ("function" in raw && typeof raw.function === "object") {
        return raw;
    }

    // Simple format: {"name": "...", "arguments": {...}}
    if ("name" in raw && typeof raw.name === "string") {
        const args = raw.arguments ?? raw.parameters ?? {};
        return {
            function: {
                name: raw.name,
                arguments: args,
            },
        };
    }

    return null;
}

function normalizeToolCallArray(raw: unknown): unknown[] | null {
    if (Array.isArray(raw)) {
        // Normalize each item in the array
        const normalized = raw
            .map((item) => {
                if (item && typeof item === "object") {
                    return normalizeToolCallObject(
                        item as Record<string, unknown>,
                    );
                }
                return null;
            })
            .filter((item) => item !== null);
        return normalized.length > 0 ? normalized : null;
    }

    if (raw && typeof raw === "object") {
        const rawObject = raw as Record<string, unknown>;

        // Check for tool_calls wrapper
        if (Array.isArray(rawObject.tool_calls)) {
            return normalizeToolCallArray(rawObject.tool_calls);
        }

        // Single tool call object
        const normalized = normalizeToolCallObject(rawObject);
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

/**
 * Parse a single tool call block content (inside <tool_call>...</tool_call>)
 * Supports multiple formats:
 * - XML key-value format (GLM): function_name\n<arg_key>k</arg_key><arg_value>v</arg_value>
 * - Simple JSON format (IQuest): {"name": "...", "arguments": {...}}
 * - OAI JSON format: {"function": {"name": "...", "arguments": "..."}}
 */
function parseXmlToolCallBlock(
    block: string,
    format?: ToolCallFormat,
): ToolCall[] {
    const trimmed = block.trim();
    if (!trimmed) {
        return [];
    }

    // If format is explicitly specified, try that first
    if (format === "xml_kv") {
        const result = parseXmlKvFormat(trimmed);
        if (result.length > 0) return result;
    } else if (format === "json_simple" || format === "json_oai") {
        const result = parseJsonToolCalls(trimmed);
        if (result) return result;
    }

    // Auto-detect format
    // Try XML key-value format first (GLM style)
    const argMatches = Array.from(trimmed.matchAll(TOOL_CALL_ARG_REGEX));
    if (argMatches.length > 0) {
        return parseXmlKvFormat(trimmed);
    }

    // Try JSON format (IQuest/OAI style)
    const parsedFromJson = parseJsonToolCalls(trimmed);
    if (parsedFromJson) {
        return parsedFromJson;
    }

    return [];
}

/**
 * Parse XML key-value format tool calls (GLM style)
 * Format: function_name\n<arg_key>k</arg_key><arg_value>v</arg_value>...
 */
function parseXmlKvFormat(block: string): ToolCall[] {
    const argMatches = Array.from(block.matchAll(TOOL_CALL_ARG_REGEX));
    if (argMatches.length === 0) {
        return [];
    }

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

function parseXmlToolCalls(
    toolCallsString: string,
    format?: ToolCallFormat,
): ToolCall[] {
    const toolCalls: ToolCall[] = [];
    for (const match of toolCallsString.matchAll(TOOL_CALL_BLOCK_REGEX)) {
        const block = match[1] ?? "";
        const parsed = parseXmlToolCallBlock(block, format);
        if (parsed.length > 0) {
            toolCalls.push(...parsed);
        }
    }
    return toolCalls;
}

export class ToolCallProcessor {
    /**
     * Parse tool calls from a JSON or XML string
     */
    static fromJson(toolCallsString: string, config?: ToolParserConfig) {
        const format = config?.format;

        // Try JSON first if format suggests it
        if (!format || format === "json_simple" || format === "json_oai") {
            const parsedJson = parseJsonToolCalls(toolCallsString);
            if (parsedJson) {
                return parsedJson;
            }
        }

        // Try XML wrapped tool calls
        const parsedXml = parseXmlToolCalls(toolCallsString, format);
        if (parsedXml.length > 0) {
            return parsedXml;
        }

        throw new Error("Unable to parse tool calls.");
    }

    /**
     * Extract tool calls from text content, returning remaining content
     */
    static extractFromText(text: string, config?: ToolParserConfig) {
        const format = config?.format;
        const startToken = config?.startToken ?? TOOL_CALL_START_TOKEN;

        // Try XML wrapped tool calls first
        const xmlToolCalls = parseXmlToolCalls(text, format);
        if (xmlToolCalls.length > 0) {
            const startIndex = text.indexOf(startToken);
            const content = startIndex >= 0
                ? text.slice(0, startIndex).trimEnd()
                : text.trimEnd();
            return { toolCalls: xmlToolCalls, content };
        }

        // Try plain JSON
        const jsonToolCalls = parseJsonToolCalls(text);
        if (jsonToolCalls) {
            return { toolCalls: jsonToolCalls, content: "" };
        }

        return { toolCalls: [], content: text };
    }
}

/**
 * Create a streaming tool call parser for incremental text processing
 */
export function createInlineToolCallParser(config?: ToolParserConfig) {
    const startToken = config?.startToken ?? TOOL_CALL_START_TOKEN;
    const endToken = config?.endToken ?? TOOL_CALL_END_TOKEN;
    const format = config?.format;

    let buffer = "";
    let toolBuffer = "";
    let inToolCall = false;
    const toolCalls: ToolCall[] = [];

    const trailingStartPrefix = (text: string) => {
        const maxLen = Math.min(text.length, startToken.length - 1);
        for (let i = maxLen; i > 0; i--) {
            if (startToken.startsWith(text.slice(-i))) {
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
                const startIdx = buffer.indexOf(startToken);
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
                const endIdx = buffer.indexOf(endToken);
                if (endIdx === -1) {
                    toolBuffer += buffer;
                    buffer = "";
                    break;
                }

                toolBuffer += buffer.slice(0, endIdx + endToken.length);
                buffer = buffer.slice(endIdx + endToken.length);
                inToolCall = false;

                const extracted = ToolCallProcessor.extractFromText(
                    toolBuffer,
                    { format, startToken, endToken },
                );
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
                { format, startToken, endToken },
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
