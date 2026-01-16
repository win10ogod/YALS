import { ToolCall } from "@/api/OAI/types/tools.ts";
import { BaseToolCallParser } from "../base.ts";
import { ToolCallExtractionResult, ToolCallParserConfig } from "../types.ts";

/**
 * Generic XML tool call parser
 *
 * Handles <tool_call>...</tool_call> format with various content formats.
 * Supports:
 * - JSON content
 * - XML-style arg_key/arg_value pairs
 * - Function name followed by arguments
 */
export class GenericXmlToolCallParser extends BaseToolCallParser {
    private readonly ARG_REGEX =
        /<arg_key>([\s\S]*?)<\/arg_key>\s*<arg_value>([\s\S]*?)<\/arg_value>/g;

    constructor(config?: Partial<ToolCallParserConfig>) {
        super({
            format: "generic_xml",
            startToken: config?.startToken ?? "<tool_call>",
            endToken: config?.endToken ?? "</tool_call>",
            parallelCalls: true,
            ...config,
        });
    }

    extractToolCalls(text: string): ToolCallExtractionResult {
        const startToken = this.getStartToken();
        const endToken = this.getEndToken();

        const regex = new RegExp(
            `${this.escapeRegex(startToken)}([\\s\\S]*?)${this.escapeRegex(endToken)}`,
            "g"
        );

        const toolCalls: ToolCall[] = [];
        let lastIndex = 0;
        let cleanContent = "";

        let match;
        while ((match = regex.exec(text)) !== null) {
            cleanContent += text.slice(lastIndex, match.index);

            try {
                const parsed = this.parseToolCallContent(match[1]);
                toolCalls.push(...parsed);
            } catch {
                cleanContent += match[0];
            }

            lastIndex = match.index + match[0].length;
        }

        cleanContent += text.slice(lastIndex);

        return {
            toolCalls,
            content: cleanContent.trim(),
        };
    }

    protected parseToolCallContent(content: string): ToolCall[] {
        const trimmed = content.trim();
        if (!trimmed) {
            return [];
        }

        // Try XML arg_key/arg_value format first
        const argMatches = Array.from(trimmed.matchAll(this.ARG_REGEX));
        if (argMatches.length > 0) {
            return this.parseXmlArgFormat(trimmed, argMatches);
        }

        // Try JSON format
        try {
            const parsed = JSON.parse(trimmed);

            if (Array.isArray(parsed)) {
                return parsed.map((item) => this.normalizeToolCall(item));
            }

            return [this.normalizeToolCall(parsed)];
        } catch {
            // Not valid JSON
        }

        // Try to extract function name and arguments from text
        return this.parseTextFormat(trimmed);
    }

    /**
     * Parse XML arg_key/arg_value format
     */
    private parseXmlArgFormat(
        content: string,
        argMatches: RegExpMatchArray[]
    ): ToolCall[] {
        const firstIndex = argMatches[0].index ?? 0;
        const name = content.slice(0, firstIndex).trim();

        if (!name) {
            return [];
        }

        const args: Record<string, unknown> = {};
        for (const match of argMatches) {
            const key = match[1]?.trim();
            if (!key) continue;

            const valueText = match[2]?.trim() ?? "";
            let value: unknown = valueText;

            // Try to parse as JSON
            try {
                value = JSON.parse(valueText);
            } catch {
                // Keep as string
            }

            args[key] = value;
        }

        return [
            {
                id: this.generateToolCallId(),
                type: "function",
                function: {
                    name,
                    arguments: JSON.stringify(args),
                },
            },
        ];
    }

    /**
     * Parse plain text format (function name on first line, then args)
     */
    private parseTextFormat(content: string): ToolCall[] {
        const lines = content.split("\n");
        if (lines.length === 0) {
            return [];
        }

        const name = lines[0].trim();
        if (!name) {
            return [];
        }

        // Try to parse remaining lines as JSON
        const argsText = lines.slice(1).join("\n").trim();
        let args: Record<string, unknown> = {};

        if (argsText) {
            try {
                args = JSON.parse(argsText);
            } catch {
                // Not valid JSON, skip
                return [];
            }
        }

        return [
            {
                id: this.generateToolCallId(),
                type: "function",
                function: {
                    name,
                    arguments: JSON.stringify(args),
                },
            },
        ];
    }

    private normalizeToolCall(data: Record<string, unknown>): ToolCall {
        // Handle {"function": {...}} format
        if (data.function && typeof data.function === "object") {
            const func = data.function as Record<string, unknown>;
            return {
                id: (data.id as string) ?? this.generateToolCallId(),
                type: "function",
                function: {
                    name: String(func.name ?? ""),
                    arguments:
                        typeof func.arguments === "string"
                            ? func.arguments
                            : JSON.stringify(func.arguments ?? {}),
                },
            };
        }

        // Handle {"name": ..., "arguments": ...} format
        return {
            id: (data.id as string) ?? this.generateToolCallId(),
            type: "function",
            function: {
                name: String(data.name ?? ""),
                arguments:
                    typeof data.arguments === "string"
                        ? data.arguments
                        : JSON.stringify(
                              data.arguments ?? data.parameters ?? {}
                          ),
            },
        };
    }

    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
}

/**
 * Generic JSON tool call parser
 *
 * Handles tool calls in pure JSON format without XML tags.
 * Looks for JSON objects or arrays with tool call structure.
 */
export class GenericJsonToolCallParser extends BaseToolCallParser {
    constructor(config?: Partial<ToolCallParserConfig>) {
        super({
            format: "generic_json",
            startToken: "",
            endToken: "",
            parallelCalls: true,
            ...config,
        });
    }

    extractToolCalls(text: string): ToolCallExtractionResult {
        const toolCalls: ToolCall[] = [];
        let content = text;

        // Try to find and parse JSON tool calls
        const jsonObjects = this.extractJsonObjects(text);

        for (const { json, startIdx, endIdx } of jsonObjects) {
            try {
                const parsed = JSON.parse(json);

                if (this.isToolCallLike(parsed)) {
                    if (Array.isArray(parsed)) {
                        for (const item of parsed) {
                            if (this.isToolCallLike(item)) {
                                toolCalls.push(this.normalizeToolCall(item));
                            }
                        }
                    } else {
                        toolCalls.push(this.normalizeToolCall(parsed));
                    }

                    // Remove from content
                    content =
                        content.slice(0, startIdx) + content.slice(endIdx);
                }
            } catch {
                // Not valid JSON
            }
        }

        return {
            toolCalls,
            content: content.trim(),
        };
    }

    /**
     * Extract all JSON objects/arrays from text
     */
    private extractJsonObjects(
        text: string
    ): Array<{ json: string; startIdx: number; endIdx: number }> {
        const results: Array<{
            json: string;
            startIdx: number;
            endIdx: number;
        }> = [];

        let i = 0;
        while (i < text.length) {
            if (text[i] === "{" || text[i] === "[") {
                const json = this.extractJsonAt(text, i);
                if (json) {
                    results.push({
                        json,
                        startIdx: i,
                        endIdx: i + json.length,
                    });
                    i += json.length;
                    continue;
                }
            }
            i++;
        }

        return results;
    }

    /**
     * Extract a complete JSON object/array starting at index
     */
    private extractJsonAt(text: string, startIdx: number): string | null {
        const openChar = text[startIdx];
        const closeChar = openChar === "{" ? "}" : "]";

        let depth = 0;
        let inString = false;
        let escape = false;

        for (let i = startIdx; i < text.length; i++) {
            const char = text[i];

            if (escape) {
                escape = false;
                continue;
            }

            if (char === "\\") {
                escape = true;
                continue;
            }

            if (char === '"') {
                inString = !inString;
                continue;
            }

            if (!inString) {
                if (char === openChar) {
                    depth++;
                } else if (char === closeChar) {
                    depth--;
                    if (depth === 0) {
                        return text.slice(startIdx, i + 1);
                    }
                }
            }
        }

        return null;
    }

    /**
     * Check if object looks like a tool call
     */
    private isToolCallLike(obj: unknown): boolean {
        if (Array.isArray(obj)) {
            return obj.some((item) => this.isToolCallLike(item));
        }

        if (typeof obj !== "object" || obj === null) {
            return false;
        }

        const record = obj as Record<string, unknown>;

        // Check for {"function": {"name": ...}} format
        if (record.function && typeof record.function === "object") {
            const func = record.function as Record<string, unknown>;
            return typeof func.name === "string";
        }

        // Check for {"name": ..., "arguments"/"parameters": ...} format
        return (
            typeof record.name === "string" &&
            (record.arguments !== undefined || record.parameters !== undefined)
        );
    }

    protected parseToolCallContent(content: string): ToolCall[] {
        const parsed = JSON.parse(content);

        if (Array.isArray(parsed)) {
            return parsed.map((item) => this.normalizeToolCall(item));
        }

        return [this.normalizeToolCall(parsed)];
    }

    private normalizeToolCall(data: Record<string, unknown>): ToolCall {
        if (data.function && typeof data.function === "object") {
            const func = data.function as Record<string, unknown>;
            return {
                id: (data.id as string) ?? this.generateToolCallId(),
                type: "function",
                function: {
                    name: String(func.name ?? ""),
                    arguments:
                        typeof func.arguments === "string"
                            ? func.arguments
                            : JSON.stringify(func.arguments ?? {}),
                },
            };
        }

        return {
            id: (data.id as string) ?? this.generateToolCallId(),
            type: "function",
            function: {
                name: String(data.name ?? ""),
                arguments:
                    typeof data.arguments === "string"
                        ? data.arguments
                        : JSON.stringify(
                              data.arguments ?? data.parameters ?? {}
                          ),
            },
        };
    }
}

/**
 * Create a generic XML tool call parser
 */
export function createGenericXmlToolCallParser(
    config?: Partial<ToolCallParserConfig>
): GenericXmlToolCallParser {
    return new GenericXmlToolCallParser(config);
}

/**
 * Create a generic JSON tool call parser
 */
export function createGenericJsonToolCallParser(
    config?: Partial<ToolCallParserConfig>
): GenericJsonToolCallParser {
    return new GenericJsonToolCallParser(config);
}
