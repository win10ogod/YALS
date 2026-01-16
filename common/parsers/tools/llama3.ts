import { ToolCall } from "@/api/OAI/types/tools.ts";
import { BaseToolCallParser } from "../base.ts";
import { ToolCallExtractionResult, ToolCallParserConfig } from "../types.ts";

/**
 * Llama 3.x tool call parser
 *
 * Handles the native Llama 3.x tool calling format.
 * Supports built-in tools: wolfram_alpha, web_search/brave_search, code_interpreter
 *
 * Example format:
 * ```
 * <|python_tag|>{"name": "get_weather", "parameters": {"location": "Tokyo"}}
 * ```
 *
 * Or JSON format:
 * ```
 * {"name": "get_weather", "parameters": {"location": "Tokyo"}}
 * ```
 */
export class Llama3ToolCallParser extends BaseToolCallParser {
    // Built-in tool names that Llama 3.x recognizes
    private readonly builtinTools = [
        "wolfram_alpha",
        "web_search",
        "brave_search",
        "code_interpreter",
    ];

    constructor(config?: Partial<ToolCallParserConfig>) {
        super({
            format: "llama3",
            startToken: "<|python_tag|>",
            endToken: "<|eom_id|>",
            parallelCalls: true,
            ...config,
        });
    }

    extractToolCalls(text: string): ToolCallExtractionResult {
        const toolCalls: ToolCall[] = [];
        let content = text;

        // Try to extract tool calls with python_tag
        const pythonTagCalls = this.extractPythonTagCalls(text);
        if (pythonTagCalls.toolCalls.length > 0) {
            toolCalls.push(...pythonTagCalls.toolCalls);
            content = pythonTagCalls.content;
        }

        // Also try to extract JSON tool calls
        const jsonCalls = this.extractJsonToolCalls(content);
        if (jsonCalls.toolCalls.length > 0) {
            toolCalls.push(...jsonCalls.toolCalls);
            content = jsonCalls.content;
        }

        return {
            toolCalls,
            content: content.trim(),
        };
    }

    /**
     * Extract tool calls using <|python_tag|> format
     */
    private extractPythonTagCalls(text: string): ToolCallExtractionResult {
        const startToken = "<|python_tag|>";
        const endTokens = ["<|eom_id|>", "<|eot_id|>", "\n\n"];

        const toolCalls: ToolCall[] = [];
        let content = text;
        let startIdx = content.indexOf(startToken);

        while (startIdx !== -1) {
            const afterStart = startIdx + startToken.length;

            // Find the end of the tool call
            let endIdx = content.length;
            for (const endToken of endTokens) {
                const idx = content.indexOf(endToken, afterStart);
                if (idx !== -1 && idx < endIdx) {
                    endIdx = idx;
                }
            }

            const toolCallContent = content.slice(afterStart, endIdx).trim();

            try {
                const parsed = this.parseToolCallContent(toolCallContent);
                toolCalls.push(...parsed);

                // Remove the tool call from content
                const fullMatch = content.slice(startIdx, endIdx);
                content = content.slice(0, startIdx) + content.slice(endIdx);
            } catch {
                // If parsing fails, move past this match
                startIdx = content.indexOf(startToken, afterStart);
                continue;
            }

            startIdx = content.indexOf(startToken);
        }

        return { toolCalls, content };
    }

    /**
     * Extract JSON tool calls (without special tags)
     */
    private extractJsonToolCalls(text: string): ToolCallExtractionResult {
        const toolCalls: ToolCall[] = [];
        let content = text;

        // Look for JSON objects that look like tool calls
        const jsonPattern = /\{[\s\S]*?"name"\s*:\s*"[^"]+"/g;
        let match;

        while ((match = jsonPattern.exec(text)) !== null) {
            // Try to extract a complete JSON object
            const startIdx = match.index;
            const jsonStr = this.extractJsonObject(text, startIdx);

            if (jsonStr) {
                try {
                    const parsed = JSON.parse(jsonStr);
                    if (this.isToolCall(parsed)) {
                        const toolCall = this.normalizeToolCall(parsed);
                        toolCalls.push(toolCall);
                        content = content.replace(jsonStr, "");
                    }
                } catch {
                    // Not valid JSON, skip
                }
            }
        }

        return { toolCalls, content };
    }

    /**
     * Extract a complete JSON object starting at the given index
     */
    private extractJsonObject(text: string, startIdx: number): string | null {
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
                if (char === "{") {
                    depth++;
                } else if (char === "}") {
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
     * Check if an object looks like a tool call
     */
    private isToolCall(obj: Record<string, unknown>): boolean {
        return (
            typeof obj.name === "string" &&
            (obj.parameters !== undefined || obj.arguments !== undefined)
        );
    }

    protected parseToolCallContent(content: string): ToolCall[] {
        const trimmed = content.trim();
        if (!trimmed) {
            return [];
        }

        const parsed = JSON.parse(trimmed);

        if (Array.isArray(parsed)) {
            return parsed.map((item) => this.normalizeToolCall(item));
        }

        return [this.normalizeToolCall(parsed)];
    }

    /**
     * Normalize to standard ToolCall format
     */
    private normalizeToolCall(data: Record<string, unknown>): ToolCall {
        const name = String(data.name ?? "");
        const args = data.parameters ?? data.arguments ?? {};

        return {
            id: (data.id as string) ?? this.generateToolCallId(),
            type: "function",
            function: {
                name,
                arguments: typeof args === "string" ? args : JSON.stringify(args),
            },
        };
    }

    /**
     * Check if a tool name is a built-in Llama tool
     */
    isBuiltinTool(name: string): boolean {
        return this.builtinTools.includes(name);
    }
}

/**
 * Create a Llama 3.x tool call parser
 */
export function createLlama3ToolCallParser(
    config?: Partial<ToolCallParserConfig>
): Llama3ToolCallParser {
    return new Llama3ToolCallParser(config);
}
