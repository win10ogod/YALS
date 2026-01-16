import { ToolCall } from "@/api/OAI/types/tools.ts";
import { BaseToolCallParser } from "../base.ts";
import { ToolCallExtractionResult, ToolCallParserConfig } from "../types.ts";

/**
 * Hermes tool call parser
 *
 * Handles the <tool_call>...</tool_call> format used by Hermes models.
 * The content inside can be JSON or have a specific structure.
 *
 * Example formats:
 * ```
 * <tool_call>
 * {"name": "get_weather", "arguments": {"location": "Tokyo"}}
 * </tool_call>
 * ```
 *
 * Or with function wrapper:
 * ```
 * <tool_call>
 * {"function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}
 * </tool_call>
 * ```
 */
export class HermesToolCallParser extends BaseToolCallParser {
    constructor(config?: Partial<ToolCallParserConfig>) {
        super({
            format: "hermes",
            startToken: "<tool_call>",
            endToken: "</tool_call>",
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
                // If parsing fails, keep the original text
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

        // Try to parse as JSON
        const parsed = JSON.parse(trimmed);

        // Handle array of tool calls
        if (Array.isArray(parsed)) {
            return parsed.map((item) => this.normalizeToolCall(item));
        }

        // Handle single tool call
        return [this.normalizeToolCall(parsed)];
    }

    /**
     * Normalize various tool call formats to standard ToolCall
     */
    private normalizeToolCall(data: Record<string, unknown>): ToolCall {
        // Format: {"function": {"name": ..., "arguments": ...}}
        if (data.function && typeof data.function === "object") {
            const func = data.function as Record<string, unknown>;
            return {
                id: (data.id as string) ?? this.generateToolCallId(),
                type: "function",
                function: {
                    name: String(func.name ?? ""),
                    arguments: typeof func.arguments === "string"
                        ? func.arguments
                        : JSON.stringify(func.arguments ?? {}),
                },
            };
        }

        // Format: {"name": ..., "arguments": ...}
        if (data.name) {
            return {
                id: (data.id as string) ?? this.generateToolCallId(),
                type: "function",
                function: {
                    name: String(data.name),
                    arguments: typeof data.arguments === "string"
                        ? data.arguments
                        : JSON.stringify(data.arguments ?? data.parameters ?? {}),
                },
            };
        }

        throw new Error("Invalid tool call format");
    }

    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
}

/**
 * Create a Hermes tool call parser
 */
export function createHermesToolCallParser(
    config?: Partial<ToolCallParserConfig>
): HermesToolCallParser {
    return new HermesToolCallParser(config);
}
