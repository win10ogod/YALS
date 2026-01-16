import * as z from "@/common/myZod.ts";
import { ToolCall } from "@/api/OAI/types/tools.ts";

/**
 * Supported reasoning/thinking parser formats
 */
export const ReasoningFormat = z.enum([
    "deepseek_r1",  // <think>...</think>
    "qwen3",        // <think>...</think> (enabled by default)
    "granite",      // IBM Granite (disabled by default)
    "generic",      // Generic <thinking>...</thinking>
    "none",         // No reasoning parsing
]);

export type ReasoningFormat = z.infer<typeof ReasoningFormat>;

/**
 * Supported tool call parser formats
 */
export const ToolCallFormat = z.enum([
    "hermes",       // <tool_call>{"name": ...}</tool_call>
    "llama3",       // Llama 3.x native format with <|python_tag|>
    "llama3_json",  // Llama 3.x JSON format
    "mistral",      // [TOOL_CALLS] format
    "qwen",         // Qwen tool call format
    "qwen_coder",   // Qwen3-Coder custom XML format
    "deepseek",     // DeepSeek format
    "deepseek_v3",  // DeepSeek V3 format
    "internlm",     // InternLM format
    "xlam",         // xLAM format with thinking tags
    "functionary",  // Functionary format
    "generic_xml",  // Generic <tool_call>...</tool_call>
    "generic_json", // Generic JSON format
    "none",         // No tool call parsing
]);

export type ToolCallFormat = z.infer<typeof ToolCallFormat>;

/**
 * Parsed content result from any parser
 */
export interface ParsedContent {
    /** Regular text content (non-thinking, non-tool) */
    text: string;

    /** Extracted thinking/reasoning content */
    thinking?: string;

    /** Extracted tool calls */
    toolCalls: ToolCall[];

    /** Whether parsing is complete (for streaming) */
    isComplete: boolean;

    /** Raw unparsed buffer (for streaming) */
    buffer?: string;
}

/**
 * Result from reasoning content extraction
 */
export interface ReasoningExtractionResult {
    /** Extracted reasoning/thinking content */
    reasoning: string;

    /** Remaining content after reasoning extraction */
    content: string;

    /** Whether we're currently inside a thinking block (streaming) */
    inThinkingBlock?: boolean;
}

/**
 * Result from tool call extraction
 */
export interface ToolCallExtractionResult {
    /** Extracted tool calls */
    toolCalls: ToolCall[];

    /** Remaining content after tool call extraction */
    content: string;

    /** Whether we're currently inside a tool call block (streaming) */
    inToolCallBlock?: boolean;
}

/**
 * Configuration for reasoning parser
 */
export interface ReasoningParserConfig {
    /** Format to use for parsing */
    format: ReasoningFormat;

    /** Start token/string for thinking block */
    startToken?: string;

    /** End token/string for thinking block */
    endToken?: string;

    /** Whether to include reasoning in response */
    includeInResponse?: boolean;

    /** Force thinking output (for models that support it) */
    forcedOpen?: boolean;
}

/**
 * Configuration for tool call parser
 */
export interface ToolCallParserConfig {
    /** Format to use for parsing */
    format: ToolCallFormat;

    /** Start token/string for tool call block */
    startToken?: string;

    /** End token/string for tool call block */
    endToken?: string;

    /** Whether parallel tool calls are supported */
    parallelCalls?: boolean;
}

/**
 * Combined parser configuration
 */
export interface ContentParserConfig {
    reasoning?: ReasoningParserConfig;
    toolCall?: ToolCallParserConfig;
}

/**
 * Statistics for token counting
 */
export interface TokenStats {
    /** Number of prompt tokens */
    promptTokens: number;

    /** Number of completion/output tokens */
    completionTokens: number;

    /** Number of thinking/reasoning tokens */
    thinkingTokens?: number;

    /** Total tokens */
    totalTokens: number;
}

/**
 * Default configurations for common model families
 */
export const MODEL_PARSER_PRESETS: Record<string, ContentParserConfig> = {
    "deepseek-r1": {
        reasoning: {
            format: "deepseek_r1",
            startToken: "<think>",
            endToken: "</think>",
            includeInResponse: true,
        },
        toolCall: {
            format: "deepseek",
        },
    },
    "deepseek-v3": {
        reasoning: {
            format: "none",
        },
        toolCall: {
            format: "deepseek_v3",
        },
    },
    "qwen3": {
        reasoning: {
            format: "qwen3",
            startToken: "<think>",
            endToken: "</think>",
            includeInResponse: true,
        },
        toolCall: {
            format: "qwen",
        },
    },
    "qwen3-coder": {
        reasoning: {
            format: "qwen3",
            startToken: "<think>",
            endToken: "</think>",
        },
        toolCall: {
            format: "qwen_coder",
        },
    },
    "llama3": {
        toolCall: {
            format: "llama3",
            parallelCalls: true,
        },
    },
    "llama3.1": {
        toolCall: {
            format: "llama3",
            parallelCalls: true,
        },
    },
    "hermes": {
        toolCall: {
            format: "hermes",
            startToken: "<tool_call>",
            endToken: "</tool_call>",
        },
    },
    "mistral": {
        toolCall: {
            format: "mistral",
        },
    },
    "granite": {
        reasoning: {
            format: "granite",
            includeInResponse: false, // Disabled by default
        },
    },
    "internlm": {
        toolCall: {
            format: "internlm",
        },
    },
    "xlam": {
        reasoning: {
            format: "generic",
            startToken: "<think>",
            endToken: "</think>",
        },
        toolCall: {
            format: "xlam",
        },
    },
    "functionary": {
        toolCall: {
            format: "functionary",
        },
    },
};
