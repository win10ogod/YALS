import { ToolCall } from "@/api/OAI/types/tools.ts";
import {
    ParsedContent,
    ReasoningExtractionResult,
    ReasoningParserConfig,
    ToolCallExtractionResult,
    ToolCallParserConfig,
} from "./types.ts";

/**
 * Base interface for all content parsers
 */
export interface ContentParser {
    /**
     * Process streaming text incrementally
     * @param text New text chunk to process
     * @returns Parsed content that can be emitted
     */
    processStreaming(text: string): ParsedContent;

    /**
     * Flush remaining buffer content
     * @returns Any remaining parsed content
     */
    flush(): ParsedContent;

    /**
     * Parse complete text (non-streaming)
     * @param text Full text to parse
     * @returns Fully parsed content
     */
    parseComplete(text: string): ParsedContent;

    /**
     * Reset parser state
     */
    reset(): void;
}

/**
 * Interface for reasoning/thinking content parsers
 */
export interface ReasoningParser {
    /**
     * Extract reasoning content from text
     * @param text Text to extract from
     * @returns Extracted reasoning and remaining content
     */
    extractReasoningContent(text: string): ReasoningExtractionResult;

    /**
     * Process streaming text for reasoning extraction
     * @param text New text chunk
     * @returns Reasoning extraction result
     */
    extractReasoningStreaming(text: string): ReasoningExtractionResult;

    /**
     * Flush remaining reasoning buffer
     * @returns Final reasoning extraction result
     */
    flushReasoning(): ReasoningExtractionResult;

    /**
     * Reset parser state
     */
    reset(): void;

    /**
     * Get the start token for thinking blocks
     */
    getStartToken(): string;

    /**
     * Get the end token for thinking blocks
     */
    getEndToken(): string;
}

/**
 * Interface for tool call parsers
 */
export interface ToolCallParser {
    /**
     * Extract tool calls from text
     * @param text Text to extract from
     * @returns Extracted tool calls and remaining content
     */
    extractToolCalls(text: string): ToolCallExtractionResult;

    /**
     * Process streaming text for tool call extraction
     * @param text New text chunk
     * @returns Tool call extraction result
     */
    extractToolCallsStreaming(text: string): ToolCallExtractionResult;

    /**
     * Flush remaining tool call buffer
     * @returns Final tool call extraction result
     */
    flushToolCalls(): ToolCallExtractionResult;

    /**
     * Reset parser state
     */
    reset(): void;

    /**
     * Get the start token for tool call blocks
     */
    getStartToken(): string;

    /**
     * Get the end token for tool call blocks
     */
    getEndToken(): string;
}

/**
 * Abstract base class for reasoning parsers with common functionality
 */
export abstract class BaseReasoningParser implements ReasoningParser {
    protected config: ReasoningParserConfig;
    protected buffer: string = "";
    protected inThinkingBlock: boolean = false;
    protected thinkingContent: string = "";

    constructor(config: ReasoningParserConfig) {
        this.config = config;
    }

    abstract extractReasoningContent(text: string): ReasoningExtractionResult;

    getStartToken(): string {
        return this.config.startToken ?? "<think>";
    }

    getEndToken(): string {
        return this.config.endToken ?? "</think>";
    }

    extractReasoningStreaming(text: string): ReasoningExtractionResult {
        this.buffer += text;

        const startToken = this.getStartToken();
        const endToken = this.getEndToken();

        let reasoning = "";
        let content = "";

        while (this.buffer.length > 0) {
            if (!this.inThinkingBlock) {
                // Look for start of thinking block
                const startIdx = this.buffer.indexOf(startToken);

                if (startIdx === -1) {
                    // Check if buffer ends with partial start token
                    const keepLen = this.trailingPrefix(this.buffer, startToken);
                    const emitLen = this.buffer.length - keepLen;

                    if (emitLen > 0) {
                        content += this.buffer.slice(0, emitLen);
                        this.buffer = this.buffer.slice(emitLen);
                    }
                    break;
                }

                // Emit content before thinking block
                content += this.buffer.slice(0, startIdx);
                this.buffer = this.buffer.slice(startIdx + startToken.length);
                this.inThinkingBlock = true;
                this.thinkingContent = "";
            } else {
                // Look for end of thinking block
                const endIdx = this.buffer.indexOf(endToken);

                if (endIdx === -1) {
                    // Still inside thinking block, buffer everything
                    this.thinkingContent += this.buffer;
                    this.buffer = "";
                    break;
                }

                // Found end of thinking block
                this.thinkingContent += this.buffer.slice(0, endIdx);
                reasoning += this.thinkingContent;
                this.buffer = this.buffer.slice(endIdx + endToken.length);
                this.inThinkingBlock = false;
                this.thinkingContent = "";
            }
        }

        return {
            reasoning,
            content,
            inThinkingBlock: this.inThinkingBlock,
        };
    }

    flushReasoning(): ReasoningExtractionResult {
        const result: ReasoningExtractionResult = {
            reasoning: this.inThinkingBlock ? this.thinkingContent : "",
            content: this.inThinkingBlock ? "" : this.buffer,
            inThinkingBlock: false,
        };

        this.reset();
        return result;
    }

    reset(): void {
        this.buffer = "";
        this.inThinkingBlock = false;
        this.thinkingContent = "";
    }

    /**
     * Check if text ends with a partial prefix of target
     */
    protected trailingPrefix(text: string, target: string): number {
        const maxLen = Math.min(text.length, target.length - 1);
        for (let i = maxLen; i > 0; i--) {
            if (target.startsWith(text.slice(-i))) {
                return i;
            }
        }
        return 0;
    }
}

/**
 * Abstract base class for tool call parsers with common functionality
 */
export abstract class BaseToolCallParser implements ToolCallParser {
    protected config: ToolCallParserConfig;
    protected buffer: string = "";
    protected inToolCallBlock: boolean = false;
    protected toolCallContent: string = "";
    protected collectedToolCalls: ToolCall[] = [];

    constructor(config: ToolCallParserConfig) {
        this.config = config;
    }

    abstract extractToolCalls(text: string): ToolCallExtractionResult;

    /**
     * Parse tool call content to ToolCall object
     * This should be implemented by specific parsers
     */
    protected abstract parseToolCallContent(content: string): ToolCall[];

    getStartToken(): string {
        return this.config.startToken ?? "<tool_call>";
    }

    getEndToken(): string {
        return this.config.endToken ?? "</tool_call>";
    }

    extractToolCallsStreaming(text: string): ToolCallExtractionResult {
        this.buffer += text;

        const startToken = this.getStartToken();
        const endToken = this.getEndToken();

        const toolCalls: ToolCall[] = [];
        let content = "";

        while (this.buffer.length > 0) {
            if (!this.inToolCallBlock) {
                // Look for start of tool call block
                const startIdx = this.buffer.indexOf(startToken);

                if (startIdx === -1) {
                    // Check if buffer ends with partial start token
                    const keepLen = this.trailingPrefix(this.buffer, startToken);
                    const emitLen = this.buffer.length - keepLen;

                    if (emitLen > 0) {
                        content += this.buffer.slice(0, emitLen);
                        this.buffer = this.buffer.slice(emitLen);
                    }
                    break;
                }

                // Emit content before tool call block
                content += this.buffer.slice(0, startIdx);
                this.buffer = this.buffer.slice(startIdx + startToken.length);
                this.inToolCallBlock = true;
                this.toolCallContent = "";
            } else {
                // Look for end of tool call block
                const endIdx = this.buffer.indexOf(endToken);

                if (endIdx === -1) {
                    // Still inside tool call block, buffer everything
                    this.toolCallContent += this.buffer;
                    this.buffer = "";
                    break;
                }

                // Found end of tool call block
                this.toolCallContent += this.buffer.slice(0, endIdx);

                // Parse the tool call content
                try {
                    const parsed = this.parseToolCallContent(this.toolCallContent);
                    toolCalls.push(...parsed);
                    this.collectedToolCalls.push(...parsed);
                } catch {
                    // If parsing fails, emit as content
                    content += startToken + this.toolCallContent + endToken;
                }

                this.buffer = this.buffer.slice(endIdx + endToken.length);
                this.inToolCallBlock = false;
                this.toolCallContent = "";
            }
        }

        return {
            toolCalls,
            content,
            inToolCallBlock: this.inToolCallBlock,
        };
    }

    flushToolCalls(): ToolCallExtractionResult {
        const result: ToolCallExtractionResult = {
            toolCalls: [],
            content: "",
            inToolCallBlock: false,
        };

        if (this.inToolCallBlock) {
            // Try to parse incomplete tool call
            try {
                const parsed = this.parseToolCallContent(this.toolCallContent);
                result.toolCalls = parsed;
            } catch {
                // If parsing fails, return as content
                result.content = this.getStartToken() + this.toolCallContent;
            }
        } else {
            result.content = this.buffer;
        }

        this.reset();
        return result;
    }

    reset(): void {
        this.buffer = "";
        this.inToolCallBlock = false;
        this.toolCallContent = "";
        this.collectedToolCalls = [];
    }

    /**
     * Check if text ends with a partial prefix of target
     */
    protected trailingPrefix(text: string, target: string): number {
        const maxLen = Math.min(text.length, target.length - 1);
        for (let i = maxLen; i > 0; i--) {
            if (target.startsWith(text.slice(-i))) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Generate a unique tool call ID
     */
    protected generateToolCallId(): string {
        return `call_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`;
    }
}

/**
 * Create default parsed content result
 */
export function createEmptyParsedContent(): ParsedContent {
    return {
        text: "",
        thinking: undefined,
        toolCalls: [],
        isComplete: true,
    };
}
