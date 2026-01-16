import { ToolCall } from "@/api/OAI/types/tools.ts";
import {
    createEmptyParsedContent,
    ReasoningParser,
    ToolCallParser,
} from "./base.ts";
import {
    createParsersFromConfig,
    createParsersFromPreset,
    detectParsersFromTemplate,
} from "./factory.ts";
import { ContentParserConfig, ParsedContent } from "./types.ts";

/**
 * Unified content processor that handles both reasoning and tool call parsing
 */
export class UnifiedContentProcessor {
    private reasoningParser: ReasoningParser | null = null;
    private toolCallParser: ToolCallParser | null = null;

    // Streaming state
    private buffer: string = "";
    private collectedThinking: string = "";
    private collectedToolCalls: ToolCall[] = [];

    constructor(
        reasoningParser?: ReasoningParser | null,
        toolCallParser?: ToolCallParser | null
    ) {
        this.reasoningParser = reasoningParser ?? null;
        this.toolCallParser = toolCallParser ?? null;
    }

    /**
     * Create a processor from configuration
     */
    static fromConfig(config: ContentParserConfig): UnifiedContentProcessor {
        const { reasoningParser, toolCallParser } =
            createParsersFromConfig(config);
        return new UnifiedContentProcessor(reasoningParser, toolCallParser);
    }

    /**
     * Create a processor from a model preset name
     */
    static fromPreset(presetName: string): UnifiedContentProcessor {
        const { reasoningParser, toolCallParser } =
            createParsersFromPreset(presetName);
        return new UnifiedContentProcessor(reasoningParser, toolCallParser);
    }

    /**
     * Create a processor by auto-detecting from template content
     */
    static fromTemplate(templateContent: string): UnifiedContentProcessor {
        const config = detectParsersFromTemplate(templateContent);
        return UnifiedContentProcessor.fromConfig(config);
    }

    /**
     * Process complete text (non-streaming)
     */
    parseComplete(text: string): ParsedContent {
        let content = text;
        let thinking: string | undefined;
        let toolCalls: ToolCall[] = [];

        // First, extract reasoning content
        if (this.reasoningParser) {
            const reasoningResult =
                this.reasoningParser.extractReasoningContent(content);
            thinking = reasoningResult.reasoning || undefined;
            content = reasoningResult.content;
        }

        // Then, extract tool calls
        if (this.toolCallParser) {
            const toolCallResult =
                this.toolCallParser.extractToolCalls(content);
            toolCalls = toolCallResult.toolCalls;
            content = toolCallResult.content;
        }

        return {
            text: content,
            thinking,
            toolCalls,
            isComplete: true,
        };
    }

    /**
     * Process streaming text incrementally
     */
    processStreaming(text: string): ParsedContent {
        this.buffer += text;

        let outputText = "";
        let streamThinking: string | undefined;

        // Process reasoning first
        if (this.reasoningParser) {
            const reasoningResult =
                this.reasoningParser.extractReasoningStreaming(this.buffer);
            this.buffer = "";

            if (reasoningResult.reasoning) {
                this.collectedThinking += reasoningResult.reasoning;
                streamThinking = reasoningResult.reasoning;
            }

            // Content goes to tool parser or output
            if (reasoningResult.content) {
                if (this.toolCallParser) {
                    this.buffer = reasoningResult.content;
                } else {
                    outputText = reasoningResult.content;
                }
            }
        }

        // Process tool calls
        if (this.toolCallParser && this.buffer) {
            const toolCallResult =
                this.toolCallParser.extractToolCallsStreaming(this.buffer);
            this.buffer = "";

            if (toolCallResult.toolCalls.length > 0) {
                this.collectedToolCalls.push(...toolCallResult.toolCalls);
            }

            outputText += toolCallResult.content;
        } else if (!this.reasoningParser) {
            // No reasoning parser, check for tool calls directly
            if (this.toolCallParser) {
                const toolCallResult =
                    this.toolCallParser.extractToolCallsStreaming(this.buffer);
                this.buffer = "";

                if (toolCallResult.toolCalls.length > 0) {
                    this.collectedToolCalls.push(...toolCallResult.toolCalls);
                }

                outputText = toolCallResult.content;
            } else {
                outputText = this.buffer;
                this.buffer = "";
            }
        }

        return {
            text: outputText,
            thinking: streamThinking,
            toolCalls: [], // Tool calls are only returned on flush
            isComplete: false,
        };
    }

    /**
     * Flush remaining buffer and return final content
     */
    flush(): ParsedContent {
        let outputText = "";
        let flushThinking: string | undefined;

        // Flush reasoning parser
        if (this.reasoningParser) {
            const reasoningResult = this.reasoningParser.flushReasoning();
            if (reasoningResult.reasoning) {
                this.collectedThinking += reasoningResult.reasoning;
                flushThinking = reasoningResult.reasoning;
            }
            if (reasoningResult.content) {
                this.buffer += reasoningResult.content;
            }
        }

        // Flush tool call parser
        if (this.toolCallParser) {
            // First process any remaining buffer
            if (this.buffer) {
                const toolCallResult =
                    this.toolCallParser.extractToolCallsStreaming(this.buffer);
                if (toolCallResult.toolCalls.length > 0) {
                    this.collectedToolCalls.push(...toolCallResult.toolCalls);
                }
                outputText += toolCallResult.content;
            }

            // Then flush the parser
            const flushResult = this.toolCallParser.flushToolCalls();
            if (flushResult.toolCalls.length > 0) {
                this.collectedToolCalls.push(...flushResult.toolCalls);
            }
            outputText += flushResult.content;
        } else {
            outputText = this.buffer;
        }

        const result: ParsedContent = {
            text: outputText,
            thinking:
                this.collectedThinking || flushThinking || undefined,
            toolCalls: this.collectedToolCalls,
            isComplete: true,
        };

        // Reset state
        this.reset();

        return result;
    }

    /**
     * Reset all parser state
     */
    reset(): void {
        this.buffer = "";
        this.collectedThinking = "";
        this.collectedToolCalls = [];

        if (this.reasoningParser) {
            this.reasoningParser.reset();
        }
        if (this.toolCallParser) {
            this.toolCallParser.reset();
        }
    }

    /**
     * Get collected thinking content so far
     */
    getCollectedThinking(): string {
        return this.collectedThinking;
    }

    /**
     * Get collected tool calls so far
     */
    getCollectedToolCalls(): ToolCall[] {
        return [...this.collectedToolCalls];
    }

    /**
     * Check if reasoning parser is active
     */
    hasReasoningParser(): boolean {
        return this.reasoningParser !== null;
    }

    /**
     * Check if tool call parser is active
     */
    hasToolCallParser(): boolean {
        return this.toolCallParser !== null;
    }

    /**
     * Get stop strings that should be added based on parsers
     */
    getStopStrings(): string[] {
        const stopStrings: string[] = [];

        if (this.reasoningParser) {
            // Add thinking end token as stop string for tool detection
            stopStrings.push(this.reasoningParser.getEndToken());
        }

        if (this.toolCallParser) {
            const startToken = this.toolCallParser.getStartToken();
            if (startToken) {
                stopStrings.push(startToken);
            }
        }

        return stopStrings;
    }
}

/**
 * Create a streaming content parser for inline processing
 *
 * This is a simpler interface similar to createInlineToolCallParser
 */
export function createStreamingContentParser(config?: ContentParserConfig) {
    const processor = config
        ? UnifiedContentProcessor.fromConfig(config)
        : new UnifiedContentProcessor();

    return {
        /**
         * Process streaming text chunk
         */
        process: (text: string): string => {
            const result = processor.processStreaming(text);
            return result.text;
        },

        /**
         * Flush remaining content
         */
        flush: (): string => {
            const result = processor.flush();
            return result.text;
        },

        /**
         * Get collected thinking content
         */
        get thinking(): string {
            return processor.getCollectedThinking();
        },

        /**
         * Get collected tool calls
         */
        get toolCalls(): ToolCall[] {
            return processor.getCollectedToolCalls();
        },

        /**
         * Reset parser state
         */
        reset: (): void => {
            processor.reset();
        },
    };
}
