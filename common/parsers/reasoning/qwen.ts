import { BaseReasoningParser } from "../base.ts";
import { ReasoningExtractionResult, ReasoningParserConfig } from "../types.ts";

/**
 * Qwen3 reasoning parser
 *
 * Handles the <think>...</think> format used by Qwen3 models.
 * Note: Thinking is enabled by default for Qwen3 series.
 * To disable, pass `enable_thinking=False` in chat_template_kwargs.
 *
 * Example output:
 * ```
 * <think>
 * The user is asking about...
 * I should consider...
 * </think>
 *
 * Here is my response...
 * ```
 */
export class Qwen3ReasoningParser extends BaseReasoningParser {
    private enableThinking: boolean;

    constructor(config?: Partial<ReasoningParserConfig> & { enableThinking?: boolean }) {
        super({
            format: "qwen3",
            startToken: "<think>",
            endToken: "</think>",
            includeInResponse: true,
            ...config,
        });
        // Qwen3 has thinking enabled by default
        this.enableThinking = config?.enableThinking ?? true;
    }

    extractReasoningContent(text: string): ReasoningExtractionResult {
        // If thinking is disabled, return text as-is
        if (!this.enableThinking) {
            return {
                reasoning: "",
                content: text,
            };
        }

        const startToken = this.getStartToken();
        const endToken = this.getEndToken();

        const regex = new RegExp(
            `${this.escapeRegex(startToken)}([\\s\\S]*?)${this.escapeRegex(endToken)}`,
            "g"
        );

        let reasoning = "";
        let lastIndex = 0;
        let cleanContent = "";

        let match;
        while ((match = regex.exec(text)) !== null) {
            cleanContent += text.slice(lastIndex, match.index);
            reasoning += match[1];
            lastIndex = match.index + match[0].length;
        }

        cleanContent += text.slice(lastIndex);

        return {
            reasoning: reasoning.trim(),
            content: cleanContent.trim(),
        };
    }

    /**
     * Check if thinking is enabled
     */
    isThinkingEnabled(): boolean {
        return this.enableThinking;
    }

    /**
     * Set thinking enabled state
     */
    setThinkingEnabled(enabled: boolean): void {
        this.enableThinking = enabled;
    }

    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
}

/**
 * Create a Qwen3 reasoning parser with default settings
 */
export function createQwen3Parser(
    config?: Partial<ReasoningParserConfig> & { enableThinking?: boolean }
): Qwen3ReasoningParser {
    return new Qwen3ReasoningParser(config);
}
