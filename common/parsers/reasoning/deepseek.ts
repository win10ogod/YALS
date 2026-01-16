import { BaseReasoningParser } from "../base.ts";
import { ReasoningExtractionResult, ReasoningParserConfig } from "../types.ts";

/**
 * DeepSeek R1 reasoning parser
 *
 * Handles the <think>...</think> format used by DeepSeek R1 models.
 *
 * Example output:
 * ```
 * <think>
 * Let me analyze this step by step...
 * First, I need to consider...
 * </think>
 *
 * Based on my analysis, the answer is...
 * ```
 */
export class DeepSeekR1ReasoningParser extends BaseReasoningParser {
    constructor(config?: Partial<ReasoningParserConfig>) {
        super({
            format: "deepseek_r1",
            startToken: "<think>",
            endToken: "</think>",
            includeInResponse: true,
            ...config,
        });
    }

    extractReasoningContent(text: string): ReasoningExtractionResult {
        const startToken = this.getStartToken();
        const endToken = this.getEndToken();

        const regex = new RegExp(
            `${this.escapeRegex(startToken)}([\\s\\S]*?)${this.escapeRegex(endToken)}`,
            "g"
        );

        let reasoning = "";
        let content = text;
        let lastIndex = 0;
        let cleanContent = "";

        // Find all thinking blocks and extract content
        let match;
        while ((match = regex.exec(text)) !== null) {
            // Add content before this thinking block
            cleanContent += text.slice(lastIndex, match.index);
            // Add the thinking content
            reasoning += match[1];
            lastIndex = match.index + match[0].length;
        }

        // Add remaining content after last thinking block
        cleanContent += text.slice(lastIndex);

        return {
            reasoning: reasoning.trim(),
            content: cleanContent.trim(),
        };
    }

    /**
     * Escape special regex characters in a string
     */
    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
}

/**
 * Create a DeepSeek R1 reasoning parser with default settings
 */
export function createDeepSeekR1Parser(
    config?: Partial<ReasoningParserConfig>
): DeepSeekR1ReasoningParser {
    return new DeepSeekR1ReasoningParser(config);
}
