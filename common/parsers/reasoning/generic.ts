import { BaseReasoningParser } from "../base.ts";
import { ReasoningExtractionResult, ReasoningParserConfig } from "../types.ts";

/**
 * Generic reasoning parser
 *
 * Handles various thinking/reasoning formats with configurable tokens.
 * Supports:
 * - <thinking>...</thinking>
 * - <reasoning>...</reasoning>
 * - Custom start/end tokens
 *
 * Example output:
 * ```
 * <thinking>
 * I need to analyze this problem...
 * </thinking>
 *
 * The solution is...
 * ```
 */
export class GenericReasoningParser extends BaseReasoningParser {
    constructor(config?: Partial<ReasoningParserConfig>) {
        super({
            format: "generic",
            startToken: config?.startToken ?? "<thinking>",
            endToken: config?.endToken ?? "</thinking>",
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

    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
}

/**
 * Create a generic reasoning parser with custom tokens
 */
export function createGenericReasoningParser(
    startToken: string = "<thinking>",
    endToken: string = "</thinking>",
    config?: Partial<ReasoningParserConfig>
): GenericReasoningParser {
    return new GenericReasoningParser({
        startToken,
        endToken,
        ...config,
    });
}
