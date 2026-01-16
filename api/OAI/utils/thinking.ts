/**
 * Thinking content parser for models that support chain-of-thought reasoning.
 * Supports formats like <think>...</think>, <thinking>...</thinking>, etc.
 */

export interface ThinkingConfig {
    startTag: string;
    endTag: string;
}

export interface ParsedThinking {
    thinking: string;
    content: string;
}

// Default thinking patterns to try
const DEFAULT_THINKING_PATTERNS: ThinkingConfig[] = [
    { startTag: "<think>", endTag: "</think>" },
    { startTag: "<thinking>", endTag: "</thinking>" },
];

/**
 * Extract thinking content from text using specified tags
 */
export function extractThinking(
    text: string,
    config?: ThinkingConfig,
): ParsedThinking {
    const patterns = config ? [config] : DEFAULT_THINKING_PATTERNS;

    for (const pattern of patterns) {
        const startIdx = text.indexOf(pattern.startTag);
        if (startIdx === -1) continue;

        const contentStart = startIdx + pattern.startTag.length;
        const endIdx = text.indexOf(pattern.endTag, contentStart);

        if (endIdx === -1) {
            // Unclosed thinking block - treat everything after start as thinking
            return {
                thinking: text.slice(contentStart).trim(),
                content: text.slice(0, startIdx).trim(),
            };
        }

        const thinking = text.slice(contentStart, endIdx).trim();
        const beforeThinking = text.slice(0, startIdx);
        const afterThinking = text.slice(endIdx + pattern.endTag.length);
        const content = (beforeThinking + afterThinking).trim();

        return { thinking, content };
    }

    // No thinking found
    return { thinking: "", content: text };
}

/**
 * Check if text contains thinking tags
 */
export function hasThinkingTags(text: string, config?: ThinkingConfig): boolean {
    const patterns = config ? [config] : DEFAULT_THINKING_PATTERNS;
    return patterns.some((p) => text.includes(p.startTag));
}

/**
 * Creates a streaming thinking parser that handles incremental text
 */
export function createStreamingThinkingParser(config?: ThinkingConfig) {
    const patterns = config ? [config] : DEFAULT_THINKING_PATTERNS;
    let buffer = "";
    let thinkingBuffer = "";
    let inThinking = false;
    let activePattern: ThinkingConfig | null = null;
    let thinkingComplete = false;

    const findMatchingPattern = (text: string): ThinkingConfig | null => {
        for (const pattern of patterns) {
            if (text.includes(pattern.startTag)) {
                return pattern;
            }
        }
        return null;
    };

    const trailingStartPrefix = (text: string, startTag: string): number => {
        const maxLen = Math.min(text.length, startTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
            if (startTag.startsWith(text.slice(-i))) {
                return i;
            }
        }
        return 0;
    };

    const process = (text: string): { content: string; thinking: string } => {
        buffer += text;
        let outputContent = "";
        let outputThinking = "";

        while (buffer.length > 0) {
            if (!inThinking) {
                // Look for start of thinking
                if (!activePattern) {
                    activePattern = findMatchingPattern(buffer);
                }

                if (!activePattern) {
                    // Check if we might have a partial start tag
                    let hasPartial = false;
                    for (const pattern of patterns) {
                        const keepLen = trailingStartPrefix(
                            buffer,
                            pattern.startTag,
                        );
                        if (keepLen > 0) {
                            const emitLen = buffer.length - keepLen;
                            if (emitLen > 0) {
                                outputContent += buffer.slice(0, emitLen);
                                buffer = buffer.slice(emitLen);
                            }
                            hasPartial = true;
                            break;
                        }
                    }
                    if (!hasPartial) {
                        outputContent += buffer;
                        buffer = "";
                    }
                    break;
                }

                const startIdx = buffer.indexOf(activePattern.startTag);
                if (startIdx === -1) {
                    // Pattern was found before but not now - shouldn't happen
                    outputContent += buffer;
                    buffer = "";
                    activePattern = null;
                    break;
                }

                // Emit content before thinking tag
                outputContent += buffer.slice(0, startIdx);
                buffer = buffer.slice(startIdx + activePattern.startTag.length);
                inThinking = true;
                thinkingBuffer = "";
            } else {
                // Inside thinking block - look for end tag
                if (!activePattern) {
                    // Shouldn't happen, but handle gracefully
                    inThinking = false;
                    continue;
                }

                const endIdx = buffer.indexOf(activePattern.endTag);
                if (endIdx === -1) {
                    // No end tag yet - buffer the thinking content
                    thinkingBuffer += buffer;
                    buffer = "";
                    break;
                }

                // Found end tag
                thinkingBuffer += buffer.slice(0, endIdx);
                outputThinking = thinkingBuffer;
                buffer = buffer.slice(endIdx + activePattern.endTag.length);
                inThinking = false;
                thinkingComplete = true;
                thinkingBuffer = "";
            }
        }

        return { content: outputContent, thinking: outputThinking };
    };

    const flush = (): { content: string; thinking: string } => {
        let outputContent = "";
        let outputThinking = "";

        if (inThinking) {
            // Unclosed thinking block
            outputThinking = thinkingBuffer + buffer;
        } else {
            outputContent = buffer;
        }

        buffer = "";
        thinkingBuffer = "";
        inThinking = false;

        return { content: outputContent, thinking: outputThinking };
    };

    const getThinkingContent = (): string => thinkingBuffer;
    const isInThinking = (): boolean => inThinking;
    const isThinkingComplete = (): boolean => thinkingComplete;

    return {
        process,
        flush,
        getThinkingContent,
        isInThinking,
        isThinkingComplete,
    };
}

/**
 * Strip thinking tags from text, keeping only the content
 */
export function stripThinkingTags(
    text: string,
    config?: ThinkingConfig,
): string {
    const { content } = extractThinking(text, config);
    return content;
}

/**
 * Wrap content in thinking tags
 */
export function wrapInThinkingTags(
    thinking: string,
    config?: ThinkingConfig,
): string {
    const pattern = config || DEFAULT_THINKING_PATTERNS[0];
    return `${pattern.startTag}${thinking}${pattern.endTag}`;
}
