import { ReasoningParser, ToolCallParser } from "./base.ts";
import {
    ContentParserConfig,
    MODEL_PARSER_PRESETS,
    ReasoningFormat,
    ReasoningParserConfig,
    ToolCallFormat,
    ToolCallParserConfig,
} from "./types.ts";
import {
    DeepSeekR1ReasoningParser,
    GenericReasoningParser,
    Qwen3ReasoningParser,
} from "./reasoning/index.ts";
import {
    GenericJsonToolCallParser,
    GenericXmlToolCallParser,
    HermesToolCallParser,
    Llama3ToolCallParser,
} from "./tools/index.ts";

/**
 * Factory for creating reasoning parsers
 */
export class ReasoningParserFactory {
    private static parsers = new Map<
        ReasoningFormat,
        new (config?: Partial<ReasoningParserConfig>) => ReasoningParser
    >([
        ["deepseek_r1", DeepSeekR1ReasoningParser],
        ["qwen3", Qwen3ReasoningParser],
        ["granite", GenericReasoningParser], // Uses generic with specific config
        ["generic", GenericReasoningParser],
    ]);

    /**
     * Create a reasoning parser for the given format
     */
    static create(
        format: ReasoningFormat,
        config?: Partial<ReasoningParserConfig>
    ): ReasoningParser | null {
        if (format === "none") {
            return null;
        }

        const ParserClass = this.parsers.get(format);
        if (!ParserClass) {
            // Default to generic
            return new GenericReasoningParser(config);
        }

        // Special handling for granite (disabled by default)
        if (format === "granite") {
            return new GenericReasoningParser({
                ...config,
                startToken: config?.startToken ?? "<think>",
                endToken: config?.endToken ?? "</think>",
                includeInResponse: config?.includeInResponse ?? false,
            });
        }

        return new ParserClass(config);
    }

    /**
     * Register a custom reasoning parser
     */
    static register(
        format: ReasoningFormat,
        parserClass: new (
            config?: Partial<ReasoningParserConfig>
        ) => ReasoningParser
    ): void {
        this.parsers.set(format, parserClass);
    }
}

/**
 * Factory for creating tool call parsers
 */
export class ToolCallParserFactory {
    private static parsers = new Map<
        ToolCallFormat,
        new (config?: Partial<ToolCallParserConfig>) => ToolCallParser
    >([
        ["hermes", HermesToolCallParser],
        ["llama3", Llama3ToolCallParser],
        ["llama3_json", Llama3ToolCallParser],
        ["qwen", GenericXmlToolCallParser], // Qwen uses similar format
        ["generic_xml", GenericXmlToolCallParser],
        ["generic_json", GenericJsonToolCallParser],
    ]);

    /**
     * Create a tool call parser for the given format
     */
    static create(
        format: ToolCallFormat,
        config?: Partial<ToolCallParserConfig>
    ): ToolCallParser | null {
        if (format === "none") {
            return null;
        }

        const ParserClass = this.parsers.get(format);
        if (!ParserClass) {
            // Default to generic XML
            return new GenericXmlToolCallParser(config);
        }

        return new ParserClass(config);
    }

    /**
     * Register a custom tool call parser
     */
    static register(
        format: ToolCallFormat,
        parserClass: new (
            config?: Partial<ToolCallParserConfig>
        ) => ToolCallParser
    ): void {
        this.parsers.set(format, parserClass);
    }
}

/**
 * Create parsers from a model preset name
 */
export function createParsersFromPreset(
    presetName: string
): {
    reasoningParser: ReasoningParser | null;
    toolCallParser: ToolCallParser | null;
} {
    const preset = MODEL_PARSER_PRESETS[presetName.toLowerCase()];

    if (!preset) {
        return {
            reasoningParser: null,
            toolCallParser: null,
        };
    }

    return createParsersFromConfig(preset);
}

/**
 * Create parsers from a configuration object
 */
export function createParsersFromConfig(config: ContentParserConfig): {
    reasoningParser: ReasoningParser | null;
    toolCallParser: ToolCallParser | null;
} {
    let reasoningParser: ReasoningParser | null = null;
    let toolCallParser: ToolCallParser | null = null;

    if (config.reasoning) {
        reasoningParser = ReasoningParserFactory.create(
            config.reasoning.format,
            config.reasoning
        );
    }

    if (config.toolCall) {
        toolCallParser = ToolCallParserFactory.create(
            config.toolCall.format,
            config.toolCall
        );
    }

    return { reasoningParser, toolCallParser };
}

/**
 * Auto-detect parser configuration from template content
 */
export function detectParsersFromTemplate(templateContent: string): ContentParserConfig {
    const config: ContentParserConfig = {};

    // Detect thinking format
    if (
        templateContent.includes("<think>") ||
        templateContent.includes("</think>")
    ) {
        // Check for specific model patterns
        if (
            templateContent.includes("deepseek") ||
            templateContent.toLowerCase().includes("deepseek")
        ) {
            config.reasoning = {
                format: "deepseek_r1",
                startToken: "<think>",
                endToken: "</think>",
            };
        } else if (
            templateContent.includes("qwen") ||
            templateContent.toLowerCase().includes("qwen")
        ) {
            config.reasoning = {
                format: "qwen3",
                startToken: "<think>",
                endToken: "</think>",
            };
        } else {
            config.reasoning = {
                format: "generic",
                startToken: "<think>",
                endToken: "</think>",
            };
        }
    } else if (
        templateContent.includes("<thinking>") ||
        templateContent.includes("</thinking>")
    ) {
        config.reasoning = {
            format: "generic",
            startToken: "<thinking>",
            endToken: "</thinking>",
        };
    }

    // Detect tool call format
    if (
        templateContent.includes("<tool_call>") ||
        templateContent.includes("</tool_call>")
    ) {
        config.toolCall = {
            format: "generic_xml",
            startToken: "<tool_call>",
            endToken: "</tool_call>",
        };
    } else if (templateContent.includes("<|python_tag|>")) {
        config.toolCall = {
            format: "llama3",
            startToken: "<|python_tag|>",
        };
    } else if (templateContent.includes("tool_calls_begin")) {
        // DeepSeek format
        config.toolCall = {
            format: "deepseek",
            startToken: "<｜tool_calls_begin｜>",
            endToken: "<｜tool_calls_end｜>",
        };
    }

    return config;
}
