// @ts-types="@/types/jinja.d.ts"
import {
    ArrayLiteral,
    BinaryExpression,
    Identifier,
    Literal,
    SetStatement,
    Template,
} from "@huggingface/jinja";
import * as z from "@/common/myZod.ts";
import * as Path from "@std/path";

// From @huggingface/jinja
export function range(start: number, stop?: number, step = 1): number[] {
    if (stop === undefined) {
        stop = start;
        start = 0;
    }

    const result: number[] = [];
    for (let i = start; i < stop; i += step) {
        result.push(i);
    }
    return result;
}

/**
 * Extended template metadata schema with support for:
 * - Stop strings
 * - Tool calling configuration
 * - Thinking/reasoning configuration
 */
const TemplateMetadataSchema = z.object({
    // Stop strings (existing)
    stop_strings: z.array(z.string()).default([]),

    // Tool calling configuration
    tool_start: z.string().optional(),
    tool_end: z.string().optional(),
    tool_format: z.enum([
        "xml",           // <tool_call>...</tool_call>
        "json",          // {"name": ..., "arguments": ...}
        "hermes",        // Hermes style
        "llama3",        // Llama 3.x native format
        "qwen",          // Qwen style
        "deepseek",      // DeepSeek style
        "generic",       // Generic format
    ]).optional(),
    supports_tools: z.boolean().optional(),
    parallel_tool_calls: z.boolean().optional(),

    // Thinking/reasoning configuration
    thinking_start: z.string().optional(),
    thinking_end: z.string().optional(),
    thinking_format: z.enum([
        "deepseek_r1",   // DeepSeek R1 <think>...</think>
        "qwen3",         // Qwen3 <think>...</think>
        "granite",       // IBM Granite
        "generic",       // Generic <thinking>...</thinking>
    ]).optional(),
    supports_thinking: z.boolean().optional(),
    thinking_enabled_by_default: z.boolean().optional(),

    // Generation configuration
    generation_prompt: z.string().optional(),
    response_prefix: z.string().optional(),
});

type TemplateMetadata = z.infer<typeof TemplateMetadataSchema>;

export class PromptTemplate {
    name: string;
    rawTemplate: string;
    template: Template;
    metadata: TemplateMetadata;

    public constructor(
        name: string,
        rawTemplate: string,
    ) {
        this.name = name;
        this.rawTemplate = rawTemplate;
        this.template = new Template(rawTemplate);
        this.metadata = this.extractMetadata(this.template);
    }

    private assignMetadataValue<K extends keyof TemplateMetadata>(
        metadata: TemplateMetadata,
        key: K,
        value: unknown,
    ) {
        metadata[key] = value as TemplateMetadata[K];
    }

    private extractMetadata(template: Template) {
        const metadata: TemplateMetadata = TemplateMetadataSchema.parse({});

        const visited = new WeakSet<object>();

        // Helper to extract literal value from AST node
        const extractLiteralValue = (node: unknown): unknown => {
            if (!node || typeof node !== "object") {
                return undefined;
            }

            const typedNode = node as { type?: string };

            // Handle string concatenation (BinaryExpression with + operator)
            if (typedNode.type === "BinaryExpression") {
                const binExpr = node as BinaryExpression;
                if (binExpr.operator?.value === "+") {
                    const left = extractLiteralValue(binExpr.left);
                    const right = extractLiteralValue(binExpr.right);
                    if (typeof left === "string" && typeof right === "string") {
                        return left + right;
                    }
                }
                return undefined;
            }

            // Handle literals
            if (typedNode.type?.endsWith?.("Literal")) {
                return (node as Literal<unknown>).value;
            }

            return undefined;
        };

        const visitNode = (node: unknown) => {
            if (!node || typeof node !== "object") {
                return;
            }

            if (visited.has(node)) {
                return;
            }
            visited.add(node);

            const statement = node as { type?: string };
            if (statement.type === "Set") {
                const setStatement = statement as SetStatement;

                const assignee = setStatement.assignee as Identifier;
                const foundMetaKey = Object.keys(TemplateMetadataSchema.shape)
                    .find(
                        (key) => key === assignee.value,
                    ) as keyof TemplateMetadata;

                if (foundMetaKey) {
                    const fieldSchema =
                        TemplateMetadataSchema.shape[foundMetaKey];

                    let result: unknown;
                    if (setStatement.value.type === "ArrayLiteral") {
                        const arrayValue = setStatement.value as ArrayLiteral;
                        result = arrayValue.value
                            .map((entry) => extractLiteralValue(entry))
                            .filter((entry) => entry !== undefined);
                    } else {
                        result = extractLiteralValue(setStatement.value);
                    }

                    const parsedValue = fieldSchema.safeParse(result);
                    if (parsedValue.success) {
                        this.assignMetadataValue(
                            metadata,
                            foundMetaKey,
                            parsedValue.data,
                        );
                    }
                }
            }

            for (const value of Object.values(node)) {
                if (Array.isArray(value)) {
                    value.forEach(visitNode);
                } else if (value && typeof value === "object") {
                    visitNode(value);
                }
            }
        };

        template.parsed.body.forEach((statement) => {
            visitNode(statement);
        });

        // Auto-detect from template content if not explicitly set
        this.autoDetectFromContent(metadata);

        return metadata;
    }

    /**
     * Auto-detect metadata from template content
     * This fills in missing metadata based on patterns found in the template
     */
    private autoDetectFromContent(metadata: TemplateMetadata): void {
        const content = this.rawTemplate;

        // Auto-detect thinking format
        if (!metadata.thinking_start && !metadata.thinking_end) {
            if (content.includes("<think>") || content.includes("</think>")) {
                metadata.thinking_start = "<think>";
                metadata.thinking_end = "</think>";
                metadata.supports_thinking = true;

                // Try to determine the format
                if (!metadata.thinking_format) {
                    const lowerContent = content.toLowerCase();
                    if (lowerContent.includes("deepseek")) {
                        metadata.thinking_format = "deepseek_r1";
                    } else if (lowerContent.includes("qwen")) {
                        metadata.thinking_format = "qwen3";
                        metadata.thinking_enabled_by_default = true;
                    } else {
                        metadata.thinking_format = "generic";
                    }
                }
            } else if (
                content.includes("<thinking>") ||
                content.includes("</thinking>")
            ) {
                metadata.thinking_start = "<thinking>";
                metadata.thinking_end = "</thinking>";
                metadata.thinking_format = "generic";
                metadata.supports_thinking = true;
            }
        }

        // Auto-detect tool call format
        if (!metadata.tool_start && !metadata.tool_end) {
            if (
                content.includes("<tool_call>") ||
                content.includes("</tool_call>")
            ) {
                metadata.tool_start = "<tool_call>";
                metadata.tool_end = "</tool_call>";
                metadata.supports_tools = true;

                if (!metadata.tool_format) {
                    metadata.tool_format = "xml";
                }
            } else if (content.includes("<|python_tag|>")) {
                metadata.tool_start = "<|python_tag|>";
                metadata.tool_format = "llama3";
                metadata.supports_tools = true;
            } else if (
                content.includes("tool_calls_begin") ||
                content.includes("｜tool_calls_begin｜")
            ) {
                // DeepSeek format with full-width characters
                metadata.tool_start = "<｜tool_calls_begin｜>";
                metadata.tool_end = "<｜tool_calls_end｜>";
                metadata.tool_format = "deepseek";
                metadata.supports_tools = true;
            } else if (content.includes("[TOOL_CALLS]")) {
                // Mistral format
                metadata.tool_start = "[TOOL_CALLS]";
                metadata.tool_format = "json";
                metadata.supports_tools = true;
            }
        }

        // Check for parallel tool calls support
        if (metadata.supports_tools && metadata.parallel_tool_calls === undefined) {
            // Check if template handles multiple tool calls
            if (
                content.includes("for tool_call in") ||
                content.includes("for call in") ||
                content.includes("tool_calls|length") ||
                content.includes("tool_calls | length")
            ) {
                metadata.parallel_tool_calls = true;
            }
        }

        // Detect generation prompt pattern
        if (!metadata.generation_prompt) {
            // Look for common patterns
            const patterns = [
                /<\|Assistant\|>/,
                /<\|assistant\|>/,
                /<\|im_start\|>assistant/,
                /\[\/INST\]/,
                /### Response:/,
                /Assistant:/,
            ];

            for (const pattern of patterns) {
                const match = content.match(pattern);
                if (match) {
                    metadata.generation_prompt = match[0];
                    break;
                }
            }
        }
    }

    static async fromFile(templatePath: string) {
        const parsedPath = Path.parse(templatePath);
        parsedPath.ext = ".jinja";
        const formattedPath = Path.format({ ...parsedPath, base: undefined });
        const rawTemplate = await Deno.readTextFile(formattedPath);
        return new PromptTemplate(parsedPath.name, rawTemplate);
    }
}
