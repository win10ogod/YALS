// @ts-types="@/types/jinja.d.ts"
import {
    ArrayLiteral,
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

const TemplateMetadataSchema = z.object({
    stop_strings: z.array(z.string()).default([]),
    tool_start: z.string().optional(),
    tool_call_start: z.string().optional(),
    tool_call_end: z.string().optional(),
    tool_calls_start: z.string().optional(),
    tool_calls_end: z.string().optional(),
    tool_call_sep: z.string().optional(),
    reasoning_start: z.string().optional(),
    reasoning_end: z.string().optional(),
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
                            .map((entry) => {
                                if (
                                    (entry as Literal<unknown>)?.type
                                        ?.endsWith?.("Literal")
                                ) {
                                    const literalValue = entry as Literal<
                                        unknown
                                    >;
                                    return literalValue.value;
                                }
                                return undefined;
                            })
                            .filter((entry) => entry !== undefined);
                    } else if (setStatement.value.type.endsWith("Literal")) {
                        const literalValue = setStatement.value as Literal<
                            unknown
                        >;
                        result = literalValue.value;
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

        this.inferMetadataFromTemplate(metadata, this.rawTemplate);
        return metadata;
    }

    private inferMetadataFromTemplate(
        metadata: TemplateMetadata,
        rawTemplate: string,
    ) {
        const setIfMissing = <K extends keyof TemplateMetadata>(
            key: K,
            value: TemplateMetadata[K],
        ) => {
            if (metadata[key] === undefined) {
                metadata[key] = value;
            }
        };

        const QWEN_TOOL_CALLS_BEGIN = "<\uFF5Ctool\u2581calls\u2581begin\uFF5C>";
        const QWEN_TOOL_CALL_BEGIN = "<\uFF5Ctool\u2581call\u2581begin\uFF5C>";
        const QWEN_TOOL_SEP = "<\uFF5Ctool\u2581sep\uFF5C>";
        const QWEN_TOOL_CALL_END = "<\uFF5Ctool\u2581call\u2581end\uFF5C>";
        const QWEN_TOOL_CALLS_END = "<\uFF5Ctool\u2581calls\u2581end\uFF5C>";

        if (rawTemplate.includes("<think>") || rawTemplate.includes("</think>")) {
            setIfMissing("reasoning_start", "<think>");
            setIfMissing("reasoning_end", "</think>");
        } else if (
            rawTemplate.includes("[THINK]") ||
            rawTemplate.includes("[/THINK]")
        ) {
            setIfMissing("reasoning_start", "[THINK]");
            setIfMissing("reasoning_end", "[/THINK]");
        }

        if (rawTemplate.includes("<tool_call>")) {
            setIfMissing("tool_call_start", "<tool_call>");
        }
        if (rawTemplate.includes("</tool_call>")) {
            setIfMissing("tool_call_end", "</tool_call>");
        }
        if (rawTemplate.includes("<tool_calls>")) {
            setIfMissing("tool_calls_start", "<tool_calls>");
        }
        if (rawTemplate.includes("</tool_calls>")) {
            setIfMissing("tool_calls_end", "</tool_calls>");
        }

        if (rawTemplate.includes("<|tool_call|>")) {
            setIfMissing("tool_call_start", "<|tool_call|>");
        }
        if (rawTemplate.includes("<|tools_prefix|>")) {
            setIfMissing("tool_calls_start", "<|tools_prefix|>");
        }
        if (rawTemplate.includes("<|tools_suffix|>")) {
            setIfMissing("tool_calls_end", "<|tools_suffix|>");
        }

        if (rawTemplate.includes("<|tool_calls_section_begin|>")) {
            setIfMissing("tool_calls_start", "<|tool_calls_section_begin|>");
        }
        if (rawTemplate.includes("<|tool_calls_section_end|>")) {
            setIfMissing("tool_calls_end", "<|tool_calls_section_end|>");
        }
        if (rawTemplate.includes("<|tool_call_begin|>")) {
            setIfMissing("tool_call_start", "<|tool_call_begin|>");
        }
        if (rawTemplate.includes("<|tool_call_end|>")) {
            setIfMissing("tool_call_end", "<|tool_call_end|>");
        }
        if (rawTemplate.includes("<|tool_call_argument_begin|>")) {
            setIfMissing("tool_call_sep", "<|tool_call_argument_begin|>");
        }

        if (
            rawTemplate.includes(QWEN_TOOL_CALLS_BEGIN) ||
            rawTemplate.includes(QWEN_TOOL_CALLS_END)
        ) {
            setIfMissing("tool_calls_start", QWEN_TOOL_CALLS_BEGIN);
            setIfMissing("tool_calls_end", QWEN_TOOL_CALLS_END);
        }
        if (rawTemplate.includes(QWEN_TOOL_CALL_BEGIN)) {
            setIfMissing("tool_call_start", QWEN_TOOL_CALL_BEGIN);
        }
        if (rawTemplate.includes(QWEN_TOOL_CALL_END)) {
            setIfMissing("tool_call_end", QWEN_TOOL_CALL_END);
        }
        if (rawTemplate.includes(QWEN_TOOL_SEP)) {
            setIfMissing("tool_call_sep", QWEN_TOOL_SEP);
        }

        if (
            rawTemplate.includes("<seed:tool_call>") ||
            rawTemplate.includes("</seed:tool_call>")
        ) {
            setIfMissing("tool_call_start", "<seed:tool_call>");
            setIfMissing("tool_call_end", "</seed:tool_call>");
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
