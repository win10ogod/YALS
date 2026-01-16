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

// Tool call format types supported by templates
export const ToolCallFormat = z.enum([
    "xml_kv",      // GLM format: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    "json_simple", // IQuest format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    "json_oai",    // OAI format: [{"function": {"name": "...", "arguments": "..."}}]
]);

export type ToolCallFormat = z.infer<typeof ToolCallFormat>;

const TemplateMetadataSchema = z.object({
    // Stop strings for the model
    stop_strings: z.array(z.string()).default([]),

    // Tool call configuration
    tool_start: z.string().optional(),
    tool_end: z.string().optional(),
    tool_call_format: ToolCallFormat.optional(),

    // Thinking/reasoning configuration
    thinking_start: z.string().optional(),
    thinking_end: z.string().optional(),
    supports_thinking: z.boolean().optional(),
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

        return metadata;
    }

    static async fromFile(templatePath: string) {
        const parsedPath = Path.parse(templatePath);
        parsedPath.ext = ".jinja";
        const formattedPath = Path.format({ ...parsedPath, base: undefined });
        const rawTemplate = await Deno.readTextFile(formattedPath);
        return new PromptTemplate(parsedPath.name, rawTemplate);
    }
}
