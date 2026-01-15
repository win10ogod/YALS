/**
 * Anthropic Messages API types
 * Compatible with Claude API format
 */

import * as z from "@/common/myZod.ts";

// ~~~ Content Block Types ~~~

export const TextContent = z.object({
    type: z.literal("text"),
    text: z.string(),
});

export type TextContent = z.infer<typeof TextContent>;

export const ImageSourceBase64 = z.object({
    type: z.literal("base64"),
    media_type: z.string(),
    data: z.string(),
});

export const ImageSourceUrl = z.object({
    type: z.literal("url"),
    url: z.string(),
});

export const ImageSource = z.union([ImageSourceBase64, ImageSourceUrl]);

export const ImageContent = z.object({
    type: z.literal("image"),
    source: ImageSource,
});

export type ImageContent = z.infer<typeof ImageContent>;

export const ToolUseContent = z.object({
    type: z.literal("tool_use"),
    id: z.string(),
    name: z.string(),
    input: z.record(z.string(), z.unknown()),
});

export type ToolUseContent = z.infer<typeof ToolUseContent>;

export const ToolResultContent = z.object({
    type: z.literal("tool_result"),
    tool_use_id: z.string(),
    content: z.union([
        z.string(),
        z.array(z.union([TextContent, ImageContent])),
    ]).optional(),
    is_error: z.boolean().optional(),
});

export type ToolResultContent = z.infer<typeof ToolResultContent>;

export const ContentBlock = z.union([
    TextContent,
    ImageContent,
    ToolUseContent,
    ToolResultContent,
]);

export type ContentBlock = z.infer<typeof ContentBlock>;

// ~~~ Message Types ~~~

export const AnthropicMessage = z.object({
    role: z.enum(["user", "assistant"]),
    content: z.union([z.string(), z.array(ContentBlock)]),
});

export type AnthropicMessage = z.infer<typeof AnthropicMessage>;

// ~~~ Tool Definition ~~~

export const ToolInputSchema = z.object({
    type: z.literal("object").optional().default("object"),
    properties: z.record(z.string(), z.unknown()).optional(),
    required: z.array(z.string()).optional(),
}).passthrough();

export const AnthropicTool = z.object({
    name: z.string(),
    description: z.string().optional(),
    input_schema: ToolInputSchema,
});

export type AnthropicTool = z.infer<typeof AnthropicTool>;

// ~~~ Tool Choice ~~~

export const ToolChoiceAuto = z.object({
    type: z.literal("auto"),
    disable_parallel_tool_use: z.boolean().optional(),
});

export const ToolChoiceAny = z.object({
    type: z.literal("any"),
    disable_parallel_tool_use: z.boolean().optional(),
});

export const ToolChoiceNamed = z.object({
    type: z.literal("tool"),
    name: z.string(),
    disable_parallel_tool_use: z.boolean().optional(),
});

export const ToolChoice = z.union([
    z.literal("auto"),
    z.literal("any"),
    z.literal("none"),
    ToolChoiceAuto,
    ToolChoiceAny,
    ToolChoiceNamed,
]);

export type ToolChoice = z.infer<typeof ToolChoice>;

// ~~~ Request Schema ~~~

export const AnthropicMessagesRequest = z.object({
    model: z.string(),
    messages: z.array(AnthropicMessage),
    max_tokens: z.number().positive(),
    system: z.union([z.string(), z.array(TextContent)]).optional(),
    stop_sequences: z.array(z.string()).optional(),
    stream: z.boolean().optional().default(false),
    temperature: z.number().min(0).max(1).optional(),
    top_p: z.number().min(0).max(1).optional(),
    top_k: z.number().int().positive().optional(),
    tools: z.array(AnthropicTool).optional(),
    tool_choice: ToolChoice.optional(),
    metadata: z.object({
        user_id: z.string().optional(),
    }).optional(),
});

export type AnthropicMessagesRequest = z.infer<typeof AnthropicMessagesRequest>;

// ~~~ Response Types ~~~

export const StopReason = z.enum([
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
]);

export type StopReason = z.infer<typeof StopReason>;

export const AnthropicUsage = z.object({
    input_tokens: z.number(),
    output_tokens: z.number(),
    cache_creation_input_tokens: z.number().optional(),
    cache_read_input_tokens: z.number().optional(),
});

export type AnthropicUsage = z.infer<typeof AnthropicUsage>;

export const AnthropicMessagesResponse = z.object({
    id: z.string(),
    type: z.literal("message"),
    role: z.literal("assistant"),
    model: z.string(),
    content: z.array(ContentBlock),
    stop_reason: StopReason.nullable(),
    stop_sequence: z.string().nullable(),
    usage: AnthropicUsage,
});

export type AnthropicMessagesResponse = z.infer<typeof AnthropicMessagesResponse>;

// ~~~ Streaming Event Types ~~~

export const StreamMessageStart = z.object({
    type: z.literal("message_start"),
    message: z.object({
        id: z.string(),
        type: z.literal("message"),
        role: z.literal("assistant"),
        model: z.string(),
        content: z.array(z.unknown()),
        stop_reason: z.null(),
        stop_sequence: z.null(),
        usage: z.object({
            input_tokens: z.number(),
            output_tokens: z.number(),
        }),
    }),
});

export type StreamMessageStart = z.infer<typeof StreamMessageStart>;

export const StreamContentBlockStart = z.object({
    type: z.literal("content_block_start"),
    index: z.number(),
    content_block: z.union([
        z.object({ type: z.literal("text"), text: z.string() }),
        z.object({
            type: z.literal("tool_use"),
            id: z.string(),
            name: z.string(),
            input: z.object({}),
        }),
    ]),
});

export type StreamContentBlockStart = z.infer<typeof StreamContentBlockStart>;

export const StreamTextDelta = z.object({
    type: z.literal("text_delta"),
    text: z.string(),
});

export const StreamInputJsonDelta = z.object({
    type: z.literal("input_json_delta"),
    partial_json: z.string(),
});

export const StreamContentBlockDelta = z.object({
    type: z.literal("content_block_delta"),
    index: z.number(),
    delta: z.union([StreamTextDelta, StreamInputJsonDelta]),
});

export type StreamContentBlockDelta = z.infer<typeof StreamContentBlockDelta>;

export const StreamContentBlockStop = z.object({
    type: z.literal("content_block_stop"),
    index: z.number(),
});

export type StreamContentBlockStop = z.infer<typeof StreamContentBlockStop>;

export const StreamMessageDelta = z.object({
    type: z.literal("message_delta"),
    delta: z.object({
        stop_reason: StopReason.nullable(),
        stop_sequence: z.string().nullable(),
    }),
    usage: z.object({
        output_tokens: z.number(),
    }),
});

export type StreamMessageDelta = z.infer<typeof StreamMessageDelta>;

export const StreamMessageStop = z.object({
    type: z.literal("message_stop"),
});

export type StreamMessageStop = z.infer<typeof StreamMessageStop>;

export const StreamPing = z.object({
    type: z.literal("ping"),
});

export const StreamError = z.object({
    type: z.literal("error"),
    error: z.object({
        type: z.string(),
        message: z.string(),
    }),
});

export type StreamEvent =
    | StreamMessageStart
    | StreamContentBlockStart
    | StreamContentBlockDelta
    | StreamContentBlockStop
    | StreamMessageDelta
    | StreamMessageStop
    | z.infer<typeof StreamPing>
    | z.infer<typeof StreamError>;

// ~~~ Error Response ~~~

export const AnthropicErrorResponse = z.object({
    type: z.literal("error"),
    error: z.object({
        type: z.string(),
        message: z.string(),
    }),
});

export type AnthropicErrorResponse = z.infer<typeof AnthropicErrorResponse>;
