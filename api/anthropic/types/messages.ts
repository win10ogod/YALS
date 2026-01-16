import * as z from "@/common/myZod.ts";

export const AnthropicError = z.object({
    type: z.string(),
    message: z.string(),
});

export const AnthropicErrorResponse = z.object({
    type: z.literal("error").default("error"),
    error: AnthropicError,
});

export const AnthropicUsage = z.object({
    input_tokens: z.number(),
    output_tokens: z.number(),
    cache_creation_input_tokens: z.number().optional(),
    cache_read_input_tokens: z.number().optional(),
});

export const AnthropicContentBlock = z.object({
    type: z.enum(["text", "image", "tool_use", "tool_result", "thinking"]),
    text: z.string().cleanOptional(),
    thinking: z.string().cleanOptional(), // For thinking blocks
    source: z.record(z.string(), z.unknown()).cleanOptional(),
    id: z.string().cleanOptional(),
    tool_use_id: z.string().cleanOptional(),
    name: z.string().cleanOptional(),
    input: z.record(z.string(), z.unknown()).cleanOptional(),
    content: z.union([
        z.string(),
        z.array(z.record(z.string(), z.unknown())),
    ]).cleanOptional(),
    is_error: z.boolean().cleanOptional(),
});

export const AnthropicMessage = z.object({
    role: z.enum(["user", "assistant"]),
    content: z.union([z.string(), z.array(AnthropicContentBlock)]),
});

export const AnthropicTool = z.object({
    name: z.string(),
    description: z.string().cleanOptional(),
    input_schema: z.record(z.string(), z.unknown()),
});

export const AnthropicToolChoice = z.object({
    type: z.enum(["auto", "any", "tool"]),
    name: z.string().cleanOptional(),
});

export const AnthropicMessagesRequest = z.object({
    model: z.string(),
    messages: z.array(AnthropicMessage),
    max_tokens: z.number().gt(0),
    metadata: z.record(z.string(), z.unknown()).cleanOptional(),
    stop_sequences: z.array(z.string()).cleanOptional(),
    stream: z.boolean().nullish().coalesce(false),
    system: z.union([z.string(), z.array(AnthropicContentBlock)]).cleanOptional(),
    temperature: z.number().cleanOptional(),
    tool_choice: AnthropicToolChoice.cleanOptional(),
    tools: z.array(AnthropicTool).cleanOptional(),
    top_k: z.number().cleanOptional(),
    top_p: z.number().cleanOptional(),
});

export type AnthropicMessagesRequest = z.infer<typeof AnthropicMessagesRequest>;

export const AnthropicMessagesResponse = z.object({
    id: z.string().default(`msg_${Date.now()}`),
    type: z.literal("message").default("message"),
    role: z.literal("assistant").default("assistant"),
    content: z.array(AnthropicContentBlock),
    model: z.string(),
    stop_reason: z.enum([
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
    ]).optional(),
    stop_sequence: z.string().optional(),
    usage: AnthropicUsage.optional(),
});

export const AnthropicDelta = z.object({
    type: z.enum(["text_delta", "input_json_delta", "thinking_delta"])
        .cleanOptional(),
    text: z.string().cleanOptional(),
    thinking: z.string().cleanOptional(), // For thinking deltas
    partial_json: z.string().cleanOptional(),
    stop_reason: z.enum([
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
    ]).optional(),
    stop_sequence: z.string().optional(),
});

export const AnthropicStreamEvent = z.object({
    type: z.enum([
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]),
    message: AnthropicMessagesResponse.cleanOptional(),
    delta: AnthropicDelta.cleanOptional(),
    content_block: AnthropicContentBlock.cleanOptional(),
    index: z.number().cleanOptional(),
    error: AnthropicError.cleanOptional(),
    usage: AnthropicUsage.cleanOptional(),
});
