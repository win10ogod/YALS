/**
 * Anthropic Messages API utilities
 */

import { SSEStreamingApi } from "hono/streaming";
import { logger } from "@/common/logging.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";
import { Model } from "@/bindings/bindings.ts";
import { MultimodalImageData, parseAnthropicImageSource } from "@/common/imageProcessing.ts";
import { MultimodalContext } from "@/bindings/multimodal.ts";

import {
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
    ContentBlock,
    ImageContent,
    StopReason,
    StreamContentBlockDelta,
    StreamContentBlockStart,
    StreamContentBlockStop,
    StreamEvent,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    TextContent,
    ToolResultContent,
    ToolUseContent,
} from "../types/messages.ts";
import { AnthropicContext } from "../types/context.ts";

/**
 * Convert Anthropic messages to prompt string and extract images
 */
export function convertAnthropicMessages(
    messages: AnthropicMessage[],
    systemPrompt?: string | TextContent[],
): { prompt: string; images: MultimodalImageData[] } {
    const images: MultimodalImageData[] = [];
    const parts: string[] = [];
    const mediaMarker = MultimodalContext.getDefaultMarker();

    // Add system prompt if present
    if (systemPrompt) {
        let systemText: string;
        if (typeof systemPrompt === "string") {
            systemText = systemPrompt;
        } else {
            systemText = systemPrompt.map((block) => block.text).join("\n");
        }
        parts.push(`<|system|>\n${systemText}\n`);
    }

    for (const message of messages) {
        const roleTag = message.role === "user" ? "<|user|>" : "<|assistant|>";
        parts.push(`${roleTag}\n`);

        if (typeof message.content === "string") {
            parts.push(message.content);
        } else {
            for (const block of message.content) {
                if (block.type === "text") {
                    parts.push((block as TextContent).text);
                } else if (block.type === "image") {
                    const imageBlock = block as ImageContent;
                    const imageData = parseAnthropicImageSource(imageBlock.source);
                    images.push(imageData);
                    parts.push(mediaMarker);
                } else if (block.type === "tool_result") {
                    const toolResult = block as ToolResultContent;
                    let content = "";
                    if (typeof toolResult.content === "string") {
                        content = toolResult.content;
                    } else if (Array.isArray(toolResult.content)) {
                        content = toolResult.content
                            .filter((c): c is TextContent => c.type === "text")
                            .map((c) => c.text)
                            .join("\n");
                    }
                    parts.push(`<tool_result id="${toolResult.tool_use_id}">\n${content}\n</tool_result>`);
                } else if (block.type === "tool_use") {
                    const toolUse = block as ToolUseContent;
                    parts.push(
                        `<tool_use id="${toolUse.id}" name="${toolUse.name}">\n` +
                        `${JSON.stringify(toolUse.input)}\n</tool_use>`,
                    );
                }
            }
        }

        parts.push("\n");
    }

    // Add generation prompt
    parts.push("<|assistant|>\n");

    return { prompt: parts.join(""), images };
}

/**
 * Convert finish reason to Anthropic format
 */
export function convertStopReason(chunk: FinishChunk): StopReason | null {
    switch (chunk.finishReason) {
        case "StopToken":
            return "end_turn";
        case "MaxNewTokens":
            return "max_tokens";
        case "StopString":
            return "stop_sequence";
        default:
            return "end_turn";
    }
}

/**
 * Create usage stats from finish chunk
 */
export function createUsage(chunk: FinishChunk): AnthropicUsage {
    return {
        input_tokens: chunk.promptTokens,
        output_tokens: chunk.genTokens,
    };
}

/**
 * Create static (non-streaming) response
 */
export function createAnthropicResponse(
    chunk: FinishChunk,
    modelName: string,
    requestId: string,
): AnthropicMessagesResponse {
    const content: ContentBlock[] = [];

    // Add text content
    if (chunk.text) {
        content.push({
            type: "text",
            text: chunk.text,
        });
    }

    // Handle tool calls if present
    if (chunk.toolCalls) {
        try {
            const toolCalls = JSON.parse(chunk.toolCalls);
            for (const call of toolCalls) {
                content.push({
                    type: "tool_use",
                    id: call.id || `toolu_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`,
                    name: call.function?.name || call.name,
                    input: call.function?.arguments
                        ? JSON.parse(call.function.arguments)
                        : call.input || {},
                });
            }
        } catch {
            // Ignore parse errors
        }
    }

    // Determine stop reason
    let stopReason = convertStopReason(chunk);
    if (content.some((c) => c.type === "tool_use")) {
        stopReason = "tool_use";
    }

    return {
        id: `msg_${requestId}`,
        type: "message",
        role: "assistant",
        model: modelName,
        content,
        stop_reason: stopReason,
        stop_sequence: chunk.stopToken || null,
        usage: createUsage(chunk),
    };
}

/**
 * Write an SSE event to the stream
 */
export async function writeStreamEvent(
    stream: SSEStreamingApi,
    event: StreamEvent,
): Promise<void> {
    await stream.writeSSE({
        event: event.type,
        data: JSON.stringify(event),
    });
}

/**
 * Stream Anthropic response
 */
export async function streamAnthropicResponse(
    ctx: AnthropicContext,
    params: AnthropicMessagesRequest,
    stream: SSEStreamingApi,
    generator: AsyncGenerator<GenerationChunk>,
): Promise<void> {
    const msgId = `msg_${ctx.requestId}`;
    let inputTokens = 0;
    let outputTokens = 0;
    let fullText = "";

    // Send message_start
    const messageStart: StreamMessageStart = {
        type: "message_start",
        message: {
            id: msgId,
            type: "message",
            role: "assistant",
            model: params.model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: {
                input_tokens: 0,
                output_tokens: 0,
            },
        },
    };
    await writeStreamEvent(stream, messageStart);

    // Send content_block_start for text
    const contentBlockStart: StreamContentBlockStart = {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
    };
    await writeStreamEvent(stream, contentBlockStart);

    // Stream text deltas
    for await (const chunk of generator) {
        if (chunk.kind === "data") {
            fullText += chunk.text;
            outputTokens++;

            const delta: StreamContentBlockDelta = {
                type: "content_block_delta",
                index: 0,
                delta: {
                    type: "text_delta",
                    text: chunk.text,
                },
            };
            await writeStreamEvent(stream, delta);
        } else if (chunk.kind === "finish") {
            inputTokens = chunk.promptTokens;
            outputTokens = chunk.genTokens;
        }
    }

    // Send content_block_stop
    const contentBlockStop: StreamContentBlockStop = {
        type: "content_block_stop",
        index: 0,
    };
    await writeStreamEvent(stream, contentBlockStop);

    // Send message_delta
    const messageDelta: StreamMessageDelta = {
        type: "message_delta",
        delta: {
            stop_reason: "end_turn",
            stop_sequence: null,
        },
        usage: {
            output_tokens: outputTokens,
        },
    };
    await writeStreamEvent(stream, messageDelta);

    // Send message_stop
    const messageStop: StreamMessageStop = {
        type: "message_stop",
    };
    await writeStreamEvent(stream, messageStop);
}

/**
 * Convert Anthropic request parameters to internal sampler format
 */
export function convertSamplerParams(params: AnthropicMessagesRequest) {
    return {
        max_tokens: params.max_tokens,
        temperature: params.temperature ?? 1.0,
        top_p: params.top_p ?? 1.0,
        top_k: params.top_k ?? 0,
        stop: params.stop_sequences || [],
        // Generation options
        min_tokens: 0,
        add_bos_token: undefined,
        ban_eos_token: false,
        seed: undefined,
        logit_bias: {},
        json_schema: undefined,
        regex_pattern: undefined,
        grammar_string: undefined,
        banned_tokens: [],
        banned_strings: [],
        // Penalty samplers
        repetition_penalty: 1.0,
        frequency_penalty: 0,
        presence_penalty: 0,
        penalty_range: -1,
        // Alphabet samplers
        min_p: 0.0,
        typical: 1.0,
        nsigma: 0.0,
        // Mirostat
        mirostat_mode: 0,
        mirostat_tau: 1.0,
        mirostat_eta: 0.0,
        // DRY
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_range: 0,
        dry_sequence_breakers: [],
        // Temperature
        temperature_last: false,
        // XTC
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        // Dynamic temperature
        max_temp: 1.0,
        min_temp: 1.0,
        temp_exponent: 1.0,
    };
}
