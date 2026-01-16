import { SSEStreamingApi } from "hono/streaming";
import { HTTPException } from "hono/http-exception";
import { Queue } from "@core/asyncutil";
import * as z from "@/common/myZod.ts";

import {
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
} from "../types/messages.ts";
import {
    ChatCompletionMessage,
    ChatCompletionMessagePart,
    ChatCompletionRequest,
    ChatCompletionResponse,
} from "@/api/OAI/types/chatCompletions.ts";
import { ToolCall, ToolSpec } from "@/api/OAI/types/tools.ts";
import {
    applyChatTemplate,
    generateChatCompletion,
} from "@/api/OAI/utils/chatCompletion.ts";
import { normalizeChatMessages } from "@/api/OAI/utils/messages.ts";
import {
    convertFinishReason,
    GenerationType,
    staticGenerate,
    streamCollector,
} from "@/api/OAI/utils/generation.ts";
import {
    createInlineToolCallParser,
    TOOL_CALL_SCHEMA,
    ToolCallProcessor,
    ToolParserConfig,
} from "@/api/OAI/utils/tools.ts";
import {
    createStreamingThinkingParser,
    extractThinking,
    ThinkingConfig,
} from "@/api/OAI/utils/thinking.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { logger } from "@/common/logging.ts";
import { OAIContext } from "@/api/OAI/types/context.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";

type AnthropicStopReason = "end_turn" | "max_tokens" | "stop_sequence" | "tool_use";
type AnthropicMessagesResponseType = z.infer<typeof AnthropicMessagesResponse>;
type AnthropicContentBlockType = z.infer<typeof AnthropicContentBlock>;
type ChatCompletionResponseType = z.infer<typeof ChatCompletionResponse>;

const STOP_REASON_MAP: Record<string, AnthropicStopReason> = {
    stop: "end_turn",
    length: "max_tokens",
    tool_calls: "tool_use",
};

/**
 * Get tool parser configuration from template metadata
 */
function getToolParserConfig(
    promptTemplate: PromptTemplate,
): ToolParserConfig {
    const metadata = promptTemplate.metadata;
    return {
        format: metadata.tool_call_format,
        startToken: metadata.tool_start ?? "<tool_call>",
        endToken: metadata.tool_end ?? "</tool_call>",
    };
}

/**
 * Get thinking parser configuration from template metadata
 */
function getThinkingConfig(
    promptTemplate: PromptTemplate,
): ThinkingConfig | undefined {
    const metadata = promptTemplate.metadata;
    if (!metadata.supports_thinking) {
        return undefined;
    }
    return {
        startTag: metadata.thinking_start ?? "<think>",
        endTag: metadata.thinking_end ?? "</think>",
    };
}

function convertAnthropicToOpenAI(
    request: AnthropicMessagesRequest,
): ChatCompletionRequest {
    const openaiMessages: ChatCompletionMessage[] = [];

    if (request.system) {
        if (typeof request.system === "string") {
            openaiMessages.push({
                role: "system",
                content: request.system,
                tool_calls: undefined,
                tool_call_id: undefined,
            });
        } else {
            const systemText = request.system
                .filter((block) => block.type === "text" && block.text)
                .map((block) => block.text)
                .join("");
            if (systemText) {
                openaiMessages.push({
                    role: "system",
                    content: systemText,
                    tool_calls: undefined,
                    tool_call_id: undefined,
                });
            }
        }
    }

    for (const msg of request.messages) {
        const openaiMessage: ChatCompletionMessage = {
            role: msg.role,
            content: "",
            tool_calls: undefined,
            tool_call_id: undefined,
        };

        if (typeof msg.content === "string") {
            openaiMessage.content = msg.content;
            openaiMessages.push(openaiMessage);
            continue;
        }

        const contentParts: ChatCompletionMessagePart[] = [];
        const toolCalls: ToolCall[] = [];
        const toolMessages: ChatCompletionMessage[] = [];

        for (const block of msg.content) {
            if (block.type === "text" && block.text) {
                contentParts.push({
                    type: "text",
                    text: block.text,
                    image_url: undefined,
                });
                continue;
            }

            if (block.type === "image" && block.source) {
                const source = block.source as Record<string, unknown>;
                if (source.type === "base64") {
                    const mediaType = String(source.media_type ?? "image/png");
                    const data = String(source.data ?? "");
                    contentParts.push({
                        type: "image_url",
                        text: undefined,
                        image_url: {
                            url: `data:${mediaType};base64,${data}`,
                            detail: undefined,
                        },
                    });
                } else if (source.type === "url") {
                    const url = String(source.url ?? "");
                    contentParts.push({
                        type: "image_url",
                        text: undefined,
                        image_url: {
                            url,
                            detail: undefined,
                        },
                    });
                }
                continue;
            }

            if (block.type === "tool_use") {
                toolCalls.push({
                    id: block.id ?? `call_${Date.now()}`,
                    type: "function",
                    function: {
                        name: block.name ?? "",
                        arguments: JSON.stringify(block.input ?? {}),
                    },
                });
                continue;
            }

            if (block.type === "tool_result") {
                if (msg.role === "user") {
                    toolMessages.push({
                        role: "tool",
                        tool_call_id: block.tool_use_id ?? block.id ?? "",
                        content: block.content ? String(block.content) : "",
                        tool_calls: undefined,
                    });
                } else {
                    contentParts.push({
                        type: "text",
                        text: block.content ? String(block.content) : "",
                        image_url: undefined,
                    });
                }
            }
        }

        if (toolCalls && toolCalls.length > 0) {
            openaiMessage.tool_calls = toolCalls;
        }

        if (
            contentParts.length === 1 &&
            (contentParts[0].type ?? "text") === "text"
        ) {
            openaiMessage.content = contentParts[0].text ?? "";
        } else if (contentParts.length > 0) {
            openaiMessage.content = contentParts;
        }

        if (openaiMessage.content || openaiMessage.tool_calls?.length) {
            openaiMessages.push(openaiMessage);
        }

        if (toolMessages.length > 0) {
            openaiMessages.push(...toolMessages);
        }
    }

    const tools = request.tools?.map<ToolSpec>((tool) => ({
        type: "function",
        function: {
            name: tool.name,
            description: tool.description ?? "",
            parameters: tool.input_schema,
        },
    }));

    const base: ChatCompletionRequest = ChatCompletionRequest.parse({
        model: request.model,
        messages: openaiMessages,
        max_tokens: request.max_tokens,
        stop: request.stop_sequences,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        stream: request.stream ?? false,
        tools,
    });

    return base;
}

function convertOpenAIResponse(
    response: ChatCompletionResponseType,
): AnthropicMessagesResponseType {
    const choice = response.choices[0];
    const message = choice.message as Record<string, unknown>;
    const contentText = typeof message.content === "string"
        ? message.content
        : "";
    const reasoningContent = message.reasoning_content as string | undefined;

    const content: AnthropicContentBlockType[] = [];

    // Add thinking block first if present (Anthropic extended thinking format)
    if (reasoningContent) {
        content.push(
            AnthropicContentBlock.parse({
                type: "thinking",
                thinking: reasoningContent,
            }),
        );
    }

    // Add text content
    if (contentText) {
        content.push(
            AnthropicContentBlock.parse({
                type: "text",
                text: contentText,
            }),
        );
    }

    // Add tool calls
    if (choice.message.tool_calls) {
        for (const toolCall of choice.message.tool_calls) {
            const args = toolCall.function.arguments;
            let input: Record<string, unknown> = {};
            if (typeof args === "string") {
                try {
                    input = JSON.parse(args);
                } catch {
                    input = {};
                }
            } else if (args && typeof args === "object") {
                input = args as Record<string, unknown>;
            }
            content.push(
                AnthropicContentBlock.parse({
                    type: "tool_use",
                    id: toolCall.id,
                    name: toolCall.function.name,
                    input,
                }),
            );
        }
    }

    // Ensure at least one content block
    if (content.length === 0) {
        content.push(
            AnthropicContentBlock.parse({
                type: "text",
                text: "",
            }),
        );
    }

    return AnthropicMessagesResponse.parse({
        id: response.id,
        model: response.model,
        content,
        stop_reason: STOP_REASON_MAP[choice.finish_reason ?? "stop"],
        usage: response.usage
            ? AnthropicUsage.parse({
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
            })
            : undefined,
    });
}

function addTemplateMetadata(
    promptTemplate: PromptTemplate,
    params: ChatCompletionRequest,
) {
    const metadata = promptTemplate.metadata;
    if (metadata.stop_strings) {
        params.stop.push(...metadata.stop_strings);
    }
    if (metadata.tool_start) {
        params.stop.push(metadata.tool_start);
    }
}

async function buildPrompt(
    ctx: OAIContext,
    params: ChatCompletionRequest,
    promptTemplate: PromptTemplate,
) {
    const { messages, media } = await normalizeChatMessages(params.messages, {
        mediaMarker: ctx.model.mediaMarker,
        decodeImages: true,
    });

    if (media.length > 0 && !ctx.model.supportsVision) {
        throw new HTTPException(422, {
            message: "The current model does not support image inputs.",
        });
    }

    const prompt = applyChatTemplate(
        ctx.model,
        promptTemplate,
        messages,
        {
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: {
                ...ctx.model.chatTemplateKwargs,
                ...params.template_vars,
            },
            tools: params.tools,
            responsePrefix: params.response_prefix,
        },
    );

    addTemplateMetadata(promptTemplate, params);

    return { prompt, media };
}

async function maybeGenerateToolCalls(
    ctx: OAIContext,
    prompt: string,
    params: ChatCompletionRequest,
    promptTemplate: PromptTemplate,
    finish: FinishChunk,
    media: Uint8Array[],
): Promise<ToolCall[]> {
    const toolStart = promptTemplate.metadata.tool_start;
    if (!toolStart || !finish.stopToken?.startsWith(toolStart)) {
        return [];
    }

    const toolParams = structuredClone(params);
    toolParams.json_schema = TOOL_CALL_SCHEMA;

    const toolCtx = {
        ...ctx,
        requestId: `${finish.requestId}-tool`,
    };

    const toolPrompt = prompt + finish.fullText;
    const toolGens = await staticGenerate(
        toolCtx,
        GenerationType.ChatCompletion,
        toolPrompt,
        toolParams,
        media,
    );

    const toolGen = toolGens[0];
    if (!toolGen?.text) {
        return [];
    }

    try {
        return ToolCallProcessor.fromJson(toolGen.text);
    } catch (error) {
        logger.warn(
            `Failed to parse tool calls for ${finish.requestId}: ${String(error)}`,
        );
        return [];
    }
}

export async function generateAnthropicMessages(
    ctx: OAIContext,
    request: AnthropicMessagesRequest,
    promptTemplate: PromptTemplate,
) {
    const openaiRequest = convertAnthropicToOpenAI(request);
    const response = await generateChatCompletion(
        ctx,
        openaiRequest,
        promptTemplate,
    );

    return convertOpenAIResponse(response);
}

export async function streamAnthropicMessages(
    ctx: OAIContext,
    request: AnthropicMessagesRequest,
    promptTemplate: PromptTemplate,
    stream: SSEStreamingApi,
) {
    logger.info(`Received streaming anthropic request ${ctx.requestId}`);
    const openaiRequest = convertAnthropicToOpenAI(request);
    const allowToolParse = !!openaiRequest.tools?.length;

    // Get parser configurations from template
    const toolParserConfig = getToolParserConfig(promptTemplate);
    const thinkingConfig = getThinkingConfig(promptTemplate);

    const inlineToolParser = allowToolParse
        ? createInlineToolCallParser(toolParserConfig)
        : null;
    const inlineThinkingParser = thinkingConfig
        ? createStreamingThinkingParser(thinkingConfig)
        : null;

    const { prompt, media } = await buildPrompt(
        ctx,
        openaiRequest,
        promptTemplate,
    );

    const genAbortController = new AbortController();
    let finished = false;
    const messageId = `msg_${crypto.randomUUID().replaceAll("-", "")}`;

    ctx.cancellationSignal.addEventListener("abort", () => {
        if (!finished) {
            genAbortController.abort();
            finished = true;
        }
    });

    let thinkingBlockStarted = false;
    let thinkingBlockIndex = -1;
    let contentBlockStarted = false;
    let contentBlockIndex = 0;

    const writeEvent = async (
        event: string,
        payload: unknown,
    ) => {
        await stream.writeSSE({
            event,
            data: JSON.stringify(payload),
        });
    };

    await writeEvent("message_start", AnthropicStreamEvent.parse({
        type: "message_start",
        message: AnthropicMessagesResponse.parse({
            id: messageId,
            model: ctx.model.path.name,
            content: [],
            usage: AnthropicUsage.parse({
                input_tokens: 0,
                output_tokens: 0,
            }),
        }),
    }));

    try {
        const queue = new Queue<GenerationChunk | Error>();
        const task = streamCollector(
            ctx,
            prompt,
            openaiRequest,
            genAbortController.signal,
            0,
            queue,
            media,
        );
        void task;

        let finishChunk: FinishChunk | null = null;

        while (true) {
            if (finished) {
                break;
            }

            const chunk = await queue.pop({
                signal: genAbortController.signal,
            });
            if (chunk instanceof Error) {
                genAbortController.abort();
                throw chunk;
            }

            if (chunk.kind === "data") {
                let textToProcess = chunk.text;
                let thinkingText = "";

                // Process through thinking parser first
                if (inlineThinkingParser) {
                    const thinkingResult = inlineThinkingParser.process(
                        textToProcess,
                    );
                    textToProcess = thinkingResult.content;
                    thinkingText = thinkingResult.thinking;

                    // Stream thinking content if we have any
                    if (thinkingText) {
                        if (!thinkingBlockStarted) {
                            thinkingBlockIndex = contentBlockIndex;
                            contentBlockIndex++;
                            await writeEvent(
                                "content_block_start",
                                AnthropicStreamEvent.parse({
                                    type: "content_block_start",
                                    index: thinkingBlockIndex,
                                    content_block: AnthropicContentBlock.parse({
                                        type: "thinking",
                                        thinking: "",
                                    }),
                                }),
                            );
                            thinkingBlockStarted = true;
                        }

                        await writeEvent(
                            "content_block_delta",
                            AnthropicStreamEvent.parse({
                                type: "content_block_delta",
                                index: thinkingBlockIndex,
                                delta: AnthropicDelta.parse({
                                    type: "thinking_delta",
                                    thinking: thinkingText,
                                }),
                            }),
                        );
                    }
                }

                // Process through tool parser
                const deltaText = inlineToolParser
                    ? inlineToolParser.process(textToProcess)
                    : textToProcess;

                if (!deltaText) {
                    continue;
                }

                // Close thinking block before starting text block
                if (
                    thinkingBlockStarted &&
                    inlineThinkingParser &&
                    !inlineThinkingParser.isInThinking()
                ) {
                    await writeEvent(
                        "content_block_stop",
                        AnthropicStreamEvent.parse({
                            type: "content_block_stop",
                            index: thinkingBlockIndex,
                        }),
                    );
                    thinkingBlockStarted = false;
                }

                if (!contentBlockStarted) {
                    await writeEvent(
                        "content_block_start",
                        AnthropicStreamEvent.parse({
                            type: "content_block_start",
                            index: contentBlockIndex,
                            content_block: AnthropicContentBlock.parse({
                                type: "text",
                                text: "",
                            }),
                        }),
                    );
                    contentBlockStarted = true;
                }

                if (deltaText) {
                    await writeEvent(
                        "content_block_delta",
                        AnthropicStreamEvent.parse({
                            type: "content_block_delta",
                            index: contentBlockIndex,
                            delta: AnthropicDelta.parse({
                                type: "text_delta",
                                text: deltaText,
                            }),
                        }),
                    );
                }
                continue;
            }

            finishChunk = chunk;
            break;
        }

        if (!finishChunk) {
            throw new Error("Generation finished without a final chunk.");
        }

        // Flush thinking parser
        if (inlineThinkingParser) {
            const flushedThinking = inlineThinkingParser.flush();
            if (flushedThinking.thinking && thinkingBlockStarted) {
                await writeEvent(
                    "content_block_delta",
                    AnthropicStreamEvent.parse({
                        type: "content_block_delta",
                        index: thinkingBlockIndex,
                        delta: AnthropicDelta.parse({
                            type: "thinking_delta",
                            thinking: flushedThinking.thinking,
                        }),
                    }),
                );
            }
        }

        // Close thinking block if still open
        if (thinkingBlockStarted) {
            await writeEvent(
                "content_block_stop",
                AnthropicStreamEvent.parse({
                    type: "content_block_stop",
                    index: thinkingBlockIndex,
                }),
            );
            thinkingBlockStarted = false;
        }

        if (inlineToolParser) {
            const flushedText = inlineToolParser.flush();
            if (flushedText) {
                if (!contentBlockStarted) {
                    await writeEvent(
                        "content_block_start",
                        AnthropicStreamEvent.parse({
                            type: "content_block_start",
                            index: contentBlockIndex,
                            content_block: AnthropicContentBlock.parse({
                                type: "text",
                                text: "",
                            }),
                        }),
                    );
                    contentBlockStarted = true;
                }

                await writeEvent(
                    "content_block_delta",
                    AnthropicStreamEvent.parse({
                        type: "content_block_delta",
                        index: contentBlockIndex,
                        delta: AnthropicDelta.parse({
                            type: "text_delta",
                            text: flushedText,
                        }),
                    }),
                );
            }
        }

        if (contentBlockStarted) {
            await writeEvent(
                "content_block_stop",
                AnthropicStreamEvent.parse({
                    type: "content_block_stop",
                    index: contentBlockIndex,
                }),
            );
            contentBlockStarted = false;
            contentBlockIndex += 1;
        }

        let toolCalls: ToolCall[] = [];
        if (inlineToolParser && inlineToolParser.toolCalls.length > 0) {
            toolCalls = inlineToolParser.toolCalls;
        } else {
            toolCalls = await maybeGenerateToolCalls(
                ctx,
                prompt,
                openaiRequest,
                promptTemplate,
                finishChunk,
                media,
            );
        }

        const toolStartToken = toolParserConfig.startToken ?? "<tool_call>";
        if (
            toolCalls.length === 0 &&
            allowToolParse &&
            finishChunk.fullText?.includes(toolStartToken)
        ) {
            const extracted = ToolCallProcessor.extractFromText(
                finishChunk.fullText,
                toolParserConfig,
            );
            toolCalls = extracted.toolCalls;
        }

        if (toolCalls.length > 0) {
            for (const toolCall of toolCalls) {
                const args = toolCall.function.arguments;
                let inputJson = "{}";
                if (typeof args === "string") {
                    inputJson = args;
                } else if (args && typeof args === "object") {
                    inputJson = JSON.stringify(args);
                }

                await writeEvent(
                    "content_block_start",
                    AnthropicStreamEvent.parse({
                        type: "content_block_start",
                        index: contentBlockIndex,
                        content_block: AnthropicContentBlock.parse({
                            type: "tool_use",
                            id: toolCall.id,
                            name: toolCall.function.name,
                            input: {},
                        }),
                    }),
                );

                await writeEvent(
                    "content_block_delta",
                    AnthropicStreamEvent.parse({
                        type: "content_block_delta",
                        index: contentBlockIndex,
                        delta: AnthropicDelta.parse({
                            type: "input_json_delta",
                            partial_json: inputJson,
                        }),
                    }),
                );

                await writeEvent(
                    "content_block_stop",
                    AnthropicStreamEvent.parse({
                        type: "content_block_stop",
                        index: contentBlockIndex,
                    }),
                );

                contentBlockIndex += 1;
            }
        }

        const stopReason = toolCalls.length > 0
            ? "tool_use"
            : STOP_REASON_MAP[convertFinishReason(finishChunk) ?? "stop"];

        await writeEvent(
            "message_delta",
            AnthropicStreamEvent.parse({
                type: "message_delta",
                delta: AnthropicDelta.parse({
                    stop_reason: stopReason,
                }),
                usage: AnthropicUsage.parse({
                    input_tokens: finishChunk.promptTokens,
                    output_tokens: finishChunk.genTokens,
                }),
            }),
        );

        await writeEvent(
            "message_stop",
            AnthropicStreamEvent.parse({
                type: "message_stop",
            }),
        );
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await writeEvent(
            "error",
            AnthropicStreamEvent.parse({
                type: "error",
                error: {
                    type: "internal_error",
                    message,
                },
            }),
        );
    }

    finished = true;
}
