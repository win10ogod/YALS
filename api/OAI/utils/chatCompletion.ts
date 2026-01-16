import { Queue } from "@core/asyncutil";
import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    staticGenerate,
    streamCollector,
} from "@/api/OAI/utils/generation.ts";
import { Model } from "@/bindings/bindings.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";
import { toGeneratorError } from "@/common/networking.ts";
import { PromptTemplate } from "@/common/templating.ts";

import {
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
} from "../types/chatCompletions.ts";
import { CancellationError } from "@/common/errors.ts";
import { logger } from "@/common/logging.ts";
import { ToolCall, ToolSpec } from "../types/tools.ts";
import {
    createInlineToolCallParser,
    TOOL_CALL_SCHEMA,
    ToolCallProcessor,
} from "./tools.ts";
import { OAIContext } from "../types/context.ts";
import { normalizeChatMessages } from "./messages.ts";
import { HTTPException } from "hono/http-exception";

interface TemplateFormatOptions {
    addBosToken?: boolean;
    banEosToken?: boolean;
    addGenerationPrompt?: boolean;
    templateVars?: Record<string, unknown>;
    tools?: ToolSpec[];
    responsePrefix?: string;
}

function createResponse(
    chunks: FinishChunk[],
    modelName: string,
    allowToolParse: boolean,
) {
    const choices: ChatCompletionRespChoice[] = [];

    for (const chunk of chunks) {
        let content = chunk.text;
        let toolCalls: ToolCall[] | undefined;

        if (chunk.toolCalls) {
            toolCalls = ToolCallProcessor.fromJson(chunk.toolCalls);
        } else if (chunk.fullText) {
            const shouldParseInline = allowToolParse ||
                chunk.fullText.includes("<tool_call>");
            if (!shouldParseInline) {
                // keep content as-is
            } else {
                const extracted = ToolCallProcessor.extractFromText(
                    chunk.fullText,
                );
                if (extracted.toolCalls.length > 0) {
                    toolCalls = extracted.toolCalls;
                    content = extracted.content;
                }
            }
        }

        const message = ChatCompletionMessage.parse({
            role: "assistant",
            content: content.length > 0 ? content : undefined,
        });

        if (toolCalls?.length) {
            message.tool_calls = toolCalls;
        }

        const finishReason = toolCalls?.length
            ? "tool_calls"
            : convertFinishReason(chunk);
        const choice = ChatCompletionRespChoice.parse({
            index: chunk.taskIdx,
            message: message,
            finish_reason: finishReason,
        });

        choices.push(choice);
    }

    const finalChunk = chunks.at(-1);
    const usage = finalChunk ? createUsageStats(finalChunk) : undefined;

    const response = ChatCompletionResponse.parse({
        choices: choices,
        model: modelName,
        usage,
    });

    return response;
}

function createStreamChunk(
    chunk: GenerationChunk,
    modelName: string,
    cmplId: string,
    allowToolParse: boolean,
) {
    const message = ChatCompletionMessage.parse({
        role: "assistant",
        content: chunk.text,
    });

    if (chunk.kind === "finish") {
        if (chunk.toolCalls) {
            message.tool_calls = ToolCallProcessor.fromJson(chunk.toolCalls);
        } else if (chunk.fullText) {
            const shouldParseInline = allowToolParse ||
                chunk.fullText.includes("<tool_call>");
            if (shouldParseInline) {
                const extracted = ToolCallProcessor.extractFromText(
                    chunk.fullText,
                );
                if (extracted.toolCalls.length > 0) {
                    message.tool_calls = extracted.toolCalls;
                }
            }
        }
    }

    const choice = ChatCompletionStreamChoice.parse({
        index: chunk.taskIdx,
        delta: message,
    });

    if (chunk.kind === "finish") {
        choice.finish_reason = message.tool_calls?.length
            ? "tool_calls"
            : convertFinishReason(chunk);
    }

    const response = ChatCompletionStreamChunk.parse({
        id: cmplId,
        choices: [choice],
        model: modelName,
    });

    return response;
}

function createUsageChunk(
    chunk: FinishChunk,
    modelName: string,
    cmplId: string,
) {
    const response = ChatCompletionStreamChunk.parse({
        id: cmplId,
        model: modelName,
        usage: createUsageStats(chunk),
    });

    return response;
}

export function applyChatTemplate(
    model: Model,
    promptTemplate: PromptTemplate,
    messages: ChatCompletionMessage[],
    options: TemplateFormatOptions = {},
): string {
    const {
        addGenerationPrompt = true,
        templateVars = {},
    } = options;

    const bosToken = model.tokenizer.bosToken;
    let prompt = promptTemplate.template.render({
        ...templateVars,
        messages: messages,
        bos_token: bosToken?.piece ?? "",
        eos_token: model.tokenizer.eosToken?.piece ?? "",
        add_generation_prompt: addGenerationPrompt,
        tools: options.tools ?? null,
    });

    if (options.responsePrefix) {
        if (addGenerationPrompt) {
            prompt += options.responsePrefix;
        } else {
            logger.warn(
                "Could not add response prefix because " +
                    "add_generation_prompt is False",
            );
        }
    }

    // Remove extra BOS token at start of prompt if present
    // Some model templates don't respect their own add_bos_token setting
    // Better to do this since a template can add BOS anywhere
    if (
        bosToken && model.tokenizer.addBosToken &&
        prompt.startsWith(bosToken.piece)
    ) {
        prompt = prompt.slice(bosToken.piece.length);
    }

    return prompt;
}

function addTemplateMetadata(
    promptTemplate: PromptTemplate,
    params: ChatCompletionRequest,
) {
    const metadata = promptTemplate.metadata;

    if (metadata.stop_strings) {
        params.stop.push(...metadata.stop_strings);
    }

    // Add tool-related stop strings
    if (metadata.tool_start) {
        params.stop.push(metadata.tool_start);
    }

    // Add thinking-related stop strings if tools are enabled
    // This allows us to detect when thinking ends and tool calls begin
    if (params.tools?.length && metadata.thinking_end) {
        // Don't add thinking_end as stop if we want to continue generating
        // after thinking. Instead, the parser will handle it.
    }
}

/**
 * Get parser configuration from template metadata
 */
function getParserConfig(metadata: PromptTemplate["metadata"]) {
    return {
        reasoning: metadata.supports_thinking
            ? {
                  format: metadata.thinking_format ?? "generic",
                  startToken: metadata.thinking_start,
                  endToken: metadata.thinking_end,
              }
            : undefined,
        toolCall: metadata.supports_tools
            ? {
                  format: metadata.tool_format ?? "xml",
                  startToken: metadata.tool_start,
                  endToken: metadata.tool_end,
              }
            : undefined,
    };
}

async function buildChatPrompt(
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

// TODO: Possibly rewrite this to unify with completions
export async function streamChatCompletion(
    ctx: OAIContext,
    params: ChatCompletionRequest,
    stream: SSEStreamingApi,
    promptTemplate: PromptTemplate,
) {
    logger.info(`Received streaming chat completion request ${ctx.requestId}`);

    const toolStart = promptTemplate.metadata.tool_start;
    const allowToolParse = !!params.tools?.length;
    const inlineToolParsers = allowToolParse
        ? new Map<number, ReturnType<typeof createInlineToolCallParser>>()
        : null;
    const cmplId = `chatcmpl-${crypto.randomUUID().replaceAll("-", "")}`;
    const genAbortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    ctx.cancellationSignal.addEventListener("abort", () => {
        if (!finished) {
            genAbortController.abort(
                new CancellationError(
                    `Streaming chat completion ${ctx.requestId} cancelled by user.`,
                ),
            );
            finished = true;
        }
    });

    const { prompt, media } = await buildChatPrompt(
        ctx,
        params,
        promptTemplate,
    );

    try {
        const queue = new Queue<GenerationChunk | Error>();
        const genTasks = [];

        for (let i = 0; i < params.n; i++) {
            const task = streamCollector(
                ctx,
                prompt,
                params,
                genAbortController.signal,
                i,
                queue,
                media,
            );

            genTasks.push(task);
        }

        let completedTasks = 0;
        while (true) {
            // Abort if the signal is set
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

            const inlineToolParser = inlineToolParsers
                ? inlineToolParsers.get(chunk.taskIdx) ??
                    createInlineToolCallParser()
                : null;
            if (inlineToolParser && inlineToolParsers) {
                inlineToolParsers.set(chunk.taskIdx, inlineToolParser);
            }

            if (chunk.kind === "finish") {
                if (inlineToolParser) {
                    const flushedText = inlineToolParser.flush();
                    if (flushedText) {
                        const flushedChunk: GenerationChunk = {
                            kind: "data",
                            text: flushedText,
                            taskIdx: chunk.taskIdx,
                            requestId: chunk.requestId,
                        };
                        const streamChunk = createStreamChunk(
                            flushedChunk,
                            ctx.model.path.name,
                            cmplId,
                            allowToolParse,
                        );
                        await stream.writeSSE({
                            data: JSON.stringify(streamChunk),
                        });
                    }
                }

                // Handle tools
                if (toolStart && chunk.stopToken) {
                    await generateToolCalls(
                        ctx,
                        prompt,
                        [chunk],
                        params,
                        promptTemplate,
                    );
                }

                if (inlineToolParser && !chunk.toolCalls) {
                    if (inlineToolParser.toolCalls.length > 0) {
                        chunk.toolCalls = JSON.stringify(
                            inlineToolParser.toolCalls,
                        );
                    }
                }

                if (inlineToolParsers) {
                    inlineToolParsers.delete(chunk.taskIdx);
                }

                completedTasks++;
            }

            if (chunk.kind === "data" && inlineToolParser) {
                const filteredText = inlineToolParser.process(chunk.text);
                if (!filteredText) {
                    continue;
                }
                const filteredChunk: GenerationChunk = {
                    ...chunk,
                    text: filteredText,
                };
                const streamChunk = createStreamChunk(
                    filteredChunk,
                    ctx.model.path.name,
                    cmplId,
                    allowToolParse,
                );
                await stream.writeSSE({ data: JSON.stringify(streamChunk) });
            } else {
                const streamChunk = createStreamChunk(
                    chunk,
                    ctx.model.path.name,
                    cmplId,
                    allowToolParse,
                );
                await stream.writeSSE({ data: JSON.stringify(streamChunk) });
            }

            // TODO: Make usage aggregated
            if (completedTasks === params.n && queue.size === 0) {
                if (
                    params.stream_options?.include_usage &&
                    chunk.kind === "finish"
                ) {
                    const usageChunk = createUsageChunk(
                        chunk,
                        ctx.model.path.name,
                        cmplId,
                    );

                    await stream.writeSSE({ data: JSON.stringify(usageChunk) });
                }

                logger.info(
                    `Finished streaming chat completion request ${ctx.requestId}`,
                );
                await stream.writeSSE({ data: "[DONE]" });

                break;
            }
        }
    } catch (error) {
        await stream.writeSSE({
            data: JSON.stringify(toGeneratorError(error)),
        });
    }

    finished = true;
}

export async function generateChatCompletion(
    ctx: OAIContext,
    params: ChatCompletionRequest,
    promptTemplate: PromptTemplate,
) {
    logger.info(`Received chat completion request ${ctx.requestId}`);

    const { prompt, media } = await buildChatPrompt(
        ctx,
        params,
        promptTemplate,
    );
    const allowToolParse = !!params.tools?.length;

    // Handle generation in the common function
    const generations = await staticGenerate(
        ctx,
        GenerationType.ChatCompletion,
        prompt,
        params,
        media,
    );

    // Check for tool calls
    await generateToolCalls(
        ctx,
        prompt,
        generations,
        params,
        promptTemplate,
    );

    const response = createResponse(
        generations,
        ctx.model.path.name,
        allowToolParse,
    );

    logger.info(`Finished chat completion request ${ctx.requestId}`);
    return response;
}

async function generateToolCalls(
    ctx: OAIContext,
    prompt: string,
    gens: FinishChunk[],
    params: ChatCompletionRequest,
    promptTemplate: PromptTemplate,
) {
    const toolGenTasks = [];
    const toolStart = promptTemplate.metadata.tool_start;
    if (!toolStart) {
        return;
    }

    const toolIdx = [];

    const toolParams = structuredClone(params);
    toolParams.json_schema = TOOL_CALL_SCHEMA;

    for (const [index, gen] of gens.entries()) {
        if (!gen.stopToken.startsWith(toolStart)) {
            continue;
        }

        logger.info(`Tool call detected for request ${gen.requestId}`);

        if (gen.fullText) {
            prompt += gen.fullText;
        }

        const toolCtx = {
            ...ctx,
            requestId: `${gen.requestId}-tool`,
        };

        const toolTask = staticGenerate(
            toolCtx,
            GenerationType.ChatCompletion,
            prompt,
            toolParams,
        );

        toolGenTasks.push(toolTask);
        toolIdx.push(index);
    }

    if (toolIdx.length > 0) {
        const toolGenResults = await Promise.allSettled(toolGenTasks);

        for (const [i, genIdx] of toolIdx.entries()) {
            const toolResult = toolGenResults[i];
            if (toolResult.status === "fulfilled" && toolResult.value[0]) {
                const toolGen = toolResult.value[0];
                gens[genIdx].toolCalls = toolGen.text;
            }
        }
    }
}
