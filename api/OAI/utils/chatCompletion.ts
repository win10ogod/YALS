import { Queue } from "@core/asyncutil";
import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    mergeFinishMetrics,
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
    TOOL_CALL_SCHEMA,
    ToolCallProcessor,
} from "./tools.ts";
import {
    createStreamingOutputParser,
    parseOutputText,
} from "./outputParsing.ts";
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
    promptTemplate: PromptTemplate,
    allowToolParse: boolean,
    includeReasoning: boolean,
) {
    const choices: ChatCompletionRespChoice[] = [];

    for (const chunk of chunks) {
        const fullText = chunk.fullText ?? chunk.text;
        const parsed = parseOutputText(fullText, {
            promptTemplate,
            allowToolParse,
            includeReasoning,
        });

        const content = parsed.content;
        let toolCalls = parsed.toolCalls;
        if (chunk.toolCalls) {
            toolCalls = ToolCallProcessor.fromJson(chunk.toolCalls);
        }

        const message = ChatCompletionMessage.parse({
            role: "assistant",
            content: content.length > 0 ? content : undefined,
            reasoning: parsed.reasoning,
            reasoning_content: parsed.reasoning,
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
    promptTemplate?: PromptTemplate,
) {
    let toolCalls: ToolCall[] | undefined;

    if (chunk.kind === "finish") {
        if (chunk.toolCalls) {
            toolCalls = ToolCallProcessor.fromJson(chunk.toolCalls);
        } else if (chunk.fullText) {
            const shouldParseInline = allowToolParse ||
                chunk.fullText.includes("<tool_call>");
            if (shouldParseInline) {
                const parsed = parseOutputText(chunk.fullText, {
                    promptTemplate,
                    allowToolParse,
                    includeReasoning: false,
                });
                if (parsed.toolCalls?.length) {
                    toolCalls = parsed.toolCalls;
                }
            }
        }
    }

    const delta: Record<string, unknown> = {
        role: "assistant",
    };
    if (chunk.text) {
        delta.content = chunk.text;
    }
    if (toolCalls?.length) {
        delta.tool_calls = toolCalls.map((toolCall, index) => {
            const args = toolCall.function.arguments;
            return {
                index,
                id: toolCall.id,
                type: toolCall.type,
                function: {
                    name: toolCall.function.name,
                    arguments: typeof args === "string"
                        ? args
                        : JSON.stringify(args ?? {}),
                },
            };
        });
    }

    const choice = ChatCompletionStreamChoice.parse({
        index: chunk.taskIdx,
        delta,
    });

    if (chunk.kind === "finish") {
        choice.finish_reason = toolCalls?.length
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

function createStreamDelta(
    delta: Record<string, unknown>,
    modelName: string,
    cmplId: string,
    taskIdx: number,
    finishReason?: string,
) {
    const choice = ChatCompletionStreamChoice.parse({
        index: taskIdx,
        delta,
    });

    if (finishReason) {
        choice.finish_reason = finishReason;
    }

    return ChatCompletionStreamChunk.parse({
        id: cmplId,
        choices: [choice],
        model: modelName,
    });
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

    if (metadata.tool_start) {
        params.stop.push(metadata.tool_start);
    }
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
    const toolParseEnabled = allowToolParse ||
        !!(promptTemplate.metadata.tool_call_start ||
            promptTemplate.metadata.tool_calls_start);
    const includeReasoning = params.include_reasoning ?? true;
    const outputParsers = new Map<
        number,
        ReturnType<typeof createStreamingOutputParser>
    >();
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

            const outputParser = outputParsers.get(chunk.taskIdx) ??
                createStreamingOutputParser({
                    promptTemplate,
                    allowToolParse: toolParseEnabled,
                    includeReasoning,
                });
            outputParsers.set(chunk.taskIdx, outputParser);

            if (chunk.kind === "finish") {
                const flushed = outputParser.flush();
                if (flushed.content || flushed.reasoning) {
                    const delta: Record<string, unknown> = {
                        role: "assistant",
                    };
                    if (flushed.content) {
                        delta.content = flushed.content;
                    }
                    if (flushed.reasoning && includeReasoning) {
                        delta.reasoning = flushed.reasoning;
                    }
                    const streamChunk = createStreamDelta(
                        delta,
                        ctx.model.path.name,
                        cmplId,
                        chunk.taskIdx,
                    );
                    await stream.writeSSE({
                        data: JSON.stringify(streamChunk),
                    });
                }

                // Handle tools
                if (toolStart && chunk.stopToken) {
                    await generateToolCalls(
                        ctx,
                        prompt,
                        [chunk],
                        params,
                        promptTemplate,
                        media,
                    );
                }

                if (!chunk.toolCalls && outputParser.toolCalls.length > 0) {
                    chunk.toolCalls = JSON.stringify(outputParser.toolCalls);
                }
                if (!chunk.toolCalls && chunk.fullText && toolParseEnabled) {
                    const parsed = parseOutputText(chunk.fullText, {
                        promptTemplate,
                        allowToolParse: toolParseEnabled,
                        includeReasoning: false,
                    });
                    if (parsed.toolCalls?.length) {
                        chunk.toolCalls = JSON.stringify(parsed.toolCalls);
                    }
                }

                outputParsers.delete(chunk.taskIdx);

                completedTasks++;
            }

            if (chunk.kind === "data") {
                const parsed = outputParser.process(chunk.text);
                if (!parsed.content && !parsed.reasoning) {
                    continue;
                }

                const delta: Record<string, unknown> = {
                    role: "assistant",
                };
                if (parsed.content) {
                    delta.content = parsed.content;
                }
                if (parsed.reasoning && includeReasoning) {
                    delta.reasoning = parsed.reasoning;
                }

                const streamChunk = createStreamDelta(
                    delta,
                    ctx.model.path.name,
                    cmplId,
                    chunk.taskIdx,
                );
                await stream.writeSSE({ data: JSON.stringify(streamChunk) });
            } else {
                const streamChunk = createStreamChunk(
                    chunk,
                    ctx.model.path.name,
                    cmplId,
                    toolParseEnabled,
                    promptTemplate,
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
    const toolParseEnabled = allowToolParse ||
        !!(promptTemplate.metadata.tool_call_start ||
            promptTemplate.metadata.tool_calls_start);
    const includeReasoning = params.include_reasoning ?? true;

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
        media,
    );

    const response = createResponse(
        generations,
        ctx.model.path.name,
        promptTemplate,
        toolParseEnabled,
        includeReasoning,
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
    multimodalData?: Uint8Array[],
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

        const toolCtx = {
            ...ctx,
            requestId: `${gen.requestId}-tool`,
        };

        const toolPrompt = gen.fullText ? prompt + gen.fullText : prompt;
        const toolTask = staticGenerate(
            toolCtx,
            GenerationType.ChatCompletion,
            toolPrompt,
            toolParams,
            multimodalData,
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
                Object.assign(
                    gens[genIdx],
                    mergeFinishMetrics(gens[genIdx], toolGen),
                );
            }
        }
    }
}
