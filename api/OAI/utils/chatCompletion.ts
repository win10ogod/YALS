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
    ToolParserConfig,
} from "./tools.ts";
import { OAIContext } from "../types/context.ts";
import { normalizeChatMessages } from "./messages.ts";
import { HTTPException } from "hono/http-exception";
import {
    createStreamingThinkingParser,
    extractThinking,
    ThinkingConfig,
} from "./thinking.ts";

interface TemplateFormatOptions {
    addBosToken?: boolean;
    banEosToken?: boolean;
    addGenerationPrompt?: boolean;
    templateVars?: Record<string, unknown>;
    tools?: ToolSpec[];
    responsePrefix?: string;
}

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

/**
 * Parse content to extract thinking and tool calls
 */
function parseGeneratedContent(
    text: string,
    fullText: string | undefined,
    toolParserConfig: ToolParserConfig,
    thinkingConfig: ThinkingConfig | undefined,
    allowToolParse: boolean,
): {
    content: string;
    thinking: string | undefined;
    toolCalls: ToolCall[] | undefined;
} {
    let content = text;
    let thinking: string | undefined;
    let toolCalls: ToolCall[] | undefined;

    const textToParse = fullText ?? text;

    // Extract thinking content first
    if (thinkingConfig) {
        const thinkingResult = extractThinking(textToParse, thinkingConfig);
        if (thinkingResult.thinking) {
            thinking = thinkingResult.thinking;
            content = thinkingResult.content;
        }
    }

    // Then extract tool calls
    const shouldParseTools = allowToolParse ||
        textToParse.includes(toolParserConfig.startToken ?? "<tool_call>");

    if (shouldParseTools) {
        const extracted = ToolCallProcessor.extractFromText(
            content,
            toolParserConfig,
        );
        if (extracted.toolCalls.length > 0) {
            toolCalls = extracted.toolCalls;
            content = extracted.content;
        }
    }

    return { content, thinking, toolCalls };
}

function createResponse(
    chunks: FinishChunk[],
    modelName: string,
    allowToolParse: boolean,
    toolParserConfig: ToolParserConfig,
    thinkingConfig: ThinkingConfig | undefined,
) {
    const choices: ChatCompletionRespChoice[] = [];

    for (const chunk of chunks) {
        let content = chunk.text;
        let toolCalls: ToolCall[] | undefined;
        let thinking: string | undefined;

        if (chunk.toolCalls) {
            toolCalls = ToolCallProcessor.fromJson(
                chunk.toolCalls,
                toolParserConfig,
            );
            // Still need to extract thinking from content
            if (thinkingConfig && chunk.fullText) {
                const thinkingResult = extractThinking(
                    chunk.fullText,
                    thinkingConfig,
                );
                if (thinkingResult.thinking) {
                    thinking = thinkingResult.thinking;
                    content = thinkingResult.content;
                }
            }
        } else if (chunk.fullText) {
            const parsed = parseGeneratedContent(
                chunk.text,
                chunk.fullText,
                toolParserConfig,
                thinkingConfig,
                allowToolParse,
            );
            content = parsed.content;
            thinking = parsed.thinking;
            toolCalls = parsed.toolCalls;
        }

        const message = ChatCompletionMessage.parse({
            role: "assistant",
            content: content.length > 0 ? content : undefined,
        });

        if (toolCalls?.length) {
            message.tool_calls = toolCalls;
        }

        // Add thinking content if present
        if (thinking) {
            message.reasoning_content = thinking;
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
    toolParserConfig: ToolParserConfig,
    thinkingConfig: ThinkingConfig | undefined,
) {
    const message = ChatCompletionMessage.parse({
        role: "assistant",
        content: chunk.text,
    });

    if (chunk.kind === "finish") {
        if (chunk.toolCalls) {
            message.tool_calls = ToolCallProcessor.fromJson(
                chunk.toolCalls,
                toolParserConfig,
            );
        } else if (chunk.fullText) {
            const parsed = parseGeneratedContent(
                chunk.text,
                chunk.fullText,
                toolParserConfig,
                thinkingConfig,
                allowToolParse,
            );
            if (parsed.toolCalls?.length) {
                message.tool_calls = parsed.toolCalls;
            }
            if (parsed.thinking) {
                message.reasoning_content = parsed.thinking;
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
    const toolParserConfig = getToolParserConfig(promptTemplate);
    const thinkingConfig = getThinkingConfig(promptTemplate);
    const allowToolParse = !!params.tools?.length;

    // Create parser factories with template-specific config
    const createToolParser = () => createInlineToolCallParser(toolParserConfig);
    const createThinkingParser = () =>
        thinkingConfig ? createStreamingThinkingParser(thinkingConfig) : null;

    const inlineToolParsers = allowToolParse
        ? new Map<number, ReturnType<typeof createInlineToolCallParser>>()
        : null;
    const inlineThinkingParsers = thinkingConfig
        ? new Map<number, ReturnType<typeof createStreamingThinkingParser>>()
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

            // Get or create parsers for this task
            const inlineToolParser = inlineToolParsers
                ? inlineToolParsers.get(chunk.taskIdx) ?? createToolParser()
                : null;
            if (inlineToolParser && inlineToolParsers) {
                inlineToolParsers.set(chunk.taskIdx, inlineToolParser);
            }

            const inlineThinkingParser = inlineThinkingParsers
                ? inlineThinkingParsers.get(chunk.taskIdx) ??
                    createThinkingParser()
                : null;
            if (inlineThinkingParser && inlineThinkingParsers) {
                inlineThinkingParsers.set(chunk.taskIdx, inlineThinkingParser);
            }

            if (chunk.kind === "finish") {
                // Flush thinking parser first
                if (inlineThinkingParser) {
                    const flushedThinking = inlineThinkingParser.flush();
                    // Thinking content is captured but not streamed to user
                    // It will be included in the final message if needed
                }

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
                            toolParserConfig,
                            thinkingConfig,
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
                if (inlineThinkingParsers) {
                    inlineThinkingParsers.delete(chunk.taskIdx);
                }

                completedTasks++;
            }

            if (chunk.kind === "data") {
                let textToStream = chunk.text;

                // Process through thinking parser first (filters out thinking tags)
                if (inlineThinkingParser) {
                    const thinkingResult = inlineThinkingParser.process(
                        textToStream,
                    );
                    textToStream = thinkingResult.content;
                    // thinkingResult.thinking is captured but not streamed
                }

                // Then process through tool parser (filters out tool tags)
                if (inlineToolParser) {
                    textToStream = inlineToolParser.process(textToStream);
                }

                if (!textToStream) {
                    continue;
                }

                const filteredChunk: GenerationChunk = {
                    ...chunk,
                    text: textToStream,
                };
                const streamChunk = createStreamChunk(
                    filteredChunk,
                    ctx.model.path.name,
                    cmplId,
                    allowToolParse,
                    toolParserConfig,
                    thinkingConfig,
                );
                await stream.writeSSE({ data: JSON.stringify(streamChunk) });
            } else {
                const streamChunk = createStreamChunk(
                    chunk,
                    ctx.model.path.name,
                    cmplId,
                    allowToolParse,
                    toolParserConfig,
                    thinkingConfig,
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
    const toolParserConfig = getToolParserConfig(promptTemplate);
    const thinkingConfig = getThinkingConfig(promptTemplate);

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
        toolParserConfig,
        thinkingConfig,
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
