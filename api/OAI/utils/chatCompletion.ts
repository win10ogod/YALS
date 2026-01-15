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
    ChatCompletionMessagePart,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
} from "../types/chatCompletions.ts";
import { CancellationError } from "@/common/errors.ts";
import { logger } from "@/common/logging.ts";
import { ToolSpec } from "../types/tools.ts";
import { TOOL_CALL_SCHEMA, ToolCallProcessor } from "./tools.ts";
import { OAIContext } from "../types/context.ts";
import {
    MultimodalImageData,
    parseOpenAIImageUrl,
    isDataUrl,
    getMediaTypeFromDataUrl,
    extractBase64FromDataUrl,
} from "@/common/imageProcessing.ts";
import { MultimodalContext } from "@/bindings/multimodal.ts";

interface TemplateFormatOptions {
    addBosToken?: boolean;
    banEosToken?: boolean;
    addGenerationPrompt?: boolean;
    templateVars?: Record<string, unknown>;
    tools?: ToolSpec[];
    responsePrefix?: string;
}

function createResponse(chunks: FinishChunk[], modelName: string) {
    const choices: ChatCompletionRespChoice[] = [];

    for (const chunk of chunks) {
        const message = ChatCompletionMessage.parse({
            role: "assistant",
            content: chunk.text,
        });

        if (chunk.toolCalls) {
            message.tool_calls = ToolCallProcessor.fromJson(chunk.toolCalls);
        }

        const choice = ChatCompletionRespChoice.parse({
            index: chunk.taskIdx,
            message: message,
            finish_reason: convertFinishReason(chunk),
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
) {
    const message = ChatCompletionMessage.parse({
        role: "assistant",
        content: chunk.text,
    });

    if (chunk.kind === "finish" && chunk.toolCalls) {
        message.tool_calls = ToolCallProcessor.fromJson(chunk.toolCalls);
    }

    const choice = ChatCompletionStreamChoice.parse({
        index: chunk.taskIdx,
        delta: message,
    });

    if (chunk.kind === "finish") {
        choice.finish_reason = convertFinishReason(chunk);
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

/** Result of processing chat messages for multimodal */
export interface ProcessedChatResult {
    prompt: string;
    images: MultimodalImageData[];
}

/**
 * Extract images from message content parts
 */
function extractImagesFromContent(
    content: ChatCompletionMessagePart[],
): { text: string; images: MultimodalImageData[] } {
    const textParts: string[] = [];
    const images: MultimodalImageData[] = [];
    const mediaMarker = MultimodalContext.getDefaultMarker();

    for (const part of content) {
        if (part.type === "text" && part.text) {
            textParts.push(part.text);
        } else if (part.type === "image_url" && part.image_url) {
            // OpenAI-style image_url
            const imageData = parseOpenAIImageUrl(part.image_url);
            images.push(imageData);
            // Insert placeholder for image
            textParts.push(mediaMarker);
        } else if (part.source?.type === "base64") {
            // Anthropic-style direct base64 source
            images.push({
                type: "base64",
                data: part.source.data,
                mediaType: part.source.media_type,
            });
            textParts.push(mediaMarker);
        }
    }

    return {
        text: textParts.join(""),
        images,
    };
}

export function applyChatTemplate(
    model: Model,
    promptTemplate: PromptTemplate,
    messages: ChatCompletionMessage[],
    options: TemplateFormatOptions = {},
): ProcessedChatResult {
    const {
        addGenerationPrompt = true,
        templateVars = {},
    } = options;

    const allImages: MultimodalImageData[] = [];

    // Process messages to extract images
    const processedMessages = messages.map((message) => {
        if (Array.isArray(message.content)) {
            const messageParts = message.content as ChatCompletionMessagePart[];
            const { text, images } = extractImagesFromContent(messageParts);
            allImages.push(...images);

            return {
                ...message,
                content: text,
            };
        }
        return message;
    });

    const bosToken = model.tokenizer.bosToken;
    let prompt = promptTemplate.template.render({
        ...templateVars,
        messages: processedMessages,
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

    return { prompt, images: allImages };
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

// TODO: Possibly rewrite this to unify with completions
export async function streamChatCompletion(
    ctx: OAIContext,
    params: ChatCompletionRequest,
    stream: SSEStreamingApi,
    promptTemplate: PromptTemplate,
) {
    logger.info(`Received streaming chat completion request ${ctx.requestId}`);

    const toolStart = promptTemplate.metadata.tool_start;
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

    const { prompt, images } = applyChatTemplate(
        ctx.model,
        promptTemplate,
        params.messages,
        {
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: params.template_vars,
            tools: params.tools,
            responsePrefix: params.response_prefix,
        },
    );

    // Log if images are present but multimodal is not available
    if (images.length > 0 && !ctx.model.multimodalContext) {
        logger.warn(
            `Request contains ${images.length} image(s) but multimodal support is not available. ` +
            `Images will be ignored. Load a model with mmproj to enable vision support.`,
        );
    }

    addTemplateMetadata(promptTemplate, params);

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

            if (chunk.kind === "finish") {
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

                completedTasks++;
            }

            const streamChunk = createStreamChunk(
                chunk,
                ctx.model.path.name,
                cmplId,
            );
            await stream.writeSSE({ data: JSON.stringify(streamChunk) });

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

    const { prompt, images } = applyChatTemplate(
        ctx.model,
        promptTemplate,
        params.messages,
        {
            addGenerationPrompt: params.add_generation_prompt,
            templateVars: params.template_vars,
            tools: params.tools,
            responsePrefix: params.response_prefix,
        },
    );

    // Log if images are present but multimodal is not available
    if (images.length > 0 && !ctx.model.multimodalContext) {
        logger.warn(
            `Request contains ${images.length} image(s) but multimodal support is not available. ` +
            `Images will be ignored. Load a model with mmproj to enable vision support.`,
        );
    }

    addTemplateMetadata(promptTemplate, params);

    // Handle generation in the common function
    const generations = await staticGenerate(
        ctx,
        GenerationType.ChatCompletion,
        prompt,
        params,
    );

    // Check for tool calls
    await generateToolCalls(
        ctx,
        prompt,
        generations,
        params,
        promptTemplate,
    );

    const response = createResponse(generations, ctx.model.path.name);

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
            prompt += prompt + gen.fullText;
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
