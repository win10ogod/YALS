import { Queue } from "@core/asyncutil";
import { SSEStreamingApi } from "hono/streaming";

import {
    convertFinishReason,
    createUsageStats,
    GenerationType,
    staticGenerate,
    streamCollector,
} from "@/api/OAI/utils/generation.ts";
import { FinishChunk, GenerationChunk } from "@/bindings/types.ts";
import { CancellationError } from "@/common/errors.ts";
import { toGeneratorError } from "@/common/networking.ts";
import { logger } from "@/common/logging.ts";
import {
    CompletionRequest,
    CompletionRespChoice,
    CompletionResponse,
} from "../types/completions.ts";
import { OAIContext } from "../types/context.ts";
import { createStreamingOutputParser, parseOutputText } from "./outputParsing.ts";

function createResponse(
    chunks: FinishChunk[],
    modelName: string,
    includeReasoning: boolean,
) {
    const choices: CompletionRespChoice[] = [];
    for (const chunk of chunks) {
        const fullText = chunk.fullText ?? chunk.text;
        const parsed = parseOutputText(fullText, {
            allowToolParse: false,
            includeReasoning,
        });
        const finishReason = convertFinishReason(chunk);
        const reasoning = parsed.reasoning || undefined;

        const choice = CompletionRespChoice.parse({
            index: chunk.taskIdx,
            text: parsed.content ?? "",
            finish_reason: finishReason,
            reasoning,
            reasoning_content: reasoning,
        });

        choices.push(choice);
    }

    const finalChunk = chunks.at(-1);
    const usage = finalChunk
        ? createUsageStats(finalChunk)
        : undefined;

    const response = CompletionResponse.parse({
        choices,
        model: modelName,
        usage,
    });

    return response;
}

export async function streamCompletion(
    ctx: OAIContext,
    params: CompletionRequest,
    stream: SSEStreamingApi,
) {
    logger.info(`Received streaming completion request ${ctx.requestId}`);

    const includeReasoning = params.include_reasoning ?? true;
    const outputParsers = new Map<
        number,
        ReturnType<typeof createStreamingOutputParser>
    >();
    const genAbortController = new AbortController();
    let finished = false;

    // If an abort happens before streaming starts
    ctx.cancellationSignal.addEventListener("abort", () => {
        if (!finished) {
            genAbortController.abort(
                new CancellationError(
                    `Streaming completion ${ctx.requestId} cancelled by user.`,
                ),
            );
            finished = true;
        }
    });

    try {
        const queue = new Queue<GenerationChunk | Error>();
        const genTasks = [];

        for (let i = 0; i < params.n; i++) {
            const task = streamCollector(
                ctx,
                params.prompt,
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

            const outputParser = outputParsers.get(chunk.taskIdx) ??
                createStreamingOutputParser({
                    allowToolParse: false,
                    includeReasoning,
                });
            outputParsers.set(chunk.taskIdx, outputParser);

            if (chunk.kind === "data") {
                const parsed = outputParser.process(chunk.text);
                const content = parsed.content ?? "";
                const reasoning = parsed.reasoning || undefined;
                if (!content && !reasoning) {
                    continue;
                }

                const choice = CompletionRespChoice.parse({
                    index: chunk.taskIdx,
                    text: content,
                    reasoning,
                    reasoning_content: reasoning,
                });
                const streamChunk = CompletionResponse.parse({
                    choices: [choice],
                    model: ctx.model.path.name,
                });
                await stream.writeSSE({ data: JSON.stringify(streamChunk) });
                continue;
            }

            const parsed = outputParser.process(chunk.text);
            const flushed = outputParser.flush();
            outputParsers.delete(chunk.taskIdx);

            const content = `${parsed.content ?? ""}${flushed.content ?? ""}`;
            const reasoning = `${parsed.reasoning ?? ""}${flushed.reasoning ?? ""}`
                .trim() || undefined;
            const choice = CompletionRespChoice.parse({
                index: chunk.taskIdx,
                text: content,
                finish_reason: convertFinishReason(chunk),
                reasoning,
                reasoning_content: reasoning,
            });
            const streamChunk = CompletionResponse.parse({
                choices: [choice],
                model: ctx.model.path.name,
                usage: createUsageStats(chunk),
            });
            await stream.writeSSE({ data: JSON.stringify(streamChunk) });

            if (chunk.kind === "finish") {
                completedTasks++;
            }

            if (completedTasks === params.n && queue.size === 0) {
                logger.info(
                    `Finished streaming completion request ${ctx.requestId}`,
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

export async function generateCompletion(
    ctx: OAIContext,
    params: CompletionRequest,
) {
    logger.info(`Received completion request ${ctx.requestId}`);

    const includeReasoning = params.include_reasoning ?? true;
    // Handle generation in the common function
    const generations = await staticGenerate(
        ctx,
        GenerationType.Completion,
        params.prompt,
        params,
    );

    const response = createResponse(
        generations,
        ctx.model.path.name,
        includeReasoning,
    );

    logger.info(`Finished completion request ${ctx.requestId}`);
    return response;
}
