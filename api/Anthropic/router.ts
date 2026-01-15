/**
 * Anthropic Messages API router
 * Compatible with Claude API format
 */

import { Hono, Context } from "hono";
import { HTTPException } from "hono/http-exception";
import { streamSSE } from "hono/streaming";
import { describeRoute } from "hono-openapi";
import { validator as sValidator } from "hono-openapi";

import { jsonContent } from "@/common/networking.ts";
import { logger } from "@/common/logging.ts";
import { PromptTemplate } from "@/common/templating.ts";
import { Model } from "@/bindings/bindings.ts";

import authMiddleware from "../middleware/authMiddleware.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";
import inlineLoadMiddleware from "../middleware/inlineLoadMiddleware.ts";

import {
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicErrorResponse,
} from "./types/messages.ts";
import { AnthropicContext } from "./types/context.ts";
import {
    convertAnthropicMessages,
    convertSamplerParams,
    createAnthropicResponse,
    streamAnthropicResponse,
} from "./utils/messages.ts";

type Variables = {
    model: Model;
    anthropicCtx: AnthropicContext;
};

const router = new Hono<{ Variables: Variables }>();

/**
 * Middleware to create Anthropic context
 */
const anthropicContextMiddleware = async (
    c: Context<{ Variables: Variables }>,
    next: () => Promise<void>,
) => {
    const model = c.get("model");
    const requestId = c.req.header("x-request-id") || crypto.randomUUID().replaceAll("-", "").slice(0, 24);

    const abortController = new AbortController();
    c.req.raw.signal.addEventListener("abort", () => {
        abortController.abort();
    });

    const anthropicCtx: AnthropicContext = {
        model,
        requestId,
        cancellationSignal: abortController.signal,
    };

    c.set("anthropicCtx", anthropicCtx);
    await next();
};

const messagesRoute = describeRoute({
    description: "Create a message using the Anthropic Messages API format",
    responses: {
        200: jsonContent(AnthropicMessagesResponse, "Message response"),
        400: jsonContent(AnthropicErrorResponse, "Bad request"),
        422: jsonContent(AnthropicErrorResponse, "Validation error"),
    },
});

router.post(
    "/v1/messages",
    messagesRoute,
    authMiddleware("api"),
    sValidator("json", AnthropicMessagesRequest),
    async (c, next) => {
        const params = c.req.valid("json");
        // Use inline load middleware to switch models if needed
        await inlineLoadMiddleware(c.req, next, params.model);
    },
    checkModelMiddleware,
    anthropicContextMiddleware,
    async (c) => {
        const params = c.req.valid("json");
        const ctx = c.get("anthropicCtx");
        const model = c.get("model");

        // Get prompt template
        let promptTemplate: PromptTemplate;
        if (model.promptTemplate) {
            promptTemplate = model.promptTemplate;
        } else {
            // Use a basic template if none is available
            promptTemplate = new PromptTemplate("anthropic_fallback", "{% for message in messages %}{{ message.content }}{% endfor %}");
        }

        // Convert messages to prompt
        const { prompt, images } = convertAnthropicMessages(
            params.messages,
            params.system,
        );

        // Log if images are present but multimodal is not available
        if (images.length > 0 && !model.multimodalContext) {
            logger.warn(
                `Request contains ${images.length} image(s) but multimodal support is not available. ` +
                `Images will be ignored. Load a model with mmproj to enable vision support.`,
            );
        }

        // Convert params to internal format
        const samplerParams = convertSamplerParams(params);

        if (params.stream) {
            logger.info(`Received streaming Anthropic messages request ${ctx.requestId}`);

            return streamSSE(c, async (stream) => {
                try {
                    const generator = model.generateGen(
                        ctx.requestId,
                        prompt,
                        samplerParams,
                        ctx.cancellationSignal,
                    );

                    await streamAnthropicResponse(ctx, params, stream, generator);

                    logger.info(`Finished streaming Anthropic messages request ${ctx.requestId}`);
                } catch (error) {
                    logger.error(`Stream error: ${error}`);
                    await stream.writeSSE({
                        event: "error",
                        data: JSON.stringify({
                            type: "error",
                            error: {
                                type: "server_error",
                                message: error instanceof Error ? error.message : "Unknown error",
                            },
                        }),
                    });
                }
            });
        } else {
            logger.info(`Received Anthropic messages request ${ctx.requestId}`);

            // Generate response
            const result = await model.generate(
                ctx.requestId,
                prompt,
                samplerParams,
                ctx.cancellationSignal,
            );

            const response = createAnthropicResponse(
                result,
                model.path.name,
                ctx.requestId,
            );

            logger.info(`Finished Anthropic messages request ${ctx.requestId}`);
            return c.json(response);
        }
    },
);

// Health check for Anthropic compatibility
router.get("/v1/messages", (c) => {
    return c.json({
        message: "Anthropic Messages API is available. Use POST to create messages.",
    });
});

export default router;
