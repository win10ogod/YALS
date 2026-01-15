import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { streamSSE } from "hono/streaming";
import { describeRoute, validator as sValidator } from "hono-openapi";

import { AnthropicMessagesRequest, AnthropicMessagesResponse } from "./types/messages.ts";
import {
    generateAnthropicMessages,
    streamAnthropicMessages,
} from "./utils/messages.ts";
import { jsonContent } from "@/common/networking.ts";
import { PromptTemplate } from "@/common/templating.ts";

import authMiddleware from "../middleware/authMiddleware.ts";
import checkModelMiddleware from "../middleware/checkModelMiddleware.ts";
import inlineLoadMiddleware from "../middleware/inlineLoadMiddleware.ts";
import oaiContextMiddleware from "../middleware/oaiContextMiddleware.ts";

const router = new Hono();

const messagesRoute = describeRoute({
    responses: {
        200: jsonContent(AnthropicMessagesResponse, "Response to messages"),
    },
});

router.post(
    "/v1/messages",
    messagesRoute,
    authMiddleware("api"),
    sValidator("json", AnthropicMessagesRequest),
    async (c, next) => {
        const params = c.req.valid("json");
        await inlineLoadMiddleware(c.req, next, params.model);
    },
    checkModelMiddleware,
    oaiContextMiddleware,
    async (c) => {
        const params = c.req.valid("json");

        let promptTemplate: PromptTemplate;
        if (c.var.model.promptTemplate) {
            promptTemplate = c.var.model.promptTemplate;
        } else {
            throw new HTTPException(422, {
                message:
                    "Messages are disabled because a prompt template isn't set.",
            });
        }

        if (params.stream) {
            return streamSSE(c, async (stream) => {
                await streamAnthropicMessages(
                    c.var.oaiCtx,
                    params,
                    promptTemplate,
                    stream,
                );
            });
        }

        const response = await generateAnthropicMessages(
            c.var.oaiCtx,
            params,
            promptTemplate,
        );
        return c.json(response);
    },
);

export default router;
