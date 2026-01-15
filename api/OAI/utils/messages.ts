import { DEFAULT_MEDIA_MARKER, resolveImageUrl } from "@/common/multimodal.ts";
import { ChatCompletionMessage, ChatCompletionMessagePart } from "../types/chatCompletions.ts";

export interface NormalizedChatMessages {
    messages: ChatCompletionMessage[];
    media: Uint8Array[];
}

export async function normalizeChatMessages(
    messages: ChatCompletionMessage[],
    options: { mediaMarker?: string; decodeImages?: boolean } = {},
): Promise<NormalizedChatMessages> {
    const mediaMarker = options.mediaMarker ?? DEFAULT_MEDIA_MARKER;
    const decodeImages = options.decodeImages ?? false;
    const media: Uint8Array[] = [];

    const normalized = await Promise.all(
        messages.map(async (message) => {
            if (!Array.isArray(message.content)) {
                return message;
            }

            const parts = message.content as ChatCompletionMessagePart[];
            let contentText = "";

            for (const part of parts) {
                const partType = part.type ?? "text";
                if (partType === "text") {
                    if (part.text) {
                        contentText += part.text;
                    }
                    continue;
                }

                if (partType === "image_url") {
                    if (!part.image_url?.url) {
                        throw new Error("Image URL is required for image parts.");
                    }
                    contentText += mediaMarker;
                    if (decodeImages) {
                        media.push(await resolveImageUrl(part.image_url.url));
                    }
                }
            }

            return {
                ...message,
                content: contentText,
            };
        }),
    );

    return { messages: normalized, media };
}
