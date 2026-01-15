/**
 * Anthropic API context type
 */

import { Model } from "@/bindings/bindings.ts";

export interface AnthropicContext {
    model: Model;
    requestId: string;
    cancellationSignal: AbortSignal;
}
