/**
 * Image processing utilities for multimodal API support
 */

import { logger } from "./logging.ts";

/** Supported image media types */
export const SUPPORTED_IMAGE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
] as const;

export type SupportedImageType = typeof SUPPORTED_IMAGE_TYPES[number];

/** Image data representation for API requests */
export interface ImageInput {
    type: "base64" | "url";
    data: string;
    mediaType?: string;
}

/** Processed image ready for multimodal context */
export interface ProcessedImage {
    bytes: Uint8Array;
    mediaType: string;
    width?: number;
    height?: number;
}

/**
 * Validate if the media type is supported
 */
export function validateImageFormat(mediaType: string): boolean {
    return SUPPORTED_IMAGE_TYPES.includes(mediaType.toLowerCase() as SupportedImageType);
}

/**
 * Extract media type from a data URL
 * @param dataUrl Data URL (e.g., "data:image/png;base64,...")
 * @returns Media type or null if not found
 */
export function getMediaTypeFromDataUrl(dataUrl: string): string | null {
    const match = dataUrl.match(/^data:([^;]+);base64,/);
    return match ? match[1] : null;
}

/**
 * Extract base64 data from a data URL
 * @param dataUrl Data URL
 * @returns Base64 string or the original string if not a data URL
 */
export function extractBase64FromDataUrl(dataUrl: string): string {
    const match = dataUrl.match(/^data:[^;]+;base64,(.+)$/);
    return match ? match[1] : dataUrl;
}

/**
 * Check if a string is a data URL
 */
export function isDataUrl(str: string): boolean {
    return str.startsWith("data:");
}

/**
 * Decode base64 image data to bytes
 * @param base64Data Base64 encoded string (with or without data URL prefix)
 * @returns Decoded bytes
 */
export function decodeBase64Image(base64Data: string): Uint8Array {
    // Remove data URL prefix if present
    const cleanData = extractBase64FromDataUrl(base64Data);

    try {
        const binaryString = atob(cleanData);
        const bytes = new Uint8Array(binaryString.length);

        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        return bytes;
    } catch (e) {
        throw new Error(`Failed to decode base64 image: ${e}`);
    }
}

/**
 * Fetch image from URL
 * @param url Image URL
 * @param timeout Timeout in milliseconds (default: 30000)
 * @returns Image bytes and media type
 */
export async function fetchImageFromUrl(
    url: string,
    timeout: number = 30000,
): Promise<{ bytes: Uint8Array; mediaType: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            signal: controller.signal,
            headers: {
                "User-Agent": "YALS/1.0 (Multimodal Image Fetch)",
            },
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }

        const contentType = response.headers.get("content-type") || "";
        const mediaType = contentType.split(";")[0].trim();

        if (!mediaType.startsWith("image/")) {
            throw new Error(`URL does not point to an image: ${mediaType}`);
        }

        if (!validateImageFormat(mediaType)) {
            throw new Error(`Unsupported image format: ${mediaType}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        return {
            bytes: new Uint8Array(arrayBuffer),
            mediaType,
        };
    } finally {
        clearTimeout(timeoutId);
    }
}

/**
 * Process an image input (base64 or URL) to bytes
 * @param image Image input data
 * @returns Processed image with bytes and media type
 */
export async function processImageInput(image: ImageInput): Promise<ProcessedImage> {
    if (image.type === "base64") {
        const bytes = decodeBase64Image(image.data);
        let mediaType = image.mediaType;

        // Try to detect media type from data URL if not provided
        if (!mediaType && isDataUrl(image.data)) {
            mediaType = getMediaTypeFromDataUrl(image.data) || undefined;
        }

        // Default to JPEG if we can't determine type
        if (!mediaType) {
            mediaType = detectImageType(bytes) || "image/jpeg";
        }

        return { bytes, mediaType };
    } else if (image.type === "url") {
        // Check if it's a data URL
        if (isDataUrl(image.data)) {
            const mediaType = getMediaTypeFromDataUrl(image.data);
            const bytes = decodeBase64Image(image.data);
            return {
                bytes,
                mediaType: mediaType || "image/jpeg",
            };
        }

        // Fetch from HTTP(S) URL
        const result = await fetchImageFromUrl(image.data);
        return {
            bytes: result.bytes,
            mediaType: result.mediaType,
        };
    }

    throw new Error(`Unknown image type: ${image.type}`);
}

/**
 * Detect image type from magic bytes
 * @param bytes Image bytes
 * @returns Media type or null if unknown
 */
export function detectImageType(bytes: Uint8Array): string | null {
    if (bytes.length < 4) return null;

    // JPEG: FF D8 FF
    if (bytes[0] === 0xFF && bytes[1] === 0xD8 && bytes[2] === 0xFF) {
        return "image/jpeg";
    }

    // PNG: 89 50 4E 47
    if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4E && bytes[3] === 0x47) {
        return "image/png";
    }

    // GIF: 47 49 46 38
    if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x38) {
        return "image/gif";
    }

    // WebP: 52 49 46 46 ... 57 45 42 50
    if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
        if (bytes.length >= 12 && bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) {
            return "image/webp";
        }
    }

    // BMP: 42 4D
    if (bytes[0] === 0x42 && bytes[1] === 0x4D) {
        return "image/bmp";
    }

    return null;
}

/**
 * Convert image input to multimodal-compatible format
 * Handles both OpenAI and Anthropic image formats
 */
export interface MultimodalImageData {
    type: "base64" | "url" | "bytes";
    data: string | Uint8Array;
    mediaType?: string;
}

/**
 * Parse OpenAI image_url format
 * Supports: data URLs, HTTP URLs, and direct base64
 */
export function parseOpenAIImageUrl(imageUrl: { url: string; detail?: string }): MultimodalImageData {
    const { url } = imageUrl;

    if (isDataUrl(url)) {
        const mediaType = getMediaTypeFromDataUrl(url);
        const base64 = extractBase64FromDataUrl(url);
        return {
            type: "base64",
            data: base64,
            mediaType: mediaType || undefined,
        };
    }

    // HTTP(S) URL
    return {
        type: "url",
        data: url,
    };
}

/**
 * Parse Anthropic image source format
 */
export function parseAnthropicImageSource(
    source: { type: string; media_type?: string; data?: string; url?: string },
): MultimodalImageData {
    if (source.type === "base64") {
        return {
            type: "base64",
            data: source.data || "",
            mediaType: source.media_type,
        };
    } else if (source.type === "url") {
        return {
            type: "url",
            data: source.url || "",
        };
    }

    throw new Error(`Unknown Anthropic image source type: ${source.type}`);
}

/**
 * Get size estimate of image in bytes
 */
export function estimateImageSize(image: MultimodalImageData): number {
    if (image.type === "bytes") {
        return (image.data as Uint8Array).length;
    } else if (image.type === "base64") {
        // Base64 is ~4/3 times the size of the original data
        return Math.ceil((image.data as string).length * 0.75);
    }
    // URL - unknown size until fetched
    return 0;
}

/**
 * Maximum image size in bytes (default: 20MB)
 */
export const MAX_IMAGE_SIZE = 20 * 1024 * 1024;

/**
 * Validate image size
 */
export function validateImageSize(bytes: Uint8Array, maxSize: number = MAX_IMAGE_SIZE): void {
    if (bytes.length > maxSize) {
        throw new Error(
            `Image size (${(bytes.length / 1024 / 1024).toFixed(2)}MB) exceeds maximum allowed size (${(maxSize / 1024 / 1024).toFixed(2)}MB)`,
        );
    }
}
