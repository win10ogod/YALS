/**
 * Multimodal (Vision/Audio) support for YALS
 * Wraps the MTMD library from llama.cpp
 */

import { hasMultimodal, lib } from "./lib.ts";
import { logger } from "@/common/logging.ts";

/** Image data representation */
export interface ImageData {
    /** Source type: base64 encoded or URL */
    type: "base64" | "url" | "bytes";
    /** The image data (base64 string, URL, or raw bytes) */
    data: string | Uint8Array;
    /** Media type (e.g., "image/jpeg", "image/png") */
    mediaType?: string;
}

/** Chunk types from mtmd */
export enum ChunkType {
    Text = 0,
    Image = 1,
    Audio = 2,
}

/** Input chunk wrapper */
export interface InputChunk {
    type: ChunkType;
    nTokens: number;
    pointer: Deno.PointerValue;
}

/** Multimodal context for processing images with text */
export class MultimodalContext {
    private ctx: Deno.PointerValue;
    private model: Deno.PointerValue;
    private _supportsVision: boolean;
    private _supportsAudio: boolean;

    private constructor(
        ctx: Deno.PointerValue,
        model: Deno.PointerValue,
        supportsVision: boolean,
        supportsAudio: boolean,
    ) {
        this.ctx = ctx;
        this.model = model;
        this._supportsVision = supportsVision;
        this._supportsAudio = supportsAudio;
    }

    /** Check if multimodal support is available */
    static isAvailable(): boolean {
        return hasMultimodal && lib.symbols.mtmd_ctx_init !== undefined;
    }

    /**
     * Initialize multimodal context from mmproj file
     * @param mmprojPath Path to the mmproj file
     * @param model Pointer to the llama model
     * @param useGpu Whether to use GPU for vision encoding
     * @param nThreads Number of threads to use
     * @param warmup Whether to run warmup pass
     */
    static async init(
        mmprojPath: string,
        model: Deno.PointerValue,
        useGpu: boolean = true,
        nThreads: number = 4,
        warmup: boolean = true,
    ): Promise<MultimodalContext | null> {
        if (!this.isAvailable()) {
            logger.warn("Multimodal support is not available in this build");
            return null;
        }

        const pathBuffer = new TextEncoder().encode(mmprojPath + "\0");

        const ctx = await lib.symbols.mtmd_ctx_init!(
            pathBuffer,
            model,
            useGpu,
            nThreads,
            warmup,
        );

        if (!ctx) {
            logger.error(`Failed to initialize multimodal context from ${mmprojPath}`);
            return null;
        }

        const supportsVision = lib.symbols.mtmd_ctx_supports_vision!(ctx);
        const supportsAudio = lib.symbols.mtmd_ctx_supports_audio!(ctx);

        logger.info(`Multimodal context initialized (vision: ${supportsVision}, audio: ${supportsAudio})`);

        return new MultimodalContext(ctx, model, supportsVision, supportsAudio);
    }

    /** Check if context supports vision (image) input */
    get supportsVision(): boolean {
        return this._supportsVision;
    }

    /** Check if context supports audio input */
    get supportsAudio(): boolean {
        return this._supportsAudio;
    }

    /** Get the default media marker string */
    static getDefaultMarker(): string {
        if (!hasMultimodal || !lib.symbols.mtmd_get_default_marker) {
            return "<__media__>";
        }
        const ptr = lib.symbols.mtmd_get_default_marker!();
        if (!ptr) return "<__media__>";
        return new Deno.UnsafePointerView(ptr).getCString();
    }

    /**
     * Create a bitmap from raw image bytes (JPEG, PNG, etc.)
     * @param imageBytes Raw image file bytes
     * @returns Bitmap pointer or null on failure
     */
    createBitmapFromBytes(imageBytes: Uint8Array): Deno.PointerValue | null {
        const bitmap = lib.symbols.mtmd_bitmap_from_bytes!(
            imageBytes,
            BigInt(imageBytes.length),
        );

        if (!bitmap) {
            logger.error("Failed to create bitmap from image bytes");
            return null;
        }

        return bitmap;
    }

    /**
     * Create a bitmap from raw RGB data
     * @param width Image width
     * @param height Image height
     * @param rgbData Raw RGB pixel data (width * height * 3 bytes)
     */
    createBitmapFromRGB(
        width: number,
        height: number,
        rgbData: Uint8Array,
    ): Deno.PointerValue | null {
        const bitmap = lib.symbols.mtmd_bitmap_create!(
            width,
            height,
            rgbData,
        );

        if (!bitmap) {
            logger.error("Failed to create bitmap from RGB data");
            return null;
        }

        return bitmap;
    }

    /**
     * Create a bitmap from ImageData
     * @param image Image data (base64, URL, or bytes)
     */
    async createBitmap(image: ImageData): Promise<Deno.PointerValue | null> {
        let imageBytes: Uint8Array;

        if (image.type === "bytes") {
            if (!(image.data instanceof Uint8Array)) {
                logger.error("Image data must be Uint8Array for bytes type");
                return null;
            }
            imageBytes = image.data;
        } else if (image.type === "base64") {
            if (typeof image.data !== "string") {
                logger.error("Image data must be string for base64 type");
                return null;
            }
            // Decode base64
            try {
                const binaryString = atob(image.data);
                imageBytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    imageBytes[i] = binaryString.charCodeAt(i);
                }
            } catch (e) {
                logger.error(`Failed to decode base64 image: ${e}`);
                return null;
            }
        } else if (image.type === "url") {
            if (typeof image.data !== "string") {
                logger.error("Image data must be string for url type");
                return null;
            }
            // Fetch URL
            try {
                const response = await fetch(image.data);
                if (!response.ok) {
                    logger.error(`Failed to fetch image from URL: ${response.status}`);
                    return null;
                }
                const arrayBuffer = await response.arrayBuffer();
                imageBytes = new Uint8Array(arrayBuffer);
            } catch (e) {
                logger.error(`Failed to fetch image from URL: ${e}`);
                return null;
            }
        } else {
            logger.error(`Unknown image type: ${image.type}`);
            return null;
        }

        return this.createBitmapFromBytes(imageBytes);
    }

    /**
     * Free a bitmap
     * @param bitmap Bitmap pointer to free
     */
    freeBitmap(bitmap: Deno.PointerValue): void {
        lib.symbols.mtmd_bitmap_destroy!(bitmap);
    }

    /**
     * Get bitmap dimensions
     * @param bitmap Bitmap pointer
     */
    getBitmapSize(bitmap: Deno.PointerValue): { width: number; height: number } {
        return {
            width: lib.symbols.mtmd_bitmap_get_width!(bitmap),
            height: lib.symbols.mtmd_bitmap_get_height!(bitmap),
        };
    }

    /**
     * Tokenize text with media placeholders and bitmaps
     * @param text Text with <__media__> markers
     * @param bitmaps Array of bitmap pointers (one per marker)
     * @param addSpecial Whether to add special tokens
     * @param parseSpecial Whether to parse special tokens in text
     * @returns Input chunks pointer or null on failure
     */
    async tokenizeWithMedia(
        text: string,
        bitmaps: Deno.PointerValue[],
        addSpecial: boolean = true,
        parseSpecial: boolean = true,
    ): Promise<Deno.PointerValue | null> {
        const textBuffer = new TextEncoder().encode(text + "\0");

        // Create pointer array for bitmaps
        const bitmapPtrs = new BigUint64Array(bitmaps.length);
        for (let i = 0; i < bitmaps.length; i++) {
            bitmapPtrs[i] = BigInt(Deno.UnsafePointer.value(bitmaps[i]));
        }

        const chunks = await lib.symbols.mtmd_tokenize_with_media!(
            this.ctx,
            textBuffer,
            addSpecial,
            parseSpecial,
            bitmapPtrs,
            BigInt(bitmaps.length),
        );

        if (!chunks) {
            logger.error("Failed to tokenize text with media");
            return null;
        }

        return chunks;
    }

    /**
     * Free input chunks
     * @param chunks Chunks pointer to free
     */
    freeChunks(chunks: Deno.PointerValue): void {
        lib.symbols.mtmd_chunks_free!(chunks);
    }

    /**
     * Get number of chunks
     * @param chunks Chunks pointer
     */
    getChunksSize(chunks: Deno.PointerValue): number {
        return Number(lib.symbols.mtmd_chunks_size!(chunks));
    }

    /**
     * Get total token count from all chunks
     * @param chunks Chunks pointer
     */
    getTotalTokens(chunks: Deno.PointerValue): number {
        return Number(lib.symbols.mtmd_chunks_get_total_tokens!(chunks));
    }

    /**
     * Get chunk at index
     * @param chunks Chunks pointer
     * @param idx Chunk index
     */
    getChunk(chunks: Deno.PointerValue, idx: number): InputChunk | null {
        const chunk = lib.symbols.mtmd_chunks_get!(chunks, BigInt(idx));
        if (!chunk) return null;

        return {
            type: lib.symbols.mtmd_chunk_get_type!(chunk) as ChunkType,
            nTokens: Number(lib.symbols.mtmd_chunk_get_n_tokens!(chunk)),
            pointer: chunk,
        };
    }

    /**
     * Get all chunks as array
     * @param chunks Chunks pointer
     */
    getAllChunks(chunks: Deno.PointerValue): InputChunk[] {
        const result: InputChunk[] = [];
        const size = this.getChunksSize(chunks);
        for (let i = 0; i < size; i++) {
            const chunk = this.getChunk(chunks, i);
            if (chunk) result.push(chunk);
        }
        return result;
    }

    /**
     * Encode a media chunk (image/audio) to get embeddings
     * @param chunk Chunk pointer
     * @returns 0 on success, error code otherwise
     */
    async encodeChunk(chunk: Deno.PointerValue): Promise<number> {
        return await lib.symbols.mtmd_encode_chunk!(this.ctx, chunk);
    }

    /**
     * Get output embeddings from the last encode pass
     * @returns Float pointer to embeddings
     */
    getOutputEmbeddings(): Deno.PointerValue | null {
        return lib.symbols.mtmd_get_output_embd!(this.ctx);
    }

    /**
     * Check if we need non-causal mask before llama_decode
     */
    useNonCausal(): boolean {
        return lib.symbols.mtmd_decode_use_non_causal!(this.ctx);
    }

    /**
     * Check if model uses M-RoPE for llama_decode
     */
    useMRoPE(): boolean {
        return lib.symbols.mtmd_decode_use_mrope!(this.ctx);
    }

    /**
     * Free the multimodal context
     */
    free(): void {
        if (this.ctx) {
            lib.symbols.mtmd_ctx_free!(this.ctx);
            this.ctx = null as unknown as Deno.PointerValue;
        }
    }
}

/**
 * Process images for multimodal generation
 * Helper class to manage bitmap lifecycle
 */
export class MultimodalProcessor {
    private mtmdCtx: MultimodalContext;
    private bitmaps: Deno.PointerValue[] = [];
    private chunks: Deno.PointerValue | null = null;

    constructor(mtmdCtx: MultimodalContext) {
        this.mtmdCtx = mtmdCtx;
    }

    /**
     * Process images and create bitmaps
     * @param images Array of image data
     */
    async processImages(images: ImageData[]): Promise<boolean> {
        this.cleanup();

        for (const image of images) {
            const bitmap = await this.mtmdCtx.createBitmap(image);
            if (!bitmap) {
                this.cleanup();
                return false;
            }
            this.bitmaps.push(bitmap);
        }

        return true;
    }

    /**
     * Tokenize text with the processed images
     * @param text Text with media markers
     */
    async tokenize(
        text: string,
        addSpecial: boolean = true,
        parseSpecial: boolean = true,
    ): Promise<boolean> {
        if (this.bitmaps.length === 0) {
            logger.warn("No images processed, tokenizing text only");
        }

        this.chunks = await this.mtmdCtx.tokenizeWithMedia(
            text,
            this.bitmaps,
            addSpecial,
            parseSpecial,
        );

        return this.chunks !== null;
    }

    /**
     * Get the tokenized chunks
     */
    getChunks(): InputChunk[] {
        if (!this.chunks) return [];
        return this.mtmdCtx.getAllChunks(this.chunks);
    }

    /**
     * Get total token count
     */
    getTotalTokens(): number {
        if (!this.chunks) return 0;
        return this.mtmdCtx.getTotalTokens(this.chunks);
    }

    /**
     * Encode all media chunks
     */
    async encodeMediaChunks(): Promise<boolean> {
        if (!this.chunks) return false;

        const allChunks = this.getChunks();
        for (const chunk of allChunks) {
            if (chunk.type === ChunkType.Image || chunk.type === ChunkType.Audio) {
                const result = await this.mtmdCtx.encodeChunk(chunk.pointer);
                if (result !== 0) {
                    logger.error(`Failed to encode media chunk: ${result}`);
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Cleanup all resources
     */
    cleanup(): void {
        if (this.chunks) {
            this.mtmdCtx.freeChunks(this.chunks);
            this.chunks = null;
        }

        for (const bitmap of this.bitmaps) {
            this.mtmdCtx.freeBitmap(bitmap);
        }
        this.bitmaps = [];
    }
}
