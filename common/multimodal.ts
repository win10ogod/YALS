export const DEFAULT_MEDIA_MARKER = "<__media__>";

function decodeBase64(input: string): Uint8Array {
    const normalized = input.replace(/\s+/g, "");
    const binary = atob(normalized);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

function parseDataUrl(url: string): { mime: string; data: string } | null {
    if (!url.startsWith("data:")) {
        return null;
    }

    const commaIndex = url.indexOf(",");
    if (commaIndex === -1) {
        return null;
    }

    const meta = url.slice(5, commaIndex);
    const data = url.slice(commaIndex + 1);
    if (!meta.includes("base64")) {
        return null;
    }

    const mime = meta.split(";")[0] || "application/octet-stream";
    return { mime, data };
}

async function loadFileUrl(url: string): Promise<Uint8Array> {
    const path = url.replace(/^file:\/\//, "");
    const filePath = decodeURIComponent(path);
    const data = await Deno.readFile(filePath);
    return data;
}

export async function resolveImageUrl(url: string): Promise<Uint8Array> {
    const dataUrl = parseDataUrl(url);
    if (dataUrl) {
        return decodeBase64(dataUrl.data);
    }

    if (url.startsWith("file://")) {
        return await loadFileUrl(url);
    }

    if (url.startsWith("http://") || url.startsWith("https://")) {
        throw new Error(
            "Remote image URLs are not supported. Use data URLs or file URLs instead.",
        );
    }

    throw new Error("Unsupported image URL format.");
}

export function resolveBase64Payload(data: string): Uint8Array {
    return decodeBase64(data);
}
