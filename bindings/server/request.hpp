#ifndef REQUEST_HPP
#define REQUEST_HPP

#include <memory>
#include <vector>

#include "llama.h"
#include "mtmd.h"

/*
 * A light abstraction over a request to fill a slot. This pends in a queue until we have free slots to take
 * the next request.
 */

struct MtmdChunksDeleter {
    void operator()(mtmd_input_chunks* ptr) const {
        if (ptr) {
            mtmd_input_chunks_free(ptr);
        }
    }
};

using MtmdChunksPtr = std::unique_ptr<mtmd_input_chunks, MtmdChunksDeleter>;

struct Request {
    int id;
    std::vector<llama_token> prompt_tokens;
    MtmdChunksPtr mtmd_chunks;
    size_t prompt_tokens_total = 0;
    llama_pos prompt_n_pos = 0;
    bool has_mtmd = false;
    InferenceArgs inference_args;

    Request() = default;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) = default;
    Request& operator=(Request&&) = default;
};

#endif // REQUEST_HPP
