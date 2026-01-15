#ifndef SLOT_HPP
#define SLOT_HPP

#include <string>
#include <vector>
#include "llama.h"
#include "tokenization.hpp"
#include "sequence_stream.hpp"
#include "generation_resources.hpp"
#include "presampler.hpp"

/*
 *  Slots are essentially just a data container holding the current inference state for a single complete inference.
 *
 *  Provides
 *  A centralized data container for the processor to manage the inference state.
 */

struct Slot {
    enum class State {
        IDLE,
        PROMPT,
        GENERATING,
        SUSPENDED,
    };

    struct SlotSnapshot {
        size_t prompt_tokens_processed{};
        int tokens_generated{};
        int n_past{};
        int i_batch{};
        llama_token last_token{};
        int ga_i{};
        bool self_extend_used{};
        std::string previous_seq_stream_buffer;
        int32_t previous_kv_pos{};

        static SlotSnapshot snapshot_slot(const Slot& slot, llama_memory_t mem, const bool during_prompt) {
            SlotSnapshot snapshot;
            snapshot.prompt_tokens_processed = slot.prompt_tokens_processed;
            snapshot.tokens_generated = slot.tokens_generated;
            snapshot.n_past = slot.n_past;
            snapshot.i_batch = slot.i_batch;
            snapshot.last_token = slot.last_token;
            snapshot.ga_i = slot.ga_i;
            snapshot.self_extend_used = slot.self_extend_used;
            snapshot.previous_seq_stream_buffer = slot.sequence_stream->sequence_buffer;

            // During the prompt because we do not call decode, we need a special case to update the kv pos for prompt
            snapshot.previous_kv_pos = during_prompt ? slot.n_past : llama_memory_seq_pos_max(mem, slot.slot_id) + 1;
            return snapshot;
        }

        int32_t rewind_slot(Slot& slot) const {
            slot.prompt_tokens_processed = prompt_tokens_processed;
            slot.tokens_generated = tokens_generated;
            slot.n_past = n_past;
            slot.i_batch = i_batch;
            slot.last_token = last_token;
            slot.ga_i = ga_i;
            slot.self_extend_used = self_extend_used;
            slot.sequence_stream->sequence_buffer = previous_seq_stream_buffer;
            return previous_kv_pos;
        }
    };

    int job_index{-1};
    int request_id{-1};
    int slot_id{0};
    uint32_t n_ctx_max{0};
    State state = State::IDLE;
    int n_keep{0};
    int n_discard{0};
    int grp_attn_n{1};
    int grp_attn_w{512};
    int ga_i{0};
    bool ctx_shift{false};
    bool add_special{true};
    bool self_extend_used{false};

    std::vector<llama_token> prompt_tokens;
    size_t prompt_tokens_processed{0};
    int tokens_generated{0};

    int n_past{0};
    int i_batch{-1};

    double slot_start_time{0.0};
    double prompt_end_time{0.0};
    double generating_end_time{0.0};

    llama_token last_token{0};
    std::string generated_text;

    TokenStreamDetokenizer* detokenizer;
    SequenceStream* sequence_stream;
    SlotSnapshot rewind_snapshot;

    llama_sampler* rule_chain{nullptr};
    Presampler presampler;
    llama_sampler* sampler{nullptr};

    GenerationResources* gen_resources{nullptr};
    class RuleStream* rule_stream{nullptr};

    bool cancelled{false};
    bool has_mtmd{false};

    explicit Slot(const llama_model* model, llama_context* ctx): presampler() {
        detokenizer = new TokenStreamDetokenizer(ctx);
        sequence_stream = new SequenceStream();
    }

    ~Slot() {
        delete detokenizer;
        delete sequence_stream;
        generation_resources_release(gen_resources);
    }

    [[nodiscard]] bool is_processing() const { return state == State::PROMPT || state == State::GENERATING; }
    [[nodiscard]] bool is_processing_prompt() const { return state == State::PROMPT; }
    [[nodiscard]] bool is_generating() const { return state == State::GENERATING; }

    void clear() {
        if (self_extend_used) {
            prompt_tokens.clear();
        }
        request_id = -1;
        state = State::IDLE;
        prompt_tokens_processed = 0;
        tokens_generated = 0;
        n_past = 0;
        i_batch = -1;
        last_token = 0;
        slot_start_time = 0;
        prompt_end_time = 0.0;
        generating_end_time = 0.0;
        generated_text.clear();
        detokenizer->reset();
        presampler.reset();
        cancelled = false;
        has_mtmd = false;
        n_keep = 0;
        n_discard = 0;
        grp_attn_n = 1;
        grp_attn_w = 512;
        ga_i = 0;
        ctx_shift = false;
        add_special = true;
        self_extend_used = false;
    }

    State previous_state{State::IDLE};
    void suspend() {
        if (state == State::SUSPENDED) return;
        previous_state = state;
        state = State::SUSPENDED;
    }

    void resume() {
        if  (state != State::SUSPENDED) return;
        state = previous_state;
    }

    void end(const int new_id, llama_context* ctx) {
        clear();
        job_index = new_id;
    }
};

#endif // SLOT_HPP
