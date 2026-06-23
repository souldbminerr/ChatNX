#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include <switch.h>

#include <ggml.h>
#include <gguf.h>
#include <llama.h>

namespace {
    constexpr const char* MODEL_DIR = "sdmc:/switch/chatnx";

    constexpr int N_CTX = 2048;
    constexpr int N_THREADS = 3;
    constexpr int N_PREDICT = 512;

    struct ModelEntry {
        std::string name;
        std::string path;
        std::string quant;
        u64         size;
    };

    std::string detect_quant(const std::string& path) {
        struct gguf_init_params gp = { true, nullptr };
        gguf_context* g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return "?";

        int counts[64] = {0};
        int64_t n = gguf_get_n_tensors(g);
        for (int64_t i = 0; i < n; ++i) {
            int t = (int)gguf_get_tensor_type(g, i);
            if (t >= 0 && t < 64) counts[t]++;
        }
        gguf_free(g);

        int best = -1, best_n = 0, any = -1, any_n = 0;
        for (int t = 0; t < 64; ++t) {
            if (counts[t] > any_n) { any_n = counts[t]; any = t; }
            if (t == (int)GGML_TYPE_F32 || t == (int)GGML_TYPE_F16) continue;
            if (counts[t] > best_n) { best_n = counts[t]; best = t; }
        }
        int chosen = best >= 0 ? best : any;
        if (chosen < 0) return "?";
        return ggml_type_name((enum ggml_type)chosen);
    }

    u64 file_size(const std::string& path) {
        struct stat st;
        return stat(path.c_str(), &st) == 0 ? (u64)st.st_size : 0;
    }

    std::vector<ModelEntry> list_models(const char* dir) {
        std::vector<ModelEntry> out;
        DIR* d = opendir(dir);
        if (!d) return out;
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string name = e->d_name;
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
                std::string path = std::string(dir) + "/" + name;
                out.push_back({ name, path, detect_quant(path), file_size(path) });
            }
        }
        closedir(d);
        return out;
    }

    bool is_application_mode() {
        AppletType t = appletGetAppletType();
        return t == AppletType_Application || t == AppletType_SystemApplication;
    }

    u64 free_ram() {
        u64 total = 0, used = 0;
        svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
        svcGetInfo(&used,  InfoType_UsedMemorySize,  CUR_PROCESS_HANDLE, 0);
        return total > used ? total - used : 0;
    }

    void log_cb(ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            printf("%s", text);
            consoleUpdate(NULL);
        }
    }

    void wait_for_exit(PadState* pad) {
        printf("\nPress + to exit.\n");
        consoleUpdate(NULL);
        while (appletMainLoop()) {
            padUpdate(pad);
            if (padGetButtonsDown(pad) & HidNpadButton_Plus) break;
            consoleUpdate(NULL);
        }
    }

    std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text,
                                      bool add_special) {
        int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                nullptr, 0, add_special, true);
        std::vector<llama_token> out(n);
        int got = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                 out.data(), n, add_special, true);
        out.resize(got < 0 ? 0 : got);
        return out;
    }

    std::string token_to_text(const llama_vocab* vocab, llama_token tok) {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n < 0) return std::string();
        return std::string(buf, n);
    }

    bool read_line(const char* guide, std::string& out) {
        SwkbdConfig kbd;
        if (R_FAILED(swkbdCreate(&kbd, 0))) return false;
        swkbdConfigMakePresetDefault(&kbd);
        swkbdConfigSetHeaderText(&kbd, "Enter your prompt");
        swkbdConfigSetGuideText(&kbd, guide);

        char buf[1024] = {0};
        Result rc = swkbdShow(&kbd, buf, sizeof(buf));
        swkbdClose(&kbd);
        if (R_FAILED(rc)) return false;

        out.assign(buf);
        return true;
    }

    struct ChatView {
        struct Line { std::string s; int len; };
        std::vector<Line> lines;
        int  width  = 79;
        int  height = 44;
        int  scroll = 0;
        bool follow = true;
        std::string color;

        void init() {
            PrintConsole* c = consoleGetDefault();
            if (c->consoleWidth  > 1) width  = c->consoleWidth  - 1;
            if (c->consoleHeight > 1) height = c->consoleHeight - 1;
            lines.assign(1, { std::string(), 0 });
            scroll = 0;
            follow = true;
            color.clear();
        }

        int max_scroll() const {
            int m = (int)lines.size() - height;
            return m > 0 ? m : 0;
        }

        void newline() { lines.push_back({ color, 0 }); }

        void append(const std::string& text) {
            for (size_t i = 0; i < text.size(); ) {
                char ch = text[i];
                if (ch == '\x1b') {
                    std::string seq = "\x1b";
                    size_t j = i + 1;
                    if (j < text.size() && text[j] == '[') {
                        seq += '['; ++j;
                        while (j < text.size() && !(text[j] >= '@' && text[j] <= '~'))
                            seq += text[j++];
                        if (j < text.size()) seq += text[j++];
                    }
                    if (!seq.empty() && seq.back() == 'm')
                        color = (seq == "\x1b[0m" || seq == "\x1b[m") ? std::string() : seq;
                    lines.back().s += seq;
                    i = j;
                } else if (ch == '\n') {
                    newline();
                    ++i;
                } else if (ch == '\r') {
                    ++i;
                } else {
                    if (lines.back().len >= width) newline();
                    lines.back().s += ch;
                    lines.back().len += 1;
                    ++i;
                }
            }
            if (follow) scroll = max_scroll();
        }

        void scroll_by(int d) {
            int ns = scroll + d;
            if (ns < 0) ns = 0;
            if (ns > max_scroll()) ns = max_scroll();
            scroll = ns;
            follow = (scroll >= max_scroll());
        }

        void to_bottom() { follow = true; scroll = max_scroll(); }

        void render() {
            consoleClear();
            int total = (int)lines.size();
            bool bar = total > height;
            int thumb_pos = 0, thumb_size = height;
            if (bar) {
                thumb_size = height * height / total;
                if (thumb_size < 1) thumb_size = 1;
                int ms = max_scroll();
                thumb_pos = ms > 0 ? scroll * (height - thumb_size) / ms : 0;
            }
            for (int r = 0; r < height; ++r) {
                int idx = scroll + r;
                printf("\x1b[%d;1H", r + 1);
                if (idx < total) printf("%s\x1b[0m", lines[idx].s.c_str());
                if (bar) {
                    printf("\x1b[%d;%dH", r + 1, width + 1);
                    bool thumb = (r >= thumb_pos && r < thumb_pos + thumb_size);
                    printf(thumb ? "\x1b[7m \x1b[0m" : "\x1b[2m|\x1b[0m");
                }
            }
            consoleUpdate(NULL);
        }
    };

    ChatView g_view;

    std::string generate(llama_context* ctx, const llama_vocab* vocab, llama_sampler* smpl,
                         const std::string& prompt, PadState* pad) {
        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
        std::vector<llama_token> toks = tokenize(vocab, prompt, is_first);
        if (toks.empty()) return std::string();

        llama_batch batch = llama_batch_get_one(toks.data(), (int32_t)toks.size());
        std::string reply;
        int n_gen = 0;
        u64 t_start = armGetSystemTick();

        for (int i = 0; i < N_PREDICT && appletMainLoop(); ++i) {
            if (llama_decode(ctx, batch) != 0) break;

            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;

            std::string piece = token_to_text(vocab, id);
            reply += piece;
            g_view.append(piece);
            g_view.render();
            ++n_gen;

            static llama_token next;
            next  = id;
            batch = llama_batch_get_one(&next, 1);

            padUpdate(pad);
            if (padGetButtonsDown(pad) & HidNpadButton_B) {
                g_view.append("\x1b[2;37m [cancelled]\x1b[0m");
                break;
            }
        }

        double secs = armTicksToNs(armGetSystemTick() - t_start) / 1.0e9;
        if (n_gen > 0 && secs > 0.0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "\n\x1b[2;37m%d tokens, %.2f/sec\x1b[0m\n", n_gen, n_gen / secs);
            g_view.append(buf);
            g_view.render();
        }

        return reply;
    }

    int select_model(PadState* pad, const std::vector<ModelEntry>& models, std::string& out_path) {
        int sel = 0;
        while (appletMainLoop()) {
            consoleClear();
            printf("ChatNX - select a model\n\n");
            for (int i = 0; i < (int)models.size(); ++i) {
                double gib = models[i].size / (1024.0 * 1024.0 * 1024.0);
                const char* arrow = (i == sel) ? "\x1b[1;36m> " : "    ";
                printf("%s%s\x1b[0m  \x1b[2;37m[%s, %.2f GiB]\x1b[0m\n",
                       arrow, models[i].name.c_str(), models[i].quant.c_str(), gib);
            }
            printf("Press + to exit");
            consoleUpdate(NULL);

            padUpdate(pad);
            u64 k = padGetButtonsDown(pad);
            if (k & HidNpadButton_Plus) return -1;
            if (k & (HidNpadButton_Down | HidNpadButton_StickLDown))
                sel = (sel + 1) % models.size();
            if (k & (HidNpadButton_Up | HidNpadButton_StickLUp))
                sel = (sel - 1 + (int)models.size()) % models.size();
            if (k & HidNpadButton_A) {
                out_path = models[sel].path;
                return sel;
            }
        }
        return -1;
    }

    bool run_chat(PadState* pad, const std::string& model_path) {
        const double mib = 1024.0 * 1024.0;

        llama_log_set(log_cb, nullptr);
        llama_backend_init();

        consoleClear();
        printf("Loading %s...\n", model_path.c_str());
        consoleUpdate(NULL);

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_mmap     = false;
        mp.use_mlock    = false;

        llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) {
            printf("Failed to load model.");
            return false;
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);
        printf("Model loaded. Free heap: %.1f MiB\n", free_ram() / mib);
        consoleUpdate(NULL);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = N_CTX;
        cp.n_threads       = N_THREADS;
        cp.n_threads_batch = N_THREADS;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            printf("Failed to create context.\n");
            llama_model_free(model);
            return false;
        }

        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        // Prevemt repatition loop
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const char* tmpl = llama_model_chat_template(model, nullptr);

        std::vector<llama_chat_message> messages;
        std::vector<char> formatted(llama_n_ctx(ctx));
        int prev_len = 0;

        messages.push_back({ "system",
            strdup("You are a helpful AI assistant. Answer the user's question directly in plain prose. "
                   "Do not use markdown, code blocks, backticks, or bullet lists. "
                   "Do not repeat the question back. Give a real answer in a few complete sentences."
                   "If the data you are being questioned on is not in your training data or you are unsure about it, tell that to the user") });


        g_view.init();
        g_view.append("\x1b[2;37mPress A to enter a prompt and + to exit\x1b[0m\n");
        g_view.render();

        while (appletMainLoop()) {
            padUpdate(pad);
            u64 kDown = padGetButtonsDown(pad);
            if (kDown & HidNpadButton_Plus) break;

            // Scroll the chat history with the D-pad or left stick (held = repeat).
            u64 kHeld = padGetButtons(pad);
            int prev_scroll = g_view.scroll;
            if (kHeld & (HidNpadButton_Up   | HidNpadButton_StickLUp))   g_view.scroll_by(-1);
            if (kHeld & (HidNpadButton_Down | HidNpadButton_StickLDown)) g_view.scroll_by(1);

            if (!(kDown & HidNpadButton_A)) {
                if (g_view.scroll != prev_scroll) g_view.render();
                else                              consoleUpdate(NULL);
                continue;
            }

            std::string input;
            if (!read_line("Message", input) || input.empty()) continue;

            g_view.to_bottom();
            char hdr[1100];
            snprintf(hdr, sizeof(hdr), "\n\x1b[1;33mYou:\x1b[0m %s\n\x1b[1;36mAI:\x1b[0m ", input.c_str());
            g_view.append(hdr);
            g_view.render();

            messages.push_back({ "user", strdup(input.c_str()) });

            int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                    true, formatted.data(), formatted.size());
            if (new_len > (int)formatted.size()) {
                formatted.resize(new_len);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                    true, formatted.data(), formatted.size());
            }
            if (new_len < 0) {
                g_view.append("\x1b[1;31mError\x1b[0m\n");
                g_view.render();
                continue;
            }

            std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
            std::string reply = generate(ctx, vocab, smpl, prompt, pad);

            messages.push_back({ "assistant", strdup(reply.c_str()) });
            prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                 false, nullptr, 0);
            if (prev_len < 0) prev_len = 0;

            g_view.append("\n\x1b[2;37mPress A to enter a prompt and + to exit\x1b[0m\n");
            g_view.render();
        }

        for (llama_chat_message& m : messages) free((void*)m.content);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return true;
    }
}

int main(int argc, char* argv[]) {
    consoleInit(NULL);
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);

    PadState pad;
    padInitializeDefault(&pad);

    if (!is_application_mode()) {
        printf("Rerun in title takeover mode. Press + to exit.\n");
        wait_for_exit(&pad);
        consoleExit(NULL);
        return 0;
    }

    printf("ChatNX\nScanning %s ...\n", MODEL_DIR);
    consoleUpdate(NULL);

    std::vector<ModelEntry> models = list_models(MODEL_DIR);
    if (models.empty()) {
        printf("\x1b[1;31mNo models found.");
        printf("Put .gguf files in %s\n", MODEL_DIR);
        wait_for_exit(&pad);
        consoleExit(NULL);
        return 0;
    }

    std::string model_path;
    if (select_model(&pad, models, model_path) < 0) {
        consoleExit(NULL);
        return 0;
    }

    run_chat(&pad, model_path);

    wait_for_exit(&pad);

    consoleExit(NULL);
    return 0;
}
