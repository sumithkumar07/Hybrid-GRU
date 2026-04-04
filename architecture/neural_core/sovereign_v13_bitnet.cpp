#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>
#include <string>
#include <fstream>
#include <cstring>
#include <ctime>
#include <omp.h>
#include "tokenizer.h"

#ifdef _WIN32
#define SOV_API __declspec(dllexport)
#else
#define SOV_API
#endif


// --- DIMENSIONS ---
static const int VOCAB = 50257; // GPT-2 Standard Vocabulary size
static const int EMBED_DIM = 256;
static const int H_DIM = 1024;
static const int GRU_CONCAT = EMBED_DIM + H_DIM;
static const int HIDDEN = 1024;

// --- BITNET 1.58b UTILS ---
// Ternary weights: -1, 0, 1 stored as int8_t for 2-bit efficiency
typedef int8_t tweight; 

inline double sov_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sov_relu(double x) { return x > 0 ? x : 0; }

void softmax(double* raw_logits, double* probs, int n) {
    double max_l = -1e9;
    for(int i=0; i<n; i++) {
        if (std::isnan(raw_logits[i]) || std::isinf(raw_logits[i])) raw_logits[i] = -50.0; // NaN Guard
        if(raw_logits[i] > max_l) max_l = raw_logits[i];
    }
    double sum = 0;
    for(int i=0; i<n; i++) {
        probs[i] = std::exp(raw_logits[i] - max_l);
        sum += probs[i];
    }
    
    // Neural Pulse: If brain is too shy, guess at random instead of hanging
    if (sum < 1e-9) {
        for(int i=0; i<n; i++) probs[i] = 1.0 / (double)n;
    } else {
        for(int i=0; i<n; i++) probs[i] /= sum;
    }
}

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

inline void rmsnorm(double* x, int n) {
    double ms = 0;
    for(int i=0; i<n; i++) ms += x[i] * x[i];
    double rrms = 1.0 / std::sqrt(ms / (double)n + 1e-8);
    for(int i=0; i<n; i++) x[i] *= rrms;
}

// Optimized Ternary Matrix Multiplication
// Instead of w*x, we do: sum += (w == 1) ? x : (w == -1) ? -x : 0
inline double t_matmul(const tweight* w, const double* x, int n, double scale) {
    double sum = 0;
    #pragma omp simd reduction(+:sum)
    for(int i=0; i<n; i++) {
        if(w[i] == 1) sum += x[i];
        else if(w[i] == -1) sum -= x[i];
    }
    return sum * scale;
}

// --- HYDRA FRAGMENTS ---
struct Fragment {
    std::string agent_id;
    float personality_bias[H_DIM];
    double h_memory[H_DIM]; 

    // Phase 7: Archival Memory (Long-Term Vector Store)
    double* h_archive = nullptr;
    int archive_count = 0;
    int archive_head = 0;
    const int ARCHIVE_CAP = 128; // Optimized for 1000-agent swarm (1MB/agent)

    Fragment() {
        h_archive = new double[ARCHIVE_CAP * H_DIM];
        std::memset(personality_bias, 0, 4 * H_DIM);
        std::memset(h_archive, 0, 8 * ARCHIVE_CAP * H_DIM);
        std::memset(h_memory, 0, 8 * H_DIM);
    }
    ~Fragment() { if(h_archive) delete[] h_archive; }
};

// --- SOVEREIGN V13 ENGINE ---
struct SovereignBlock {
    // Ternary Weights (BitNet Inference)
    tweight* We; // [VOCAB * EMBED_DIM]
    tweight* Wz; // [H_DIM * GRU_CONCAT]
    tweight* Wr; // [H_DIM * GRU_CONCAT]
    tweight* Wh; // [H_DIM * GRU_CONCAT]
    tweight* Wo; // [VOCAB * HIDDEN]
    
    // Scale Factors
    double s_e, s_z, s_r, s_h, s_o;
    
    // Biases
    double *bz, *br, *bh, *bo;

    // Latent Buffers (Only for Training)
    double *L_We=nullptr, *L_Wz=nullptr, *L_Wr=nullptr, *L_Wh=nullptr, *L_Wo=nullptr;
    
    // Phase 4: Shared Swarm Memory
    double *hive_context;

    SovereignBlock() {
        We = new tweight[VOCAB * EMBED_DIM];
        Wz = new tweight[H_DIM * GRU_CONCAT];
        Wr = new tweight[H_DIM * GRU_CONCAT];
        Wh = new tweight[H_DIM * GRU_CONCAT];
        Wo = new tweight[VOCAB * HIDDEN];
        
        bz = new double[H_DIM]; br = new double[H_DIM]; bh = new double[H_DIM]; bo = new double[VOCAB];
        
        // Phase 2: Xavier/Glorot Initialization for BitNet 1.58b
        s_e = 0.1; // Embedding scale remains controlled
        s_z = std::sqrt(1.0 / (double)GRU_CONCAT);
        s_r = std::sqrt(1.0 / (double)GRU_CONCAT);
        s_h = std::sqrt(1.0 / (double)GRU_CONCAT);
        s_o = std::sqrt(1.0 / (double)HIDDEN);
        
        hive_context = new double[H_DIM];
        std::memset(hive_context, 0, 8 * H_DIM);
    }

    void enable_training() {
        if (!L_We) {
            L_We = new double[VOCAB * EMBED_DIM];
            L_Wz = new double[H_DIM * GRU_CONCAT];
            L_Wr = new double[H_DIM * GRU_CONCAT];
            L_Wh = new double[H_DIM * GRU_CONCAT];
            L_Wo = new double[VOCAB * HIDDEN];
            // Initialize latent buffers from ternary weights to maintain state
            auto init_latent = [&](double* latent, tweight* ternary, int n, double scale) {
                for(int i=0; i<n; i++) latent[i] = (ternary[i] == 1) ? scale : (ternary[i] == -1) ? -scale : 0;
            };
            init_latent(L_We, We, VOCAB * EMBED_DIM, s_e);
            init_latent(L_Wz, Wz, H_DIM * GRU_CONCAT, s_z);
            init_latent(L_Wr, Wr, H_DIM * GRU_CONCAT, s_r);
            init_latent(L_Wh, Wh, H_DIM * GRU_CONCAT, s_h);
            init_latent(L_Wo, Wo, VOCAB * HIDDEN, s_o);
        }
    }

    void quantize() {
        // Phase 2: BitNet 1.58b Quantization with Mean-Absolute Scaling
        auto q = [](double* latent, tweight* ternary, int n, double& scale) {
            double sum = 0, abs_sum = 0; 
            for(int i=0; i<n; i++) {
                sum += latent[i];
                abs_sum += std::abs(latent[i]);
            }
            double avg = sum / n;
            scale = abs_sum / n; // Dynamic Scale factor
            if (scale < 1e-6) scale = 0.01;
            
            for(int i=0; i<n; i++) {
                double val = latent[i] - avg;
                ternary[i] = (val > 0.5 * scale) ? 1 : (val < -0.5 * scale) ? -1 : 0;
            }
        };
        if (L_We) {
            q(L_We, We, VOCAB * EMBED_DIM, s_e);
            q(L_Wz, Wz, H_DIM * GRU_CONCAT, s_z);
            q(L_Wr, Wr, H_DIM * GRU_CONCAT, s_r);
            q(L_Wh, Wh, H_DIM * GRU_CONCAT, s_h);
            q(L_Wo, Wo, VOCAB * HIDDEN, s_o);
        }
    }

    void save(const char* p) {
        // Obsolete (Latent format) - Use sovereign_save_compact
    }
};

struct Agent {
    SovereignBlock* m;
    Fragment* f;
    double h[H_DIM];
    std::mt19937 gen;
    
    Agent(SovereignBlock* master, Fragment* frag, int seed) 
        : m(master), f(frag), gen(seed) { 
        std::memset(h, 0, 8*H_DIM); 
    }
    ~Agent() {}
};

static WordTokenizer* g_tok = nullptr;

#ifdef _WIN32
#define SOV_API __declspec(dllexport)
#else
#define SOV_API 
#endif

extern "C" {

SOV_API void sovereign_save_compact(void* master, const char* path) {
        if(!master || !path) return;
        SovereignBlock* m = (SovereignBlock*)master;
        std::ofstream f(path, std::ios::binary); if(!f) return;
        
        // Pack back from ternary to 2-bit storage
        auto pack = [&](tweight* ternary, int n, double scale) {
            f.write((char*)&scale, 8); // Save Scale Factor
            int packed_size = (n + 3) / 4;
            unsigned char* packed = new unsigned char[packed_size];
            std::memset(packed, 0, packed_size);
            for(int i=0; i<n; i++) {
                unsigned char val = (ternary[i] == 1) ? 2 : (ternary[i] == -1) ? 0 : 1;
                packed[i/4] |= (val << (2 * (i%4)));
            }
            f.write((char*)packed, packed_size);
            delete[] packed;
        };
        
        pack(m->We, VOCAB * EMBED_DIM, m->s_e);
        pack(m->Wz, H_DIM * GRU_CONCAT, m->s_z); f.write((char*)m->bz, 8 * H_DIM);
        pack(m->Wr, H_DIM * GRU_CONCAT, m->s_r); f.write((char*)m->br, 8 * H_DIM);
        pack(m->Wh, H_DIM * GRU_CONCAT, m->s_h); f.write((char*)m->bh, 8 * H_DIM);
        pack(m->Wo, VOCAB * HIDDEN, m->s_o); f.write((char*)m->bo, 8 * VOCAB);
    }

    SOV_API void sovereign_load_compact(void* master, const char* path) {
        if(!master || !path) return;
        SovereignBlock* m = (SovereignBlock*)master;
        std::ifstream f(path, std::ios::binary); if(!f) return;
        
        // Unpack directly into ternary without latent bloat
        auto unpack = [&](tweight* ternary, int n, double& scale) {
            f.read((char*)&scale, 8); // Load Scale Factor (8 bytes)
            int packed_size = (n + 3) / 4;
            unsigned char* packed = new unsigned char[packed_size];
            f.read((char*)packed, packed_size);
            
            #pragma omp parallel for
            for(int i=0; i<packed_size; i++) {
                unsigned char b = packed[i];
                for(int j=0; j<4; j++) {
                    int idx = i*4 + j;
                    if(idx < n) {
                        int val = (b >> (j*2)) & 0x03;
                        ternary[idx] = (val == 0) ? -1 : (val == 1) ? 0 : (val == 2) ? 1 : 0;
                    }
                }
            }
            delete[] packed;
        };

        unpack(m->We, VOCAB * EMBED_DIM, m->s_e);
        unpack(m->Wz, H_DIM * GRU_CONCAT, m->s_z); f.read((char*)m->bz, 8 * H_DIM);
        unpack(m->Wr, H_DIM * GRU_CONCAT, m->s_r); f.read((char*)m->br, 8 * H_DIM);
        unpack(m->Wh, H_DIM * GRU_CONCAT, m->s_h); f.read((char*)m->bh, 8 * H_DIM);
        unpack(m->Wo, VOCAB * HIDDEN, m->s_o); f.read((char*)m->bo, 8 * VOCAB);
    }


    SOV_API void sovereign_hive_broadcast(void* master, double* vector) {
        if(!master || !vector) return;
        SovereignBlock* m = (SovereignBlock*)master;
        std::memcpy(m->hive_context, vector, 8 * H_DIM);
    }

    SOV_API void sovereign_hive_consensus(void* master, double* agent_h, double factor) {
        if(!master || !agent_h) return;
        SovereignBlock* m = (SovereignBlock*)master;
        for(int i=0; i<H_DIM; i++) {
            m->hive_context[i] = (1.0 - factor)*m->hive_context[i] + factor*agent_h[i];
        }
        rmsnorm(m->hive_context, H_DIM);
    }

    SOV_API int sovereign_tokenize(const char* word) {
        if(!g_tok || !word) return TOK_PAD;
        std::vector<int> tokens = g_tok->encode(std::string(word));
        // encode returns [START, T1, T2, ..., END]
        // We want the first actual token if it exists
        if (tokens.size() > 2) return tokens[1];
        return TOK_UNK;
    }

    SOV_API const char* sovereign_detokenize(int idx) {
        if(!g_tok || idx < 0 || idx >= (int)g_tok->id_to_word.size()) return "";
        return g_tok->id_to_word[idx].c_str();
    }

    SOV_API void* sovereign_init_master() {
        if(!g_tok) { g_tok = new WordTokenizer(); g_tok->load_vocab("vocab.txt"); }
        SovereignBlock* b = new SovereignBlock(); 
        // Zero-RAM randomization: initialize ternary directly or enable training temporarily
        std::mt19937 g(42);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for(int i=0; i<VOCAB*EMBED_DIM; i++) b->We[i] = (d(g) > 0) ? 1 : -1;
        for(int i=0; i<H_DIM*GRU_CONCAT; i++) b->Wz[i] = (d(g) > 0) ? 1 : -1;
        for(int i=0; i<H_DIM*GRU_CONCAT; i++) b->Wr[i] = (d(g) > 0) ? 1 : -1;
        for(int i=0; i<H_DIM*GRU_CONCAT; i++) b->Wh[i] = (d(g) > 0) ? 1 : -1;
        for(int i=0; i<VOCAB*HIDDEN; i++)     b->Wo[i] = (d(g) > 0) ? 1 : -1;

        std::memset(b->bz, 0, 8 * H_DIM);
        std::memset(b->br, 0, 8 * H_DIM);
        std::memset(b->bh, 0, 8 * H_DIM);
        std::memset(b->bo, 0, 8 * VOCAB);
        return (void*)b;
    }

    SOV_API void sovereign_free_master(void* master) {
        if(!master) return;
        delete (SovereignBlock*)master;
    }

    SOV_API void sovereign_train_step_distill(void* agent_ptr, int input_token, float* teacher_probs, float lr) {
        if(!agent_ptr || !teacher_probs) return;
        Agent* a = (Agent*)agent_ptr;
        
        thread_local double s_logits[VOCAB];
        thread_local double s_probs[VOCAB];
        
        // Feed-Forward Logits
        for(int v=0; v<VOCAB; v++) {
            s_logits[v] = t_matmul(&a->m->Wo[v * HIDDEN], a->h, HIDDEN, a->m->s_o) + a->m->bo[v];
        }
        softmax(s_logits, s_probs, VOCAB);

        // 1. Output Gradient (Softmax -> Logits)
        double dh[H_DIM];
        std::memset(dh, 0, 8 * H_DIM);
        
        #pragma omp parallel for
        for(int v=0; v<VOCAB; v++) {
            double grad = s_probs[v] - (double)teacher_probs[v];
            for(int k=0; k<HIDDEN; k++) {
                // Update Wo (Latent)
                a->m->L_Wo[v * HIDDEN + k] -= lr * (grad * a->h[k] + 0.001 * a->m->L_Wo[v * HIDDEN + k]);
                // Accumulate dh for recurrent backprop
                // We use the ternary value or a surrogate for the gradient
                tweight w = a->m->Wo[v * HIDDEN + k];
                dh[k] += grad * ((w == 1) ? a->m->s_o : (w == -1) ? -a->m->s_o : 0);
            }
            a->m->bo[v] -= lr * grad;
        }

        // 2. Simple GRU BPTT (1-step approximation)
        // Updating Wz, Wr, Wh based on how they contributed to the current h
        #pragma omp parallel for
        for(int k=0; k<H_DIM; k++) {
            double dh_local = dh[k] * lr; // Scaled local gradient
            // We'll update the first 128 elements of the concat buffer (Simplified contribution)
            for(int j=0; j<GRU_CONCAT; j++) {
                a->m->L_Wz[k * GRU_CONCAT + j] -= dh_local * 0.1; 
                a->m->L_Wr[k * GRU_CONCAT + j] -= dh_local * 0.1;
                a->m->L_Wh[k * GRU_CONCAT + j] -= dh_local * 0.1;
            }
        }
    }

    SOV_API void sovereign_train_distill_bulk(void* agent_ptr, int* tokens, int n, float lr) {
        if(!agent_ptr || !tokens) return;
        Agent* a = (Agent*)agent_ptr;
        a->m->enable_training();
        
        for(int i=0; i<n; i++) {
            // Simulate single-token distribution (Targeted)
            float target[VOCAB];
            std::memset(target, 0, 4 * VOCAB);
            target[tokens[i]] = 1.0f;
            sovereign_train_step_distill(agent_ptr, tokens[i], target, lr);
        }
        
        // Single quantization pass for the entire batch (CRITICAL FOR SPEED)
        a->m->quantize();
    }

    SOV_API void* sovereign_init_fragment(const char* id) {
        if(!id) return nullptr;
        Fragment* f = new Fragment();
        f->agent_id = std::string(id);
        std::memset(f->personality_bias, 0, 4 * H_DIM);
        std::memset(f->h_memory, 0, 8 * H_DIM); // Initialize memory anchor to zero
        return (void*)f;
    }

    SOV_API void sovereign_set_fragment_bias(void* fragment_ptr, float* bias_data) {
        if(!fragment_ptr || !bias_data) return;
        std::memcpy(((Fragment*)fragment_ptr)->personality_bias, bias_data, 4 * H_DIM);
    }

    SOV_API float* sovereign_get_fragment_bias(void* fragment_ptr) {
        if(!fragment_ptr) return nullptr;
        return ((Fragment*)fragment_ptr)->personality_bias;
    }

    SOV_API void sovereign_agent_save_state(void* agent_ptr, void* fragment_ptr) {
        if(!agent_ptr || !fragment_ptr) return;
        Agent* a = (Agent*)agent_ptr;
        Fragment* f = (Fragment*)fragment_ptr;
        std::memcpy(f->h_memory, a->h, 8 * H_DIM);
    }

    SOV_API void sovereign_agent_load_state(void* agent_ptr, void* fragment_ptr) {
        if(!agent_ptr || !fragment_ptr) return;
        Agent* a = (Agent*)agent_ptr;
        Fragment* f = (Fragment*)fragment_ptr;
        std::memcpy(a->h, f->h_memory, 8 * H_DIM);
    }

    SOV_API void sovereign_agent_set_fragment(void* agent_ptr, void* fragment_ptr) {
        if(!agent_ptr || !fragment_ptr) return;
        Agent* a = (Agent*)agent_ptr;
        
        // Auto-Save outgoing state if fragment exists
        if(a->f) sovereign_agent_save_state(agent_ptr, (void*)a->f);
        
        // Switch Fragment
        a->f = (Fragment*)fragment_ptr;
        
        // Auto-Load incoming state (Memory Anchor)
        sovereign_agent_load_state(agent_ptr, fragment_ptr);
    }

    SOV_API void* sovereign_init_agent(const char* id, void* master, int seed) {
        if(!id || !master) return nullptr;
        Fragment* f = new Fragment();
        f->agent_id = std::string(id);
        for(int i=0; i<H_DIM; i++) f->personality_bias[i] = 0.0f; 
        return (void*)new Agent((SovereignBlock*)master, f, seed);
    }

    SOV_API void sovereign_agent_observe(void* agent_ptr, const char* text) {
        if(!text || !agent_ptr) return;
        Agent* a = (Agent*)agent_ptr;
        std::vector<int> tokens = g_tok->encode(std::string(text));
        for(int tok : tokens) {
            if (tok < 0 || tok >= VOCAB) tok = 0;
            double gru_in[EMBED_DIM];
            for(int d=0; d<EMBED_DIM; d++) {
                tweight w = a->m->We[tok * EMBED_DIM + d];
                gru_in[d] = (w == 1) ? a->m->s_e : (w == -1) ? -a->m->s_e : 0;
            }
            double h_new[H_DIM];
            double zsv[H_DIM], rsv[H_DIM];
            double concat[GRU_CONCAT];
            std::memcpy(concat, gru_in, sizeof(double) * EMBED_DIM);
            std::memcpy(concat + EMBED_DIM, a->h, sizeof(double) * H_DIM);

            #pragma omp parallel for
            for(int k=0; k<H_DIM; k++) {
                zsv[k] = sov_sigmoid(a->m->bz[k] + t_matmul(&a->m->Wz[k * GRU_CONCAT], concat, GRU_CONCAT, a->m->s_z));
                rsv[k] = sov_sigmoid(a->m->br[k] + t_matmul(&a->m->Wr[k * GRU_CONCAT], concat, GRU_CONCAT, a->m->s_r));
            }

            double concat_h[GRU_CONCAT];
            std::memcpy(concat_h, gru_in, sizeof(double) * EMBED_DIM);
            for(int j=0; j<H_DIM; j++) concat_h[EMBED_DIM + j] = rsv[j] * a->h[j];

            #pragma omp parallel for
            for(int k=0; k<H_DIM; k++) {
                double phc = a->m->bh[k] + t_matmul(&a->m->Wh[k * GRU_CONCAT], concat_h, GRU_CONCAT, a->m->s_h);
                h_new[k] = (1.0-zsv[k])*a->h[k] + zsv[k]*std::tanh(phc + (a->f->personality_bias[k] * 0.1f) + (a->m->hive_context[k] * 0.1f));
            }
            rmsnorm(h_new, H_DIM);
            std::memcpy(a->h, h_new, 8*H_DIM);
        }
    }

    SOV_API const char* sovereign_agent_act(void* agent_ptr, int max_len, double temp) {
        if(!agent_ptr) return "";
        Agent* a = (Agent*)agent_ptr;
        std::vector<int> response;
        int last_tok = 2;
        thread_local double s_logits[VOCAB];
        thread_local double s_probs[VOCAB];
        thread_local char out[8192]; 

        for(int i=0; i<max_len; i++) {
            double gru_in[EMBED_DIM];
            for(int d=0; d<EMBED_DIM; d++) {
                tweight w = a->m->We[last_tok * EMBED_DIM + d];
                gru_in[d] = (w == 1) ? a->m->s_e : (w == -1) ? -a->m->s_e : 0;
            }
            double h_new[H_DIM];
            double zsv[H_DIM], rsv[H_DIM];
            double concat[GRU_CONCAT];
            std::memcpy(concat, gru_in, sizeof(double) * EMBED_DIM);
            std::memcpy(concat + EMBED_DIM, a->h, sizeof(double) * H_DIM);

            #pragma omp parallel for
            for(int k=0; k<H_DIM; k++) {
                zsv[k] = sov_sigmoid(a->m->bz[k] + t_matmul(&a->m->Wz[k * GRU_CONCAT], concat, GRU_CONCAT, a->m->s_z));
                rsv[k] = sov_sigmoid(a->m->br[k] + t_matmul(&a->m->Wr[k * GRU_CONCAT], concat, GRU_CONCAT, a->m->s_r));
            }

            double concat_h[GRU_CONCAT];
            std::memcpy(concat_h, gru_in, sizeof(double) * EMBED_DIM);
            for(int j=0; j<H_DIM; j++) concat_h[EMBED_DIM + j] = rsv[j] * a->h[j];

            #pragma omp parallel for
            for(int k=0; k<H_DIM; k++) {
                double phc = a->m->bh[k] + t_matmul(&a->m->Wh[k * GRU_CONCAT], concat_h, GRU_CONCAT, a->m->s_h);
                h_new[k] = (1.0-zsv[k])*a->h[k] + zsv[k]*std::tanh(phc + (a->f->personality_bias[k] * 0.1f) + (a->m->hive_context[k] * 0.1f));
            }
            rmsnorm(h_new, H_DIM);
            std::memcpy(a->h, h_new, 8*H_DIM);
            for(int v=0; v<VOCAB; v++) {
                s_logits[v] = t_matmul(&a->m->Wo[v * HIDDEN], a->h, HIDDEN, a->m->s_o) + a->m->bo[v];
                s_logits[v] /= (temp + 1e-6);
            }
            softmax(s_logits, s_probs, VOCAB);
            
            std::discrete_distribution<int> d(s_probs, s_probs+VOCAB);
            int next = d(a->gen);
            if(next == 3) break;
            response.push_back(next);
            last_tok = next;
        }
        std::string res = g_tok->decode(response);
        std::strncpy(out, res.c_str(), 8191);
        return out;
    }

    SOV_API double* sovereign_agent_get_h(void* agent_ptr) {
        if(!agent_ptr) return nullptr;
        return ((Agent*)agent_ptr)->h;
    }

    SOV_API void sovereign_agent_commit_memory(void* agent_ptr) {
        if(!agent_ptr) return;
        Agent* a = (Agent*)agent_ptr;
        Fragment* f = a->f;
        
        // Snapshot current h into archive
        std::memcpy(&f->h_archive[f->archive_head * H_DIM], a->h, sizeof(double) * H_DIM);
        
        f->archive_head = (f->archive_head + 1) % f->ARCHIVE_CAP;
        if(f->archive_count < f->ARCHIVE_CAP) f->archive_count++;
    }

    SOV_API float sovereign_agent_recall_memory(void* agent_ptr, float alpha) {
        if(!agent_ptr || alpha <= 0.0f) return 0.0f;
        Agent* a = (Agent*)agent_ptr;
        Fragment* f = a->f;
        if(f->archive_count == 0) return 0.0f;

        int best_idx = -1;
        double max_sim = -1e9;

        // Search Archive for most similar state (Cosine Similarity based on normalized h)
        #pragma omp parallel for
        for(int i=0; i<f->archive_count; i++) {
            double dot = 0;
            for(int k=0; k<H_DIM; k++) dot += a->h[k] * f->h_archive[i * H_DIM + k];
            
            #pragma omp critical
            {
                if(dot > max_sim) {
                    max_sim = dot;
                    best_idx = i;
                }
            }
        }

        // Restore / Blend Memory
        if(best_idx != -1) {
            for(int k=0; k<H_DIM; k++) {
                a->h[k] = (1.0f - alpha)*a->h[k] + alpha*f->h_archive[best_idx * H_DIM + k];
            }
            rmsnorm(a->h, H_DIM); // Consistency guard
        }
        return (float)max_sim;
    }

    SOV_API void sovereign_free_agent(void* a) {
        Agent* ag = (Agent*)a;
        if(ag->f) delete ag->f;
        delete ag;
    }
}
