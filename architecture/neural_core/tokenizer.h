#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cctype>

// Special token IDs
static const int TOK_PAD   = 0;
static const int TOK_UNK   = 1;
static const int TOK_START = 2;
static const int TOK_END   = 3;

class WordTokenizer {
public:
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    int vocab_size;

    WordTokenizer() : vocab_size(4) {
        reset_specials();
    }

    void reset_specials() {
        word_to_id.clear();
        id_to_word.clear();
        id_to_word.push_back("[PAD]");
        id_to_word.push_back("[UNK]");
        id_to_word.push_back("[START]");
        id_to_word.push_back("[END]");
        word_to_id["[PAD]"]   = TOK_PAD;
        word_to_id["[UNK]"]   = TOK_UNK;
        word_to_id["[START]"] = TOK_START;
        word_to_id["[END]"]   = TOK_END;
        vocab_size = 4;
    }

    // --- Build vocabulary from a text corpus (Updated for Phase 3) ---
    void build_vocab(const std::string& text, int max_vocab = 5000) {
        reset_specials();
        std::unordered_map<std::string, int> freq;
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            word = normalize(word);
            if (!word.empty()) freq[word]++;
        }

        std::vector<std::pair<std::string, int>> sorted_words(freq.begin(), freq.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < sorted_words.size(); i++) {
            if (vocab_size >= max_vocab) break;
            const std::string& w = sorted_words[i].first;
            if (word_to_id.find(w) == word_to_id.end()) {
                word_to_id[w] = vocab_size;
                id_to_word.push_back(w);
                vocab_size++;
            }
        }
    }

    bool save_vocab(const std::string& path) {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        // Don't save specials to the file, keep the file clean
        for (size_t i = 4; i < id_to_word.size(); i++) f << id_to_word[i] << "\n";
        f.close();
        return true;
    }

    bool load_vocab(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        reset_specials(); // Ensure specials are at 0-3
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (word_to_id.find(line) == word_to_id.end()) {
                word_to_id[line] = vocab_size;
                id_to_word.push_back(line);
                vocab_size++;
            }
        }
        std::cout << "[TOKENIZER] Sane Load complete. Vocab: " << vocab_size << " (Specials at 0-3)\n";
        return true;
    }

    // --- Phase 3: Greedy MaxMatch Subword Encoding ---
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(TOK_START);
        
        std::string normalized = normalize_full(text);
        size_t start = 0;
        while (start < normalized.length()) {
            bool found = false;
            // Greedy Longest Match
            for (size_t len = std::min((size_t)20, normalized.length() - start); len > 0; len--) {
                std::string sub = normalized.substr(start, len);
                auto it = word_to_id.find(sub);
                if (it != word_to_id.end()) {
                    tokens.push_back(it->second);
                    start += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Fallback: Skip one character or treat as [UNK]
                // (In Phase 3.5 we should add ALL ASCII to the vocab to avoid this)
                tokens.push_back(TOK_UNK);
                start++;
            }
        }
        
        tokens.push_back(TOK_END);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int t : tokens) {
            if (t == TOK_PAD || t == TOK_START) continue;
            if (t == TOK_END) break;
            if (t >= 0 && t < (int)id_to_word.size()) {
                // Heuristic: If it's a subword/punc, don't necessarily add space
                // For Phase 3 simplicity, we still add space between chunks
                if (!result.empty()) result += " ";
                result += id_to_word[t];
            }
        }
        return result;
    }

    std::string decode_token(int id) {
        if (id >= 0 && id < (int)id_to_word.size()) return id_to_word[id];
        return "[UNK]";
    }

private:
    // Simple normalization for vocab building
    std::string normalize(const std::string& raw) {
        std::string w;
        for (char c : raw) {
            if (std::isalnum(c) || c == '\'') w += std::tolower(c);
        }
        return w;
    }

    // Phase 3: Full normalization preserving symbols
    std::string normalize_full(const std::string& raw) {
        std::string w;
        for (char c : raw) {
            if (std::isspace(c)) continue;
            w += std::tolower(c);
        }
        return w;
    }
};
