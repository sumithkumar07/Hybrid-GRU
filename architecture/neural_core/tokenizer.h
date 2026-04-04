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
        // Reserve special tokens
        id_to_word.push_back("[PAD]");
        id_to_word.push_back("[UNK]");
        id_to_word.push_back("[START]");
        id_to_word.push_back("[END]");
        word_to_id["[PAD]"]   = TOK_PAD;
        word_to_id["[UNK]"]   = TOK_UNK;
        word_to_id["[START]"] = TOK_START;
        word_to_id["[END]"]   = TOK_END;
    }

    // --- Build vocabulary from a text corpus ---
    void build_vocab(const std::string& text, int max_vocab = 5000) {
        // Count word frequencies
        std::unordered_map<std::string, int> freq;
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            word = normalize(word);
            if (!word.empty()) freq[word]++;
        }

        // Sort by frequency (descending)
        std::vector<std::pair<std::string, int>> sorted_words(freq.begin(), freq.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // Take top N words (minus the 4 special tokens)
        int count = 0;
        for (size_t i = 0; i < sorted_words.size(); i++) {
            if (count >= max_vocab - 4) break;
            const std::string& w = sorted_words[i].first;
            if (word_to_id.find(w) == word_to_id.end()) {
                int id = (int)id_to_word.size();
                word_to_id[w] = id;
                id_to_word.push_back(w);
                count++;
            }
        }
        vocab_size = (int)id_to_word.size();
        std::cout << "[TOKENIZER] Built vocabulary: " << vocab_size << " words\n";
    }

    // --- Save vocabulary to file ---
    bool save_vocab(const std::string& path) {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        for (auto& w : id_to_word) f << w << "\n";
        f.close();
        return true;
    }

    // --- Load vocabulary from file ---
    bool load_vocab(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        word_to_id.clear();
        id_to_word.clear();
        std::string line;
        int id = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            word_to_id[line] = id;
            id_to_word.push_back(line);
            id++;
        }
        vocab_size = id;
        f.close();
        std::cout << "[TOKENIZER] Loaded vocabulary: " << vocab_size << " words from " << path << "\n";
        return true;
    }

    // --- Encode text to token IDs ---
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(TOK_START);
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            word = normalize(word);
            if (word.empty()) continue;
            auto it = word_to_id.find(word);
            tokens.push_back(it != word_to_id.end() ? it->second : TOK_UNK);
        }
        tokens.push_back(TOK_END);
        return tokens;
    }

    // --- Decode token IDs to text ---
    std::string decode(const std::vector<int>& tokens) {
        std::string result;
        for (int t : tokens) {
            if (t == TOK_PAD || t == TOK_START) continue;
            if (t == TOK_END) break;
            if (t >= 0 && t < (int)id_to_word.size()) {
                if (!result.empty()) result += " ";
                result += id_to_word[t];
            }
        }
        return result;
    }

    // --- Decode a single token ---
    std::string decode_token(int id) {
        if (id >= 0 && id < (int)id_to_word.size()) return id_to_word[id];
        return "[UNK]";
    }

private:
    // Lowercase and strip trailing punctuation (keep the word clean)
    std::string normalize(const std::string& raw) {
        std::string w;
        for (char c : raw) {
            if (std::isalpha(c) || c == '\'') {
                w += std::tolower(c);
            }
        }
        return w;
    }
};
