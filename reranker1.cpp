// C++ version of reranker1.py using ONNX Runtime
// This file implements a text reranker using a CrossEncoder model via ONNX Runtime
//
// Prerequisites:
// - ONNX Runtime C++ headers and library
// - An ONNX model exported from cross-encoder/ms-marco-MiniLM-L6-v2
// - A vocabulary file (vocab.txt) from the model
//
// Build example:
//   cmake -B build && cmake --build build
//
// Usage:
//   ./reranker1 <model.onnx> <vocab.txt>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Simple tokenizer that mimics basic BERT WordPiece tokenization
class SimpleTokenizer {
public:
    explicit SimpleTokenizer(const std::string& vocab_path) {
        loadVocab(vocab_path);
    }

    // Load vocabulary from file (one token per line)
    void loadVocab(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
        }

        std::string line;
        int32_t idx = 0;
        while (std::getline(file, line)) {
            // Remove trailing carriage return if present (Windows line endings)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            vocab_[line] = idx++;
        }

        // Ensure special tokens exist
        if (vocab_.find("[CLS]") == vocab_.end() ||
            vocab_.find("[SEP]") == vocab_.end() ||
            vocab_.find("[UNK]") == vocab_.end() ||
            vocab_.find("[PAD]") == vocab_.end()) {
            throw std::runtime_error("Vocabulary missing required special tokens");
        }

        cls_id_ = vocab_["[CLS]"];
        sep_id_ = vocab_["[SEP]"];
        unk_id_ = vocab_["[UNK]"];
        pad_id_ = vocab_["[PAD]"];
    }

    // Basic tokenization: lowercase, split on whitespace and punctuation
    std::vector<std::string> basicTokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string current_token;

        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
            } else if (std::ispunct(static_cast<unsigned char>(c))) {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
                tokens.push_back(std::string(1, c));
            } else {
                current_token += std::tolower(static_cast<unsigned char>(c));
            }
        }

        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }

        return tokens;
    }

    // WordPiece tokenization
    std::vector<int32_t> wordPieceTokenize(const std::string& word) const {
        std::vector<int32_t> output_ids;

        if (word.empty()) {
            return output_ids;
        }

        // Check if the whole word is in vocabulary
        if (vocab_.find(word) != vocab_.end()) {
            output_ids.push_back(vocab_.at(word));
            return output_ids;
        }

        // Try to break down into subwords
        size_t start = 0;
        while (start < word.length()) {
            size_t end = word.length();
            bool found = false;

            while (start < end) {
                std::string substr = word.substr(start, end - start);
                if (start > 0) {
                    substr = "##" + substr;
                }

                if (vocab_.find(substr) != vocab_.end()) {
                    output_ids.push_back(vocab_.at(substr));
                    found = true;
                    start = end;
                    break;
                }
                end--;
            }

            if (!found) {
                // Character not found, use [UNK]
                output_ids.push_back(unk_id_);
                start++;
            }
        }

        return output_ids;
    }

    // Encode a pair of texts (query, passage) for CrossEncoder
    // Returns input_ids, attention_mask, token_type_ids
    struct EncodedInput {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids;
    };

    EncodedInput encode(const std::string& text_a, const std::string& text_b,
                        size_t max_length = 512) const {
        EncodedInput result;

        // Tokenize both texts
        std::vector<std::string> tokens_a = basicTokenize(text_a);
        std::vector<std::string> tokens_b = basicTokenize(text_b);

        // Convert to IDs with WordPiece
        std::vector<int32_t> ids_a;
        for (const auto& token : tokens_a) {
            auto sub_ids = wordPieceTokenize(token);
            ids_a.insert(ids_a.end(), sub_ids.begin(), sub_ids.end());
        }

        std::vector<int32_t> ids_b;
        for (const auto& token : tokens_b) {
            auto sub_ids = wordPieceTokenize(token);
            ids_b.insert(ids_b.end(), sub_ids.begin(), sub_ids.end());
        }

        // Truncate if necessary (simple truncation from the end)
        // Reserve 3 spots for [CLS], [SEP], [SEP]
        size_t max_tokens = max_length - 3;
        size_t total_tokens = ids_a.size() + ids_b.size();

        if (total_tokens > max_tokens) {
            // Truncate the longer sequence first
            while (ids_a.size() + ids_b.size() > max_tokens) {
                if (ids_b.size() > ids_a.size()) {
                    ids_b.pop_back();
                } else {
                    ids_a.pop_back();
                }
            }
        }

        // Build final sequence: [CLS] text_a [SEP] text_b [SEP]
        result.input_ids.push_back(cls_id_);
        result.token_type_ids.push_back(0);
        result.attention_mask.push_back(1);

        for (int32_t id : ids_a) {
            result.input_ids.push_back(id);
            result.token_type_ids.push_back(0);
            result.attention_mask.push_back(1);
        }

        result.input_ids.push_back(sep_id_);
        result.token_type_ids.push_back(0);
        result.attention_mask.push_back(1);

        for (int32_t id : ids_b) {
            result.input_ids.push_back(id);
            result.token_type_ids.push_back(1);
            result.attention_mask.push_back(1);
        }

        result.input_ids.push_back(sep_id_);
        result.token_type_ids.push_back(1);
        result.attention_mask.push_back(1);

        return result;
    }

    int32_t getPadId() const { return pad_id_; }

private:
    std::unordered_map<std::string, int32_t> vocab_;
    int32_t cls_id_ = 0;
    int32_t sep_id_ = 0;
    int32_t unk_id_ = 0;
    int32_t pad_id_ = 0;
};

// CrossEncoder class using ONNX Runtime
class CrossEncoder {
public:
    CrossEncoder(const std::string& model_path, const std::string& vocab_path)
        : tokenizer_(vocab_path),
          env_(ORT_LOGGING_LEVEL_WARNING, "CrossEncoder"),
          session_(nullptr) {
        
        Ort::SessionOptions session_options;
        // Use 0 to let ONNX Runtime determine optimal thread count based on available cores
        session_options.SetIntraOpNumThreads(0);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(std::string(name.get()));
        }

        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(name.get()));
        }

        // Build c_str pointers for ONNX Runtime API
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }
    }

    // Predict scores for query-passage pairs
    std::vector<float> predict(const std::vector<std::pair<std::string, std::string>>& pairs) {
        if (pairs.empty()) {
            return {};
        }

        // Encode all pairs
        std::vector<SimpleTokenizer::EncodedInput> encoded;
        size_t max_len = 0;

        for (const auto& pair : pairs) {
            auto enc = tokenizer_.encode(pair.first, pair.second);
            max_len = std::max(max_len, enc.input_ids.size());
            encoded.push_back(std::move(enc));
        }

        // Pad all sequences to the same length
        int32_t pad_id = tokenizer_.getPadId();
        for (auto& enc : encoded) {
            while (enc.input_ids.size() < max_len) {
                enc.input_ids.push_back(pad_id);
                enc.attention_mask.push_back(0);
                enc.token_type_ids.push_back(0);
            }
        }

        // Create batched tensors
        size_t batch_size = pairs.size();
        std::vector<int64_t> input_ids_flat;
        std::vector<int64_t> attention_mask_flat;
        std::vector<int64_t> token_type_ids_flat;

        for (const auto& enc : encoded) {
            input_ids_flat.insert(input_ids_flat.end(), 
                                  enc.input_ids.begin(), enc.input_ids.end());
            attention_mask_flat.insert(attention_mask_flat.end(), 
                                       enc.attention_mask.begin(), enc.attention_mask.end());
            token_type_ids_flat.insert(token_type_ids_flat.end(), 
                                       enc.token_type_ids.begin(), enc.token_type_ids.end());
        }

        // Create ONNX tensors
        std::array<int64_t, 2> input_shape = {static_cast<int64_t>(batch_size), 
                                               static_cast<int64_t>(max_len)};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_flat.data(), input_ids_flat.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask_flat.data(), attention_mask_flat.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, token_type_ids_flat.data(), token_type_ids_flat.size(),
            input_shape.data(), input_shape.size()));

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), input_tensors.data(), input_tensors.size(),
            output_names_.data(), output_names_.size());

        // Extract scores from output
        // The CrossEncoder model outputs logits of shape [batch_size, 1] or [batch_size]
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::vector<float> scores;
        if (output_shape.size() == 2) {
            // Shape [batch_size, 1] - take the first column
            for (size_t i = 0; i < batch_size; i++) {
                scores.push_back(output_data[i * output_shape[1]]);
            }
        } else {
            // Shape [batch_size]
            for (size_t i = 0; i < batch_size; i++) {
                scores.push_back(output_data[i]);
            }
        }

        return scores;
    }

    // Rank structure for results
    struct RankResult {
        size_t corpus_id;
        float score;
        std::string text;
    };

    // Rank passages for a query
    std::vector<RankResult> rank(const std::string& query, 
                                  const std::vector<std::string>& passages,
                                  bool return_documents = false) {
        // Create query-passage pairs
        std::vector<std::pair<std::string, std::string>> pairs;
        for (const auto& passage : passages) {
            pairs.emplace_back(query, passage);
        }

        // Get scores
        std::vector<float> scores = predict(pairs);

        // Create ranking results
        std::vector<RankResult> results;
        for (size_t i = 0; i < passages.size(); i++) {
            RankResult result;
            result.corpus_id = i;
            result.score = scores[i];
            result.text = return_documents ? passages[i] : "";
            results.push_back(result);
        }

        // Sort by score descending
        std::sort(results.begin(), results.end(),
                  [](const RankResult& a, const RankResult& b) {
                      return a.score > b.score;
                  });

        return results;
    }

private:
    SimpleTokenizer tokenizer_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

// Print usage information
void printUsage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model.onnx> <vocab.txt>\n";
    std::cerr << "\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  model.onnx  - Path to the ONNX model file (exported from cross-encoder/ms-marco-MiniLM-L6-v2)\n";
    std::cerr << "  vocab.txt   - Path to the vocabulary file from the model\n";
    std::cerr << "\n";
    std::cerr << "Example:\n";
    std::cerr << "  " << program_name << " ./model.onnx ./vocab.txt\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string vocab_path = argv[2];

    try {
        // 1. Load a pretrained CrossEncoder model (ONNX version)
        std::cout << "Loading model from: " << model_path << std::endl;
        std::cout << "Loading vocabulary from: " << vocab_path << std::endl;

        CrossEncoder model(model_path, vocab_path);

        // The texts for which to predict similarity scores
        std::string query = "How many people live in Berlin?";
        std::vector<std::string> passages = {
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
            "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs."
        };

        // 2a. Predict scores for pairs of texts
        std::vector<std::pair<std::string, std::string>> pairs;
        for (const auto& passage : passages) {
            pairs.emplace_back(query, passage);
        }

        std::vector<float> scores = model.predict(pairs);

        std::cout << "[";
        for (size_t i = 0; i < scores.size(); i++) {
            std::cout << scores[i];
            if (i < scores.size() - 1) std::cout << " ";
        }
        std::cout << "]" << std::endl;
        // Expected output similar to: [8.607139 5.506266 6.352977]

        std::cout << "#####################################################################################################" << std::endl;

        // 2b. Rank a list of passages for a query
        auto ranks = model.rank(query, passages, true);

        std::cout << "Query: " << query << std::endl;
        for (const auto& rank : ranks) {
            std::cout << "- #" << rank.corpus_id << " (" 
                      << std::fixed << std::setprecision(2) << rank.score 
                      << "): " << rank.text << std::endl;
        }
        /*
        Expected output similar to:
        Query: How many people live in Berlin?
        - #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
        - #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
        - #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
        */

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
