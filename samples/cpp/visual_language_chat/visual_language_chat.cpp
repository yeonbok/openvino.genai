// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
//#include "openvino/genai/perf_metrics.hpp"
#include <filesystem>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES>");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    std::string device = "GPU";  // GPU can be used as well
    ov::AnyMap enable_compile_cache;
//    if (device == "GPU") {
//        // Cache compiled models on disk for GPU to save time on the
//        // next run. It's not beneficial for CPU.
//        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
//    }
    std::cout << "Run for " << device << std::endl;
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt;

    pipe.start_chat();
    std::cout << "question:\n";

    std::getline(std::cin, prompt);
    std::cout << "Prompt : " << prompt << std::endl;
    std::cout << std::fixed << std::setprecision(3) << std::endl;
    auto pipeline_start_time = std::chrono::steady_clock::now();
    auto res = pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));
    auto pipeline_end_time = std::chrono::steady_clock::now();
    auto pipeline_total_time = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_time - pipeline_start_time).count();
    std::cout << "\n======================================" << std::endl;
    std::cout << "# input tokens : " << res.perf_metrics.get_num_input_tokens() << std::endl;
    std::cout << "# generated tokens : " << res.perf_metrics.get_num_generated_tokens() << std::endl;
    std::cout << "pipeline_total_time : " << pipeline_total_time / 1000 << " ms" << std::endl;
//    std::cout << "generate_duration_mean : " << res.perf_metrics.get_generate_duration().mean <<  "mcs" << std::endl;
//    std::cout << "inference_duration_mean : " << res.perf_metrics.get_inference_duration().mean << " mcs " << std::endl;
//    std::cout << "tokenization_duration_mean : " << res.perf_metrics.get_tokenization_duration().mean << " mcs " << std::endl;
//    std::cout << "detokenization_duration_mean : " << res.perf_metrics.get_detokenization_duration().mean << " mcs " << std::endl;
    auto vlm_perf = static_cast<ov::genai::VLMPerfMetrics>(res.perf_metrics);
    auto raw_perf = vlm_perf.raw_metrics;
    for (auto i = 0; i < vlm_perf.vlm_raw_metrics.prepare_embeddings_durations.size(); ++i) {
        std::cout << "prepare_embeddings[" << i << "] : " << vlm_perf.vlm_raw_metrics.prepare_embeddings_durations[i].count()/1000 << " ms " << std::endl;
    }
    for (auto i = 0; i < raw_perf.generate_durations.size(); ++i) {
        std::cout << "gen_durations[" << i << "] : " << raw_perf.generate_durations[i].count()/1000 << " ms " << std::endl;
    }
 
    for (auto i = 0; i < raw_perf.tokenization_durations.size(); ++i) {
        std::cout << "token[" << i << "] : " << raw_perf.tokenization_durations[i].count() << " mcs " << std::endl;
    }
    for (auto i = 0; i < raw_perf.detokenization_durations.size(); ++i) {
        std::cout << "detoken[" << i << "] : " << raw_perf.detokenization_durations[i].count() << " mcs " << std::endl;
    }


    std::cout << "\n----------\n"
        "question:\n";
    while (std::getline(std::cin, prompt)) {
        std::cout << "Prompt : " << prompt << std::endl;
        auto res = pipe.generate(prompt,
                      ov::genai::generation_config(generation_config),
                      ov::genai::streamer(print_subword));
        std::cout << "\n----------\n"
            "question:\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
