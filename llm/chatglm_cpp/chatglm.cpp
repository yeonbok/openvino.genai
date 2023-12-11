// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

const std::string sentences[] =
{
    //"my pc sound is too low",
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
};

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest & tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int64_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int64_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << openvino_extensions::unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

}

#define COMPILE_FROM_XML 1

int main(int argc, char* argv[]) try {
    if (argc < 5 || argc > 6) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<infer device>' <plugin_for_first_infer(default is same as infer device>)");
    }
    bool use_cpu_plugin_for_first_infer = false;
    std::cout << ov::get_openvino_version() << std::endl;
    auto model = argv[1];
    auto tokenizer_model = argv[2];
    auto detokenizer_model = argv[3];
    auto default_device = argv[4];
    auto first_infer_device = argv[5];
    if (argc == 6) {
        use_cpu_plugin_for_first_infer = (std::string(first_infer_device) == "CPU");
    }


    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    auto input_ids = tokenizer.get_tensor("input_ids");
    auto attention_mask = tokenizer.get_tensor("attention_mask");
    ov::InferRequest detokenizer = core.compile_model(detokenizer_model, "CPU").create_infer_request();
    constexpr size_t BATCH_SIZE = 1;

    double total_time = 0;
    int count = 0;
    auto startTime = Time::now();
    ov::CompiledModel compilemodel = core.compile_model(model, default_device, ov::cache_dir("tmp_cache"));
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "[" << default_device << "] Compile LLM model took " << duration_ms << " ms" << std::endl;

    // Extra cpu plugin for first inference
    ov::CompiledModel compilemodel_cpu;
    ov::InferRequest ireq_cpu;
    if (use_cpu_plugin_for_first_infer) {
        auto startTime = Time::now();
        compilemodel_cpu = std::move(core.compile_model(model, first_infer_device, ov::cache_dir("")));
        auto duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "[" << first_infer_device << "] Compile LLM model took " << duration_ms << " ms" << std::endl;
    }
    ov::InferRequest ireq = compilemodel.create_infer_request();
    if (use_cpu_plugin_for_first_infer) {
        ireq_cpu = std::move(compilemodel_cpu.create_infer_request());
    }

    auto inputs = use_cpu_plugin_for_first_infer ? compilemodel_cpu.inputs() : compilemodel.inputs();

    int selected_input = -1;
    if (std::getenv("INPUT_ID") != nullptr) {
        selected_input = std::atoi(std::getenv("INPUT_ID"));
        std::cout << "Run selected input " << selected_input << std::endl;
    } else {
        std::cout << "Run all input " << selected_input << std::endl;
    }
    for (int s = 0; s < (selected_input > 0 ? 1 : sizeof(sentences)); ++s) {
        const auto& input_text = selected_input < 0 ? sentences[s] : sentences[selected_input];
        total_time = 0;
        count = 0;
        auto prompt_text ="<|user|> " + input_text + " <|assitant|>";
        std::cout << " #### sentence: index " << prompt_text << std::endl;
        tokenize(tokenizer, prompt_text);
        input_ids = tokenizer.get_tensor("input_ids");
        attention_mask = tokenizer.get_tensor("attention_mask");
        std::cout << "input lenghth " << input_ids.get_size() << std::endl;

        for (auto &input : inputs) {
            for (const std::string& name : input.get_names()) {
                if (name.rfind("past_key_values", 0) == 0)
                {
                    ov::PartialShape shape = input.get_partial_shape().get_min_shape();
                    shape[1] = BATCH_SIZE;
                    use_cpu_plugin_for_first_infer ? ireq_cpu.get_tensor(input).set_shape(shape.get_shape()) : ireq.get_tensor(input).set_shape(shape.get_shape());;
                    break;
                }
            }
        }
        size_t vocab_size;
        float* logits;
        int64_t out_token;
        if (use_cpu_plugin_for_first_infer) {
            ireq_cpu.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
            ireq_cpu.get_tensor("attention_mask").set_shape(attention_mask.get_shape());
            std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq_cpu.get_tensor("input_ids").data<int64_t>());
            std::fill_n(ireq_cpu.get_tensor("attention_mask").data<int64_t>(), attention_mask.get_size(), 1);
            ireq_cpu.get_tensor("position_ids").set_shape(input_ids.get_shape());
            std::iota(ireq_cpu.get_tensor("position_ids").data<int64_t>(), ireq_cpu.get_tensor("position_ids").data<int64_t>() + ireq_cpu.get_tensor("position_ids").get_size(), 0);

            startTime = Time::now();
            ireq_cpu.infer();
            duration_ms = get_duration_ms_until_now(startTime);
            std::cout << "First token took " << duration_ms << " ms" << " (" << first_infer_device << ")" << std::endl;

            vocab_size = ireq_cpu.get_tensor("logits").get_shape().back();
            logits = ireq_cpu.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
            out_token = std::max_element(logits, logits + vocab_size) - logits;
        } else {
            ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
            ireq.get_tensor("attention_mask").set_shape(attention_mask.get_shape());
            std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
            std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), attention_mask.get_size(), 1);
            ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
            std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);

            startTime = Time::now();
            ireq.infer();
            duration_ms = get_duration_ms_until_now(startTime);
            std::cout << "First token took " << duration_ms << " ms" << " (" << default_device << ")" << std::endl;

            vocab_size = ireq.get_tensor("logits").get_shape().back();
            logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
            out_token = std::max_element(logits, logits + vocab_size) - logits;
        }
        ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, 1 });
        ireq.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

        constexpr int64_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the detokenizer for now
        //while (out_token != SPECIAL_EOS_TOKEN) {
        int gen_count = -1;
        if (std::getenv("GEN_COUNT") != nullptr) {
            gen_count = std::atoi(std::getenv("GEN_COUNT"));
            std::cout << "Generate " << gen_count << "tokens" << std::endl;
        } else {
            std::cout << "Generate tokens as many as possible" << std::endl;
        }
        while (gen_count > 0 ? count < gen_count : out_token != SPECIAL_EOS_TOKEN) {
            startTime = Time::now();
            ireq.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            if (count == 0 && use_cpu_plugin_for_first_infer) {
                ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq_cpu.get_tensor("attention_mask").get_shape()[1] + 1 });
                std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq_cpu.get_tensor("attention_mask").get_size(), 1);
                ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq_cpu.get_tensor("attention_mask").get_size() - 2;
            } else {
                ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
                std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
                ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
            }
            for (auto& input : inputs) {
                for (const std::string& name : input.get_names()) {
                    if (name.rfind("past_key_values", 0) == 0)
                    {
                        if (count == 0 && use_cpu_plugin_for_first_infer)
                            ireq.set_tensor(input, ireq_cpu.get_tensor("present" + name.substr(15)));
                        else
                            ireq.set_tensor(input, ireq.get_tensor("present" + name.substr(15)));
                        break;
                    }
                }
            }
            ireq.start_async();
            ireq.wait();
            duration_ms = get_duration_ms_until_now(startTime);
            count += 1;
            total_time += duration_ms;

            print_token(detokenizer, out_token);
            logits = ireq.get_tensor("logits").data<float>();
            out_token = std::max_element(logits, logits + vocab_size) - logits;
        }
        std::cout << '\n';

        if (count > 0) {
            std::cout << "[" << default_device << "] Other Avg inference took total " << total_time << " ms token num " << count << " avg " << total_time / (count) << " ms" << std::endl;
        }
        if (std::getenv("INPUT_ID") != nullptr)
            break;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
