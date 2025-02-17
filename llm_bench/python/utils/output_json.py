import json


def write_result(report_file, model, framework, device, model_args, iter_data_list, pretrain_time, model_precision):
    metadata = {'model': model, 'framework': framework, 'device': device, 'precision': model_precision,
                'num_beams': model_args['num_beams'], 'batch_size': model_args['batch_size']}
    result = []
    total_iters = len(iter_data_list)
    for i in range(total_iters):
        iter_data = iter_data_list[i]
        generation_time = iter_data['generation_time']
        latency = iter_data['latency']
        first_latency = iter_data['first_token_latency']
        other_latency = iter_data['other_tokens_avg_latency']
        first_token_infer_latency = iter_data['first_token_infer_latency']
        other_token_infer_latency = iter_data['other_tokens_infer_avg_latency']
        rss_mem = iter_data['max_rss_mem_consumption']
        shared_mem = iter_data['max_shared_mem_consumption']

        result_md5 = []
        for idx_md5 in range(len(iter_data['result_md5'])):
            result_md5.append(iter_data['result_md5'][idx_md5])

        res_data = {
            'iteration': iter_data['iteration'],
            'input_size': iter_data['input_size'],
            'infer_count': iter_data['infer_count'],
            'generation_time': round(generation_time, 5) if generation_time != '' else generation_time,
            'output_size': iter_data['output_size'],
            'latency': round(latency, 5) if latency != '' else latency,
            'result_md5': result_md5,
            'first_latency': round(first_latency, 5) if first_latency != '' else first_latency,
            'second_avg_latency': round(other_latency, 5) if other_latency != '' else other_latency,
            'first_infer_latency': round(first_token_infer_latency, 5) if first_token_infer_latency != '' else first_token_infer_latency,
            'second_infer_avg_latency': round(other_token_infer_latency, 5) if other_token_infer_latency != '' else other_token_infer_latency,
            'max_rss_mem': round(rss_mem, 5) if rss_mem != '' else -1,
            'max_shared_mem': round(shared_mem, 5) if shared_mem != '' else -1,
            'prompt_idx': iter_data['prompt_idx'],
        }

        result.append(res_data)

    output_result = {'metadata': metadata, "perfdata": {'compile_time': pretrain_time, 'results': result}}

    with open(report_file, 'w') as outfile:
        json.dump(output_result, outfile)
