"""
    entrance.py - user entrance for the platform
    author: Qizheng Zhang (qizhengz@uchicago.edu)
            Kuntai Du (kuntai@uchicago.edu)
"""

import os
import subprocess
import yaml
import sys

# dirty fix
sys.path.append('../')

def load_configuration():
    """read configuration information from yaml file

    Returns:
        dict: information of the yaml file
    """
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader)
    return config_info


def execute_single(single_instance):
    """execute an atomic instance

    Args:
        single_instance (dict): the instance to be executed
    """
    # unpacking
    baseline = single_instance['method']

    # to be fixed:
    # gt and mpeg must be run for dds regardless of whether they are in config

    # branching based on baselines
    if baseline == 'gt':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')

        # skip if result file already exists
        result_file_name = f"{video_name}_gt"
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        # otherwise, start execution
        else:
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'

            subprocess.run(['python', '../play_video.py', 
                            yaml.dump(single_instance)])

    # assume we are running emulation
    elif baseline == 'mpeg':
        # unpacking
        video_name = single_instance['video_name']
        mpeg_qp = single_instance['low_qp']
        mpeg_resolution = single_instance['low_resolution']
        original_images_dir = os.path.join(data_dir, video_name, 'src')

        # skip if result file already exists
        result_file_name = f"{video_name}_mpeg_{mpeg_resolution}_{mpeg_qp}"
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'
            single_instance['ground_truth'] = f'results/{video_name}_gt'

            subprocess.run(['python', '../play_video.py',
                            yaml.dump(single_instance)])

    elif baseline == 'dds':
        # unpacking
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        low_qp = single_instance['low_qp']
        high_qp = single_instance['high_qp']
        low_res = single_instance['low_resolution']
        high_res = single_instance['high_resolution']
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']
        context = single_instance['context']
        padding = single_instance['padding']
        iou_thresh = single_instance['iou_thresh']
        grouping = single_instance['grouping']
        ctx_mode = single_instance['ctx_mode']
        max_ctx = single_instance['max_ctx']
        merge_rpn = 'merge-rpn' if single_instance['merge_rpn'] else 'nomerge-rpn'
        merge_thresh = single_instance['merge_thresh']
        reduced = '_reduced' if single_instance['reduced'] else ''
        used_context = '_ctxconvert' if single_instance['use_context'] else '_noctxconvert'
        normalized = '_normalized' if single_instance['normalize'] else ''

        # skip if result file already exists
        # You could customize the way to serialize the parameters into filename by yourself
        result_file_name = (f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                            f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                            f"{prune_score}_{objfilter_iou}_{size_obj}_"
                            f"{context}_{padding}{normalized}_{iou_thresh}{reduced}{used_context}_"
                            f"{grouping}_{ctx_mode}_{max_ctx}_{merge_rpn}_{merge_thresh}")
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'
            single_instance['ground_truth'] = f'results/{video_name}_gt'
            single_instance['low_results_path'] = f'results/{video_name}_mpeg_{low_res}_{low_qp}'
            single_instance['base_dds_path'] = (f"results/"
                                                f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                                                f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                                                f"{prune_score}_{objfilter_iou}_{size_obj}")
            single_instance['low_dds_path'] = f'results/{video_name}_dds_low'

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5000'
                
            subprocess.run(['python', '../play_video.py',
                                yaml.dump(single_instance)])
    
    elif baseline == 'base-dds':
        video_name = single_instance['video_name']
        original_images_dir = os.path.join(data_dir, video_name, 'src')
        low_qp = single_instance['low_qp']
        high_qp = single_instance['high_qp']
        low_res = single_instance['low_resolution']
        high_res = single_instance['high_resolution']
        rpn_enlarge_ratio = single_instance['rpn_enlarge_ratio']
        batch_size = single_instance['batch_size']
        prune_score = single_instance['prune_score']
        objfilter_iou = single_instance['objfilter_iou']
        size_obj = single_instance['size_obj']

        # skip if result file already exists
        # You could customize the way to serialize the parameters into filename by yourself
        result_file_name = (f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                            f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                            f"{prune_score}_{objfilter_iou}_{size_obj}")
        if single_instance['overwrite'] == False and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            single_instance['video_name'] = f'results/{result_file_name}'
            single_instance['high_images_path'] = f'{original_images_dir}'
            single_instance['outfile'] = 'stats'
            single_instance['ground_truth'] = f'results/{video_name}_gt'
            single_instance['low_results_path'] = f'results/{video_name}_mpeg_{low_res}_{low_qp}'
            single_instance['base_dds_path'] = (f"results/"
                                                f"{video_name}_dds_{low_res}_{high_res}_{low_qp}_{high_qp}_"
                                                f"{rpn_enlarge_ratio}_twosides_batch_{batch_size}_"
                                                f"{prune_score}_{objfilter_iou}_{size_obj}")
            single_instance['low_dds_path'] = f'results/{video_name}_dds_low'

            if single_instance["mode"] == 'implementation':
                assert single_instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5000'
                
            subprocess.run(['python', '../play_video.py',
                                yaml.dump(single_instance)])
    

def parameter_sweeping(instances, new_instance, keys):
    """recursive function for parameter sweeping

    Args:
        instances (dict): the instance in process
        new_instance (dict): recursive parameter
        keys (list): keys of the instance in process
    """
    if keys == []: # base case
        execute_single(new_instance)
    else: # recursive step
        curr_key = keys[0]
        if (isinstance(instances[curr_key], list)): 
            # need parameter sweeping
            for each_parameter in instances[curr_key]:
                # replace the list with a single value
                new_instance[curr_key] = each_parameter
                # proceed with the other parameters in keys
                parameter_sweeping(instances, new_instance, keys[1:])
        else: # no need for parameter sweeping
            new_instance[curr_key] = instances[curr_key]
            parameter_sweeping(instances, new_instance, keys[1:])


def execute_all(config_info):
    """execute all instances based on user's config info and default config info

    Args:
        config_info (dict): configuration information from the yaml file
    """
    all_instances = config_info['instances']
    default = config_info['default']
    print('found %d instances' % (len(all_instances)))
    for single_instance in all_instances:

        # propagate default config to current instance
        for key in default.keys():
            if key not in single_instance.keys():
                single_instance[key] = default[key]

        keys = list(single_instance.keys())
        new_instance = {} # initially empty
        parameter_sweeping(single_instance, new_instance, keys)


if __name__ == "__main__":
    # load configuration information (only once)
    config_info = load_configuration()
    data_dir = config_info['data_dir']
    
    execute_all(config_info)