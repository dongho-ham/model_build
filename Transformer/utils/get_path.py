import os
import time

def get_save_folder_path(args):
    base = args.result_folder
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    save_folder_path = os.path.join(base, current_time)

    return save_folder_path
