import multiprocessing as mp
import torch
import time


def run(task):
    """
    Accept a task, run it and return the results.
    :return: The results of one task.
    """
    return task()


def _device_run(handler, func_to_call, func_args, queue):
    """
    Helper function for device_run.
    :param mp.Queue queue: It's where the results will be saved.
    """
    torch.cuda.set_device(handler.device_id)
    rst = func_to_call(**func_args)
    torch.cuda.empty_cache()
    queue.put(rst)


def device_run(handler, func_to_call, func_args):
    """
    Use device to run a sub-task.
    :param Handler handler: The handler of the task, which is used to acquire the device
    semaphore.
    :param func func_to_call: The function to be called. CAUTION: This is different from
    the func_to_call in Task class.
    :param dict func_args: Args for the func_to_call.
    :return: The result of func_to_call
    """
    queue = mp.Queue()
    process = mp.Process(target=_device_run, args=[handler, func_to_call, func_args, queue])
    process.daemon = True
    handler.acquire_device(cuda_request=False)
    process.start()
    rst = queue.get()
    process.terminate()
    time.sleep(1)
    handler.release_device(cuda_request=False)
    return rst
