import time
from .color_text import colored
import multiprocessing as mp
import torch

from .run_func import device_run
from mp_manager.config import *


class Handler(object):
    def __init__(self, *, process_id, task_id=None,
                 device_id=None, verbose=100, device_semaphore=None,
                 uni_process=False):
        """
        Handler is provided to functions as a helper object. Currently it could be used to
        1. Fancy message printing. The printed message will be labeled with the process id and timestamps,
        and will be stored into mailbox, which will be finally returned to the main process, so you don't
        need to worry that they will be lost.
        2. Device management. Every time you want to use a device, you have to acquire a semaphore from
        handler. This is done through calling the handler's use_device_to_run method.
        :param int process_id: Process ID.
        :param int task_id: Task ID.
        :param int device_id: Device ID.
        :param int verbose: Verbose level.
        :param mp.Semaphore device_semaphore: Device resource semaphore.
        :param bool uni_process: Only one process? If so, no need to acquire or release device.
        """

        # -1 for main process
        self.process_id = process_id
        self.task_id = task_id
        self.device_id = device_id
        # start time
        self.verbose = verbose
        self.since = time.time()
        self.device_semaphore = device_semaphore
        self.uni_process = uni_process

        # To save the print messages.
        self.msg_archive = list()

    def print(self, msg='', verbose=0):
        """
        Print messages with labels.
        :param str msg: Message itself.
        :param int verbose: Verbose level.
        """
        elapse = time.time() - self.since
        msg = str(msg)
        # Save the message regardless whether it's too verbose or not.
        self.msg_archive.append([elapse, msg])
        if verbose > self.verbose:
            return

        # Use fancy mode.
        if self.process_id >= 0:
            prefix = r'[Child {process_id:>2}|{elapse:6.0f}]: '
        else:
            prefix = r'[Parent  |{elapse:6.0f}]: '

        prefix = prefix.format(elapse=elapse, process_id=self.process_id)
        prefix_color = parent_color if self.process_id < 0 else child_color
        to_print = colored(prefix, prefix_color) + msg
        print(to_print)

    def mail_box(self):
        """
        Get all the printed msgs.
        """
        return self.msg_archive

    def acquire_device(self, silent=False, cuda_request=True):
        """
        Acquire the device semaphore.
        :param bool silent: Don't output any message.
        :param bool cuda_request: Use torch.cuda.set_device to setup.
        """
        if self.device_semaphore is None or self.uni_process:
            return

        self.device_semaphore.acquire()
        if cuda_request:
            torch.cuda.set_device(self.device_id)

        if self.verbose < 2 or silent:
            return
        msg = 'acquire device {device_id:}'\
            .format(process_id=self.process_id, device_id=self.device_id)
        self.print(colored(msg, parent_color))

    def release_device(self, silent=False, cuda_request=True):
        """
        Release the device semaphore.
        :param bool silent: Don't output any message.
        :param bool cuda_request: clean all torch cache before release. Not necessary to do.
        """
        if self.device_semaphore is None or self.uni_process:
            return
        if cuda_request:
            torch.cuda.empty_cache()
        self.device_semaphore.release()

        if self.verbose < 2 or silent:
            return
        msg = 'release device {device_id:}' \
            .format(process_id=self.process_id, device_id=self.device_id)
        self.print(colored(msg, parent_color))

    def use_device_to_run(self, func_to_call, func_args):
        """
        Use device to run functions.
        When called, this function will acquire the device first, then do the job.
        :param func func_to_call: The function hat will be called.
        :param dict func_args: The args for func_args.
        :return:
        """
        if self.uni_process:
            return func_to_call(**func_args)
        return device_run(self, func_to_call, func_args)
