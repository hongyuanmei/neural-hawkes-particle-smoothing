import multiprocessing as mp
import torch
import os

from .task import Task
from .handler import Handler
from .run_func import run
from .my_mp import MyPool as Pool


class Manager(object):
    def __init__(self, *, device_count=None, verbose=3, max_processes=None,
                 max_process_per_device=1, visible_devices=None, marcc=False):
        """
        Manager for multiprocessing.
        Since some jobs require both cpu and gpu computation, we can do them in parallel.
        Using TorchMpMgr, you could do the following things:
        1. Multiprocessing: Distribute your jobs to multiple CPUs.
        2. Flexible GPU management: Set a maximum number of jobs in parallel for each GPU in order not
        to exceed the GPU memory limit.
        3. Friendly interface: Highlight for different types of message.
        :param int device_count: Maximum number of GPUs that could be used. If not set, manager will call
        torch.cuda.device_count() to get the GPU number.
        :param int verbose: Verbose level for output message. The higher, the more verbose. By default it
        outputs all the messages.
        :param int max_processes: Maximum number of workers. By default, manager will call mp.count() to
        get the number processor cores.
        :param int max_process_per_device: The maximum number of GPU jobs that could be run in a single
        GPU at the same time. By default, it's 1, which means each device could only do one job at a time.
        :param list[int] visible_devices: The visible GPUs. The environment variable CUDA_VISIBLE_DEVICES
        will be set if this is given. By default every device is visible.
        :param bool marcc: Marcc mode. Currently it's not supported.
        """

        assert not marcc, "MARCC is not supported now"

        # Torch doesn't support fork mode
        mp.set_start_method('spawn')
        # The manager process (main process) is assigned with a special process id, -1.
        self.handler = Handler(process_id=-1, verbose=verbose)

        if visible_devices is not None:
            # If visible_device is set, CUDA_VISIBLE_DEVICES will be used.
            # E.g., if visible_devices=[0, 1, 3], only GPU:0, GPU:1, GPU:3 will be visible to torch.
            visible_devices = [str(item) for item in visible_devices]
            self.handler.print('Only GPU {} are visible'.format(' '.join(visible_devices)))
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(visible_devices)

        # If device_count is not given, set as the number of GPUs.
        device_count = device_count or torch.cuda.device_count()
        self.n_device = device_count

        # If max_processes is not given, set as the number of CPU cores.
        if max_processes > 0:
            self.max_processes = max_processes
        else:
            self.max_processes = mp.cpu_count()

        self.max_device_per_process = max_process_per_device

        self.verbose = verbose
        verbose > 1 and self.handler.print('Device manager is on!')

        self.tasks = list()

        self.mp_manager = mp.Manager()

        # Device usage counter.
        # This is an int array. Len(device_usage) == device_count
        # E.g., if device_usage[2] == 3, there are 3 processes using GPU:2 (GPU:2 is the third
        # GPU among the visible devices)
        self.device_usage = self.mp_manager.Array('i', [0] * device_count)
        # The process lock for device_usage. Every time a process want to modify device_usage,
        # it should acquire this lock.
        self.device_usage_lock = self.mp_manager.Lock()

        # Device resource control
        self.device_semaphores = [self.mp_manager.Semaphore(max_process_per_device)
                                  for _ in range(device_count)]

    def add_task(self, identifier, func_to_call, func_args):
        """
        Add a task to manager. A task should consists a function and its args.
        CAUTION: The task will not be executed util the run function is called.
        For example, suppose we have a function:
        >>> def add(a, b):
        >>>     return a + b
        and a device manager mgr, you should add tasks like this:
        >>> mgr.add_task('first task', add, {'a': 1, 'b': 2})
        >>> mgr.add_task([2, 3], add, {'a': 2, 'b': 3})
        :param identifier: Can be any type. The identifier for every task. When all the tasks finished,
        their results will be returned along with this identifier.
        :param func func_to_call: The function to be called.
        :param dict func_args: The args for the func_to_call. Should be constructed as a dict.
        :return:
        """
        new_task = Task(
            identifier=identifier,
            func_to_call=func_to_call,
            func_args=func_args,
            device_semaphores=self.device_semaphores,
            device_usage=self.device_usage,
            device_usage_lock=self.device_usage_lock,
            parent_handler=self.handler,
            task_id=len(self.tasks),
            verbose=self.verbose,
            uni_process=(self.max_processes == 1)
        )
        self.tasks.append(new_task)

    def run(self):
        """
        Execute all the tasks.
        :return: 
        1. The output messages, which were printed on the screen.
        2. The results of all the tasks, which are identified by the provided identifiers.
        """

        # Print global information.
        msg = '{} available devices.'.format(self.n_device)
        self.handler.print(msg, 1)

        msg = '{} tasks to do.'.format(len(self.tasks))
        self.handler.print(msg, 1)

        msg = '{} processes are initialized.'.format(self.max_processes)
        self.handler.print(msg, 1)

        msg = 'Each device supports at most {} processes.'.format(self.max_device_per_process)
        self.handler.print(msg, 1)

        # If the max_processes is set as 1, there is no need to do multiprocessing, so all the
        # processes will be run on the main process.
        # This is designed for debugging.
        if self.max_processes > 1:
            with Pool(processes=self.max_processes) as pool:
                rst = pool.map(run, self.tasks)
        else:
            rst = list()
            for task in self.tasks:
                rst.append(run(task))

        # msg is a list containing all the printed messages, including those of the main process.
        # Suppose there are n tasks, the msg will be a list with the size of n+1
        msg = list()
        msg.append(self.handler.mail_box())
        for task_id in range(len(self.tasks)):
            msg.append(rst[task_id][0])
            # There is no need to store the msg in each result.
            rst[task_id] = rst[task_id][1:]

        # Reset the task list.
        self.tasks = list()
        return msg, rst
