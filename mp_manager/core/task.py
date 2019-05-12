import multiprocessing as mp

from .handler import Handler


class Task(object):
    def __init__(self, *, identifier, func_to_call, func_args, device_semaphores,
                 device_usage, device_usage_lock, parent_handler, task_id, verbose,
                 uni_process=False):
        """
        Basically speaking, a task is comprised of a function to call and its args.
        A task will provide the function with an additional arg `handler', which could offer some
        functionality, including the fancy print function, a device-run function.
        :param identifier: Can be any type.
        :param func func_to_call: This function will be called when executing the task.
        :param dict func_args: The args for func_args.
        :param list[mp.Semaphore] device_semaphores: Semaphore for device resource.
        :param list device_usage: Recording the usage of each device.
        :param mp.Lock device_usage_lock: Lock for device_usage variable.
        :param Handler parent_handler: The handler of the main process. This is used to print out some
        global information.
        :param int task_id: The id for this task.
        :param int verbose: Verbose level. 3 is the highest.
        :param bool uni_process: If true, there is only one process running now. This is mode is on when
        max_processes == 1 in the manager. If this mode is on, the device_run method will not be called.
        """
        self.identifier = identifier
        self.function_to_call = func_to_call
        self.function_args = func_args

        self.device_semaphores = device_semaphores
        self.device_usage_lock = device_usage_lock
        self.device_usage = device_usage
        self.n_device = len(device_usage)

        self.task_id = task_id
        self.parent_handler = parent_handler
        self.verbose = verbose

        self.uni_process = uni_process

    def __call__(self):
        """
        Execute the task.
        :return:
        1. the printed messages
        2. the identifier
        3. results
        """
        # Get the process id
        process_name = mp.current_process().name
        if '-' in process_name:
            process_id = int(process_name.split('-')[1])
        else:
            process_id = -1

        # n_device == 0 means that we are going to use CPU
        if self.n_device > 0:
            # If there are more than one devices, try to find the device with lowest usage.
            self.device_usage_lock.acquire()
            try:
                # GPU:device_id has the lowest usage
                device_id = -1
                min_usage = float('inf')
                for idx, value in enumerate(self.device_usage):
                    if value <= min_usage:
                        # I use <= instead of < so the rear devices have higher priority.
                        device_id = idx
                        min_usage = value
                # update the record
                self.device_usage[device_id] += 1
            finally:
                self.device_usage_lock.release()

            semaphore_to_use = self.device_semaphores[device_id]

            to_print = 'Process #{} begins to do task #{} with device #{}'
            to_print = to_print.format(process_id, self.task_id, device_id)
        else:
            # use CPU
            to_print = 'Process #{} begins to do task #{}'
            to_print = to_print.format(process_id, self.task_id)
            device_id = -1
            # Set semaphore as None since there is no need to acquire CPU device
            semaphore_to_use = None

        handler = Handler(
            process_id=process_id,
            device_id=device_id,
            task_id=self.task_id,
            verbose=self.verbose,
            device_semaphore=semaphore_to_use,
            uni_process=self.uni_process
        )

        self.parent_handler.print(msg=to_print, verbose=1) 

        try:
            rst = self.function_to_call(**self.function_args, handler=handler)
        except TypeError as type_error:
            # Type error is raised maybe because handler is not acceptable by function_to_call.
            if "unexpected keyword argument 'handler'" in str(type_error):
                rst = self.function_to_call(**self.function_args)
            else:
                raise type_error

        # After using, update the device record
        if self.n_device > 0 and not self.uni_process:
            self.device_usage_lock.acquire()
            try:
                self.device_usage[device_id] -= 1
            finally:
                self.device_usage_lock.release()

        to_print = 'Process #{} finishes task #{}'
        to_print = to_print.format(process_id, self.task_id)
        self.parent_handler.print(msg=to_print, verbose=1)

        return handler.mail_box(), self.identifier, rst
