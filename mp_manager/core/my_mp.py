import multiprocessing as mp
from multiprocessing import pool


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(pool.Pool):
    """
    A specially designed Pool class, whose processes are all not daemon.
    A daemon process is not allowed to spawn more sub-processes, so it is not
    favorable.
    """
    Process = NoDaemonProcess
