import sys

class Logger(object):
    def __init__(self, local_rank):
        self.terminal = sys.stdout
        self.file = None
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(msg)
            self.file.flush()
    def flush(self):
        pass

class AverageMeter (object):
    def __init__(self):
        self.reset ()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count