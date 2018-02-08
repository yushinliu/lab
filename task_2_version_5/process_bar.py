import sys, time

class ShowProcess():

    i = 0 # now process
    max_steps = 0 # total processing steps
    max_arrow = 50 # length of bar

    # initialize , need to know total steps
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    # monitor function
    # performance [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #compute how many'>'
        num_line = self.max_arrow - num_arrow #compute how many'-'
        percent = self.i * 100.0 / self.max_steps # compute finishing processxx.xx%
        process_bar = str('[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r') #out put string£¬'\r' 
        sys.stdout.write(process_bar) #output to terminal
        sys.stdout.flush()

    def close(self, words='done'):
        print('done')
        self.i = 0