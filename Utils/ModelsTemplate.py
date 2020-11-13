from torch import Size
from Utils.Logger import LoggerTimer, StepLogger

class ModelTimer:
    def __init__(self, net, name, device, criterion, optimizer, lr_scheduler=None, logger = None):
        super(ModelTimer, self).__init__()
        self.net = net
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logs = logger if logger else LoggerTimer()
        self.name = name
  
    def update(self, acc, loss, time, train, size):
        self.logs.update(acc, loss, time, train, size)

    def reset(self, train):
        self.logs.reset(train)
    
    def save_results(self):
        self.logs.save_results(self.name)

class ModelTrackLayersParam(ModelTimer):
    def __init__(self, net, name, device, criterion, optimizer, layers, lr_scheduler=None):
        ModelTimer.__init__(self, net, name, device, criterion, optimizer, lr_scheduler)
        self.num_layers = len(layers)
        self.hidden_units = sum(layers)
        self.total_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def save_results(self):
        self.logs.save_results(self.name, self.num_layers, self.total_parameters, self.hidden_units)

class ModelStepLogger(ModelTimer):
    def __init__(self, net, name, device, criterion, optimizer, layers, lr_scheduler=None):
        ModelTimer.__init__(self, net, name, device, criterion, optimizer, lr_scheduler, StepLogger())

    