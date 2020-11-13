from Utils.Meters import AverageMeter, AccuracyMeter
import pickle

class LoggerTimer(object):
    
    def __init__(self) -> None:
        self.train_acc_meter = AccuracyMeter()
        self.val_acc_meter = AccuracyMeter()
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        self.train_epoch_time_meter = AverageMeter()
        self.val_epoch_time_meter = AverageMeter()

        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []
        self.train_timer = []
        self.val_timer = []

    def update(self, acc, loss, time, train, N=1):
        if train:
            self.train_acc_meter.update(acc, N)
            self.train_loss_meter.update(loss, N)
            self.train_epoch_time_meter.update(time, 1)
        else:
            self.val_acc_meter.update(acc, N)
            self.val_loss_meter.update(loss, N)
            self.val_epoch_time_meter.update(time, 1)

    def reset(self, train):

        if train:
            self.train_acc.append(self.train_acc_meter.avg)
            self.train_loss.append(self.train_loss_meter.avg)
            self.train_timer.append(self.train_epoch_time_meter.sum)
            print('Train Loss: {:.4f}; Accuracy: {:.4f}  EpochTime: {:.4f}'.format(self.train_loss_meter.avg, self.train_acc_meter.avg, self.train_epoch_time_meter.sum))
        else:
            self.val_acc.append(self.val_acc_meter.avg)
            self.val_loss.append(self.val_loss_meter.avg)
            self.val_timer.append(self.val_epoch_time_meter.sum)
            print('Val Loss: {:.4f}; Val Accuracy: {:.4f}  EpochTime: {:.4f}'.format(self.val_loss_meter.avg, self.val_acc_meter.avg, self.val_epoch_time_meter.sum))

        self.train_acc_meter.reset()
        self.val_acc_meter.reset()
        self.train_loss_meter.reset()
        self.val_loss_meter.reset()
        self.train_epoch_time_meter.reset()
        self.val_epoch_time_meter.reset()


    def save_results(self, name, num_layers=None, num_parameters=None, num_hidden=None):
        with open('{0}.pickle'.format(name), 'wb') as f:
            data = {'train_acc':self.train_acc, 'val_acc': self.val_acc, 'train_loss': self.train_loss, 'val_loss': self.val_loss, 'train_time': self.train_timer, 'val_time': self.val_timer ,'layers': num_layers,
            'parameters': num_parameters, 'num_hidden': num_hidden }
            pickle.dump(data, f)

class StepLogger(LoggerTimer):
    def __init__(self) -> None:
        LoggerTimer.__init__(self)
        self.step_acc = []
        self.step_loss = []
    
    def update(self, acc, loss, time, train, N):
        LoggerTimer.update(self, acc, loss, time, train, N)

    def save_results(self, name, num_layers=None, num_parameters=None, num_hidden=None):
        LoggerTimer.save_results(self, name, num_layers, num_parameters, num_hidden)
        with open('{0}-StepLogs.pickle'.format(name), 'wb') as f:
            data = {'train_acc':self.train_acc, 'val_acc': self.val_acc, 'train_loss': self.train_loss, 'val_loss': self.val_loss, 'train_time': self.train_timer, 'val_time': self.val_timer ,'layers': num_layers,
            'parameters': num_parameters, 'num_hidden': num_hidden }
            pickle.dump(data, f)


    