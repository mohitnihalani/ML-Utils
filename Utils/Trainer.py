import torch
import time

def _flatten_duplicates(inputs, target, batch_first=True, expand_target=True):
    duplicates = inputs.size(1)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    inputs = inputs.flatten(0, 1)

    target = target.unsqueeze(dim = 1)
    target = target.view(-1, 1).expand(-1, duplicates)
    target = target.flatten(0, 1)
    return inputs, target

def calculate_accuracy(predictions, labels):
  return torch.sum(predictions == labels.data)

def ba_test(model, data_loader):
    """Test over the whole dataset"""

    model.net.eval()  # set model in evaluation mode

    start = time.time()
    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU

      
        inputs = inputs.to(model.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(model.device, non_blocking=True, dtype=torch.long)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model.net(inputs)
           
            _, predictions = torch.max(outputs, 1)
            loss = model.criterion(outputs, labels)

        accuracy = calculate_accuracy(predictions, labels)
        # statistics
        model.update(accuracy.double().item(), loss.item(), time.time() - start, False, inputs.size(0))
        start = time.time()
    model.reset(train=False)

def ba_train(model,data_loader,duplicates):
    """Train for a single epoch"""

    # set model to training mode
    model.net.train()

    start = time.time()
    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        
        if duplicates > 1:
          inputs, labels = _flatten_duplicates(inputs, labels, batch_first=True, expand_target=True)


        # send the input/labels to the GPU
        inputs = inputs.to(model.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(model.device, non_blocking=True, dtype=torch.long)
        # zero the parameter gradients
        model.optimizer.zero_grad()
   
        
        with torch.set_grad_enabled(True):
          outputs = model.net(inputs)
          _, predictions = torch.max(outputs, 1)

          loss = model.criterion(outputs, labels)

          loss.backward()
          model.optimizer.step()

        accuracy = calculate_accuracy(predictions, labels)
        model.update(accuracy.double().item(), loss.item(), time.time() - start, True, inputs.size(0))
        start = time.time()
    
    if (model.lr_scheduler):
        model.lr_scheduler.step()

    model.reset(train=True)

def train_epoch(model,data_loader):
    """Train for a single epoch"""

    # set model to training mode
    model.net.train()

    start = time.time()
    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(model.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(model.device, non_blocking=True, dtype=torch.long)
        # zero the parameter gradients
        model.optimizer.zero_grad()


        with torch.set_grad_enabled(True):
            # forward
            outputs = model.net(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = model.criterion(outputs, labels)

            # backward
            loss.backward()
            model.optimizer.step()
           
        accuracy = calculate_accuracy(predictions, labels)
        model.update(accuracy.double().item(), loss.item(), time.time() - start, True, inputs.size(0))
        start = time.time()
    

    if model.lr_scheduler:
        model.lr_scheduler.step()

    model.reset(train=True)



def train_reg_epoch(model,data_loader):
    """Train for a single epoch"""

    # set model to training mode
    model.net.train()

    start = time.time()
    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(model.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(model.device, non_blocking=True, dtype=torch.float)
        # zero the parameter gradients
        model.optimizer.zero_grad()


        with torch.set_grad_enabled(True):
            # forward
            outputs = model.net(inputs).squeeze(1)
            loss = model.criterion(outputs, labels)
            #print(labels)
            #print(outputs)
            # backward
            loss.backward()
            model.optimizer.step()
        
        accuracy = torch.sqrt(loss)*input.size(0)
        model.update(accuracy.item(), loss.item(), time.time() - start, True, inputs.size(0))
        start = time.time()
    

    if model.lr_scheduler:
        model.lr_scheduler.step()

    model.reset(train=True)

def test_epoch(model, data_loader):
    """Test over the whole dataset"""

    model.net.eval()  # set model in evaluation mode

    start = time.time()
    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU

      
        inputs = inputs.to(model.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(model.device, non_blocking=True, dtype=torch.long)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model.net(inputs)
           
            _, predictions = torch.max(outputs, 1)
            loss = model.criterion(outputs, labels)

        accuracy = calculate_accuracy(predictions, labels)
        # statistics
        model.update(accuracy.double().item(), loss.item(), time.time() - start, False, inputs.size(0))
        start = time.time()
    model.reset(train=False)

def test_reg_epoch(model, data_loader):
    """Test over the whole dataset"""

    model.net.eval()  # set model in evaluation mode

    start = time.time()
    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU

      
        inputs = inputs.to(model.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(model.device, non_blocking=True, dtype=torch.float)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model.net(inputs).squeeze(1)
           
            loss = model.criterion(outputs, labels)

        accuracy = torch.sqrt(loss)*input.size(0)
        # statistics
        model.update(accuracy.item(), loss.item(), time.time() - start, False, inputs.size(0))
        start = time.time()
    model.reset(train=False)