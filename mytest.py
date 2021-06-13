import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import TensorDataset
import numpy as np

def mytest(dataset_name,source_path,target_path):
    assert dataset_name in ['source', 'target']

    model_root = 'models'

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    if dataset_name == 'target':
        
        target_x_path=os.path.join(target_path,'freqDomain/freqSignal.csv')
        target_y_path=os.path.join(target_path,'label.csv')
        target_x=np.loadtxt(target_x_path,delimiter=',',dtype=np.float32)
        target_y=np.loadtxt(target_y_path,delimiter=',',dtype=np.int64)
        target_x=torch.from_numpy(target_x)
        target_y=torch.from_numpy(target_y)
        dataset=TensorDataset(target_x,target_y)
    else:
        source_x_path=os.path.join(source_path, 'freqDomain/freqSignal.csv')
        source_y_path=os.path.join(source_path,'label.csv')
        source_x=np.loadtxt(source_x_path,delimiter=',',dtype=np.float32)
        source_y=np.loadtxt(source_y_path,delimiter=',',dtype=np.int64)
        source_x=torch.from_numpy(source_x)
        source_y=torch.from_numpy(source_y)
        dataset=TensorDataset(source_x,source_y)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
