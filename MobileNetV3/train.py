import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from data.dataloader import LoadData
from torch.autograd import Variable
from torchvision.models import mobilenet_v3_large
from torchtoolbox.tools import mixup_data, mixup_criterion
from torchtoolbox.transform import Cutout

# 设置全局参数
learning_rate = 1e-4
num_classes = 12
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = r'D:\Intern\datasets\statue_roi\train'
test_path = r'D:\Intern\datasets\statue_roi\val'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    Cutout(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
model_ft = mobilenet_v3_large(weights='DEFAULT')
# print(model_ft)
num_ftrs = model_ft.classifier[3].in_features
model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
model_ft.to(DEVICE)
# print(model_ft)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=20,
                                                       eta_min=1e-9)

dataset_train = LoadData(train_path, transforms=transform, train=True)
dataset_test = LoadData(test_path, transforms=transform_test, train=False)
# 读取数据
# print(dataset_train.imgs)

# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

# 定义训练过程
alpha = 0.2


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device,
                               non_blocking=True), target.to(device,
                                                             non_blocking=True)
        data, labels_a, labels_b, lam = mixup_data(data, target, alpha)
        optimizer.zero_grad()
        output = model(data)
        loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.
                  format(epoch, (batch_idx + 1) * len(data),
                         len(train_loader.dataset),
                         100. * (batch_idx + 1) / len(train_loader),
                         loss.item(), lr))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))


ACC = 0
min_loss = 1e8


# 验证过程
def val(model, device, test_loader):
    global ACC, min_loss
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(
                device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avg_loss = test_loss / len(test_loader)
        print(
            '\nValidation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
            format(avg_loss, correct, len(test_loader.dataset), 100 * acc))
        if acc >= ACC and avg_loss <= min_loss:
            torch.save(
                model_ft,
                'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pt')
            ACC = acc
            min_loss = avg_loss


# 训练

for epoch in range(1, EPOCHS + 1):
    train(model_ft, DEVICE, train_loader, optimizer, epoch)
    cosine_schedule.step()
    val(model_ft, DEVICE, test_loader)
