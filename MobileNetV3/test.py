import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt


model_path = "statue_person_v4.pt"  # 测试模型路径
test_path = 'D:\\Intern\\datasets\\statue_roi\\new_truncated_508'    # 测试数据路径
# test_path = 'D:\\Intern\\datasets\\statue\\images\\test_human'    # 测试数据路径
txt_path = 'D:\\Intern\\MobileNetV3'  # 测试结果保存路径

classes = ('Statue', 'Person')  # 此处类别顺序应与dataloader.py中的定义保持完全一致
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path)
model.eval()
model.to(DEVICE)


testList = os.listdir(test_path)
f = open(f'{txt_path}/test_result.txt', 'w')
for file in testList:
    img = Image.open(test_path + '/' + file).convert('RGB')
    img = transform_test(img)
    # plt.imshow(img.permute(1, 2, 0))
    # plt.show()
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    # print(img)

    # print('input_shape: ', img.shape)
    out = model(img)
    # print('out: ', out)
    # Predict
    _, pred = torch.max(out.data, 1)
    f.write('Image: {}\tPrediction: {}\n'.format(file, classes[pred.data.item()]))
    print('Image: {}\tPrediction: {}'.format(file, classes[pred.data.item()]))
f.close()
print("\nTest completed.")
print("Results saved to %s" % txt_path)
