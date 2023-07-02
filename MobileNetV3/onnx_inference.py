import torch
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os


def inference(model_path, test_path, txt_path):
    '''
        Predict one image
    '''
    classes = ('Statue', 'Person')  # 此处类别顺序应与dataloader.py中的定义保持完全一致
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    testList = os.listdir(test_path)
    f = open(f'{txt_path}/test_result.txt', 'w')
    for file in testList:
        img = Image.open(test_path + '/' + file)
        img = transform_test(img)
        # plt.imshow(img.permute(1, 2, 0))
        # plt.show()
        img.unsqueeze_(0)
        img = Variable(img)
        output = ort_session.run(['output'], {'input': img.detach().numpy()})
        print(output)
        # predict
        pred_class = np.argmax(output[0])
        f.write('Image: {}\tPrediction: {}\n'.format(file, classes[pred_class]))
        print('Image: {}\tPrediction: {}'.format(file, classes[pred_class]))
    f.close()
    print("\nTest completed.")


if __name__ == '__main__':
    # onnx路径
    model_path = "statue_person_v3.onnx"  # 测试模型路径
    test_path = 'D:\\Intern\\MobileNetV3-NCNN\\test_images'  # 测试数据路径
    txt_path = 'D:\\Intern\\MobileNetV3'  # 测试结果保存路径
    # 推理
    inference(model_path, test_path, txt_path)