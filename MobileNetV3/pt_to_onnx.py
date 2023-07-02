import torch
import onnx
# torch.backends.cudnn.benchmark = False


def pt_2_onnx(onnx_export_path):
    """
    将pytorch模型导出为onnx, 导出时pytorch内部使用的是trace或者script先执行一次模型推理,然后记录下网络图结构
    :return:
    """
    model = torch.load('D:\\Intern\\statue_person_v3.pt')
    print(model)
    model.eval()
    model.cuda()
    input_names = ["input"]
    output_names = ["output"]
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    torch.onnx.export(model,
                      dummy_input,
                      onnx_export_path,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True)


def check_onnx(onnx_export_path):
    # Load the ONNX model
    model = onnx.load(onnx_export_path)

    # Check if the model is well formed
    try:
        onnx.checker.check_model(model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")

    # A human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))


if __name__ == '__main__':
    onnx_export_path = 'D:\\Intern\\statue_person_v3.onnx'
    pt_2_onnx(onnx_export_path)
    check_onnx(onnx_export_path)