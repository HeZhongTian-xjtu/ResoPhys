import importlib
import torch

def build_net(args, device):
    model_name = args.model
    module = importlib.import_module(name=f"model.{model_name}")
    net = module.load_model(args, device)
    return net



def load_model(args, device):
    return BioPhysNet(frames=args.T, device=device)


if __name__ == "__main__":
    model = BioPhysNet(frames=160, device="cuda:0")
    model.cuda()

    import time

    import torchkeras
    input_data = {"video": torch.rand(1, 3, 160, 64, 64).cuda(), "seg": torch.rand(1, 160, 64, 64).cuda()}
    start_time = time.time()
    with torch.no_grad():  # 确保在推理期间不计算梯度
        for i in range(100):
            y = model(input_data)
    end_time = time.time()

    inference_time = (end_time - start_time) / 100
    print("160 frames inference time: %.4f s" % inference_time)
    print("30s inference time: %.4f s" % (30 * 30 /160 * inference_time))
    print("inference fps: %.4f fps" % (1 /((inference_time)/160)))
    print("inference time per frame: %.4f ms" % ((inference_time)/160 * 1000))
    # print(y["rppg"].shape)
    # torchkeras.summary(model, input_data)

    # from thop import profile
    # flops, params = profile(model, (input_data,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))