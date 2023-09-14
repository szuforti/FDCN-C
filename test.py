from preprocessing.HGD.cut_cat_dataloader3 import *
from Model.FDCN import fdcn

if __name__ == "__main__":
    model_path = r"models/S6.pth"
    data_list = r"data"
    subject = [6]
    model_par = {
        "filter_time_length": 40,
        "final_conv_length": 20,
        "in_chans": 44,
        "input_window_samples": 1000,
        "n_classes": 4,
        "n_filters_time": 40,
        "n_filters_spat": 40,
        "pool_time_length": 75,
        "pool_time_stride": 15,
        "drop_prob": 0.3,
        "conv_nonlin": "ELSE",
        "pool_nonlin": "ELSE",
        "log": True
    }
    model = fdcn(**model_par)
    model.load_state_dict(torch.load(model_path)["net"])
    Source_data = HGD_dataset(data_list=data_list, subjects=subject)
    test_loader = Source_data.test_loader(batch_size=16,)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    test_acc_num = 0
    for i, data in enumerate(test_loader):
        print(i)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.type(torch.cuda.FloatTensor)
        out = model(inputs)
        _, pred = torch.max(torch.mean(out.data, dim=2).squeeze(), dim=1)
        pred_label = pred.cpu().numpy()
        correct = pred.eq(labels.data).cpu().sum().item()
        test_acc_num = correct + test_acc_num
    classifier_acc = 100.0 * test_acc_num / (i+1)*16