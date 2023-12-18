#coding=utf-8
import pickle
import torch
from torch.utils.data import TensorDataset,DataLoader


def get_loader(filename,batch_size):
    with open(filename,'rb') as file:
        a = pickle.load(file)
    sequence_list = list(a)[0]
    label_list = list(a)[1]
    tensor_X = torch.tensor(sequence_list)
    tensor_X = tensor_X.permute(0, 2, 1)
    tensor_Y = torch.tensor(label_list)
    dataset = TensorDataset(tensor_X, tensor_Y)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = get_loader('yelp_test.pickle',batch_size=64)

model = torch.load('best_model.pth')
model = model.to(device)
model.eval()
correct_predictions = 0
total_samples = 0
for j, (X_test, Y_test) in enumerate(test_loader):
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    prediction = model(X_test)
    # 为计算准确率做准备
    prediction = torch.argmax(prediction, dim=1)
    correct_predictions += torch.eq(prediction, Y_test.squeeze(dim=-1)).sum()
    total_samples = total_samples + len(Y_test)


accuracy = correct_predictions / total_samples

print(f'测试集上的准确率: {accuracy * 100:.4f}%')