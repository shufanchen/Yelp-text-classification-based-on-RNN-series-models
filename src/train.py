#coding=utf-8
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from model import Model
import matplotlib.pyplot as plt
import time

#参数设置
# model_type = 'RNN'
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 640,100,16,2,0.1,0.001 #35.3,33.9
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 320,200,16,2,0.1,0.0001 #35.2,33.4
#model_type = 'LSTM'
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 640,100,16,2,0.1,0.001 #55.7,53.9,16
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 640,200,128,2,0.1,0.001 #59.6,57.6,7
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 640,200,128,1,0,0.001 #61.08,59,7
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 320,200,128,1,0,0.0005 #60.1,56.3
# batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 128,200,128,2,0.2,0.0005 #60.5,57.5

model_type = 'GRU'
#batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 320,200,128,1,0,0.0005 #60.6,56.4
#batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 128,200,128,2,0.2,0.0005 #62.5,59.3
#batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 128,200,192,3,0.2,0.0001 #60.9,57.8
batch_size,embedding_size, hidden_size, num_layers,dropout,lr = 320,200,128,2,0.3,0.0005 #61.6,60
epochs = 30


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

train_loader = get_loader('yelp_train.pickle',batch_size)
val_loader = get_loader('yelp_val.pickle',batch_size)

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('training on:',device)
model = Model(embedding_size, hidden_size, num_layers, dropout, model_type)
model = model.to(device)
loss_list = []
loss_list1 = []
train_loss_list = []
val_loss_list = []
val_acc_list = []
train_time = []
val_time = []
acc_best = 0
model_best = []
early_stop = 0

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    # training
    model.train()
    start1 = time.time()
    for i, (X, Y) in enumerate(train_loader):
        X,Y= X.to(device),Y.to(device)
        out = model(X)
        loss = criterion(out, Y.squeeze(dim=-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    end1 = time.time()
    time_epoch = end1 - start1
    train_time.append(time_epoch)
    loss_epoch = np.mean(np.array(loss_list)) #一个epoch中平均每个batch的loss
    train_loss_list.append(loss_epoch)
    print('epoch = ', epoch,'train_loss = ', loss_epoch.item(),'epoch_time = ', time_epoch)

    # validation
    model.eval()
    total_samples = 0
    correct_predictions = 0
    start2 = time.time()
    for j, (X_val, Y_val) in enumerate(val_loader):
        X_val, Y_val = X_val.to(device), Y_val.to(device)
        prediction = model(X_val)
        loss = criterion(prediction, Y_val.squeeze(dim=-1))
        loss_list1.append(loss.item())
        #为计算准确率做准备
        prediction = torch.argmax(prediction, dim=1)
        correct_predictions += torch.eq(prediction, Y_val.squeeze(dim=-1)).sum()
        total_samples = total_samples + len(Y_val)
    val_loss_list.append(np.mean(np.array(loss_list1)))
    end2 = time.time()
    val_time.append(end2 - start2)
    accuracy = correct_predictions / total_samples
    val_acc_list.append(accuracy.cpu())
    print(f'验证集上的准确率: {accuracy * 100:.4f}%','验证所消耗时间:',sum(val_time))
    if accuracy > acc_best:
        model_best = model
        acc_best = accuracy
        early_stop = 0
    else:
        early_stop+=1
    if early_stop > 4:  #连续多次在验证集上没有准确率的提升就早停，防止过拟合
        print(acc_best)
        torch.save(model_best,'best_model.pth')
        break


# 绘制训练损失曲线
plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss', marker='o')

# 绘制验证损失曲线
plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Validation Loss', marker='o')

# 设置标题、x轴标签、y轴标签和图例
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 保存图形到当前目录下
plt.savefig('training_validation_loss_plot.png')

# 清空当前图形，以便绘制下一个图形
plt.clf()

# 绘制验证集上准确率变化曲线
plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, label='Validation Accuracy', marker='o')

# 设置标题、x轴标签、y轴标签和图例
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 保存图形到当前目录下
plt.savefig('validation_accuracy_plot.png')



