import torch
import torch.nn as nn
import torch.optim as optim #优化器
from torch.utils.data import DataLoader #数据加载
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from model.cnn import simplecnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  #裁剪为224*224
    transforms.ToTensor(),    #把图片转化为tensor张量0-1像素值
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #标准化，3通道

])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  #裁剪为224*224
    transforms.ToTensor(),    #把图片转化为tensor张量0-1像素值
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #标准化，3通道

])

trainset=datasets.ImageFolder(root=os.path.join(r'dataset\COVID_19_Radiography_Dataset','train'),
                            transform=train_transform
                            )

testset=datasets.ImageFolder(root=os.path.join(r'dataset\COVID_19_Radiography_Dataset','test'),
                            transform=test_transform
                            )

train_loader=DataLoader(trainset,batch_size=32,num_workers=0,shuffle=True)
test_loader=DataLoader(testset,batch_size=32,num_workers=0,shuffle=False)

def train(model,train_loader,criterion,optimizer,num_epochs):
    print(device)
    best_acc=0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for inputs,labels in tqdm(train_loader,desc=f"epoch:{epoch+1}/{num_epochs}",unit="batch"):
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*inputs.size(0)    #用loss*批次大小，得到该批次的loss
        epoch_loss=running_loss/len(train_loader.dataset)   #总损失/总数据集大小，得到每轮损失
        print(f"epoch:{epoch+1}/{num_epochs},loss:{epoch_loss:.4f}")

        accuracy=evaluate(model,test_loader,criterion)
        if accuracy>best_acc:
            best_acc=accuracy
            save_model(model,save_path)
            print(f"model saved with best acc",best_acc)


def evaluate(model,test_loader,criterion):
    model.eval()
    test_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for inputs,labels in tqdm(test_loader,desc="evaluating",unit="batch"):
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            test_loss+=loss.item()*inputs.size(0)
            _,predicted=torch.max(outputs,1)    #predicted是索引
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    avg_loss=test_loss/len(test_loader.dataset)
    accuracy=100*correct/total
    print(f"Test Loss:{avg_loss:.4f},Accuracy:{accuracy:.2f}%")
    return accuracy

def save_model(model,save_pth):
    torch.save(model.state_dict(),save_path)

if __name__ == '__main__':
    num_epochs=10
    learning_rate=0.001
    num_classes=4
    save_path=r"model_pth\best.pth"
    model=simplecnn(num_classes).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    train(model,train_loader,criterion,optimizer,num_epochs)
    evaluate(model,test_loader,criterion)