import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader,TensorDataset,random_split 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import OneHotEncoder 
import numpy as np 

data = load_iris() 
X = data.data
y = data.target.reshape(-1,1) 

encoder = OneHotEncoder(sparse_output=False) 
y = encoder.fit_transform(y) 

X = torch.tensor(X,dtype=torch.float32) 
y = torch.tensor(y,dtype=torch.float32) 

dataset = TensorDataset(X,y) 
train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class ANN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(ANN,self).__init__() 
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
input_dim = X.shape[1] 
hidden_dim = 10 
output_dim = y.shape[1] 
model = ANN(input_dim,hidden_dim,output_dim) 

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(),lr=0.001) 
num_epochs = 100 
for epoch in range(num_epochs):
    for inputs,targets in train_loader:
        outputs = model(inputs) 
        loss = criterion(outputs,torch.argmax(targets,dim=1)) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval() 
with torch.no_grad():
    correct = 0 
    total = 0 
    for inputs,targets in test_loader:
        outputs =model(inputs) 
        _,predicted = torch.max(outputs.data,1) 
        total += targets.size(0) 
        correct += (predicted == torch.argmax(targets,dim=1)).sum().item() 
    print(f'Test Accuracy:{100 * correct / total:.2f}%')