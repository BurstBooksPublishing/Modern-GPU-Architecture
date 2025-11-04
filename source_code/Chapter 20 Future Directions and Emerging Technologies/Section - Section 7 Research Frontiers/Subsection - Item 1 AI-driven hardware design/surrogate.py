import torch, torch.nn as nn, torch.optim as optim
# simple surrogate: maps arch vector x -> latency estimate
class Surrogate(nn.Module):
    def __init__(self, n): super().__init__(); self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze()
# x: continuous arch params (e.g., SM_count, TF_width, L1_kb) scaled
surrogate=Surrogate(n=6); opt=optim.Adam(surrogate.parameters(),lr=1e-3)
for epoch in range(200):                      # train on measured data (omitted)
    loss = ...                                # MSE to measured latency
    opt.zero_grad(); loss.backward(); opt.step()
# propose new arch by gradient descent on surrogate + penalty (toy example)
x = torch.randn(6, requires_grad=True)       # initial proposal
for it in range(50):
    pred = surrogate(x)
    penalty = 0.1*torch.relu(x-10).sum()      # enforce upper bounds
    obj = pred + penalty
    obj.backward()
    with torch.no_grad(): x -= 1e-2 * x.grad; x.grad.zero_()
# x rounded to nearest legal configuration before RTL evaluation