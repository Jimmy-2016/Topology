
from utils import *


pts = (torch.rand((200, 2)) * 2 - 1).requires_grad_()
opt = torch.optim.SGD([pts], lr=1)
scheduler = LambdaLR(opt, [lambda epoch: 10./(10+epoch)])
for idx in range(600):
    opt.zero_grad()
    myloss(pts).backward()
    opt.step()
    scheduler.step()
    # Draw every 100 epochs
    if idx % 100 == 99:
        P = pts.detach().numpy()
        plt.scatter(P[:, 0], P[:, 1])
        plt.show()



