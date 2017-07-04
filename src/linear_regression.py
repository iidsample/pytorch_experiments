from itertools import count

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 4 is the polynomial degree

w_t = torch.randn(4,1) * 5
b_t = torch.randn(1) * 5
def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1,5)])

def create_batch():
    "create a batch of size 32"
    random = torch.randn(32, 1)
    # random = random.unsqueeze(1)
    x = torch.cat([random ** i for i in range(1,5)], 1)
    y = x.mm(w_t) + b_t[0]
    return Variable(x), Variable(y)

def poly_desc(W,b):
    """
    Create string description
    """
    result = 'y = '
    for i,w in enumerate(W):
        result += '{:+.2f} x^{}'.format(w, len(W) -i)
    result += '{:+.2f}'.format(b[0])
    return result

# torch model
fc = torch.nn.Linear(w_t.size(0), 1)
for batch_index in count(1):
    " count in itertools is a fancy way of just getting infinite series"
    batch_x, batch_y = create_batch()
    fc.zero_grad() # each iteration to reset the gradient
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.data[0]
    output.backward()
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)

    if loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_index))
print('learned function : \t' + poly_desc(fc.weight.data.view(-1),
                                          fc.bias.data))
print('Actual function:' + poly_desc(w_t.view(-1), b_t))
