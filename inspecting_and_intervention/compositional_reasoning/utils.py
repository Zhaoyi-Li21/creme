import torch
import torch.nn as nn
torch.manual_seed(0)

def model_name2path(model_name:str)->str:
    if model_name == "gpt2-xl":
        return "/root/autodl-tmp/zhaoyi/huggingface_models/gpt2-xl"
    elif model_name == "gpt-j-6b":
        return "/root/autodl-tmp/zhaoyi/huggingface_models/gpt-j-6b"
    elif model_name == "openalpaca-3b":
        return "/root/autodl-tmp/zhaoyi/huggingface_models/openalpaca3b"
    elif model_name == "llama2-7b":
        return "/root/autodl-tmp/zhaoyi/huggingface_models/llama2-7b-hf"
    elif model_name == "redpajama-7b":
        return "/root/autodl-tmp/zhaoyi/huggingface_models/RedPajama-INCITE-7B-Instruct"

def get_lm_type(model_name:str)->str:
    if "gpt" in model_name:
        # GPT2-small,base,large,xl; GPT-J-6B
        lm_type = "gpt"
    elif "llama" in model_name or "alpaca" in model_name:
        # OpenAlpaca-3B, LLaMA-2-7B
        lm_type = "llama"
    else: 
        raise Exception("model:{} is currently not covered in the model list.".format(model_name))
    return lm_type
    
def get_distance(h1, h2, type='L2'):
    # h1.shape = h2.shape = [hidden_size]
    if type == 'L2': return torch.sqrt(torch.sum(torch.pow((h1-h2), 2)))
    elif type == 'L1': return torch.sum(torch.abs(h1-h2))
    elif type == 'Cosine': return torch.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).squeeze(0)
    elif type == 'IP': return torch.dot(h1, h2)


def test():
    h1 = torch.tensor([1., 2., 3.], dtype=torch.float)
    h2 = torch.tensor([1., 2., 3.], dtype=torch.float)
    print(get_distance(h1, h2, 'IP'))


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(MLP, self).__init__()
        self.up_proj = nn.Linear(input_dim, hid_dim)
        self.down_proj = nn.Linear(hid_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.down_proj(self.relu(self.up_proj(x)))
        
def test_2():
    mlp = MLP(4, 6, 4)
    p_new = torch.tensor([0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05], dtype=torch.float)


    p_old = torch.tensor([0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25, 0.1], dtype=torch.float)
    
    W = torch.rand(10, 4)

    x = torch.tensor([1., 1., 1., 1.], requires_grad=True)
    opt = torch.optim.SGD([x], 1e-2)
    for i in range(100000):
        z = mlp(x)
        y = torch.matmul(W, z)
        out = torch.softmax(y, 0)

        loss = torch.norm((out-p_new)[0], p=2) * 10 + torch.norm((out-p_old)[1:-1], p=2) + torch.norm((out-p_old)[-1], p=2) * 10

        loss.backward()        
        if i % 1000 == 0:
            print('epoch:',i,'grad:', x.grad)
        # x.data = x.data - lr * x.grad.data
        # x.grad.data.zero_()
        opt.step()
        opt.zero_grad()
        # print(x)
        if i % 20 == 0:
            print('epoch:',i,'loss:',loss)

    print(out) 

# test_2()