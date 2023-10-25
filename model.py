from torch import nn
import torch
import torch.nn.functional as F
import math

class GraphNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, step=1):
        super().__init__()
        
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))

        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        
        inputs = torch.cat([input_in, input_out], 2)
        
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        
        newgate = torch.tanh(i_n + resetgate * h_n)
        
        hidden_new = newgate + inputgate * (hidden - newgate)
        
        return hidden_new

    def forward(self, A, hidden):
        
        for _ in range(self.step):
            hidden = self.GNNCell(A, hidden)
            
        return hidden



class SRGNN(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hidden_size = parameters.hiddenSize
        self.n_node = parameters.n_node
        self.batch_size = parameters.batchSize
        self.nonhybrid = parameters.nonhybrid
        
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        
        self.gnn = GraphNeuralNetwork(self.hidden_size, step=parameters.step)
        
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
            
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def forward_pass(model, i, data, device):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    
    alias_inputs = torch.Tensor(alias_inputs).long().to(device)
    items = torch.Tensor(items).long().to(device)
    A = torch.Tensor(A).float().to(device)
    mask = torch.Tensor(mask).long().to(device)
    
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    return targets, model.compute_scores(seq_hidden, mask)