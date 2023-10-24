from model import *


def build(parameters, resume):
    model = SRGNN(parameters=parameters)
    if resume is not None:
        model = torch.load(resume)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=parameters.learningRate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters.lr_dc_step, gamma=parameters.lr_dc)
    
    return model, loss_function, optimizer, scheduler

    
def train_step(model, train_data, optimizer, scheduler, loss_fn, device):
    scheduler.step()
    model.train()
    
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    
    for i, j in zip(slices, np.arange(len(slices))):
        
        optimizer.zero_grad()
        
        targets, scores = forward_pass(model, i, train_data)
        targets = torch.Tensor(targets).long().to(device)
        
        loss = loss_fn(scores, targets - 1)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss



def train_model(epochs, model, train_data, optimizer, scheduler, loss_fn, device):
    best_result = [0, 0] # [hit, mrr]
    best_epoch = [0, 0] # [hit_epoch, mrr_epoch]
    
    for epoch in range(epochs):
        start = time.time()
        hit, mrr = train_step(model, train_data, optimizer, scheduler, loss_fn, device)
        end = time.time()
        print(f'Epoch [{epoch}/{epochs}]\tRecall@20:\t{hit:.4f}\tMMR@20:\t{mrr:.4f}\tTime:\t{end - start:.2f}s')
     
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch

    metrics = {'best_epoch': best_epoch, 
               'best_result': best_result}
    
    return model, metrics



def test_model(model, test_data):
    model.eval()
    
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    
    with torch.no_grad():
        for i in slices:
            targets, scores = forward_pass(model, i, test_data)
            sub_scores = scores.topk(20)[1]
            sub_scores = sub_scores.cpu().detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                
                hit.append(np.isin(target - 1, score))
                
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                    
                else:
                    mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                    
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    
    return hit, mrr