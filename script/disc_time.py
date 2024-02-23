import torch
import time
from torch.utils.data import DataLoader
from models import get_model  
from data import get_dataloader


model_name = 'rnn'

dataset_kwargs = {
    'input_folder' : '/home/rtlink/robros/dataset/0215_norm/0215_collision/input_data', 
    'target_folder': '/home/rtlink/robros/dataset/0215_norm/0215_collision/target_data', 
    'collision_folder' : '/home/rtlink/robros/dataset/0215_norm/0215_collision/collision',
    'num_joints' : 7,
    'seq_len': 3000, 
    'offset' : 3000
}

trainset = get_dataloader(name=model_name, train=True, **dataset_kwargs)
validset = get_dataloader(name=model_name, train=False, **dataset_kwargs)

loader_kwargs = dict(
    batch_size = 4,
    drop_last = True
)

valid_loader = DataLoader(validset, **loader_kwargs)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = '/home/rtlink/robros/log/0221/discriminator/rnn/rn_1/ckpt.pt'
 
checkpoint = torch.load(ckpt_path, map_location=device)
model_CFG = checkpoint['cfg']

model_kwargs = dict(
    hidden_size=512, 
    num_joints=7, 
    num_layers=20,
    nhead=8,
    num_encoder_layers=6,
    )

model_name = "fc"


model = get_model(model_name, **model_kwargs).cuda()
 
model.eval()
 
 
# start_time = time.time()
 
# with torch.no_grad():
#     for batch_idx, (input, target, collision) in enumerate(valid_loader):
#         input, target, collision = input.to(device), target.to(device), collision.to(device)
#         output = model(input)
 
# total_time = time.time() - start_time
 
# average_time = total_time / len(valid_loader)
 
# print(f'Average Inference Time per Batch: {average_time:.4f} seconds')

num_tests = 100
times = []
 
with torch.no_grad():
    for _, input_data in zip(range(num_tests), valid_loader):
        input_data = input_data[0].to(device) 
        start_time = time.time()
        _ = model(input_data)
        end_time = time.time()
        times.append(end_time - start_time)
 
avg_time = sum(times) / len(times) * 1000  # 밀리초 단위로 변환
 
print(f"Average inference time: {avg_time:.2f} ms")