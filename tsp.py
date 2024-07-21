import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from model import MambaFull, generate_data, seq2seq_generate_tour, compute_tour_length, Bandau_Pointer, Dot_Pointer, train_step
from datetime import datetime
import argparse
import wandb
from torch.profiler import profile, record_function, ProfilerActivity


#login to wandb
wandb.login()
#parser
parser = argparse.ArgumentParser(description='Train Mamba model')
parser.add_argument('--bsz', type=int, default=600, help='Batch size')
parser.add_argument('--d_model', type=int, default=64, help='Model dimension')#ensure that this is a multiple of 2 if fourier_scale is not None
parser.add_argument('--coord_dim', type=int, default=2, help='Coordinate dimension')
parser.add_argument('--nb_layers', type=int, default=3, help='Number of layers in the model')
parser.add_argument('--mlp_cls', type=str, default='gatedmlp', help='Type of mlp to use')#set as 'identity' or 'gatedmlp'
parser.add_argument('--city_count', type=int, default=5, help='Number of cities')
parser.add_argument('--fourier_scale', type=float, default=None, help='Fourier scale')#If set as None a standard Linear map is used else a gaussian fourier feature mapping is used
parser.add_argument('--start', type=float, default=2.0, help='Start token')
parser.add_argument('--city_range', type=str, default='0,0', help='Range of cities to be used when generating data')
parser.add_argument('--non_det', type=bool, default=False, help='Set to True if you want to use a non-deterministic baseline')

parser.add_argument('--wandb', action='store_false', help='Call argument if you do not want to log to wandb')
parser.add_argument('--project_name', type=str, default='Mamba_big', help='Name of project in wandb')
parser.add_argument('--action', type=str, default='tour', help="Select if action is defined to be 'tour' or 'next_city'")

parser.add_argument('--nb_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--nb_batch_per_epoch', type=int, default=10, help='Number of batches per epoch')

parser.add_argument('--test_size', type=int, default=2000, help='Size of test data')
parser.add_argument('--save_loc', type=str, default='checkpoints/not_named', help='Location to save model')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
parser.add_argument('--recycle_data', type=int, default=0, help='Recycle data')
parser.add_argument('--model_name', type=str, default='Full', help='Model name')
parser.add_argument('--mamba2', type=bool, default=False, help='choose if mamba2 is used')
parser.add_argument('--reverse', type=bool, default=False, help='Reverse even model layers')
parser.add_argument('--reverse_start', type=bool, default=False, help='Set to True if you want to reverse the input')
parser.add_argument('--last_layer', type=str, default='pointer', help='Select whether the last layer is identity or pointer')
parser.add_argument('--test_folder_name', type=str, default=None, help='Name of folder where test data is stored')

parser.add_argument('--profiler', type=bool, default=False, help='Set to True if you want to profile the model')
parser.add_argument('--memory_snapshot', type=bool, default=False, help='Set to True if you want to profile the model')
parser.add_argument('--pynvml', type=bool, default=False, help='Set to True if you want to profile the model')
parser.add_argument('--gpu_id', type=int, default=-1, help='The GPU ID to get information')
# Define model parameters and hyperparameters
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

args=DotDict() 

# Update the DotDict instance with the parsed arguments
parsed_args = parser.parse_args()
for key, value in vars(parsed_args).items():
    setattr(args, key, value)

if args.test_folder_name is None and (args.start).is_integer():
    args.test_data_loc=f'data/start_{int(args.start)}/{args.test_size}_{args.city_count}_{args.coord_dim}.pt'
else:
    args.test_data_loc=f'data/{args.test_folder_name}/{args.test_size}_{args.city_count}_{args.coord_dim}.pt'

#Load checkpoint
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
else:
    checkpoint = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.fourier_scale is None:
    args.B = None
else:
    if checkpoint:
        args.B = checkpoint['args'].B
    else:
        args.B = torch.randn(args.d_model // 2, 2).to(device) * args.fourier_scale

args.city_range = tuple(map(int, args.city_range.split(',')))
args['x_flipped']=False
if args.reverse_start and not args.reverse:
    args['x_flipped']=True
elif args.reverse_start and args.reverse:
    if args.nb_layers%2!=0:
        args['x_flipped']=True
elif args.reverse and not args.reverse_start:
    if args.nb_layers%2==0:
        args['x_flipped']=True

if args.wandb:
    project_name = args.project_name
    if args.profiler:
        project_name = 'Mamba_profiler'
    run = wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        name=args.save_loc.split('/')[-1],
        # Track hyperparameters and run metadata
        config=args,
    )

if args.pynvml:
    if args.gpu_id == -1:
        raise ValueError("Please provide a GPU ID")
    print('gpu id= ',args.gpu_id)
    from gpu_stats import GPULogger
    gpu_logger = GPULogger(args.gpu_id)
else:
    gpu_logger=None

if args.memory_snapshot:
    torch.cuda.memory._record_memory_history()

#load train and baseline model, where baseline is used to reduce variance in loss function as per the REINFORCE algorithm. 
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls,args.B, args.reverse,args.reverse_start,args.mamba2,args.last_layer).to(device)
model_baseline = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls,args.B, args.reverse,args.reverse_start,args.mamba2,args.last_layer).to(device)
for param in model_baseline.parameters():
    param.requires_grad = False
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model_train.parameters(), lr=1e-4)

# Load checkpoint
if checkpoint:
    if 'model_baseline_state_dict' in checkpoint.keys():
        model_train.load_state_dict(checkpoint['model_train_state_dict'])
        model_baseline.load_state_dict(checkpoint['model_baseline_state_dict'])
    else:
        model_train.load_state_dict(checkpoint['model_state_dict'])
        model_baseline.load_state_dict(checkpoint['model_state_dict']) 

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    tot_time_ckpt = checkpoint['time_tot']
    start_epoch = checkpoint['epoch']
    mean_tour_length_list = checkpoint['mean_tour_length_list']
    mean_tour_length_best = min([i.item() for i in checkpoint['mean_tour_length_list']])
    if 'time_to_reach_best' in checkpoint.keys():    
        best_time = checkpoint['time_to_reach_best']
    else:
        best_time = 0
else:
    tot_time_ckpt, start_epoch = 0,0
    mean_tour_length_list = [] 
    mean_tour_length_best = float('inf') 
    best_time = 0
    model_baseline.load_state_dict(model_train.state_dict())

model_baseline.eval()
if False:
    with open(f'logs_param/parameters_{args.last_layer}_{args.city_count}_lay_{args.nb_layers}_dim_{args.d_model}.txt', 'w') as f:
        for name, param in model_train.named_parameters():
            f.write(f"Parameter: {name}, Size: {param.size()}\n")
        total_params = sum(p.numel() for p in model_train.parameters())
        f.write(f"Total number of parameters: {total_params}\n")
print('Total number of parameters:', sum(p.numel() for p in model_train.parameters()))
print('Using determinisitic Baseline')

# Load test data
test_data = torch.load(args.test_data_loc).to(device)
test_data_batches = torch.split(test_data, args.bsz)

print(args)

start_training_time = time.time()
now = datetime.now()
date_time = now.strftime("%d-%m_%H-%M")

if args.pynvml:
    gpu_logger.start_gpu_logging(f'{args.save_loc}_gpu_stats.csv', interval_ms=25)

# Training loop
for epoch in tqdm(range(start_epoch,args.nb_epochs)):
    model_train.train()
    i= 0 # Tracks the number of steps before we generate new data
    start = time.time()
    #L_train_train is the average tour length of the train model on the train data
    L_train_train_total = 0
    L_baseline_train_total = 0
    for step in range(args.nb_batch_per_epoch):
        if args.pynvml: gpu_logger.log_event(f'Epoch {epoch}, Step {step} start')
        if i == 0:
            if args.pynvml: gpu_logger.log_event(f'Generating data')
            #Inputs will have size (bsz, seq_len, coord_dim)
            if args.city_range==(0,0):
                inputs = generate_data(device, args.bsz, args.city_count, args.coord_dim,start=args.start)
            else:
                generate_city_count = np.random.randint(args.city_range[0],args.city_range[1]+1)
                inputs = generate_data(device, args.bsz, generate_city_count, args.coord_dim,start=args.start)
            i=args.recycle_data
        else: i-=1

        if args.profiler and epoch>args.nb_epochs-2:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,profile_memory=True, with_stack=True) as prof:
                with record_function("model_inference"):
                    L_train_train_total, L_baseline_train_total = train_step(model_train, model_baseline, inputs, optimizer, device,L_train_train_total,L_baseline_train_total,gpu_logger,args.action,args.non_det)
                prof.step()  # Denotes step end
            print('Epoch:', epoch, ' Step:', step)
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            prof.export_chrome_trace(f'{args.save_loc}.json')
        else:
            L_train_train_total, L_baseline_train_total = train_step(model_train, model_baseline, inputs, optimizer, device,L_train_train_total,L_baseline_train_total,gpu_logger,args.action,args.non_det)
        
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt

    ###################
    # Evaluate train model and baseline
    ###################
    model_train.eval()
    with torch.no_grad():
        #L_train is the average tour length of the train model on the test data
        L_train_total = 0
        L_baseline_total = 0
        
        # Compute tour for model and baseline for test data, making it sure its split to not overload the gpu
        for batch_num,test_data_batch in enumerate(test_data_batches):
            if args.pynvml: gpu_logger.log_event(f'Epoch {epoch}, Generating tour on test data batch {batch_num}')
            tour_train, _ = seq2seq_generate_tour(device, model_train,test_data_batch, lastlayer=args.last_layer,deterministic=True)
            tour_baseline, _ = seq2seq_generate_tour(device, model_baseline, test_data_batch, lastlayer=args.last_layer, deterministic=True)

            if args.pynvml: gpu_logger.log_event(f'Epoch {epoch}, computing tour length on test data batch {batch_num}')
            # Get the lengths of the tours and add to the accumulators
            L_train_total += compute_tour_length(test_data_batch, tour_train).sum()
            L_baseline_total += compute_tour_length(test_data_batch, tour_baseline).sum()

        # Compute the average tour lengths
        L_train = L_train_total / args.test_size
        L_baseline = L_baseline_total / args.test_size
        if args.pynvml: gpu_logger.log_event(f'Epoch {epoch} saving checkpoint and updating baseline if train is better')

    #print(f'Epoch {epoch}, test tour length train: {L_train}, test tour length baseline: {L_baseline}, time one epoch: {time_one_epoch}, time tot: {time_tot}')
    if args.wandb:
        wandb.log({
            "test_tour length train": float(L_train),
            "test_tour length baseline": float(L_baseline),
            "time one epoch": float(time_one_epoch),
            "time tot": float(time_tot),
            "train_tour length train": float(L_train_train_total),
            "train_tour length baseline": float(L_baseline_train_total)
        })

    mean_tour_length_list.append(L_train)
    #If our city range does not vary our test set is a good indicator of the performance of our model
    #however if there exists a city range we have to create a new test set to evaluate the model to update the baseline
    #this new test set is created by generating a random number of cities within the city range
    #Not yet implemented, will complete if needed
    if True:# city_range==[0,0]:
        # evaluate train model and baseline and update if train model is better
        if L_train < L_baseline:
            model_baseline.load_state_dict( model_train.state_dict() )
            best_time = time_tot
    else:
        #create a new dev set to evaluate the model and to see if the train model is better than the baseline
        L_train_dev_total = 0
        L_baseline_dev_total = 0




    # Save checkpoint every 10,000 epochs
    if L_train < mean_tour_length_best or epoch % 10 == 0:
        if L_train < mean_tour_length_best:
            mean_tour_length_best = L_train

        # Append to filename
        filename = f"file_{date_time}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_train_state_dict': model_train.state_dict(),
            'model_baseline_state_dict': model_baseline.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean_tour_length_list': mean_tour_length_list,
            'args': args,
            'time_tot': time_tot,
            'time_to_reach_best': best_time,
        }
        torch.save(checkpoint, f'{args.save_loc}_{date_time}.pt' )

if args.pynvml: 
    gpu_logger.stop_logging()
    gpu_logger.export_events(f'{args.save_loc}_events.csv')

if args.memory_snapshot:
    torch.cuda.memory._dump_snapshot(f'{args.save_loc}_memory_snapshot.pickle')
    torch.cuda.memory._record_memory_history(enabled=None)
