from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
from ranger import Ranger
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import time
import json

# Start the timer
start_time = time.time()

# Parse the configuration file path
parser = ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
args = parser.parse_args()

# Load the configuration
config = load_config_from_yaml(args.config_path)

# Initialize the accelerator
accelerator = Accelerator(mixed_precision='fp16')

# Set CUDA devices and create necessary directories
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
os.makedirs('predictions', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('subs', exist_ok=True)

# Load and process data
data = pl.read_csv(f"{config.input_dir}/test_sequences.csv")
lengths = data['sequence'].apply(len).to_list()
data = data.with_columns(pl.Series('sequence_length', lengths))
data = data.sort('sequence_length', descending=True)

test_ids = data['sequence_id'].to_list()
sequences = data['sequence'].to_list()
attention_mask = torch.tensor(get_distance_mask(max(lengths))).float()

data_dict = {'sequences': sequences, 'sequence_ids': test_ids, "attention_mask": attention_mask}
assert len(test_ids) == len(data)

# Create the dataset and dataloader
val_dataset = TestRNAdataset(np.arange(len(data)), data_dict, k=config.k)
val_loader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False,
                        collate_fn=Custom_Collate_Obj_test(), num_workers=min(config.batch_size, 32))

# Load models
models = []
for i in range(1):  # Adjust range if multiple models are used
    model = RibonanzaNet(config).cuda()
    model.eval()
    model.load_state_dict(torch.load(f"models/model{i}.pt", map_location='gpu'))
    models.append(model)

# Prepare models and dataloader with accelerator
models, val_loader = accelerator.prepare(models, val_loader)

# Inference
preds = []
model.eval()
tbar = tqdm(val_loader)

for idx, batch in enumerate(tbar):
    src = batch['sequence'].cuda()
    masks = batch['masks'].bool().cuda()
    length = batch['length']

    src_flipped = src.clone()
    for batch_idx in range(len(src)):
        src_flipped[batch_idx, :length[batch_idx]] = src_flipped[batch_idx, :length[batch_idx]].flip(0)

    with torch.no_grad():
        with accelerator.autocast():
            output = []
            for model in models:
                output.append(model(src, masks))
                if config.use_flip_aug:
                    flipped_output = model(src_flipped, masks)
                    for batch_idx in range(len(flipped_output)):
                        flipped_output[batch_idx, :length[batch_idx]] = flipped_output[batch_idx, :length[batch_idx]].flip(0)
                    output.append(flipped_output)
            output = torch.stack(output).mean(0)

    output = accelerator.pad_across_processes(output, 1)
    all_output = accelerator.gather(output).cpu().numpy()
    preds.append(all_output)

# Save predictions if the current process is the main process
if accelerator.is_local_main_process:
    preds_dict = {}
    for i, id in tqdm(enumerate(test_ids)):
        batch_number = i // (config.test_batch_size * accelerator.num_processes)
        in_batch_index = i % (config.test_batch_size * accelerator.num_processes)
        preds_dict[id] = preds[batch_number][in_batch_index]

    with open("preds.p", 'wb+') as f:
        pickle.dump(preds_dict, f)

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open("inference_stats.json", 'w') as file:
        json.dump({'Total_execution_time': elapsed_time}, file, indent=4)
