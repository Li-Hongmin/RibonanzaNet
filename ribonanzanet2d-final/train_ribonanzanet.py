import argparse
import torch
from torch.utils.data import DataLoader
from ranger import Ranger
from tqdm import tqdm
from ribonanzanet_utils import *
import os 
os.environ["HOME"] = "/work/02/gs58/d58004"  # Change this to an appropriate path
os.environ["TRANSFORMERS_CACHE"] = "/work/02/gs58/d58004/cache"  # Set a writable path for cache

def freeze_existing_layers(model, previous_state_dict):
    # Freeze all layers present in the previous state dict
    for name, param in model.named_parameters():
        if name in previous_state_dict:
            param.requires_grad = False
        else:
            param.requires_grad = True
    # Unfreeze the last layer
    for param in model.decoder.parameters():
        param.requires_grad = True
def unfreeze_all_layers(model):
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, save_path, schedule=None):
    best_loss = np.inf
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        tbar = tqdm(train_loader)
        total_loss = 0
        for idx, batch in enumerate(tbar):
            sequence = batch['sequence'].to(device)
            labels = batch['labels'].float().to(device)
            seq_length = batch['length'].to(device)
            output = model(sequence)
            
            final_output, final_labels = slice_outputs(output, labels, seq_length)
            
            loss = criterion(final_output, final_labels).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1)}")
        
        if schedule:
            schedule.step()
            
        model.eval()
        val_loss, val_preds = evaluate_model(model, val_loader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_preds = val_preds
            save_model_path = f"{save_path}FinetuneDeg-epoch.pt"
            torch.save(model.state_dict(), save_model_path)
    return save_model_path

def evaluate_model(model, val_loader, criterion):
    val_loss = 0
    val_preds = []
    device = next(model.parameters()).device
    tbar = tqdm(val_loader)
    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].to(device)
        labels = batch['labels'].float().to(device)
        with torch.no_grad():
            output = model(sequence)
            loss = criterion(output[:, :68], labels).mean()
        val_loss += loss.item()
        val_preds.append([labels.cpu().numpy(), output.cpu().numpy()])
    val_loss /= len(tbar)
    print(f"Validation loss: {val_loss}")
    return val_loss, val_preds

def main(args):
    set_seed()

    # Load configuration and initialize model
    config = load_config_from_yaml(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = finetuned_RibonanzaNet(config, use_mamba_end = args.use_mamba_end).to(device)
    
    # Load previous model state
    print(f"Loading model from {args.model_path}")
    previous_state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(previous_state_dict, strict=False)

    # Freeze existing layers and unfreeze new layers
    print("Freezing existing layers")
    freeze_existing_layers(model, previous_state_dict)

    # Load and process data
    print("Loading and processing data")
    data, data_noisy, test107, test130 = load_data(args.train_pseudo_path, args.test_pseudo_107_path, args.test_pseudo_130_path, args.noisy_threshold)
    train_split, val_split = split_data(data)

    # Update pseudo labels for noisy data
    data_noisy = update_pseudo_labels(data_noisy)
    test107 = update_pseudo_labels(test107)
    test130 = update_pseudo_labels(test130)
    train_split = augment_real_with_pseudo(train_split)

    train_step3, highSN = prepare_training_data(train_split, data_noisy, test107, test130, args.sn_threshold)

    # Create data loaders
    train_loader3 = DataLoader(RNA_Dataset(train_step3, args.max_seq_length), batch_size=args.batch_size, shuffle=True)
    highSN_loader = DataLoader(RNA_Dataset(highSN, 68), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RNA_Dataset(val_split, 68), batch_size=args.batch_size, shuffle=False)
    
    # Initial training with pseudo labels by freezing existing layers and unfreezing last layers
    optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay, lr=args.lr*10)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(highSN_loader))

    # Save path with parameters
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = f"{args.save_dir}/pseudo_lr{args.lr}-epochs{args.epochs}-wd{args.weight_decay}-max_seq_length{args.max_seq_length}-sn_threshold{args.sn_threshold}-noisy_threshold{args.noisy_threshold}-batch_size{args.batch_size}-use_mamba{config.use_mamba}-use_mamba_end{args.use_mamba_end}-0-freezed-"
    last_model_path = train_model(model, train_loader3, val_loader, epochs=args.epochs, optimizer=optimizer, criterion=MCRMAE, save_path=save_path, schedule=schedule)
    
    # Unfreeze all layers
    print("Unfreezing all layers")
    unfreeze_all_layers(model)
    # Retrain the model without freezing any layers
    model.load_state_dict(torch.load(last_model_path, map_location=device), strict=False)
    save_path = f"{args.save_dir}/pseudo_lr{args.lr}-epochs{args.epochs}-wd{args.weight_decay}-max_seq_length{args.max_seq_length}-sn_threshold{args.sn_threshold}-noisy_threshold{args.noisy_threshold}-batch_size{args.batch_size}-use_mamba{config.use_mamba}-use_mamba_end{args.use_mamba_end}-1-unfreezed-"
    
    last_model_path = train_model(model, train_loader3, val_loader, epochs=args.epochs, optimizer=optimizer, criterion=MCRMAE, save_path=save_path)

    # Annealed training with high SN data
    print("Annealed training with high SN data")
    model.load_state_dict(torch.load(last_model_path, map_location=device), strict=False)
    optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay, lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(highSN_loader))
    save_path = f"{args.save_dir}/highSN_lr{args.lr}-epochs{args.epochs}-wd{args.weight_decay}-max_seq_length{args.max_seq_length}-sn_threshold{args.sn_threshold}-noisy_threshold{args.noisy_threshold}-batch_size{args.batch_size}-use_mamba{config.use_mamba}-use_mamba_end{args.use_mamba_end}-2-annealed-"
    train_model(model, highSN_loader, val_loader, epochs=args.epochs, optimizer=optimizer, criterion=MCRMAE, save_path=save_path, schedule=schedule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RibonanzaNet with pseudo labels")
    parser.add_argument("--config_path", type=str, default="configs/pairwise.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--model_path", type=str, default="../ribonanzanet-weights/RibonanzaNet-Deg.pt", help="Path to the initial model state dict")
    parser.add_argument("--train_pseudo_path", type=str, default="train_pseudo.json", help="Path to the train pseudo JSON file")
    parser.add_argument("--test_pseudo_107_path", type=str, default="test_pseudo_107.json", help="Path to the test pseudo 107 JSON file")
    parser.add_argument("--test_pseudo_130_path", type=str, default="test_pseudo_130.json", help="Path to the test pseudo 130 JSON file")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Path to save the trained model state dict")
    parser.add_argument("--sn_threshold", type=float, default=5.0, help="Signal-to-noise threshold for high SN filtering")
    parser.add_argument("--noisy_threshold", type=float, default=1.0, help="Threshold for noisy data filtering")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--max_seq_length", type=int, default=130, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # parser.add_argument("--use_mamba_start", type=bool, default=False, help="Use Mamba2 at beginning of the model")
    parser.add_argument("--use_mamba_end", type=bool, default=False, help="Use Mamba2 at end of the model")
    
    args = parser.parse_args()
    main(args)
