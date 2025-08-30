# Modified validation.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import *
from model import *

path_arr = [
    './dataset/city_A_challengedata.csv.gz',
    './dataset/city_B_challengedata.csv.gz',
    './dataset/city_C_challengedata.csv.gz',
    './dataset/city_D_challengedata.csv.gz'
]

city_names = ['A', 'B', 'C', 'D']

def Test(args):
    """
    Performs inference on the test set for cities B, C, and D using a single pre-trained GRULocationPredictor model.
    The test set contains 3000 users for cities B, C, and D.
    Note: The TestSet class typically does not provide ground-truth labels unless modified.
    """
    if args.cities:
        city_indices = []
        for city in args.cities:
            city_upper = city.upper()
            if city_upper in ['B', 'C', 'D']:
                city_indices.append(city_names.index(city_upper))
            else:
                print(f"Warning: Test set is only available for cities B, C, and D. Skipping city {city}.")
        
        if not city_indices:
            print("No valid cities specified for testing. Exiting.")
            return
    else:
        city_indices = [1, 2, 3]
        print("No cities specified. Processing default cities: B, C, D")

    os.makedirs(args.save_path, exist_ok=True)

    # Load the pre-trained model once
    device = torch.device(f'cuda:{args.cuda}')
    model = GRULocationPredictor(args.layers_num, args.embed_size, args.cityembed_size).to(device)
    pretrained_model_path = os.path.join(args.pth_dir, 'best_GRU_pretrain.pth')
    try:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Loaded pre-trained model from {pretrained_model_path}")
    except FileNotFoundError:
        print(f"Error: Pre-trained model file {pretrained_model_path} not found. Exiting.")
        return

    model.eval()

    for city_idx in city_indices:
        city_letter = city_names[city_idx]
        print(f"\n=== Processing City {city_letter} (Test Set) ===")
        
        generated_name = f'city_{city_letter}_test_generated.csv.gz'
        
        dataset_test = TestSet(path_arr[city_idx])
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers)
        
        all_generated = []
        
        with torch.no_grad():
            for data in tqdm(dataloader_test, desc=f"City {city_letter}"):
                data['uid'] = data['uid'].to(device)
                data['d'] = data['d'].to(device)
                data['t'] = data['t'].to(device)
                data['input_x'] = data['input_x'].to(device)
                data['input_y'] = data['input_y'].to(device)
                data['time_delta'] = data['time_delta'].to(device)
                data['city'] = data['city'].to(device)
                data['len'] = data['len'].to(device)
                
                output = model(data['d'], data['t'], data['input_x'], data['input_y'], 
                              data['time_delta'], data['len'], data['city'])
                
                assert torch.all((data['input_x'] == 201) == (data['input_y'] == 201))
                pred_mask = (data['input_x'] == 201)
                output = output[pred_mask]
                
                pred_x = torch.argmax(output[:, 0, :], dim=-1)
                pred_y = torch.argmax(output[:, 1, :], dim=-1)
                pred = torch.stack((pred_x, pred_y), dim=-1)
                
                generated = torch.cat((data['uid'][pred_mask].unsqueeze(-1),
                                     data['d'][pred_mask].unsqueeze(-1)-1,
                                     data['t'][pred_mask].unsqueeze(-1)-1,
                                     pred+1), dim=-1).cpu().tolist()
                
                for point in generated:
                    all_generated.append(point)
        
        columns = ['uid', 'd', 't', 'x', 'y']
        generated_df = pd.DataFrame(all_generated, columns=columns)
        generated_save_path = os.path.join(args.save_path, generated_name)
        generated_df.to_csv(generated_save_path, index=False, compression='gzip')
        
        print(f"Generated test predictions saved to: {generated_save_path}")
        print(f"Total generated points: {len(all_generated)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_dir', type=str, default='/content/drive/MyDrive', help='Directory containing the pre-trained model file')
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive', help='Directory to save generated and reference CSV files')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--cityembed_size', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--cities', nargs='*', type=str, help='Cities to process (e.g., --cities B C D). If not specified, defaults to B C D')
    
    args = parser.parse_args()
    Test(args)