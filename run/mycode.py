import sys
sys.path.append('..')
import argparse
import os
import torch
import transformers
from csv import reader
from torchvision import transforms
from PIL import Image
import numpy as np
import tqdm
from parse_config import ConfigParser
from utils import state_dict_data_parallel_fix


print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path, model, tokenizer, cls_arr, device):
    """
    Classify a single image and return the predicted action class.
    
    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): Loaded HierVL model.
        tokenizer (transformers.AutoTokenizer): Tokenizer for text input.
        cls_arr (list): List of action classes.
        device (torch.device): Device to run the model on.

    Returns:
        str: Predicted action class.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)  # (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(1)  # Add temporal dimension: (1, 1, C, H, W)

    # Prepare text embeddings for action classes
    data_cls = tokenizer(cls_arr, return_tensors='pt', padding=True, truncation=True)
    data_cls = {key: val.to(device) for key, val in data_cls.items()}

    # Forward pass through the model
    model.eval()
    text_embed = None
    vid_embed = None
    with torch.no_grad():
        dict_cls = {'text': data_cls, 'video': image_tensor}
        text_embed, vid_embed = model(dict_cls, return_embeds=True)

    # Compute similarity between text and image embeddings
    text_embeds = text_embed.cpu().detach()
    vid_embed = vid_embed.cpu().detach()

    sims = torch.nn.functional.cosine_similarity(text_embeds, vid_embed.unsqueeze(1), dim=-1)
    predicted_idx = torch.argmax(sims).item()

    return cls_arr[predicted_idx]

def run_single_image():
    args = argparse.ArgumentParser(description='PyTorch Action Recognition for Single Image')

    args.add_argument('-r', '--resume',
                      default='/kaggle/input/hiervl-charades/charades_hiervl_sa.pth',
                      help='Path to latest checkpoint (default: None)')
    args.add_argument('-gpu', '--gpu', default=1, type=str,
                      help='Indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--image', required=True, type=str,
                      help='Path to the input image')
    
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True, eval_mode='charades')

    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load action classes
    cls_arr = []
    with open('E:/CharadesEgo_v1_480/CharadesEgo/Charades_v1_classes.txt', 'r') as charades:
        csv_reader = list(reader(charades))
    for line in csv_reader:
        cls_arr.append(line[0][5:])

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # Load model
    import model.model as module_arch
    model = config.initialize('arch', module_arch)

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print('Using random weights')

    model = model.to(device)

    # Classify the input image
    predicted_class = classify_image(args.image, model, tokenizer, cls_arr, device)
    print(f'Predicted Action Class: {predicted_class}')

if __name__ == '__main__':
    run_single_image()
