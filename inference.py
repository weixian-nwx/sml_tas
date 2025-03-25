import torch
from torchvision import transforms
from PIL import Image
from model import Generator
import os
import argparse

# Define function to load model
def load_model(checkpoint_path, device):
    model = Generator()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

# Define function to preprocess image
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize image to 512x512
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Define function to postprocess and save image
def postprocess_and_save(tensor, output_path):
    tensor = (tensor.squeeze(0).detach().cpu() + 1) / 2  # From [-1,1] to [0,1]
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(output_path)
    print(f"Cartoonized image saved to {output_path}")

# Main function
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(os.path.join(args.checkpoint_dir, 'generator.pth'), device)

    # Preprocess image
    input_image = preprocess_image(args.input_path, device)

    # Generate output
    with torch.no_grad():
        output_image = model(input_image)

    # Save result
    postprocess_and_save(output_image, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save cartoonized output image')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory of model checkpoint')
    args = parser.parse_args()
    main(args)
