import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

class SimpleDeblurNet(nn.Module):
    """A lightweight CNN that learns to sharpen blurry images."""
    def __init__(self):
        super(SimpleDeblurNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x + residual  # residual learning (sharp output)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDeblurNet().to(device)
model.eval()

# Optional: load pretrained weights if you have them
# model.load_state_dict(torch.load("simple_deblur.pth", map_location=device))


def deblur_image(image):
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # Convert back to numpy
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    # Apply additional OpenCV sharpening for finer details
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(output_image, -1, kernel)
    # Denoise slightly to remove artifacts
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    return final
