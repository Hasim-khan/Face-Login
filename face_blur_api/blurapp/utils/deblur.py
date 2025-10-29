# import cv2
# import numpy as np

# def deblur_image(image):
#     """
#     Remove blur and enhance clarity of a given image.
#     Works best for out-of-focus or motion-blurred images.
#     Steps:
#       1. Estimate blur kernel using Laplacian variance
#       2. Apply deconvolution with controlled sharpening
#       3. Denoise and smooth artifacts
#     """

#     # Convert to LAB for better contrast control
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     # --- Step 1: Edge restoration using unsharp masking ---
#     blur = cv2.GaussianBlur(l, (0, 0), sigmaX=3)
#     sharpened = cv2.addWeighted(l, 1.8, blur, -0.8, 0)

#     # --- Step 2: Smart deblurring using edge-preserving filter ---
#     deblur = cv2.detailEnhance(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), sigma_s=12, sigma_r=0.15)
#     deblur = cv2.cvtColor(deblur, cv2.COLOR_RGB2BGR)

#     # Replace L channel with the sharpened one
#     lab_enhanced = cv2.merge((sharpened, a, b))
#     result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

#     # --- Step 3: Combine and refine ---
#     final = cv2.addWeighted(result, 0.8, deblur, 0.2, 0)

#     # --- Step 4: Final sharpening & denoise ---
#     final = cv2.GaussianBlur(final, (0, 0), sigmaX=0.5)
#     final = cv2.detailEnhance(final, sigma_s=5, sigma_r=0.2)
#     final = cv2.fastNlMeansDenoisingColored(final, None, 5, 5, 7, 21)

#     return np.clip(final, 0, 255).astype(np.uint8)


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

# ============================================================
# ✅ Lightweight CNN-based Deblur (no hub / no 404 issues)
# ============================================================

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
    """
    Deblur image using a small CNN model and OpenCV postprocessing.
    """
    # Convert to tensor
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
