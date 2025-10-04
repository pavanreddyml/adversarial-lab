import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from adversarial_lab.core.noise_generators.dataset import TrojanImageNoiseGenerator

# Input image path (28x28 grayscale)
in_path = r'C:\Users\Pavan Reddy\Desktop\adversarial-lab\examples\data\digits\1\1.png'

# Load sample as numpy (grayscale)
sample = np.array(Image.open(in_path).convert("L"), dtype=np.uint8)  # (28, 28)

# 3x3 white patch trojan
trojan_patch = np.full((3, 3), 255, dtype=np.uint8)

# Create generator
gen = TrojanImageNoiseGenerator(
    trojans=[trojan_patch],
    size=(3, 3),
    position=(15.0, 10.0),
    rotation=(0.5, 0.7),
    keep_aspect_ratio=False,
    fit_to_size=True,
    coerce_out_of_bound=True,
    alpha=0.2
)

# Apply trojan
trojaned = gen.apply_noise(sample, trojan_id=0)

# Plot with matplotlib (no saving)
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(sample, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("With 3x3 Trojan")
plt.imshow(trojaned, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.tight_layout()
plt.show()
