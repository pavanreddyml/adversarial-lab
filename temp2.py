import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from adversarial_lab.core.noise_generators.dataset import TrojanImageNoiseGenerator
from adversarial_lab.handlers.images_from_directory import ImagesFromDirectory
from adversarial_lab.attacker.data.trojan_attaker import TrojanAttacker

# Setup handler for a directory of images (change path as needed)
handler = ImagesFromDirectory(
    directory_path=r'C:\Users\Pavan Reddy\Desktop\Train',
    output_path=r'C:\Users\Pavan Reddy\Desktop\ttroj',
    include_alpha=False,
    batch_size=16,
    overwrite= True
)

trojan_patch_white = np.full((3, 3), 255, dtype=np.uint8)
trojan_patch_black = np.full((3, 3), 156, dtype=np.uint8)

# Create noise generator with two trojans
gen = TrojanImageNoiseGenerator(
    trojans=[trojan_patch_white, trojan_patch_black],
    size=(8, 8),
    position=(15.0, 10.0),
    rotation=(0.5, 0.7),
    keep_aspect_ratio=False,
    fit_to_size=True,
    coerce_out_of_bound=True,
)

# Poison 30% with trojan 0 (white), 20% with trojan 1 (black) for class '1'
class_trojan_map = {
    'digit_0': {0: 0.3, 1: 0.2},
    "digit_1": {0: 0.3, 1: 0.2}
}

for i in os.listdir(r'C:\Users\Pavan Reddy\Desktop\Train'):
    if i not in class_trojan_map:
        class_trojan_map[i] = {}

attacker = TrojanAttacker(
    handler=handler,
    noise_generator=gen,
    verbose=2,
    copy_remaining=True,
)
attacker.attack(class_trojan_map=class_trojan_map)
