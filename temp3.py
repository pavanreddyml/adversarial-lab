import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from adversarial_lab.core.noise_generators.dataset import TrojanImageNoiseGenerator
from adversarial_lab.handlers.images_from_directory import ImagesFromDirectory
from adversarial_lab.attacker.data.trojan_attaker import TrojanAttacker

gen = TrojanImageNoiseGenerator(
    trojans=[r"C:\Users\Pavan Reddy\Desktop\Black-And-White-PNG.png"],
    size=(500, 500),
    position=(0.45, 0.3),
    rotation=(00, 00),
    keep_aspect_ratio=False,
    fit_to_size=True,
    coerce_out_of_bound=True,
    alpha=0.9,
)

sample = np.full((1080, 1920, 3), 128, dtype=np.uint8)

_ = gen.apply_noise(sample, trojan_id=0)

plt.imshow(_)
plt.show()
