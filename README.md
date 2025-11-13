# Pixel Diffusion Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/code/jalpan04/pixel-diffusion-model)
[![Hugging Face](https://img.shields.io/badge/🤗-Demo-yellow.svg)](https://huggingface.co/spaces/jalpan04/Pixel_Diffusion)

A conditional Denoising Diffusion Probabilistic Model (DDPM) for generating 16x16 pixel art sprites with class-based control and real-time visualization.

**[Try the Live Demo on Hugging Face](https://huggingface.co/spaces/jalpan04/Pixel_Diffusion)** | **[View Training Notebook on Kaggle](https://www.kaggle.com/code/jalpan04/pixel-diffusion-model)**

---

## Overview

This project operates in two phases: a **training phase** (detailed in `Training.ipynb`) and an **inference/application phase** (detailed in `app.py`). The model from the first phase is loaded into the second to create an interactive application for generating pixel art sprites.

---

## How It Works: A Detailed Breakdown

The core of this project is a conditional Denoising Diffusion Probabilistic Model (DDPM). The process can be broken down into data handling, model architecture, training, and inference.

### 1. Data and Scheduling

* **Data Handling:** The model is trained on 16x16 pixel art sprites. The `PixelArtDataset` class in the training notebook is custom-built for this data.
* **Noise Schedule:** A `DiffusionSchedule` class implements a **cosine noise schedule**. This defines how noise is added to an image over `T=1000` timesteps. The model's job is to learn how to reverse this process, starting from pure noise and gradually denoising it back to a clean image.

### 2. The Model: `ContextUNet`

The model's "brain" is the `ContextUNet`. This architecture is specifically designed to handle and be controlled by external information.

* **U-Net Structure:** It is a standard U-Net with a downsampling path, a bottleneck, and an upsampling path. Skip-connections link the downsampling layers to the upsampling layers, which helps the model preserve fine details (crucial for pixel art).
* **Context Injection:** This is the "Context" part of the name. The model is given three pieces of information at every step:
    1.  **The Noisy Image (`x_t`):** The current image at timestep `t`.
    2.  **The Timestep (`t`):** The model needs to know *how much* noise is in the image to remove the correct amount. The timestep `t` is passed through its own small neural network (`time_mlp`) to create a "time embedding".
    3.  **The Class Condition (`c`):** This is the *control* mechanism. The desired class (e.g., "Characters" or "Monsters") is provided as an integer ID. This ID is passed through an `nn.Embedding` layer (`label_emb`) to create a "class embedding".
* **Embedding Combination:** The time embedding and class embedding are added together (`emb = t_emb + c_emb`). This combined context vector is then injected into every single `ResidualBlock` throughout the U-Net. This means at every stage of processing, the model is constantly reminded of *what* it is supposed to be drawing and *how much* denoising it needs to do.

### 3. Training: Learning to Denoise

The training loop in `Training.ipynb` teaches the model its core task.

1.  A clean image `x` and its label `c` are loaded from the dataset.
2.  A random timestep `t` (from 1 to 1000) is chosen.
3.  The correct amount of noise for timestep `t` (defined by the cosine schedule) is added to the clean image `x`, creating the noisy image `x_t`.
4.  The noisy image `x_t`, the timestep `t`, and the label `c` are all fed into the `ContextUNet`.
5.  The model's goal is to predict the *original noise* that was added.
6.  The loss is a simple Mean Squared Error (`F.mse_loss`) between the model's predicted noise and the actual noise.

### 4. Inference: Guided Generation

The `app.py` file uses the trained model to generate new images. This is where **Classifier-Free Guidance (CFG)** comes into play, a technique that allows for explicit control over the generation.

1.  **Start:** The process begins with a 16x16 tensor of pure random noise (`x = torch.randn(...)`).
2.  **Denoising Loop:** The model iterates backward from timestep `T-1` down to `0`.
3.  **CFG at each step:** For each step in the loop, the model *runs twice*:
    * **Conditional Run:** It predicts the noise using the user's chosen category (e.g., "Characters"). This is `eps_cond`.
    * **Unconditional Run:** It predicts the noise using a special "null" class. This is `eps_uncond`.
4.  **Guidance:** The final noise prediction is a guided combination of the two:
    `eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)`.
    The `guidance_scale` (the slider in the UI) determines how strongly the model "sticks" to the category. A high value forces the model to strictly follow the prompt, while a low value allows for more creative (but less accurate) results.
5.  **Step:** The model uses this guided `eps` to clean the image by a small amount, producing the image for the next, less-noisy step.
6.  **Finish:** After all 1000 steps, the noise is gone, and the final clean image remains.

---

## Key Improvements That Made It Better

Several specific design choices in these files contribute to the model's success, especially for this specific task.

1.  **Cosine Noise Schedule:** Instead of a simple linear schedule, the training uses `make_cosine_schedule`. A cosine schedule adds noise more gradually and is known to improve sample quality and training stability, especially for smaller diffusion models.

2.  **Classifier-Free Guidance (CFG):** This is the most significant improvement for usability. To make CFG work, the model was *trained* to handle it. In `Training.ipynb`, 10% of the time (`p_uncond = 0.1`), the true class label was randomly replaced with a `NULL_CLASS_IDX`. This forced the model to learn how to denoise *both with and without* a class, enabling the guided inference method in `app.py`.

3.  **Exponential Moving Average (EMA):** Training can be noisy, and the model's weights at the very last step might not be the best. The `EMA` class in `Training.ipynb` keeps a "shadow" copy of the model's weights, which is a slowly-updating average. This `ema_shadow.pth` file, which is loaded by `app.py`, contains these averaged weights, which are less "jumpy" and almost always produce higher-quality, more stable-looking final images.

4.  **Appropriate Interpolation:** When loading the data in `Training.ipynb`, the `T.Resize` transform *explicitly* uses `interpolation=Image.NEAREST`. For pixel art, using standard (bilinear) interpolation would create blurry, averaged colors, corrupting the data. `NEAREST` preserves the sharp, blocky nature of pixel art, leading to a much better-trained model. This same method is used in `app.py` to scale the 16x16 output to 256x256 for viewing.

5.  **Attention Blocks:** The `ContextUNet` isn't just `Conv2d` layers. In its deeper (smaller resolution) layers, it uses `AttentionBlock` modules. This allows the model to learn long-range spatial relationships—for example, to understand that a pixel on the left side of the image (e.g., a "hand") is related to a pixel on the right side (e.g., a "shoulder").

6.  **Live-Updating Generator:** For the `app.py`, the inference loop `sample_loop_generator` is a Python *generator* (it uses `yield`). Instead of only returning the final image after 1000 steps, it *yields* its prediction of the *clean image* every 20 steps. The Gradio UI catches these yielded images, allowing the user to see the image "fade in" from noise in real-time. This is a major user experience improvement that makes the underlying process visible.

---

## Technical Details

- **Architecture:** Conditional U-Net with attention blocks
- **Training Steps:** 1000 diffusion timesteps
- **Resolution:** 16x16 pixels (upscaled to 256x256 for display)
- **Guidance Method:** Classifier-Free Guidance (CFG)
- **Noise Schedule:** Cosine schedule for improved quality

---
![Generated Examples](https://github.com/Jalpan04/Pixel-Diffusion-Model/blob/main/examples.png)
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This implementation draws inspiration from modern diffusion model research, including DDPM and classifier-free guidance techniques.
