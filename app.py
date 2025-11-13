import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

print("Loading libraries...")


# --- 1. ALL MODEL & SCHEDULE CODE (Must be identical to training) ---

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, out_ch), SiLU())

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_mlp(t_emb).view(h.shape[0], -1, 1, 1)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = x.view(b, c, h * w)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        w_ = torch.bmm(q.permute(0, 2, 1), k) * (c ** (-0.5))
        w_ = torch.softmax(w_, dim=-1)
        out = torch.bmm(w_, v.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.proj_out(out)
        out = out.view(b, c, h, w)
        return out + x_in


class ContextUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, channel_mults=(1, 2, 2), num_res_blocks=2,
                 img_size=16, num_classes=10):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.num_classes = num_classes
        time_emb_dim = base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.label_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        chs = [base_ch] + [base_ch * m for m in channel_mults]
        self.init_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        in_channels = chs[0]
        use_attention = (False, True, True)
        for i in range(len(channel_mults)):
            out_channels = chs[i + 1]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_channels, out_channels, time_emb_dim))
                in_channels = out_channels
            attn = AttentionBlock(out_channels) if use_attention[i] else nn.Identity()
            self.downs.append(nn.ModuleDict({
                'blocks': blocks,
                'attn': attn,
                'downsample': nn.AvgPool2d(2) if (i < len(channel_mults) - 1) else nn.Identity()
            }))

        self.mid = nn.ModuleDict({
            'block1': ResidualBlock(in_channels, in_channels, time_emb_dim),
            'attn': AttentionBlock(in_channels),
            'block2': ResidualBlock(in_channels, in_channels, time_emb_dim)
        })

        self.ups = nn.ModuleList()
        for i in reversed(range(len(channel_mults))):
            out_channels = chs[i]
            skip_channels = chs[i + 1]

            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                if j == 0:
                    block_in_ch = in_channels + skip_channels
                else:
                    block_in_ch = out_channels

                blocks.append(ResidualBlock(block_in_ch, out_channels, time_emb_dim))
            in_channels = out_channels

            attn = AttentionBlock(out_channels) if use_attention[i] else nn.Identity()

            upsample = nn.Upsample(scale_factor=2, mode='nearest') if (i < len(channel_mults) - 1) else nn.Identity()

            self.ups.append(nn.ModuleDict({
                'blocks': blocks,
                'attn': attn,
                'upsample': upsample
            }))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t_norm, c):
        B = x.shape[0]
        t = t_norm.view(B, 1)
        t_emb = self.time_mlp(t)
        c_emb = self.label_emb(c)
        emb = t_emb + c_emb

        hs = []
        h = self.init_conv(x)
        hs.append(h)

        for layer in self.downs:
            for block in layer['blocks']:
                h = block(h, emb)
            h = layer['attn'](h)
            hs.append(h)
            h = layer['downsample'](h)

        h = self.mid['block1'](h, emb)
        h = self.mid['attn'](h)
        h = self.mid['block2'](h, emb)

        for layer in self.ups:
            h = layer['upsample'](h)
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            for block in layer['blocks']:
                h = block(h, emb)
            h = layer['attn'](h)

        out = self.out_conv(h)
        return out


def make_cosine_schedule(timesteps: int, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0001, 0.999)
    return betas


class DiffusionSchedule:
    def __init__(self, timesteps: int):
        self.timesteps = timesteps
        device = torch.device('cpu')  # Will be moved to target device later
        betas = make_cosine_schedule(timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self


# --- 2. CFG SAMPLING LOOP (NOW A GENERATOR) ---

@torch.no_grad()
def sample_loop_generator(model: ContextUNet, schedule: DiffusionSchedule, img_size: int, n_samples: int,
                          cond_label: int, null_class_idx: int, guidance_scale: float, device='cuda'):
    model.eval()
    T = schedule.timesteps
    shape = (n_samples, 3, img_size, img_size)
    x = torch.randn(shape, device=device)

    c = torch.full((n_samples,), cond_label, dtype=torch.long, device=device)
    c_uncond = torch.full((n_samples,), null_class_idx, dtype=torch.long, device=device)

    # --- How many steps to show? ---
    num_yield_steps = 50  # Show 50 frames of the generation
    yield_every_n_steps = max(1, T // num_yield_steps)
    # ---

    for i in tqdm(range(T - 1, -1, -1), desc=f"Sampling (w={guidance_scale})", disable=True):
        t_norm = torch.full((n_samples,), i / T, device=device)
        eps_cond = model(x, t_norm, c)
        eps_uncond = model(x, t_norm, c_uncond)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        sqrt_alphas_cumprod = schedule.sqrt_alphas_cumprod[i].to(device)
        sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[i].to(device)

        # Predict the "clean" image (x0)
        x0_pred = (x - sqrt_one_minus * eps) / sqrt_alphas_cumprod
        x0_pred = x0_pred.clamp(-1., 1.)

        posterior_mean = (
                schedule.posterior_mean_coef1[i].to(device) * x0_pred +
                schedule.posterior_mean_coef2[i].to(device) * x
        )

        if i > 0:
            noise = torch.randn_like(x)
            var = schedule.posterior_variance[i].to(device)
            x = posterior_mean + torch.sqrt(var) * noise
        else:
            x = posterior_mean  # Final step

        # --- YIELD THE PREDICTED CLEAN IMAGE & STATUS ---
        if (i % yield_every_n_steps == 0) or (i == 0):
            yield x0_pred.clamp(-1, 1), f"Denoising Step {T - i} / {T}"

    # Final yield of the actual final image
    yield x.clamp(-1, 1), "Generation Complete!"


# --- 3. YOUR APP CONFIGURATION (UPDATED!) ---

YOUR_NUM_CLASSES = 5

# This dictionary maps the name in the dropdown to the Class ID
# (Based on the inspection image you provided)
YOUR_LABEL_MAP = {
    "Characters": 0,
    "Monsters / Creatures": 1,
    "Fruits / Food": 2,
    "Equipment / Armor": 3,
    "Fighters / Player Classes": 4,
}
# ---

# --- App constants ---
TIMESTEPS = 1000
IMG_SIZE = 16
BASE_CH = 64
EMA_WEIGHTS_PATH = "ema_shadow.pth"
NULL_CLASS_IDX = YOUR_NUM_CLASSES
CATEGORY_CHOICES = list(YOUR_LABEL_MAP.keys())

# --- 4. Load Model and Schedule ---
print("Setting up model and schedule...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

schedule = DiffusionSchedule(TIMESTEPS).to(device)

model = ContextUNet(
    in_ch=3,
    base_ch=BASE_CH,
    channel_mults=(1, 2, 2),
    img_size=IMG_SIZE,
    num_classes=YOUR_NUM_CLASSES
).to(device)

try:
    shadow_state_dict = torch.load(EMA_WEIGHTS_PATH, map_location=device)
    for name, p in model.named_parameters():
        if name in shadow_state_dict:
            p.data.copy_(shadow_state_dict[name].data)
    model.eval()
    print(f"✅ Successfully loaded EMA weights from {EMA_WEIGHTS_PATH}")
except FileNotFoundError:
    print(f"❌ Error: EMA weights file not found at {EMA_WEIGHTS_PATH}")
    print("Please make sure 'ema_shadow.pth' is in the same folder as app.py")
    exit()
except Exception as e:
    print(f"❌ Error loading EMA weights: {e}")
    exit()


# --- 5. Define Gradio Function (NOW A GENERATOR) ---

def generate_image(category_name: str, guidance: float):
    """Gradio will call this function"""

    class_label = YOUR_LABEL_MAP.get(category_name, 0)
    print(f"Generating: '{category_name}' (Class ID {class_label}) with w={guidance}")

    # Call the new generator function
    sample_generator = sample_loop_generator(
        model=model,
        schedule=schedule,
        img_size=IMG_SIZE,
        n_samples=1,
        cond_label=class_label,
        null_class_idx=NULL_CLASS_IDX,
        guidance_scale=guidance,
        device=device
    )

    # Loop through the yielded samples and yield processed images
    for samples_tensor, status_message in sample_generator:
        # Process the sample
        img_tensor = (samples_tensor[0] + 1) / 2  # Denormalize [-1,1] -> [0,1]
        img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray((img_tensor * 255).astype(np.uint8))
        img_pil_large = img_pil.resize((256, 256), Image.NEAREST)

        # Yield the processed image and the status message
        yield img_pil_large, status_message

    # --- 6. Create the UPDATED Gradio Interface ---


print("Launching Gradio interface...")

with gr.Blocks(theme=gr.themes.Monochrome()) as iface:
    gr.Markdown("# Pixel Diffusion")
    gr.Markdown("Watch the AI generate a new sprite from pure noise! Select a category and see it come to life.")

    with gr.Row():
        with gr.Column(scale=1):
            category_input = gr.Dropdown(
                choices=CATEGORY_CHOICES,
                label="Sprite Category",
                value=CATEGORY_CHOICES[0]  # Default to the first item
            )

            guidance_input = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                step=0.5,
                value=7.5,
                label="Guidance Scale (w)",
                info="Higher = follows class strictly. Lower = more creative."
            )

            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(
                image_mode="RGB",
                width=256,
                height=256,
                interactive=False
            )
            # This Textbox will act as our loading bar
            generation_status = gr.Textbox(
                label="Generation Status",
                value="Ready",
                interactive=False
            )

    # Connect the button click to the generator function
    # It now correctly outputs to BOTH components
    submit_btn.click(
        fn=generate_image,
        inputs=[category_input, guidance_input],
        outputs=[output_image, generation_status]
    )


if __name__ == "__main__":
    iface.launch(share=True)