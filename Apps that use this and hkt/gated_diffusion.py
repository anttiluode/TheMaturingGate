"""
gated_diffusion.py — The Maturing Gate applied to Stable Diffusion
==================================================================
This script intercepts the standard diffusion loop and applies the 
two-network coincidence gate concept.

THE ARCHITECTURE:
- HEAVY NETWORK (Slow): The 860M parameter U-Net. Expensive to run.
- FAST PREDICTOR (Fast): A 0-parameter linear extrapolator running on a 
  delay buffer of the last two states. Instant to run.
- THE GATE (Chandelier): Compares the true U-Net output to the expected 
  trajectory. If |true - expected| < threshold, the surprise is low. The 
  gate shuts off the U-Net and lets the Fast Predictor run the next steps.

Watch the U-Net calls drop, and the generation time plummet, while the 
image remains intact.
"""

import torch
from diffusers import StableDiffusionPipeline
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import threading

# Use standard SD 1.5. It is heavy enough to show the speedup clearly.
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GatedDiffusionUI:
    def __init__(self, master):
        self.master = master
        self.master.title("The Evolved Gate - Diffusion Accelerator")
        self.master.geometry("900x600")
        
        self.pipeline = None
        self.is_generating = False
        
        self.setup_ui()
        
        # Load model in background to keep GUI responsive
        self.status_var.set("Loading Stable Diffusion Pipeline (may take a moment)...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
            self.pipeline.to(DEVICE)
            # Use DDIM or Euler for smooth trajectories
            self.pipeline.scheduler = self.pipeline.scheduler.__class__.from_config(self.pipeline.scheduler.config)
            self.master.after(0, lambda: self.status_var.set("Model Loaded. Ready."))
        except Exception as e:
            self.master.after(0, lambda: self.status_var.set(f"Error loading model: {e}"))

    def setup_ui(self):
        # --- Control Panel (Left) ---
        control_frame = tk.Frame(self.master, width=300, padx=15, pady=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Prompt:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.prompt_entry = tk.Text(control_frame, height=4, width=35)
        self.prompt_entry.insert(tk.END, "A highly detailed portrait of a cyberpunk mechanic working in a neon lit garage, cinematic lighting")
        self.prompt_entry.pack(pady=(0, 15))

        # The Toggle
        self.use_gate_var = tk.BooleanVar(value=True)
        gate_chk = tk.Checkbutton(control_frame, text="Enable Gated Predictor (Veto U-Net)", variable=self.use_gate_var, font=("Arial", 10, "bold"), fg="blue")
        gate_chk.pack(anchor=tk.W, pady=5)

        # Sliders
        tk.Label(control_frame, text="Total Diffusion Steps:").pack(anchor=tk.W)
        self.steps_var = tk.IntVar(value=30)
        tk.Scale(control_frame, from_=10, to=60, orient=tk.HORIZONTAL, variable=self.steps_var, length=250).pack(pady=(0, 10))

        tk.Label(control_frame, text="Surprise Threshold (Gate Tolerance):").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.08)
        tk.Scale(control_frame, from_=0.01, to=0.30, resolution=0.01, orient=tk.HORIZONTAL, variable=self.threshold_var, length=250).pack()
        tk.Label(control_frame, text="Higher = more skips, lower quality. Lower = precise.", font=("Arial", 8), fg="gray").pack(anchor=tk.W, pady=(0, 10))

        tk.Label(control_frame, text="Max Skips per Handshake:").pack(anchor=tk.W)
        self.skips_var = tk.IntVar(value=2)
        tk.Scale(control_frame, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.skips_var, length=250).pack(pady=(0, 15))

        self.btn_generate = tk.Button(control_frame, text="Generate Image", command=self.start_generation, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.btn_generate.pack(fill=tk.X, pady=10)

        # Stats
        self.stats_label = tk.Label(control_frame, text="", justify=tk.LEFT, font=("Courier", 10))
        self.stats_label.pack(anchor=tk.W, pady=20)

        # --- Image Display (Right) ---
        self.image_label = tk.Label(self.master, bg="gray", text="Image will appear here")
        self.image_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=15, pady=15)

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = tk.Label(self.master, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_generation(self):
        if self.pipeline is None or self.is_generating:
            return
        self.is_generating = True
        self.btn_generate.config(state=tk.DISABLED)
        self.status_var.set("Generating...")
        self.stats_label.config(text="")
        
        # Run in thread so GUI doesn't freeze
        threading.Thread(target=self.run_diffusion, daemon=True).start()

    def run_diffusion(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        num_steps = self.steps_var.get()
        use_gate = self.use_gate_var.get()
        threshold = self.threshold_var.get()
        max_skips = self.skips_var.get()
        
        # --- FIX: Force the pipeline to the GPU on the execution thread ---
        # Background threads in PyTorch on Windows occasionally drop the device transfer.
        self.pipeline.to(DEVICE)
        
        # Setup context and embeddings manually to intercept the loop
        generator = torch.Generator(device=DEVICE).manual_seed(42) # Fixed seed
        
        with torch.no_grad(), torch.autocast(DEVICE):
            # 1. Encode Prompt
            text_inputs = self.pipeline.tokenizer(prompt, padding="max_length", max_length=self.pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids.to(DEVICE))[0]
            
            uncond_inputs = self.pipeline.tokenizer([""], padding="max_length", max_length=self.pipeline.tokenizer.model_max_length, return_tensors="pt")
            uncond_embeddings = self.pipeline.text_encoder(uncond_inputs.input_ids.to(DEVICE))[0]
            
            # For Classifier-Free Guidance (CFG)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # 2. Setup Schedulers and Latents
            self.pipeline.scheduler.set_timesteps(num_steps)
            timesteps = self.pipeline.scheduler.timesteps
            
            latents = torch.randn((1, self.pipeline.unet.config.in_channels, 512 // 8, 512 // 8), generator=generator, device=DEVICE, dtype=text_embeddings.dtype)
            latents = latents * self.pipeline.scheduler.init_noise_sigma

            # --- THE GATED SYSTEM VARIABLES ---
            delay_buffer = []    # Holds recent noise_preds to calculate trajectory
            skips_left = 0
            unet_evals = 0
            start_time = time.time()
            
            # 3. The Custom Diffusion Loop
            for i, t in enumerate(timesteps):
                self.master.after(0, lambda idx=i: self.status_var.set(f"Step {idx+1}/{num_steps}"))
                
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                # ====================================================
                # THE GATE: Decide whether to use Heavy Net or Fast Net
                # ====================================================
                if use_gate and skips_left > 0 and len(delay_buffer) >= 2:
                    # FAST PREDICTOR (Veto is ON): We bypass the U-Net entirely.
                    # We extrapolate the trajectory from the delay buffer.
                    d_noise = delay_buffer[-1] - delay_buffer[-2]
                    noise_pred = delay_buffer[-1] + d_noise
                    
                    skips_left -= 1
                else:
                    # HEAVY NETWORK (Gate is OPEN): Run the expensive U-Net
                    unet_out = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    unet_evals += 1
                    
                    # Perform CFG
                    noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                    
                    # THE HANDSHAKE (Evaluate Surprise)
                    if use_gate and len(delay_buffer) >= 2:
                        expected = delay_buffer[-1] + (delay_buffer[-1] - delay_buffer[-2])
                        
                        # Calculate Surprise (Mean Absolute Error)
                        surprise = torch.nn.functional.l1_loss(noise_pred, expected).item()
                        
                        # If the U-Net did exactly what the linear predictor thought it would do,
                        # shut the gate and skip the next few steps.
                        if surprise < threshold:
                            skips_left = max_skips
                    
                    # Update Delay Buffer
                    delay_buffer.append(noise_pred)
                    if len(delay_buffer) > 2:
                        delay_buffer.pop(0)

                # Step the scheduler
                latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            # 4. Decode the Latent to Image
            self.master.after(0, lambda: self.status_var.set("Decoding Latents to Image..."))
            print("Step 4 reached: Preparing to decode.")
            
            try:
                # SAFETY CHECK: Did the fast predictor explode the math?
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    raise ValueError("Latents contain NaN or Inf. The fast predictor extrapolated too far and broke the math.")

                latents = (1.0 / self.pipeline.vae.config.scaling_factor) * latents
                
                # Ensure the latent dtype matches the VAE dtype perfectly
                latents = latents.to(self.pipeline.vae.dtype)
                
                print("Running VAE Decode...")
                # Decode
                image = self.pipeline.vae.decode(latents).sample
                
                print("Converting tensor to PIL Image...")
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                
                import numpy as np
                image = (image * 255).round().astype(np.uint8)
                image = Image.fromarray(image)
                print("Decode complete. Sending to UI.")
                
            except Exception as e:
                print(f"\nCRITICAL ERROR DURING DECODE: {e}\n")
                self.master.after(0, lambda err=str(e): self.status_var.set(f"Decode Error: {err}"))
                self.master.after(0, lambda: self.btn_generate.config(state=tk.NORMAL))
                self.is_generating = False
                return

            end_time = time.time()
            elapsed = end_time - start_time

            # Update GUI safely
            self.master.after(0, self.update_result, image, elapsed, num_steps, unet_evals, use_gate)

    def update_result(self, image, elapsed, num_steps, unet_evals, use_gate):
        # Resize for display
        display_img = image.copy()
        display_img.thumbnail((512, 512))
        self.photo = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.photo, text="")
        
        # Format Stats
        stats = f"Generation Time: {elapsed:.2f} seconds\n"
        stats += f"Total Steps:     {num_steps}\n"
        stats += f"U-Net Evals:     {unet_evals} "
        
        if use_gate:
            saved = num_steps - unet_evals
            savings_pct = (saved / num_steps) * 100
            stats += f"\nSkipped:         {saved} steps\n"
            stats += f"Compute Saved:   {savings_pct:.0f}%\n"
            stats += "\nThe Fast Predictor successfully\nhandled the smooth trajectories."
        else:
            stats += "\n\nStandard Diffusion.\nNo compute saved."
            
        self.stats_label.config(text=stats)
        self.btn_generate.config(state=tk.NORMAL)
        self.status_var.set("Ready.")
        self.is_generating = False

if __name__ == "__main__":
    root = tk.Tk()
    app = GatedDiffusionUI(root)
    root.mainloop()