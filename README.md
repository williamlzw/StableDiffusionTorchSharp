# Introduction
Stable Diffusion model v1.5 for TorchSharp.  
The cpu requires a minimum of 16GB of memory.
# download code
https://github.com/kjsman/stable-diffusion-pytorch

# download checkpoint and put it into python code
https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/tree/main/data.v20221029.tar

# convert checkpoint to torchsharp
python export_torchsharp.py

# inference model with torchsharp
run c# program

