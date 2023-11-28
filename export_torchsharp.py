import torch
from stable_diffusion_pytorch import model_loader
import exportsd

def convert_clip_to_sharp():
    device = torch.device('cpu')
    f = open("clip.dat", "wb")
    clip = model_loader.load_clip(device)
    clip.eval()
    exportsd.save_state_dict(clip.to("cpu").state_dict(), f)
    f.close()


def convert_unet_to_sharp():
    device = torch.device('cpu')
    f = open("unet.dat", "wb")
    unet = model_loader.load_diffusion(device)
    unet.eval()
    exportsd.save_state_dict(unet.to("cpu").state_dict(), f)
    f.close()


def convert_decoder_to_sharp():
    device = torch.device('cpu')
    f = open("decoder.dat", "wb")
    decoder = model_loader.load_decoder(device)
    decoder.eval()
    exportsd.save_state_dict(decoder.to("cpu").state_dict(), f)
    f.close()


def convert_encoder_to_sharp():
    device = torch.device('cpu')
    f = open("encoder.dat", "wb")
    encoder = model_loader.load_encoder(device)
    encoder.eval()
    exportsd.save_state_dict(encoder.to("cpu").state_dict(), f)
    f.close()


def test():
    from stable_diffusion_pytorch import Diffusion
    # diffusion = Diffusion()
    # for (k,v) in diffusion.state_dict().items():
    # 	print(k, v.shape)
    static_dict = torch.load(
        'G:/stable-code/stable-diffusion-pytorch-main/data/ckpt/diffusion.pt')
    for (k, v) in static_dict.items():
        print(k, v.shape)