import numpy as np

def psnr(ref, dec, max_val=255.0):
    mse = np.mean((ref.astype(np.float32) - dec.astype(np.float32))**2)
    if mse == 0: 
        return float('inf')  # identical
    return 10.0 * np.log10((max_val**2) / mse)

def ssim_simple(ref, dec, K1=0.01, K2=0.03, L=255):
    # simple single-window SSIM for demonstration (not multi-scale)
    C1 = (K1*L)**2; C2 = (K2*L)**2
    mu_x = ref.mean(); mu_y = dec.mean()
    sigma_x2 = ((ref - mu_x)**2).mean(); sigma_y2 = ((dec - mu_y)**2).mean()
    sigma_xy = ((ref - mu_x)*(dec - mu_y)).mean()
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(num / den)

# usage: load frames as numpy arrays and call psnr(ref, dec) and ssim_simple(ref, dec)