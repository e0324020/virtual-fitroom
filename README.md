# fitroom
Abstract—We present Virtual Fitroom, a web-based virtual
try-on system that combines (i) real-time body-conforming gar-
ment overlay on a live webcam stream with (ii) diffusion-
based photorealistic “Snap” rendering, and (iii) automated retail
size recommendation from webcam-derived body measurements.
The system estimates anthropometric measurements from 33
pose landmarks, extracts garment-only cutouts from e-commerce
product images using clothing-parsing segmentation, and deforms
garments onto the user’s body contour using a 36-strip affine
warp for the live view. For the photorealistic mode, we deploy
mask-based CatVTON – concatenating the masked person and
the reference garment in the latent space of a fine-tuned Stable
Diffusion inpainting UNet – entirely within a 6 GB consumer
GPU (measured peak 3.3 GiB). A ResNet-18 garment classifier
(94.7% validation accuracy on 742 annotated product images)
and per-garment size charts drive the size recommender. The
dual-mode design addresses the fundamental latency gap between
AR filters (≥30 fps, 0 ms server cost) and diffusion try-on (28 s
at 768 × 576, 20 DDIM steps): the live mode provides immediate
silhouette feedback while the Snap delivers true fabric drape
on demand. All inference runs locally; no image ever leaves the
device
