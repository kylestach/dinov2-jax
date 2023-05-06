# DINOv2 JAX
This repository contains a port of FAIR's [DINOv2](https://dinov2.metademolab.com/) to JAX, intended for running inference against the pretrained DINO weights.

Use `dino_weights.py` for loading pretrained weights into a ViT-S JAX model (with the same modifications as are made in the DINO paper).

> **Warning**: There are currently some minor discrepancies between the output of the JAX model and the original model. The results are mostly identical, and the difference is likely down to numerical differences in the JAX and pytorch implementations, but there are no guarantees of correctness.