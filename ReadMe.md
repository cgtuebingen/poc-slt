# POC-SLT: Partial Object Completion with SDF Latent Transformers

**Authors**: Faezeh Zakeri, Raphael Braun, Lukas Ruppert, and Hendrik P.A. Lensch

**Conference**: [CRV 2025](https://crv.pubpub.org/pub/yanc7d1w)

**Arxiv**: [Arxiv 2024](https://arxiv.org/abs/2411.05419)

**Code Repository**: [poc-slt](https://github.com/cgtuebingen/poc-slt)

---

## Project Progress Checklist

- [ ] Upload pretrained models
- [ ] Upload Dataset LMDB
- [ ] Upload train and validation splits
- [ ] Release main codebase
- [ ] Add evaluation scripts
- [ ] Add Demo Results and videos
- [ ] Clean up and document configs
- [ ] Add citation and license info

---

## Abstract

3D geometric shape completion hinges on representation learning and a deep understanding of geometric data. Without profound insights into the three-dimensional nature of the data, this task remains unattainable. Our work addresses this challenge of 3D shape completion given partial observations by proposing a transformer operating on a latent space representing Signed Distance Fields (SDFs). Instead of a monolithic volume, the SDF of an object is partitioned into smaller high-resolution patches leading to a sequence of latent codes. The approach relies on a smooth latent space encoding learned via a variational autoencoder (VAE), trained on millions of 3D patches. We employ an efficient masked autoencoder transformer to complete partial sequences into comprehensive shapes in latent space. Our approach is extensively evaluated on partial observations from ShapeNet and the ABC dataset where only fractions of the objects are given. The proposed POC-SLT architecture compares favorably with several baseline state-of-the-art methods, demonstrating a significant improvement in 3D shape completion, both qualitatively and quantitatively.

---

## Project Structure

```bash
├── data/
├── src/
├── results/
├── requirements.txt
├── train.py
├── eval.py
└── README.md
```


## ? Citation

If you use this work, please cite:

> **Zakeri, Faezeh**, Braun, Raphael, Ruppert, Lukas, and Lensch, Hendrik P.A.
> *POC-SLT: Partial Object Completion with SDF Latent Transformers.*
> *Proceedings of the Conference on Robots and Vision*, May 27, 2025.
> [https://crv.pubpub.org/pub/yanc7d1w](https://crv.pubpub.org/pub/yanc7d1w)

You can also find citation metadata in the [CITATION.cff](./CITATION.cff) file or by clicking **?Cite this repository?** on the right.
