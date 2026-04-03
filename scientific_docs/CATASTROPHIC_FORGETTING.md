# When to Use Self-Supervised Learning vs. ImageNet

## The "Catastrophic Forgetting" Phenomenon

If your goal is absolute performance (maximizing mIoU), it might seem logical to combine **ImageNet weights** with **Self-Supervised Learning (SSL)** sequentially (e.g., ImageNet Init $\rightarrow$ SSL Pre-training $\rightarrow$ Supervised Fine-Tuning). 

However, running this exact sequence on a CNN often results in **poorer performance** than bypassing SSL and jumping straight from ImageNet to Fine-Tuning.

### Why Does This Happen?

1.  **ImageNet is a "Hard" Task:** ImageNet weights were developed by training on 14 million images to classify 1,000 extremely diverse, complex classes (dogs, cars, skyscrapers). The resulting encoder possesses incredibly deep, mathematically rich semantic filters.
2.  **CNN MAE is an "Easier" Task:** When you run the Masked Autoencoder (MAE) pre-training, you enforce a localized pixel-reconstruction task (MSE Loss). Because standard CNNs (like U-Net++) have overlapping sliding windows and dense convolutions, the network "cheats" by learning to locally interpolate visible pixels into the masked regions.
3.  **The Overwrite (Forgetting):** Because local pixel interpolation is a fundamentally "simpler" objective than 1,000-class global semantic classification, the optimizer actively dismantles the complex ImageNet filters, replacing them with basic local-color-averaging filters. You are essentially taking a highly educated model and forcing it to spend epochs on a rudimentary interpolation task—by the end, it actively forgets its original deep features.

## The Theory of Mislabeled Ground Truth
A fascinating secondary reason SSL might appear to perform "worse" on your validation metrics relates to the conflict between **learned physical boundaries** and **human labeling error**.

*   **Standard Supervised Models:** Standard models are heavily penalized for disagreeing with the human label. If a human hastily labels an entire patchy area as "Woodland" (even though 40% of the pixels clearly look like "Bareland"), a supervised-only model will learn to ignore the visual difference and just predict the dominant human class to minimize its loss.
*   **SSL Models:** MAE pre-training forces a model to become hyper-aware of actual, physical textures and hard edges independent of any human bias. When fine-tuned on sloppy labels, an SSL-trained model will implicitly resist the human polygon because it *knows* the pixels represent bare land.
*   **The Paradoxical Result:** The standard supervised model scores a higher mIoU on the test set because it learned to accurately replicate the human's sloppy polygons, while the SSL model appears to score worse on paper, despite actually producing a more physically accurate and highly-resolved segmentation map.

## Recommendations for Your Workflow

### Objective A: Maximize Absolute mIoU (The pragmatic approach)
If your primary goal is the highest metric possible to achieve peak benchmarking scores:
*   **Skip SSL entirely.** Use `ENCODER_WEIGHTS: imagenet`, and proceed directly to normal supervised transfer learning (Frozen Head Swap $\rightarrow$ Unfrozen Fine-tuning). The massive external scale of ImageNet provides a robust, "stupid but effective" foundation for general accuracy.


### Objective B: Scientific Proof of Architecture (The research approach)
If your thesis goal is to prove that **Self-Supervised Learning is a powerful alternative to external datasets** for remote sensing:
*   **Turn off ImageNet.** Set `ENCODER_WEIGHTS: null` in your `config/config.yaml`.
*   Establish two baselines:
    1.  **Random-Init Supervised Baseline:** Train on the dataset starting from `null` weights. The results will be relatively poor.
    2.  **Your SSL Contribution:** Run `pretrain.py` from `null` weights, then fine-tune. Show that your SSL methodology significantly outperforms the baseline, proving your MAE architecture successfully builds deep geometric features entirely from scratch using only satellite data.

## Note on Self-Training (Pseudo-Labeling)

In the **Self-Training** workflow, the choice of **Teacher** is paramount:
1.  **ImageNet Teacher**: If you use a model trained only on ImageNet as a teacher, its pseudo-labels will be biased towards generic features.
2.  **Domain-Specific Teacher**: If you use a model already fine-tuned on your satellite data (Stage 3), its pseudo-labels will be much more refined for the specific spectral signatures of your classes.

**Recommendation**: Always use your best fine-tuned model (from Stage 3) as the Teacher. This creates a "positive feedback loop" where the model's existing domain knowledge is used to label new data, which in turn reinforces the model's understanding of that Domain.
