/* Single source of truth for the course outline.
   Used by index.html (roadmap + progress) and by every chapter (prev/next nav
   and the sidebar). Order here IS the dependency order: each chapter uses what
   the previous one built. */

const PARTS = [
  {
    n: "Part 0", t: "Learning to See Tensors",
    d: "The visual foundation everything else stands on",
    ch: [
      { id: "01", f: "ch01-tensors.html", t: "Tensors, Shapes, and How to Actually See Them",
        s: "Axes, rank, the mental model that makes every later shape obvious", tag: "60 min" },
      { id: "02", f: "ch02-matmul.html", t: "Matrix Multiplication as a Machine",
        s: "Broadcasting, batched matmul, and einsum — the three things that confuse everyone", tag: "70 min" },
    ]
  },
  {
    n: "Part I", t: "From Text to Vectors",
    d: "How a string becomes something a network can multiply",
    ch: [
      { id: "03", f: "ch03-tokenization.html", t: "Tokenization: Text → Integers",
        s: "Byte-pair encoding built from scratch, and why tokenizers cause weird bugs", tag: "50 min" },
      { id: "04", f: "ch04-embeddings.html", t: "Embeddings and the Residual Stream",
        s: "The lookup table, weight tying, and the most useful mental model of a transformer", tag: "45 min" },
    ]
  },
  {
    n: "Part II", t: "Attention",
    d: "The core mechanism, derived rather than asserted",
    ch: [
      { id: "05", f: "ch05-attention.html", t: "Attention From Scratch",
        s: "Soft dictionary lookup, Q/K/V, why √d, masking — every dimension traced", tag: "90 min" },
      { id: "06", f: "ch06-mha.html", t: "Multi-Head Attention: The Reshape Dance",
        s: "The view/transpose sequence that trips up everybody, drawn step by step", tag: "70 min" },
      { id: "07", f: "ch07-positions.html", t: "Position: Sinusoidal → Learned → RoPE → ALiBi",
        s: "Why attention is position-blind, and the rotation trick every modern model uses", tag: "75 min" },
    ]
  },
  {
    n: "Part III", t: "The Block and the Model",
    d: "Forward, backward, then the loop that uses both",
    ch: [
      { id: "08", f: "ch08-block.html", t: "The Transformer Block",
        s: "Residuals, LayerNorm vs RMSNorm, pre-LN vs post-LN, MLP, GELU vs SwiGLU", tag: "70 min" },
      { id: "09", f: "ch09-architectures.html", t: "Whole Architectures + End-to-End Shape Trace",
        s: "Encoder-only, decoder-only, encoder-decoder; every tensor from token id to logit", tag: "70 min" },
      { id: "10", f: "ch10-backprop.html", t: "The Backward Pass, Derived",
        s: "Every gradient by hand: softmax, cross-entropy, RMSNorm, attention — checked against autograd", tag: "100 min" },
      { id: "11", f: "ch11-training.html", t: "Training a Transformer",
        s: "Cross-entropy, teacher forcing, AdamW, warmup+cosine, and the tricks that matter", tag: "80 min" },
    ]
  },
  {
    n: "Part IV", t: "Making It Fast",
    d: "Where most real-world engineering effort actually goes",
    ch: [
      { id: "12", f: "ch12-decoding.html", t: "Decoding and Sampling",
        s: "Greedy, temperature, top-k, top-p, min-p, beam — and what each does to the distribution", tag: "55 min" },
      { id: "13", f: "ch13-kvcache.html", t: "The KV Cache and the Memory Wall",
        s: "Prefill vs decode, why generation is memory-bound, and the arithmetic that proves it", tag: "65 min" },
      { id: "14", f: "ch14-gqa-mla.html", t: "MQA, GQA, and Multi-Head Latent Attention",
        s: "Shrinking the cache: from 1 KV head to DeepSeek's 512-dim latent", tag: "80 min" },
      { id: "15", f: "ch15-flashattention.html", t: "FlashAttention and the Online Softmax",
        s: "The IO-aware algorithm, derived — plus FA2, FA3, FA4 (2026)", tag: "80 min" },
      { id: "16", f: "ch16-serving.html", t: "Serving at Scale",
        s: "PagedAttention, continuous batching, speculative decoding, quantization", tag: "60 min" },
    ]
  },
  {
    n: "Part V", t: "Making It Big",
    d: "Sparsity, scale, and the 2026 frontier",
    ch: [
      { id: "17", f: "ch17-moe.html", t: "Mixture of Experts",
        s: "Routing, top-k, load balancing, shared experts, and why 671B can run like 37B", tag: "70 min" },
      { id: "18", f: "ch18-scaling.html", t: "Scaling Laws and Long Context",
        s: "Kaplan vs Chinchilla, compute-optimal training, RoPE scaling, YaRN, sliding windows", tag: "65 min" },
      { id: "19", f: "ch19-frontier.html", t: "The 2026 Frontier",
        s: "DeepSeek Sparse Attention, Mamba-2 hybrids, and what is actually shipping today", tag: "60 min" },
    ]
  },
  {
    n: "Part VI", t: "Making It Useful",
    d: "Turning a text predictor into an assistant",
    ch: [
      { id: "20", f: "ch20-sft-rlhf.html", t: "SFT, RLHF, and PPO",
        s: "The three-stage pipeline, reward models, and why RL entered the picture at all", tag: "70 min" },
      { id: "21", f: "ch21-dpo.html", t: "DPO From First Principles",
        s: "Bradley-Terry → the KL-constrained optimum → the DPO loss, derived line by line", tag: "85 min" },
      { id: "22", f: "ch22-grpo.html", t: "DPO Variants, GRPO, and RLVR",
        s: "IPO, KTO, ORPO, SimPO, then the reasoning stack: GRPO, DAPO, verifiable rewards", tag: "70 min" },
    ]
  },
  {
    n: "Part VII", t: "Capstone",
    d: "Put every piece together",
    ch: [
      { id: "23", f: "ch23-capstone.html", t: "Build a Modern LLM End to End",
        s: "RMSNorm + RoPE + GQA + SwiGLU + KV cache, trained, then DPO-aligned", tag: "4-8 hrs" },
    ]
  },
];

/* flat list for prev/next */
const CHAPTERS = PARTS.flatMap(p => p.ch);
