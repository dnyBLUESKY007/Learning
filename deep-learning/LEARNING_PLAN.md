# Deep Learning & LLM Learning Plan

A systematic path from zero to cutting-edge LLM engineering. Each phase builds on the previous. Track completed items with `[x]`.

---

## Phase 1: Mathematical Foundations

**Goal:** Build the math intuition needed to understand and implement DL from scratch.

### Linear Algebra
- [ ] Vectors, matrices, tensors — shapes, broadcasting rules
- [ ] Matrix multiplication, dot products, outer products
- [ ] Eigenvalues/eigenvectors, SVD (critical for understanding attention and LoRA)
- [ ] Norms (L1, L2, Frobenius)

### Calculus & Optimization
- [ ] Partial derivatives, gradients, Jacobians, Hessians
- [ ] Chain rule — the mathematical basis of backpropagation
- [ ] Gradient descent variants: SGD, momentum, Adam, AdamW
- [ ] Learning rate schedules: cosine decay, warmup

### Probability & Statistics
- [ ] Probability distributions: Gaussian, Categorical, Bernoulli
- [ ] Maximum likelihood estimation (MLE)
- [ ] KL divergence, cross-entropy — why they're the same for classification
- [ ] Bayes' theorem and its role in sampling

**Resources:**
- *Mathematics for Machine Learning* (Deisenroth et al.) — free PDF
- 3Blue1Brown: Essence of Linear Algebra + Calculus series (YouTube)
- Fast.ai computational linear algebra course

**Code:** `foundations/` — implement operations from scratch in NumPy to solidify intuition

---

## Phase 2: Deep Learning Foundations

**Goal:** Understand neural networks deeply enough to implement training from scratch.

### Building Blocks
- [ ] Perceptron, MLP — forward pass by hand
- [ ] Activation functions: ReLU, GELU, SiLU — and why they matter
- [ ] Loss functions: cross-entropy, MSE, RLHF losses
- [ ] Initialization strategies: Xavier, He — why bad init breaks training

### Backpropagation
- [ ] Implement autograd from scratch (a tiny `micrograd`)
- [ ] Computational graphs — how PyTorch builds them
- [ ] Gradient flow: vanishing/exploding gradients
- [ ] Gradient clipping, gradient checkpointing

### Regularization & Normalization
- [ ] Dropout, weight decay
- [ ] BatchNorm, LayerNorm, RMSNorm — when to use which
- [ ] Residual connections — why they revolutionized deep networks

### PyTorch Fundamentals
- [ ] Tensor operations, `.grad`, `requires_grad`
- [ ] `nn.Module`, `forward()`, `backward()`
- [ ] DataLoader, Dataset, custom collate functions
- [ ] `torch.no_grad()`, `.detach()`, memory management
- [ ] Mixed precision training (`torch.cuda.amp`)

**Resources:**
- Andrej Karpathy: *Neural Networks: Zero to Hero* (YouTube) — essential
- *Deep Learning* by Goodfellow, Bengio, Courville (the "DL Bible")
- PyTorch official tutorials

**Code:** `neural-networks/` — implement MLP, autograd engine, train on MNIST/CIFAR-10

---

## Phase 3: Transformer Architecture

**Goal:** Understand every component of the transformer deeply enough to implement GPT from scratch.

### Attention Mechanism
- [ ] Self-attention: Q, K, V matrices — what they represent geometrically
- [ ] Scaled dot-product attention — why the sqrt(d_k) scaling
- [ ] Multi-head attention — why multiple heads help
- [ ] Causal masking for autoregressive models
- [ ] KV-cache — what it is and why it's critical for inference

### Transformer Components
- [ ] Positional encodings: sinusoidal vs learned vs RoPE vs ALiBi
- [ ] Feed-forward sublayer (MLP in transformer blocks)
- [ ] Pre-norm vs post-norm — why modern LLMs use pre-norm
- [ ] The full GPT-2 architecture (decoder-only transformer)

### Tokenization
- [ ] Byte-Pair Encoding (BPE) — implement from scratch
- [ ] SentencePiece, tiktoken
- [ ] Vocabulary size tradeoffs

### Implement GPT from Scratch
- [ ] Follow Karpathy's `nanoGPT` — study every line
- [ ] Train a character-level language model on Shakespeare
- [ ] Scale up to a small GPT-2 (124M params)

**Resources:**
- Andrej Karpathy: *Let's build GPT from scratch* (YouTube) — mandatory
- *Attention Is All You Need* paper (Vaswani et al., 2017)
- Illustrated Transformer by Jay Alammar
- `nanoGPT` repo

**Code:** `transformers/` — GPT implementation, tokenizer, training on small corpus

---

## Phase 4: Training at Scale

**Goal:** Understand how modern LLMs are trained and be able to run/modify training pipelines.

### Pretraining
- [ ] Data pipelines: tokenization at scale, packing sequences
- [ ] The Chinchilla scaling laws — optimal tokens per parameter
- [ ] Gradient accumulation for large batch sizes
- [ ] Learning rate warmup + cosine decay schedule

### Distributed Training
- [ ] Data parallelism (DDP) — PyTorch `DistributedDataParallel`
- [ ] Tensor parallelism — splitting weight matrices across GPUs
- [ ] Pipeline parallelism — splitting model layers across GPUs
- [ ] ZeRO optimizer stages (ZeRO-1, 2, 3) — DeepSpeed/FSDP
- [ ] Mixed precision: FP16 vs BF16 — when each is appropriate

### Fine-Tuning
- [ ] Full fine-tuning vs parameter-efficient methods
- [ ] LoRA and QLoRA — how low-rank adaptation works mathematically
- [ ] Instruction tuning (SFT — Supervised Fine-Tuning)
- [ ] RLHF overview: reward modeling, PPO for LLMs
- [ ] DPO (Direct Preference Optimization) — simpler RLHF alternative

**Resources:**
- DeepSpeed and Megatron-LM documentation
- *Scaling Laws for Neural Language Models* (Kaplan et al.)
- *Training language models to follow instructions with human feedback* (InstructGPT paper)
- LoRA paper (Hu et al., 2022)
- HuggingFace `transformers` + `trl` libraries

**Code:** `training/` — DDP training loop, LoRA implementation, fine-tune a small model

---

## Phase 5: Inference Optimization

**Goal:** Understand how to serve LLMs efficiently — critical for production and understanding system design.

### Quantization
- [ ] INT8 quantization: post-training quantization (PTQ) vs quantization-aware training (QAT)
- [ ] GPTQ — weight-only quantization for LLMs
- [ ] AWQ (Activation-aware Weight Quantization)
- [ ] bitsandbytes library for 4-bit/8-bit inference

### Efficient Attention
- [ ] FlashAttention — fused kernel, IO-aware computation
- [ ] FlashAttention-2 and 3 improvements
- [ ] Paged attention (used in vLLM) — for KV-cache memory management
- [ ] Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

### Inference Serving
- [ ] Batching strategies: static batching vs continuous batching
- [ ] KV-cache memory management and eviction
- [ ] Speculative decoding — using a draft model to speed up sampling
- [ ] vLLM internals — PagedAttention in practice
- [ ] llama.cpp — CPU inference with GGUF format

### Decoding Strategies
- [ ] Greedy, beam search, top-k, top-p (nucleus), temperature
- [ ] Repetition penalty, presence penalty
- [ ] Structured outputs / constrained decoding (outlines, guidance)

**Resources:**
- FlashAttention papers (Dao et al., 2022/2023)
- vLLM blog post and paper
- llama.cpp repo and documentation

**Code:** `inference/` — implement KV-cache, benchmark different decoding strategies

---

## Phase 6: Cutting-Edge LLM Technologies

**Goal:** Be productive with the latest LLM capabilities and research directions.

### Retrieval-Augmented Generation (RAG)
- [ ] Dense retrieval: embedding models, FAISS/vector databases
- [ ] Chunking strategies and their impact on retrieval quality
- [ ] Hybrid search: dense + sparse (BM25)
- [ ] Re-ranking with cross-encoders
- [ ] Advanced RAG: query rewriting, hypothetical document embeddings (HyDE)

### LLM Tool Use & Function Calling
- [ ] How function calling works in OpenAI/Anthropic APIs
- [ ] Tool use prompting patterns
- [ ] Structured output parsing (JSON mode, Pydantic)
- [ ] Code execution tools, web search integration
- [ ] Model Context Protocol (MCP) — standardized tool interface

### Agentic AI
- [ ] ReAct framework (Reasoning + Acting)
- [ ] Chain-of-thought (CoT) and step-by-step reasoning
- [ ] Tree of Thoughts (ToT) and Monte Carlo Tree Search for reasoning
- [ ] Multi-agent frameworks: AutoGen, CrewAI, LangGraph
- [ ] Long-horizon planning: how agents decompose complex tasks
- [ ] Memory systems: in-context, external (vector DB), episodic
- [ ] Self-reflection and self-correction in agents

### Multimodality
- [ ] Vision-language models: CLIP, LLaVA architecture
- [ ] How image tokens are integrated into transformer input
- [ ] Diffusion models basics (relevant for multimodal generation)

### Current Research Frontiers (2024-2025)
- [ ] Long context models: positional interpolation, YaRN, and context extension
- [ ] Mixture of Experts (MoE): sparse activation, routing, Mixtral architecture
- [ ] Test-time compute: inference-time scaling, o1/o3 reasoning models
- [ ] State Space Models (Mamba, SSM) — alternatives to attention
- [ ] Continuous batching and disaggregated prefill/decode (prefill-decode disaggregation)

**Resources:**
- ReAct paper (Yao et al., 2022)
- Anthropic research blog
- Papers With Code — track SOTA
- AI News / Latent Space podcast for staying current

**Code:** `llm-advanced/` — build a minimal agent with tool use, implement RAG from scratch

---

## Suggested Learning Order

For a complete beginner going deep on LLMs, recommended sequence:

1. **Phase 1** (2-3 weeks): Math foundations — don't skip, especially backprop math
2. **Phase 2** (3-4 weeks): DL + PyTorch — build micrograd, train on MNIST
3. **Phase 3** (3-4 weeks): Transformers — implement GPT from scratch (nanoGPT)
4. **Phase 5 partial** (1-2 weeks): KV-cache + basic inference while transformer knowledge is fresh
5. **Phase 4** (3-4 weeks): Training at scale — LoRA, distributed training
6. **Phase 5** (2-3 weeks): Full inference optimization
7. **Phase 6** (ongoing): Current tech — RAG, agents, tool use

## Parallel Tracks

These can be done alongside the main sequence:
- **Rust for ML systems**: After finishing rustlings, explore candle (HuggingFace's Rust ML framework) and tch-rs (PyTorch Rust bindings) — highly relevant for inference engines
- **CUDA/GPU programming**: After Phase 2, learn CUDA basics to understand what FlashAttention actually optimizes
- **Reading papers**: Start with landmark papers at each phase — don't save them for the end
