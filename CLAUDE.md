# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

A personal learning tracker for tracking code and notes from learning various topics. Currently contains Rust (rustlings exercises) and is expanding to cover deep learning and LLMs.

## Repository Structure

```
Learning/
├── rust/
│   └── rustlings/          # Rustlings exercises (forked from rust-lang/rustlings)
│       ├── exercises/       # Exercise files with TODO stubs to fill in
│       ├── solutions/       # Reference solutions
│       └── Cargo.toml       # All exercises registered as separate binaries
├── deep-learning/
│   ├── LEARNING_PLAN.md     # The structured DL/LLM learning plan
│   ├── foundations/         # Math, linear algebra, calculus notebooks
│   ├── neural-networks/     # Implementing NNs from scratch
│   ├── transformers/        # Transformer architecture and attention
│   ├── training/            # Training loops, optimizers, distributed training
│   ├── inference/           # Inference optimization, quantization, serving
│   └── llm-advanced/        # Agentic AI, tool use, RAG, fine-tuning
└── readme.md
```

## Rustlings

Rustlings is a collection of exercises to get you used to reading and writing Rust code. Progress is tracked in `rust/rustlings/.rustlings-state.txt` (currently 25/94 exercises complete).

**Run interactive session:**
```bash
cd rust/rustlings && rustlings
```

**Check a specific exercise:**
```bash
cd rust/rustlings && rustlings run <exercise_name>
# e.g.: rustlings run variables1
```

**Check all exercises (compile check):**
```bash
cd rust/rustlings && cargo check
```

**Build a single exercise binary:**
```bash
cd rust/rustlings && cargo build --bin <exercise_name>
```

Exercise files live in `exercises/NN_topic/topicN.rs`. Each file has a `// TODO` or `todo!()` to replace. Solutions are in `solutions/` for reference.

## Deep Learning Code Conventions

- Python code should use PyTorch unless otherwise specified
- Jupyter notebooks go in topic subdirectories with numbered prefixes (e.g. `01_matrix_ops.ipynb`)
- Standalone scripts go as `.py` files in the same subdirectory
- Each topic directory should have a `README.md` explaining what was learned and key resources used
