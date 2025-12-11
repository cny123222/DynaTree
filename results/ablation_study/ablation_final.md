======================================================================
FINAL ABLATION STUDY: HEAD-AWARE ATTENTION MASK
======================================================================

======================================================================
GROUP: A_window_only
======================================================================

  Running: A_window_512...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=512):32
  Layer 1: uniform(w=512):32
  Layer 2: uniform(w=512):32
  Layer 3: uniform(w=512):32
  Layer 4: uniform(w=512):32
  ... (showing first 5 of 32 layers)

PPL: 13.47, Acc: 47.85%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:27<00:00, 11.43it/s]
    PPL: 13.47, Acc: 47.85%, Eff.Ctx: 512.0

  Running: A_window_256...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=256):32
  Layer 1: uniform(w=256):32
  Layer 2: uniform(w=256):32
  Layer 3: uniform(w=256):32
  Layer 4: uniform(w=256):32
  ... (showing first 5 of 32 layers)

PPL: 21.09, Acc: 43.14%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:43<00:00,  9.61it/s]
    PPL: 21.09, Acc: 43.14%, Eff.Ctx: 256.0

  Running: A_window_128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=128):32
  Layer 1: uniform(w=128):32
  Layer 2: uniform(w=128):32
  Layer 3: uniform(w=128):32
  Layer 4: uniform(w=128):32
  ... (showing first 5 of 32 layers)

PPL: 26.28, Acc: 40.94%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [02:13<00:00,  7.46it/s]
    PPL: 26.28, Acc: 40.94%, Eff.Ctx: 128.0

  Running: A_window_64...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=64):32
  Layer 1: uniform(w=64):32
  Layer 2: uniform(w=64):32
  Layer 3: uniform(w=64):32
  Layer 4: uniform(w=64):32
  ... (showing first 5 of 32 layers)

PPL: 38.72, Acc: 36.04%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:49<00:00,  9.15it/s]
    PPL: 38.72, Acc: 36.04%, Eff.Ctx: 64.0

======================================================================
GROUP: B_streaming
======================================================================

  Running: B_streaming_512...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=508):32
  Layer 1: uniform(w=508):32
  Layer 2: uniform(w=508):32
  Layer 3: uniform(w=508):32
  Layer 4: uniform(w=508):32
  ... (showing first 5 of 32 layers)

PPL: 8.81, Acc: 52.45%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.49it/s]
    PPL: 8.81, Acc: 52.45%, Eff.Ctx: 512.0

  Running: B_streaming_256...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=252):32
  Layer 1: uniform(w=252):32
  Layer 2: uniform(w=252):32
  Layer 3: uniform(w=252):32
  Layer 4: uniform(w=252):32
  ... (showing first 5 of 32 layers)

PPL: 9.14, Acc: 51.95%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.49it/s]
    PPL: 9.14, Acc: 51.95%, Eff.Ctx: 256.0

  Running: B_streaming_128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=124):32
  Layer 1: uniform(w=124):32
  Layer 2: uniform(w=124):32
  Layer 3: uniform(w=124):32
  Layer 4: uniform(w=124):32
  ... (showing first 5 of 32 layers)

PPL: 9.52, Acc: 52.05%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.57it/s]
    PPL: 9.52, Acc: 52.05%, Eff.Ctx: 128.0

  Running: B_streaming_64...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: uniform(w=60):32
  Layer 1: uniform(w=60):32
  Layer 2: uniform(w=60):32
  Layer 3: uniform(w=60):32
  Layer 4: uniform(w=60):32
  ... (showing first 5 of 32 layers)

PPL: 10.57, Acc: 51.05%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [02:07<00:00,  7.85it/s]
    PPL: 10.57, Acc: 51.05%, Eff.Ctx: 64.0

======================================================================
GROUP: G_confidence_sweep
======================================================================

  Running: G_conf0.3_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):15, gathering(w=256):14, mixed(w=64):3
  Layer 1: gathering(w=128):5, gathering(w=256):8, mixed(w=64):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):12, gathering(w=256):12, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=128):1, positional(w=16):12
  Layer 4: gathering(w=128):3, gathering(w=256):1, mixed(w=64):25, positional(w=16):3
  ... (showing first 5 of 32 layers)

PPL: 11.21, Acc: 50.65%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [02:38<00:00,  6.30it/s]
    PPL: 11.21, Acc: 50.65%, Eff.Ctx: 61.6

  Running: G_conf0.4_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):25, gathering(w=256):4, mixed(w=64):3
  Layer 1: gathering(w=128):10, gathering(w=256):3, mixed(w=64):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):16, gathering(w=256):8, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=128):3, positional(w=16):10
  Layer 4: gathering(w=128):3, gathering(w=256):1, mixed(w=64):25, positional(w=128):1, positional(w=16):2
  ... (showing first 5 of 32 layers)

PPL: 11.01, Acc: 50.35%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [02:14<00:00,  7.44it/s]
    PPL: 11.01, Acc: 50.35%, Eff.Ctx: 61.8

  Running: G_conf0.5_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=64):3
  Layer 1: gathering(w=128):13, mixed(w=64):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):23, gathering(w=256):1, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=128):12, positional(w=16):1
  Layer 4: gathering(w=128):3, gathering(w=256):1, mixed(w=64):25, positional(w=128):2, positional(w=16):1
  ... (showing first 5 of 32 layers)

PPL: 10.94, Acc: 50.45%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:49<00:00,  9.12it/s]
    PPL: 10.94, Acc: 50.45%, Eff.Ctx: 73.6

  Running: G_conf0.55_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=128):13
  Layer 4: gathering(w=128):3, gathering(w=256):1, mixed(w=128):25, positional(w=128):2, positional(w=16):1
  ... (showing first 5 of 32 layers)

PPL: 9.76, Acc: 52.25%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.57it/s]
    PPL: 9.76, Acc: 52.25%, Eff.Ctx: 120.6

  Running: G_conf0.6_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=128):13
  Layer 4: gathering(w=128):4, mixed(w=128):25, positional(w=128):2, positional(w=16):1
  ... (showing first 5 of 32 layers)

PPL: 9.57, Acc: 51.85%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.57it/s]
    PPL: 9.57, Acc: 51.85%, Eff.Ctx: 128.3

  Running: G_conf0.65_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=128):13
  Layer 4: gathering(w=128):4, mixed(w=128):25, positional(w=128):2, positional(w=16):1
  ... (showing first 5 of 32 layers)

PPL: 9.63, Acc: 51.35%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.55it/s]
    PPL: 9.63, Acc: 51.35%, Eff.Ctx: 129.9

  Running: G_conf0.7_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=128):1, positional(w=16):2
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=128):13
  Layer 4: gathering(w=128):4, mixed(w=128):25, positional(w=128):2, positional(w=16):1
  ... (showing first 5 of 32 layers)

PPL: 9.63, Acc: 51.35%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.56it/s]
    PPL: 9.63, Acc: 51.35%, Eff.Ctx: 130.5

  Running: G_conf0.8_base128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=128):2, positional(w=16):1
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=128):13
  Layer 4: gathering(w=128):4, mixed(w=128):25, positional(w=128):3
  ... (showing first 5 of 32 layers)

PPL: 9.54, Acc: 51.35%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.56it/s]
    PPL: 9.54, Acc: 51.35%, Eff.Ctx: 131.9

  Running: G_conf0.5_base64...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=64):29, mixed(w=64):3
  Layer 1: gathering(w=64):13, mixed(w=64):16, positional(w=64):1, positional(w=8):2
  Layer 2: gathering(w=128):1, gathering(w=64):23, mixed(w=64):8
  Layer 3: gathering(w=64):2, mixed(w=64):17, positional(w=64):12, positional(w=8):1
  Layer 4: gathering(w=128):1, gathering(w=64):3, mixed(w=64):25, positional(w=64):2, positional(w=8):1
  ... (showing first 5 of 32 layers)

PPL: 11.27, Acc: 50.25%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.55it/s]
    PPL: 11.27, Acc: 50.25%, Eff.Ctx: 56.5

  Running: G_conf0.6_base64...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=64):29, mixed(w=64):3
  Layer 1: gathering(w=64):13, mixed(w=64):16, positional(w=64):1, positional(w=8):2
  Layer 2: gathering(w=64):24, mixed(w=64):8
  Layer 3: gathering(w=64):2, mixed(w=64):17, positional(w=64):13
  Layer 4: gathering(w=64):4, mixed(w=64):25, positional(w=64):2, positional(w=8):1
  ... (showing first 5 of 32 layers)

PPL: 10.47, Acc: 50.65%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.52it/s]
    PPL: 10.47, Acc: 50.65%, Eff.Ctx: 66.1

  Running: G_conf0.7_base64...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=64):29, mixed(w=64):3
  Layer 1: gathering(w=64):13, mixed(w=64):16, positional(w=64):1, positional(w=8):2
  Layer 2: gathering(w=64):24, mixed(w=64):8
  Layer 3: gathering(w=64):2, mixed(w=64):17, positional(w=64):13
  Layer 4: gathering(w=64):4, mixed(w=64):25, positional(w=64):2, positional(w=8):1
  ... (showing first 5 of 32 layers)

PPL: 10.42, Acc: 51.05%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.57it/s]
    PPL: 10.42, Acc: 51.05%, Eff.Ctx: 67.2

======================================================================
GROUP: I_inverse_positional
======================================================================

  Running: I_pos128_mix64_g128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=64):3
  Layer 1: gathering(w=128):13, mixed(w=64):16, positional(w=128):3
  Layer 2: gathering(w=128):24, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=128):13
  Layer 4: gathering(w=128):4, mixed(w=64):25, positional(w=128):3
  ... (showing first 5 of 32 layers)

PPL: 10.06, Acc: 51.25%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.58it/s]
    PPL: 10.06, Acc: 51.25%, Eff.Ctx: 96.6

  Running: I_pos256_mix64_g128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=64):3
  Layer 1: gathering(w=128):13, mixed(w=64):16, positional(w=256):3
  Layer 2: gathering(w=128):24, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=256):13
  Layer 4: gathering(w=128):4, mixed(w=64):25, positional(w=256):3
  ... (showing first 5 of 32 layers)

PPL: 9.73, Acc: 52.45%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.56it/s]
    PPL: 9.73, Acc: 52.45%, Eff.Ctx: 142.9

  Running: I_pos512_mix64_g128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=64):3
  Layer 1: gathering(w=128):13, mixed(w=64):16, positional(w=512):3
  Layer 2: gathering(w=128):24, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=512):13
  Layer 4: gathering(w=128):4, mixed(w=64):25, positional(w=512):3
  ... (showing first 5 of 32 layers)

PPL: 9.62, Acc: 52.15%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:43<00:00,  9.67it/s]
    PPL: 9.62, Acc: 52.15%, Eff.Ctx: 235.4

  Running: I_pos256_mix128_g128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=256):3
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=256):13
  Layer 4: gathering(w=128):4, mixed(w=128):25, positional(w=256):3
  ... (showing first 5 of 32 layers)

PPL: 9.34, Acc: 52.55%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.56it/s]
    PPL: 9.34, Acc: 52.55%, Eff.Ctx: 178.2

======================================================================
GROUP: J_small_gathering
======================================================================

  Running: J_pos16_mix64_g64...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=64):29, mixed(w=64):3
  Layer 1: gathering(w=64):13, mixed(w=64):16, positional(w=16):3
  Layer 2: gathering(w=64):24, mixed(w=64):8
  Layer 3: gathering(w=64):2, mixed(w=64):17, positional(w=16):13
  Layer 4: gathering(w=64):4, mixed(w=64):25, positional(w=16):3
  ... (showing first 5 of 32 layers)

PPL: 11.44, Acc: 50.35%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.54it/s]
    PPL: 11.44, Acc: 50.35%, Eff.Ctx: 50.7

  Running: J_pos16_mix64_g128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=64):3
  Layer 1: gathering(w=128):13, mixed(w=64):16, positional(w=16):3
  Layer 2: gathering(w=128):24, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=16):13
  Layer 4: gathering(w=128):4, mixed(w=64):25, positional(w=16):3
  ... (showing first 5 of 32 layers)

PPL: 11.27, Acc: 50.75%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.55it/s]
    PPL: 11.27, Acc: 50.75%, Eff.Ctx: 56.2

  Running: J_pos16_mix64_g256...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=256):29, mixed(w=64):3
  Layer 1: gathering(w=256):13, mixed(w=64):16, positional(w=16):3
  Layer 2: gathering(w=256):24, mixed(w=64):8
  Layer 3: gathering(w=256):2, mixed(w=64):17, positional(w=16):13
  Layer 4: gathering(w=256):4, mixed(w=64):25, positional(w=16):3
  ... (showing first 5 of 32 layers)

PPL: 11.34, Acc: 50.45%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.55it/s]
    PPL: 11.34, Acc: 50.45%, Eff.Ctx: 67.2

  Running: J_pos64_mix64_g128...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=64):3
  Layer 1: gathering(w=128):13, mixed(w=64):16, positional(w=64):3
  Layer 2: gathering(w=128):24, mixed(w=64):8
  Layer 3: gathering(w=128):2, mixed(w=64):17, positional(w=64):13
  Layer 4: gathering(w=128):4, mixed(w=64):25, positional(w=64):3
  ... (showing first 5 of 32 layers)

PPL: 10.43, Acc: 51.45%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.55it/s]
    PPL: 10.43, Acc: 51.45%, Eff.Ctx: 73.5

======================================================================
GROUP: K_uniform_aware
======================================================================

  Running: K_uniform64_aware...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=64):29, mixed(w=64):3
  Layer 1: gathering(w=64):13, mixed(w=64):16, positional(w=64):3
  Layer 2: gathering(w=64):24, mixed(w=64):8
  Layer 3: gathering(w=64):2, mixed(w=64):17, positional(w=64):13
  Layer 4: gathering(w=64):4, mixed(w=64):25, positional(w=64):3
  ... (showing first 5 of 32 layers)

PPL: 10.42, Acc: 50.75%: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.58it/s]
    PPL: 10.42, Acc: 50.75%, Eff.Ctx: 68.0

  Running: K_uniform128_aware...
Evaluating (head-aware mask):   0%|                                                                                                    | 0/999 [00:00<?, ?it/s][PerLayerMaskInjector] Registered 32 hooks

[Per-Layer Mask Config] Head type distribution per layer:
  Layer 0: gathering(w=128):29, mixed(w=128):3
  Layer 1: gathering(w=128):13, mixed(w=128):16, positional(w=128):3
  Layer 2: gathering(w=128):24, mixed(w=128):8
  Layer 3: gathering(w=128):2, mixed(w=128):17, positional(w=128):13
  Layer 4: gathering(w=128):4, mixed(w=128):25, positional(w=128):3
  ... (showing first 5 of 32 layers)

PPL: 9.60, Acc: 52.45%: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 999/999 [01:26<00:00, 11.56it/s]
    PPL: 9.60, Acc: 52.45%, Eff.Ctx: 132.0

======================================================================
ABLATION RESULTS SUMMARY
======================================================================

--- A_window_only ---
Name                           PPL      Acc    Eff.Ctx
-------------------------------------------------------
A_window_64                  38.72  36.04%       64.0
A_window_128                 26.28  40.94%      128.0
A_window_256                 21.09  43.14%      256.0
A_window_512                 13.47  47.85%      512.0

--- B_streaming ---
Name                           PPL      Acc    Eff.Ctx
-------------------------------------------------------
B_streaming_64               10.57  51.05%       64.0
B_streaming_128               9.52  52.05%      128.0
B_streaming_256               9.14  51.95%      256.0
B_streaming_512               8.81  52.45%      512.0

--- G_confidence_sweep ---
Name                           PPL      Acc    Eff.Ctx
-------------------------------------------------------
G_conf0.5_base64             11.27  50.25%       56.5
G_conf0.3_base128            11.21  50.65%       61.6
G_conf0.4_base128            11.01  50.35%       61.8
G_conf0.6_base64             10.47  50.65%       66.1
G_conf0.7_base64             10.42  51.05%       67.2
G_conf0.5_base128            10.94  50.45%       73.6
G_conf0.55_base128            9.76  52.25%      120.6
G_conf0.6_base128             9.57  51.85%      128.3
G_conf0.65_base128            9.63  51.35%      129.9
G_conf0.7_base128             9.63  51.35%      130.5
G_conf0.8_base128             9.54  51.35%      131.9

--- I_inverse_positional ---
Name                           PPL      Acc    Eff.Ctx
-------------------------------------------------------
I_pos128_mix64_g128          10.06  51.25%       96.6
I_pos256_mix64_g128           9.73  52.45%      142.9
I_pos256_mix128_g128          9.34  52.55%      178.2
I_pos512_mix64_g128           9.62  52.15%      235.4

--- J_small_gathering ---
Name                           PPL      Acc    Eff.Ctx
-------------------------------------------------------
J_pos16_mix64_g64            11.44  50.35%       50.7
J_pos16_mix64_g128           11.27  50.75%       56.2
J_pos16_mix64_g256           11.34  50.45%       67.2
J_pos64_mix64_g128           10.43  51.45%       73.5

--- K_uniform_aware ---
Name                           PPL      Acc    Eff.Ctx
-------------------------------------------------------
K_uniform64_aware            10.42  50.75%       68.0
K_uniform128_aware            9.60  52.45%      132.0

======================================================================
KEY COMPARISONS
======================================================================

Baseline (B_streaming_128): PPL=9.52, Ctx=128

Confidence-based vs StreamingLLM_128:
  G_conf0.8_base128: PPL=9.54, Ctx=132 (+0.2% worse)
  G_conf0.6_base128: PPL=9.57, Ctx=128 (+0.6% worse)
  G_conf0.65_base128: PPL=9.63, Ctx=130 (+1.2% worse)
  G_conf0.7_base128: PPL=9.63, Ctx=130 (+1.2% worse)
  G_conf0.55_base128: PPL=9.76, Ctx=121 (+2.6% worse)
  G_conf0.7_base64: PPL=10.42, Ctx=67 (+9.5% worse)
  G_conf0.6_base64: PPL=10.47, Ctx=66 (+10.1% worse)
  G_conf0.5_base128: PPL=10.94, Ctx=74 (+15.0% worse)
  G_conf0.4_base128: PPL=11.01, Ctx=62 (+15.7% worse)
  G_conf0.3_base128: PPL=11.21, Ctx=62 (+17.8% worse)
  G_conf0.5_base64: PPL=11.27, Ctx=56 (+18.4% worse)

Inverse (large positional) vs StreamingLLM:
  I_pos256_mix128_g128: PPL=9.34, Ctx=178
    vs B_streaming_128: -1.8%
  I_pos512_mix64_g128: PPL=9.62, Ctx=235
    vs B_streaming_256: +5.2%
  I_pos256_mix64_g128: PPL=9.73, Ctx=143
    vs B_streaming_128: +2.3%
  I_pos128_mix64_g128: PPL=10.06, Ctx=97
    vs B_streaming_128: +5.7%

Small gathering window experiments:
  J_pos64_mix64_g128: PPL=10.43, Ctx=74 (-1.3%)
  J_pos16_mix64_g128: PPL=11.27, Ctx=56 (+6.6%)
  J_pos16_mix64_g256: PPL=11.34, Ctx=67 (+7.3%)
  J_pos16_mix64_g64: PPL=11.44, Ctx=51 (+8.3%)

*** BEST RESULT: B_streaming_512 ***
    PPL: 8.81, Acc: 52.45%, Ctx: 512