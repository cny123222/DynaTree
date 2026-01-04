# Timeline Comparison Figure - Drawing Guide

## 🎯 Goal

Create a timeline comparison showing how three decoding methods execute over time:
1. **Autoregressive (AR)** - one token at a time, serial
2. **Linear Speculative** - draft then verify, 2-phase
3. **DynaTree** - tree draft + parallel verify

**Reference**: Similar to Figure 1(b) in the SpecInfer paper

---

## 📐 Overall Structure

**Three horizontal sections** stacked vertically, each showing one method's execution timeline.

**Style**: Academic paper diagram
- Serif font (Times New Roman style)
- Muted colors (avoid bright neon)
- Clean boxes and arrows
- Professional look

**Output**: PDF or high-res PNG (300 DPI)

---

## 🔵 Section (a): Autoregressive Decoding

**Concept**: Serial execution - generate one token at a time

**Visual**:
```
[LLM→t₁] → [LLM→t₂] → [LLM→t₃] → [LLM→t₄] → [LLM→t₅]
```

**Style**:
- 5 blue boxes in a row
- Single arrows (→) connecting them
- Each box shows "LLM" producing one token

**Message**: Slow due to sequential execution

---

## 🟢 Section (b): Linear Speculative Decoding

**Concept**: Two phases - draft generates sequence, LLM verifies in parallel

**Visual**:
```
[Draft: t₁→t₂→t₃→t₄→t₅→t₆] ⇒ [LLM Verify] ⇒ [Next Draft: t₃→...]
                                   ↓
                              ✓ Accept: t₁,t₂
```

**Style**:
- Green box (Draft)
- Orange box (Verify)
- Double arrows (⇒) for batch processing
- Show accepted tokens with checkmark

**Message**: Reduces rounds but draft is still sequential

---

## 🟣 Section (c): DynaTree (Tree-based)

**Concept**: Draft generates tree of candidates, LLM verifies all paths at once

**Visual**:
```
[Draft Tree:    ] ⇒ [LLM TreeVerify] ⇒ [Next Tree: ...]
     t₁                    ↓
    /│\              ✓ Accept path:
  t₂ t₃ t₄            t₁→t₂→t₃
  /│ │\ ...
```

**Style**:
- Purple box (Draft Tree)
- Red box (TreeVerify)
- Show tree structure inside first box (can be simplified)
- Highlight accepted path in green

**Message**: Maximum parallelism through multi-path exploration

---

## 🎨 Color Scheme

Use muted academic colors:

| Method | Primary Color | Box Fill |
|--------|--------------|----------|
| AR | Dark blue (#1976D2) | Light blue (#E3F2FD) |
| Linear Draft | Dark green (#388E3C) | Light green (#E8F5E9) |
| Linear Verify | Dark orange (#F57C00) | Light orange (#FFF3E0) |
| Tree Draft | Dark purple (#7B1FA2) | Light purple (#F3E5F5) |
| Tree Verify | Dark red (#C62828) | Light red (#FFEBEE) |

**Accents**:
- Accepted tokens: Green (#2E7D32)
- Regular text: Dark gray (#333333)
- Description: Gray (#666666)

---

## 💡 Key Design Principles

1. **Visual contrast**: Three methods should look distinctly different
2. **Left-to-right flow**: Show time progression clearly
3. **Emphasis on parallelism**: Use double arrows (⇒) for batch operations
4. **Highlight acceptance**: Show which tokens are accepted (✓)
5. **Simple tree**: Don't overcomplicate the tree structure - a simple 2-3 level sketch is fine

---

## 📝 Text to Include

**Titles** (top of each section):
- (a) Autoregressive Decoding (AR)
- (b) Linear Speculative Decoding
- (c) DynaTree (Tree-based Speculative Decoding)

**Descriptions** (below each section, optional):
- (a): "Serial execution, one token per step"
- (b): "Draft generates sequence, LLM verifies batch"
- (c): "Tree draft explores multiple paths, parallel verification"

---

## 🌳 How to Draw the Tree (Section c)

**Simple approach** (recommended):
```
     t₁
    /│\
  t₂ₐ t₂ᵦ t₂ᶜ
  │  │  │
  ... ... ...
```
Just show 2 levels clearly, indicate more levels with "..."

**Alternative**: Text-based
```
Multi-path tree
Depth: 3-7 levels
Branch: 2-3 per node
```

**Don't worry about perfection** - the concept matters more than exact tree structure.

---

## ✨ What Makes a Good Timeline Figure

✅ **Clear visual distinction** between the three methods
✅ **Obvious parallelism** in Linear and DynaTree (use ⇒ arrows)
✅ **Serial nature** of AR is apparent (use → arrows)
✅ **Tree structure** is recognizable (even if simplified)
✅ **Professional look** (muted colors, serif font, clean layout)

❌ Don't over-complicate - simplicity wins
❌ Don't use neon/bright colors - keep it academic
❌ Don't make text too small - readability first

---

## 🎯 Success Criteria

**A reader should immediately see:**
1. AR processes sequentially (slow)
2. Linear has 2 phases (better, but draft is sequential)
3. DynaTree explores tree in parallel (fastest, most parallelism)

**The figure should tell the story**: "DynaTree achieves maximum parallelism by exploring multiple candidate paths simultaneously."

---

## 🚀 Your Creative Freedom

Feel free to:
- Adjust box sizes for better visual balance
- Choose your own layout (vertical sections, side-by-side, etc.)
- Simplify the tree structure as needed
- Add subtle visual enhancements (shadows, gradients, etc.)
- Use your judgment on spacing and proportions

**Just keep the core concept clear!**

---

## 📦 Deliverable

Export as:
- **PDF** (preferred, vector format)
- Or **PNG** at 300 DPI minimum

Reasonable size: around 1200×900px or larger

---

**Remember**: The goal is to communicate the concept, not to create a pixel-perfect replica. Use your creativity to make it clear and professional! 🎨

