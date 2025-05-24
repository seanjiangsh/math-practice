# Converting Between Rectangular and Polar Equations

## 🔁 Conversion Overview

### 🟦 Rectangular → Polar

Use these identities:

- $x = r\cos(\theta)$
- $y = r\sin(\theta)$
- $r = \sqrt{x^2 + y^2}$
- $\tan(\theta) = \frac{y}{x}$ _(watch quadrants!)_

### 🟨 Polar → Rectangular

Use direct substitution:

- $x = r\cos(\theta)$
- $y = r\sin(\theta)$
- $x^2 + y^2 = r^2$

---

## 🔧 General Tips Table

| Type                | Rectangular Form      | Polar Form                         | Tip                                      |
| ------------------- | --------------------- | ---------------------------------- | ---------------------------------------- |
| Circle              | $x^2 + y^2 = r_0^2$   | $r = r_0$                          | Use $r = \sqrt{x^2 + y^2}$               |
| Line through origin | $y = mx$              | $\theta = \tan^{-1}(m)$            | Replace $y/x$ with $\tan(\theta)$        |
| Vertical line       | $x = a$               | $r = a / \cos(\theta)$             | Use $x = r\cos(\theta)$                  |
| Horizontal line     | $y = b$               | $r = b / \sin(\theta)$             | Use $y = r\sin(\theta)$                  |
| Spiral              | e.g., $y = x\tan(kx)$ | $r = f(\theta)$                    | Often cleaner in polar form              |
| Conic section       | Quadratic in $x, y$   | $r = \frac{ed}{1 \pm e\cos\theta}$ | Used for ellipses, parabolas, hyperbolas |

---

## ✅ Conversion Strategy

- Look for $x^2 + y^2$ → convert to $r^2$
- Look for $\frac{y}{x}$ → becomes $\tan(\theta)$
- Isolate $r$ when converting **to polar**
- Replace $r\cos(\theta)$, $r\sin(\theta)$ when converting **to rectangular**
