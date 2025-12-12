# 🌳 Trig Identity Derivation Tree (from Euler’s Formula)

### **Root (foundation)**

* Euler’s Formula:

  $$
  e^{i\theta} = \cos\theta + i\sin\theta
  $$

---

### **Level 1: Basic properties**

* Conjugate:

  $$
  e^{-i\theta} = \cos\theta - i\sin\theta
  $$
* Symmetry:

  $$
  \cos(-\theta)=\cos\theta, \quad \sin(-\theta)=-\sin\theta
  $$
* Cosine and sine as exponentials:

  $$
  \cos\theta = \tfrac{1}{2}(e^{i\theta}+e^{-i\theta}), \quad
  \sin\theta = \tfrac{1}{2i}(e^{i\theta}-e^{-i\theta})
  $$

---

### **Level 2: Sum & Difference identities**

From

$$
\frac{e^{i\alpha}}{e^{i\beta}} = e^{i(\alpha-\beta)} \quad \text{and} \quad e^{i\alpha}e^{i\beta}=e^{i(\alpha+\beta)}
$$

We get:

* $\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta$
* $\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$

---

### **Level 3: Double & Multiple Angles**



**Start** with Euler’s formula and the product rule:

$$
\text{cis}(\theta)=e^{i\theta}=\cos\theta+i\sin\theta,\qquad
e^{i(\alpha+\beta)}=e^{i\alpha}e^{i\beta}.
$$

So for doubling the angle:

$$
\text{cis}(2\theta)=e^{i(2\theta)}=e^{i\theta}e^{i\theta}=\big(\cos\theta+i\sin\theta\big)\big(\cos\theta+i\sin\theta\big).
$$

**Expand the product** $(\cos\theta+i\sin\theta)^2$:

$$
(\cos\theta+i\sin\theta)^2
= \cos^2\theta + 2i\cos\theta\sin\theta + i^2\sin^2\theta.
$$

Since $i^2=-1$, this becomes

$$
= (\cos^2\theta - \sin^2\theta) + i(2\sin\theta\cos\theta).
$$

But the left side is also $\text{cis}(2\theta)=\cos(2\theta)+i\sin(2\theta)$.
**Compare real and imaginary parts**:

* Real part:

  $$
  \cos(2\theta)=\cos^2\theta-\sin^2\theta.
  $$

* Imaginary part:

  $$
  \sin(2\theta)=2\sin\theta\cos\theta.
  $$


### Useful equivalent forms

Use $\cos^2\theta+\sin^2\theta=1$ to get alternate expressions.

From $\cos(2\theta)=\cos^2\theta-\sin^2\theta$:

* replace $\cos^2\theta=1-\sin^2\theta$:

  $$
  \cos(2\theta)=(1-\sin^2\theta)-\sin^2\theta = 1-2\sin^2\theta.
  $$

* or replace $\sin^2\theta=1-\cos^2\theta$:

  $$
  \cos(2\theta)=\cos^2\theta-(1-\cos^2\theta)=2\cos^2\theta-1.
  $$

So you have three common forms:

$$
\cos(2\theta)=\cos^2\theta-\sin^2\theta=1-2\sin^2\theta=2\cos^2\theta-1.
$$


### Double-angle for tangent

Divide $\sin(2\theta)$ by $\cos(2\theta)$:

$$
\tan(2\theta)=\frac{2\sin\theta\cos\theta}{\cos^2\theta-\sin^2\theta}.
$$

Divide numerator and denominator by $\cos^2\theta$ (assuming $\cos\theta\neq0$):

$$
\tan(2\theta)=\frac{2\tan\theta}{1-\tan^2\theta}.
$$

---

### **Level 4: Half-Angle & Power-Reduction**

From double-angle rearrangements:

* $\sin^2\theta = \tfrac{1}{2}(1-\cos(2\theta))$
* $\cos^2\theta = \tfrac{1}{2}(1+\cos(2\theta))$

So for half-angle:

* $\sin\frac{\theta}{2} = \pm \sqrt{\tfrac{1-\cos\theta}{2}}$
* $\cos\frac{\theta}{2} = \pm \sqrt{\tfrac{1+\cos\theta}{2}}$

---

### **Level 5: Product-to-Sum / Sum-to-Product**

Using exponential forms of sine & cosine:

Example:

$$
\cos\alpha \cos\beta = \tfrac{1}{2}[\cos(\alpha+\beta)+\cos(\alpha-\beta)]
$$

$$
\sin\alpha \sin\beta = \tfrac{1}{2}[\cos(\alpha-\beta)-\cos(\alpha+\beta)]
$$

$$
\sin\alpha \cos\beta = \tfrac{1}{2}[\sin(\alpha+\beta)+\sin(\alpha-\beta)]
$$

---

### **Level 6: Tangent Identities**

From $\tan\theta = \frac{\sin\theta}{\cos\theta}$, using sum/difference:

$$
\tan(\alpha\pm\beta) = \frac{\tan\alpha \pm \tan\beta}{1 \mp \tan\alpha\tan\beta}
$$

Also double-angle for tangent:

$$
\tan(2\theta) = \frac{2\tan\theta}{1-\tan^2\theta}
$$

---

### **Level 7: Special cases & symmetries**

* Pythagorean identities: $\sin^2\theta+\cos^2\theta=1$ (from magnitude $|e^{i\theta}|=1$)
* Reciprocal identities: $\sec\theta = 1/\cos\theta$, etc.
* Cofunction identities: $\sin(\tfrac{\pi}{2}-\theta)=\cos\theta$, etc.

---

✨ So the whole **ecosystem of trig identities** cascades naturally from just one seed:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

---
