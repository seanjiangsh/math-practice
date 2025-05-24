# 🧠 Solving Complex Equations with Euler’s Formula and Roots of Unity

## 🔹 Euler’s Formula

Euler’s Formula connects exponential and trigonometric functions:

\[
e^{ix} = \cos(x) + i\sin(x)
\]

Where:

- \( e \) is Euler’s number
- \( i \) is the imaginary unit
- \( x \) is a real number (in radians)
- \( e^{ix} \) lies on the **unit circle** in the complex plane at angle \( x \)

---

## 🔹 Unit Circle and Complex Plane

Each complex number on the unit circle can be represented as:
\[
z = e^{i\theta} = \cos(\theta) + i\sin(\theta)
\]

This forms a **circle of radius 1** around the origin.

---

## 🔹 Solving \( x^n = 1 \): Roots of Unity

To solve:
\[
x^n = 1
\]

We find **n complex roots** evenly spaced on the unit circle.

### ✅ Steps:

1. **Write 1 in exponential form**:
   \[
   1 = e^{2k\pi i}, \text{ for } k \in \mathbb{Z}
   \]

2. **Solve \( x^n = e^{2k\pi i} \)**:
   \[
   x = e^{\frac{2k\pi i}{n}} \quad \text{for } k = 0, 1, 2, ..., n-1
   \]

3. **Each root** is:
   \[
   x_k = \cos\left(\frac{2k\pi}{n}\right) + i\sin\left(\frac{2k\pi}{n}\right)
   \]

---

## 🧾 Example: Solve \( x^8 = 1 \)

1. \( x = e^{2k\pi i / 8} \) for \( k = 0 \) to \( 7 \)
2. Roots:
   - \( x_0 = e^{0i} = 1 \)
   - \( x_1 = e^{\pi i/4} = \cos(\pi/4) + i\sin(\pi/4) \)
   - \( x_2 = e^{\pi i/2} = i \)
   - ...
   - \( x_7 = e^{7\pi i/4} \)

These 8 points lie equally spaced on the unit circle.

---

## 🛠 Applications

- Useful in factoring \( x^n - 1 \)
- Applies in signal processing, discrete Fourier transforms
- Helps visualize **rotation and symmetry** in complex roots

---
