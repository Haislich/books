# Fourier Series

The Fourier series represents a periodic function as a sum of sine and cosine functions, or equivalently, as a sum of complex exponentials. A function \(f(t)\) that is periodic over a period \(T\) can be represented as:

\[
f(t) = a_0 + \sum_{n=1}^\infty \left(a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right)\right)
\]

Where the Fourier coefficients \(a_n\) and \(b_n\) are given by:

\[
a_n = \frac{2}{T} \int_0^T f(t) \cos\left(\frac{2\pi n t}{T}\right) \, dt
\]
\[
b_n = \frac{2}{T} \int_0^T f(t) \sin\left(\frac{2\pi n t}{T}\right) \, dt
\]
