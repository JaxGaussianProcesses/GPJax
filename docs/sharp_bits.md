# 🔪 The sharp bits

## Pseudo-randomness

Can briefly acknowledge and then point to the Jax docs for more information.

## Float64

The need for Float64 when inverting the Gram matrix

## Positive-definiteness

The need for jitter in the kernel Gram matrix

## Slow-to-evaluate

More than several thousand data points will require the use of inducing points - don't try and use the ConjugateMLL objective on a million data points.
