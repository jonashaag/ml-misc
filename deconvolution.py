"""
Gradient descent based deconvolution using TensorFlow.

From a convolution

    convolved = signal * kernel

where * denotes the convolution operation, given 'convolved' and 'signal',
reconstruct 'kernel' using gradient descent.

Much better accuracy than e.g. Wiener deconvolution.

It works by training a model to compute 'convolved' (output) from 'signal'
(input) using only convolution, which forces the model to learn 'kernel'.

Uses Wiener deconvolution as a shortcut to speed up the process, but can also
learn entirely unknown 'kernel'.

Example for accuracy gain using this process (mean squared errors between actual
'kernel' and reconstructed 'kernel'):

    Wiener deconvolution: 8.6173e-07
    10k steps: 1.1445e-07
    20k steps: 6.5040e-08
    30k steps: 3.5857e-08

    10k steps without Wiener deconvolution initialization: 1.5045e-05
"""
import numpy as np
import scipy
import tensorflow as tf


def wiener_deconvolution(signal, kernel, snr):
    kernel = np.hstack(kernel, np.zeros(len(signal) - len(kernel)))
    H = np.fft.fft(kernel)
    deconvolved = np.real(
        np.fft.ifft(np.fft.fft(signal) * np.conj(H) / (H * np.conj(H) + snr ** 2))
    )
    return deconvolved


class IrSimilarityMetric(tf.keras.metrics.Metric):
    """Track accuracy of kernel approximation during training.

    Print the mean squared error between actual kernel and reconstructed kernel
    during training. This of course requires that the actual (expected) kernel
    is known, so it is only meaningful during development.
    """

    def __init__(self, actual_kernel, model, *a, **k):
        super().__init__(*a, **k)
        self._model = model
        # TensorFlow convolution kernels are flipped
        self._actual_kernel = actual_kernel[::-1]

    def result(self):
        return tf.math.reduce_mean(
            (tf.reshape(self._model.weights[0], kernel_shape) - self._actual_kernel)
            ** 2
        )

    update_state = reset_states = lambda *a, **k: None


PADDING = "valid"

signal = ...
kernel = ...
kernel_shape = kernel.shape
convolved = scipy.signal.convolve(signal, kernel, PADDING)


# Starting here, we pretend to not know the actual kernel.
# Remove this line to use IrSimilarityMetric.
del kernel


# Wiener deconvolution initialization.
# If you skip this, training will still converge, but take longer.
# If you are using a batch dimension (see below), pass only a single element
# to wiener_deconvolution() here, or somehow interpolate the multiple results
# from multiple calls to wiener_deconvolution() into a single result of shape (k,)).
kernel_deconvolved_wiener = wiener_deconvolution(
    convolved,
    signal[kernel_shape[0] - 1:] if PADDING == "valid" else NotImplemented,
    1e-1,
)[:kernel_shape[0]]


# We train a model to transform 'signal' into 'convolved'.
# The model also supports batching: You may provide multiple pairs of 'signal'
# and 'convolved' through an optional batch dimension (first dimension), in
# which case the model will try to find a 'kernel' that fits all of the pairs.
ds = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensors(signal),
        tf.data.Dataset.from_tensors(convolved[..., kernel_shape[0]:]),
    )
)

inp = tf.keras.Input(shape=signal.shape[-1:])
inp_chan = tf.expand_dims(inp, -1)
kernel_r = tf.keras.layers.Conv1D(
    1,
    kernel_shape,
    strides=1,
    padding=PADDING,
    use_bias=False,
    # Remove the 'kernel_initializer' argument if not using Wiener initialization.
    kernel_initializer=tf.constant_initializer(
        # TensorFlow convolution kernels are flipped
        kernel_deconvolved_wiener[::-1].reshape(-1, 1, 1)
    ),
)(inp_chan)[..., 0]
model = tf.keras.Model(inputs=inp, outputs=kernel_r)


model.compile(
    tf.keras.optimizers.Adam(lr=1e-4),
    "mse",
    metrics=[IrSimilarityMetric(kernel, model, name="kernel")],
)
print("Error before refinement:", model.evaluate(ds.batch(1)))
model.fit(
    ds.repeat().batch(1), steps_per_epoch=100_000,
)

# Reconstructed 'kernel' in SciPy order
tf.reshape(model.weights[0], kernel_shape)[...,::-1]
