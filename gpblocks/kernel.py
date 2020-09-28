

# if __name__ == '__main__':
#     x = tf.reshape(tf.linspace(-1, 1, 100), (-1, 1))
#     kern = SquaredExponential(lengthscale=[0.1])
#     D = kern._distance(x, x)
#     assert D.shape[0] == D.shape[1]
#     assert D.shape[0] == x.shape[0]
#     assert tf.reduce_sum(tf.abs(tf.linalg.diag_part(D))) == 0.0
#     K = kern.compute_gram(x, x)
#     kern.lengthscale=[to_default_float(1.0)]
#     K2 = kern.compute_gram(x, x)
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
#     ax = axes.ravel()
#     ax[0].matshow(K)
#     ax[1].matshow(K2)
#     plt.show()
#     K3 = kern(x, x)
#     print(K3-K2)
