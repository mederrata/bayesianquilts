
import tensorflow as tf

from bayesianquilts.tf.parameter import Decomposed, Interactions

def demo():
    interaction = Interactions(
    [
        ('sex', 2),
        ('race', 5),
        ('smoking', 2)
    ], exclusions=[]
    )

    beta = Decomposed(
        param_shape=[100],
        interactions=interaction,
        name='beta'
    )
    t, l, s = beta.generate_tensors(batch_shape=[5])
    beta.set_params(t)
    indices = [
        [1, 1, 1],
        [0, 3, 0],
        [0, 2, 1],
        [1, 0, 1]
    ]

    beta_effective = beta.lookup(indices)
    print(beta_effective.shape)

    @tf.function
    def test_lookup_graph():
        return beta.lookup(indices)

    x = tf.ones([100])
    print(beta.dot_sum(indices, x))

    @tf.function
    def test_sumparts_graph():
        return beta.dot_sum(indices, x)

    out = test_lookup_graph()
    print(out.shape)

    out_sum = test_sumparts_graph()
    print(out_sum.shape)
    return 


if __name__ == "__main__":
    demo()