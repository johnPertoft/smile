import tensorflow as tf


def reuse_scope_after_first_use(scope):
    def wrapper_with_scope(ops_fn):
        should_reuse = False

        def wrapped(*args):
            nonlocal should_reuse
            with tf.variable_scope(scope, reuse=should_reuse):
                result = ops_fn(*args)
            should_reuse = True
            return result

        return wrapped
    return wrapper_with_scope
