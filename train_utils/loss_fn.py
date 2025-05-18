import jax.numpy as jnp
import optax


def hinge_loss_fn(model,data,target):
    outputs = model(data)
    loss = jnp.mean(optax.hinge_loss(outputs,target))
    return loss,outputs


def mse_loss_fn(model,data,target):
    outputs = model(data)
    loss = jnp.mean((outputs-target)**2)
    return loss,outputs

def cross_entropy_loss_fn(model,data,target):
    outputs = model(data)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=outputs, labels=target).mean()
    return loss,outputs
