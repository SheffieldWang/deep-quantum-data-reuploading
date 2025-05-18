import jax.numpy as jnp

class MetricComputer:
    def __init__(self):
        self.metric_map = {
            'binary_accuracy': self.compute_binary_accuracy,
            'accuracy': self.compute_accuracy, 
            'abs_error': self.compute_abs_error,
            'pred_error': self.compute_pred_error
        }

    def compute_binary_accuracy(self,outputs,targets):
        # Check if logits sum to 1    
        predictions = jnp.sign(outputs)
        accuracy = jnp.sum(predictions == targets) / len(targets)
        return accuracy
    
    def compute_accuracy(self,outputs,targets):
        predictions = jnp.argmax(outputs,axis=1)
        accuracy = jnp.sum(predictions == targets) / len(targets)
        return accuracy

    def compute_pred_error(self,outputs,targets):
        # due to the property of quantum measurement, the outputs are logits
        logits = outputs
        prediction_probability = logits[jnp.arange(len(targets)),targets] 
        pred_error = jnp.sum(1.0-prediction_probability) / len(targets)
        return pred_error

    def compute_abs_error(self,outputs,targets):
        predictions = outputs
        abs_error = jnp.mean(jnp.abs(predictions - targets))
        return abs_error
        
    def compute_metric(self, metric_name, outputs, targets):
        if metric_name not in self.metric_map:
            raise ValueError(f"Unsupported metric: {metric_name}")
        return self.metric_map[metric_name](outputs, targets)