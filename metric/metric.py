class Metrics:
    def __init__(self):
        self.metrics = {}
        self.indices = {}
        self.index_types = {}
        self.stats = {}
        
    def register_metric(self, name, split='train', index_type='iteration'):
        """Register a new metric for tracking with specified index type"""
        if split not in self.metrics:
            self.metrics[split] = {}
            self.indices[split] = {}
            self.index_types[split] = {}
            self.stats[split] = {}
        self.metrics[split][name] = []
        self.indices[split][name] = []
        self.index_types[split][name] = index_type
        self.stats[split][name] = {'max': float('-inf'), 
                                  'min': float('inf'),
                                  'mean': 0.0,
                                  'std': 0.0}

    def update(self, name, value, index=None, split='train', index_type='iteration'):
        """Update a metric value with optional index"""
        if split not in self.metrics or name not in self.metrics[split]:
            raise ValueError(f"Metric {name} for split {split} has not been registered. Please call register_metric() first.")
            
        value = float(value)
        self.metrics[split][name].append(value)
        self.indices[split][name].append(index if index is not None else len(self.indices[split][name]))
        
        # Update statistics
        values = self.metrics[split][name]
        self.stats[split][name]['max'] = max(self.stats[split][name]['max'], value)
        self.stats[split][name]['min'] = min(self.stats[split][name]['min'], value)
        self.stats[split][name]['mean'] = sum(values) / len(values)
        
        # Calculate standard deviation
        if len(values) > 1:
            mean = self.stats[split][name]['mean']
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            self.stats[split][name]['std'] = variance ** 0.5

    def get_metrics(self, split='train'):
        """Get all metrics for a split"""
        if split not in self.metrics:
            raise ValueError(f"Split {split} has not been registered. Please call register_metric() first.")
        return {
            'values': self.metrics[split],
            'indices': self.indices[split],
            'index_types': self.index_types[split],
            'stats': self.stats[split]
        }

    def get_metric(self, name, split='train', index=None):
        """Get a specific metric and its indices
        
        Args:
            name: metric name
            split: data split (train/test)
            index: if provided, return the value at this index
                  can also be 'min' or 'max' to get min/max values
            
        Returns:
            If index is None: returns full metric dict
            If index is provided: returns value at that index
            If index is 'min'/'max': returns min/max value
        """
        if split not in self.metrics or name not in self.metrics[split]:
            raise ValueError(f"Metric {name} for split {split} has not been registered. Please call register_metric() first.")
            
        if index is not None:
            if index == 'min':
                return self.stats[split][name]['min']
            elif index == 'max':
                return self.stats[split][name]['max']
            # Return value at the specified index
            if -len(self.metrics[split][name]) <= index < len(self.metrics[split][name]):
                return self.metrics[split][name][index]
            raise ValueError(f"Index {index} is out of bounds for metric {name} in split {split}.")
            
        return {
            'values': self.metrics[split][name],
            'indices': self.indices[split][name], 
            'index_type': self.index_types[split][name],
            'stats': self.stats[split][name]
        }

    def reset(self):
        """Reset all metrics"""
        self.__init__()
