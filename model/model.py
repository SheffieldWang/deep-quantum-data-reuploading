import pennylane as qml
import jax
import jax.numpy as jnp
import warnings 
from flax import nnx
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)



class BaseQuantumCircuit(nnx.Module):
    def __init__(self, 
                 n_qubits=1,
                 device_name='default.qubit', 
                 interface='jax',
                 measurement_type='probs',
                 hamiltonian=None,
                 measure_wires=None):
        """
        Initialize the quantum model.

        Parameters:
        - n_qubits (int): Number of qubits.
        - n_reps (int): Number of repetitions.
        - n_layers (int): Number of layers per repetition.
        - device_name (str): Name of the quantum device.
        - interface (str): Interface type, e.g., 'jax'.
        """      
        self.n_qubits = n_qubits
        self.interface = interface
        self.device = qml.device(device_name, wires=self.n_qubits)
        self.measurement_type = measurement_type
        self.hamiltonian = hamiltonian 
        self.measure_wires = measure_wires

        self._validate_measurement_settings()
        
    def _validate_measurement_settings(self):
        """Validate measurement settings"""
        if self.measurement_type not in ['probs', 'hamiltonian', 'density', 'state']:
            raise ValueError("Measurement type must be 'probs', 'hamiltonian', 'density' or 'state'")
        if self.measurement_type == 'hamiltonian' and self.hamiltonian is None:
            raise ValueError("Hamiltonian is not provided")
        if (self.measurement_type == 'probs' or self.measurement_type == 'density') and self.measure_wires is None:
            warnings.warn("Measure wires are not provided, using all qubits")
            self.measure_wires = range(self.n_qubits)    


    def __call__(self,x):
        if self.measurement_type == 'probs':
            batched_probs = jax.vmap(self.probs, in_axes=(0), out_axes=0)
            return batched_probs(x)
        elif self.measurement_type == 'hamiltonian':
            batched_expvalH = jax.vmap(self.expvalH, in_axes=(0), out_axes=0)
            return batched_expvalH(x)
        elif self.measurement_type == 'density':
            batched_density = jax.vmap(self.density_matrix, in_axes=(0), out_axes=0)
            return batched_density(x)
        elif self.measurement_type == 'state':
            batched_state = jax.vmap(self.state, in_axes=(0), out_axes=0)
            return batched_state(x) 
        else:
            raise ValueError("Invalid measurement type")
        

    def ansatz(self, x):
        """Abstract method to be implemented in subclass"""
        raise NotImplementedError("Ansatz must be implemented in subclass")
    
              
                    
    def draw_circuit(self, x):
        """
        Draw quantum circuit diagram

        Parameters:
        - x (array): Input data

        Returns:
        - Quantum circuit diagram
        """
        @qml.qnode(self.device, interface=self.interface)
        def circuit_to_draw(x):
            self.ansatz(x,self.params)
            return qml.probs(wires=0)

        # Use PennyLane's draw_mpl() function to draw circuit
        # style: Chart style, can be 'default' or 'pennylane'
        # decimals: Number of decimal places to display
        # fontsize: Font size
        # max_length: Maximum display length
        fig = qml.draw_mpl(circuit_to_draw, 
                          style='pennylane',
                          decimals=2,
                          fontsize=13)(x)
        return fig

    def probs(self, x):
        """
        Calculate and return measurement probabilities.

        Parameters:
        - x (array): Input data
        - wires (int or list): Quantum wires to measure, defaults to 0

        Returns:
        - array: Measurement probabilities
        """
            
        @qml.qnode(self.device, interface=self.interface)
        def qnode_probs(x):
            self.ansatz(x)
            return qml.probs(wires=self.measure_wires)
        return qnode_probs(x)
    
    def expvalH(self, x):
        """
        Calculate and return the measurement expectation value.

        Args:
            - x (array): Input data

        Returns:
            - array: Measurement expectation value
        """
            
        @qml.qnode(self.device, interface=self.interface)
        def qnode_expvalH(x):
            self.ansatz(x)
            return qml.expval(self.hamiltonian)
        return qnode_expvalH(x)

    def state(self, x):
        """
        Get and return the quantum state.

        Args:
            - x (array): Input data

        Returns:
            - array: Quantum state
        """
        @qml.qnode(self.device, interface=self.interface)
        def qnode_state(x):
            self.ansatz(x)
            return qml.state()
        return qnode_state(x)
    
    
    def vn_entropy(self, x):
        """
        Calculate and return the von Neumann entropy.
        """
        @qml.qnode(self.device, interface=self.interface)
        def qnode_vn_entropy(x):
            self.ansatz(x)
            entropies = []
            for i in range(self.n_qubits):
                entropies.append(qml.vn_entropy(wires=[i]))
            return entropies
        return jnp.mean(jnp.array(qnode_vn_entropy(x)))



    def density_matrix(self, x):
        """
        Get and return the density matrix.

        Args:
            - x (array): Input data

        Returns:
            - array: Density matrix
        """
        @qml.qnode(self.device, interface=self.interface)
        def qnode_density(x):
            self.ansatz(x)
            return qml.density_matrix(wires=self.measure_wires)
        return qnode_density(x)


class ParameterizedQuantumCircuit(BaseQuantumCircuit):
    """Base class for quantum circuits with trainable parameters"""
    def __init__(self, 
                 n_qubits=1, 
                 **kwargs):
        super().__init__(n_qubits, **kwargs)
        self.params = None  # Will be initialized in subclass
    def update_params(self, new_params):
        """
        Update model parameters

        Parameters:
        - new_params: New parameter values, must match shape of self.params
        """
        if self.params is None:
            raise ValueError("Parameters not initialized")
        if new_params.shape != self.params.shape:
            raise ValueError(f"Parameter shape mismatch. Expected: {self.params.shape}, Got: {new_params.shape}")
        self.params = nnx.Param(new_params)
        
    def get_params(self):
        """
        Get current model parameters

        Returns:
        - Copy of current model parameters
        """
        if self.params is None:
            raise ValueError("Parameters not initialized")
        return jax.numpy.array(self.params)
    
    
    def save_params(self, filepath):
        """
        Save model parameters to file

        Parameters:
        - filepath: File path to save parameters, recommended to use .npy format
        """
        if self.params is None:
            raise ValueError("Parameters not initialized")
        
        params_numpy = jnp.array(self.params)
        jnp.save(filepath, params_numpy)



class ZeroPaddingCircuit(ParameterizedQuantumCircuit):
    """Quantum circuit with zero padding"""
    def __init__(self,
                 n_qubits=1,
                 n_reps=1,
                 n_layers=1,
                 max_layers=100,
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits,**kwargs)
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.n_layers = n_layers
        self.max_layers = max_layers
        self.shape = (self.n_reps, self.max_layers, self.n_qubits, 3)
        key = nnx.Rngs(seed).params()
        self.params = nnx.Param(jax.random.normal(key, self.shape))

    def ansatz(self, x):
        for p in range(self.n_reps):
            # Data layers
            for l in range(self.n_layers):
                for n in range(self.n_qubits):
                    qml.Rot(x[l][n][0], x[l][n][1], x[l][n][2], wires=n)
                    qml.Rot(self.params[p][l][n][0], self.params[p][l][n][1], self.params[p][l][n][2], wires=n)
                if self.n_qubits > 1:
                    for n in range(self.n_qubits):
                        qml.CNOT(wires=[n,(n+1)%self.n_qubits])
            # Zero padding layers
            for l in range(self.n_layers, self.max_layers):
                for n in range(self.n_qubits):
                    # qml.Rot(0.0, 0.0, 0.0, wires=n)
                    qml.Rot(self.params[p][l][n][0], self.params[p][l][n][1], self.params[p][l][n][2], wires=n)
                if self.n_qubits > 1:
                    for n in range(self.n_qubits):
                        qml.CNOT(wires=[n,(n+1)%self.n_qubits])


                    


class GeneralCircuit(ParameterizedQuantumCircuit):
    """General quantum circuit"""
    def __init__(self,
                 n_qubits=1,
                 n_reps=1,
                 n_layers=1,
                 seed=0,
                 **kwargs):
        super().__init__(n_qubits,**kwargs)
        self.n_qubits = n_qubits
        self.n_reps = n_reps
        self.n_layers = n_layers
        self.shape = (self.n_reps, self.n_layers, self.n_qubits, 3)
        key = nnx.Rngs(seed).params()
        self.params = nnx.Param(jax.random.normal(key, self.shape))

    def ansatz(self, x):
        for p in range(self.n_reps):
            for l in range(self.n_layers):
                for n in range(self.n_qubits):
                    qml.Rot(x[l][n][0], x[l][n][1], x[l][n][2], wires=n)
                    qml.Rot(self.params[p][l][n][0], self.params[p][l][n][1], self.params[p][l][n][2], wires=n)
                if self.n_qubits > 1:   
                    for n in range(self.n_qubits):
                        qml.CNOT(wires=[n,(n+1)%self.n_qubits])




class DataReuploading(nnx.Module):
    def __init__(self, ansatz_type="general", **kwargs):
        super().__init__()
        
        
        # Select appropriate circuit class based on ansatz_type
        circuit_classes = {
            'zero_padding': ZeroPaddingCircuit,
            'general': GeneralCircuit
        }

        CircuitClass = circuit_classes.get(ansatz_type, BaseQuantumCircuit)
        self.quantum_model = CircuitClass(**kwargs)

        
    def __call__(self, x):
        return self.quantum_model(x)