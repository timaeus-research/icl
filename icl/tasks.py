from typing import Generic, TypeVar
from itertools import combinations

import torch
from devinfra.utils.device import DeviceOrDeviceLiteral

T = TypeVar('T', 'GaussianTaskDistribution', 'DiscreteTaskDistribution', 'SingletonTaskDistribution')


def apply_transformations(ws: torch.Tensor, xs: torch.Tensor, error: float, device: DeviceOrDeviceLiteral = "cpu"):
    B, K, D = xs.shape

    errors = torch.normal(
        mean=0.,
        std=error,
        size=(B, K, 1,),
        device=device,
    )
    ys = xs @ ws.view(B, D, 1) + errors # B K D @ B D . + B K 1 -> B K 1

    return ys


class RegressionSequenceDistribution(Generic[T]):
    """
    Represents a synthetic in-context regression data set, where each token
    sequence is an i.i.d. sample of inputs and outputs of the form:
        `[ x_1, y_1, x_2, y_2, ..., x_K, y_K ]`
    where `y_i = w . x_i + N(0, noise_variance)` and `w` is sampled from
    `task_distribution`.

    Parameters:

    * `task_distribution : TaskDistribution`
        the distribution of true parameters `w` underlying each sequence.
    * `noise_variance : float >= 0`
        variance for gaussian error added to regression outputs.

    Fields:

    * `task_distribution : TaskDistribution`
        the distribution of true parameters `w` underlying each sequence.
    * `noise_variance : float >= 0`
        variance for gaussian error added to regression outputs.
    """
    task_distribution: T

    def __init__(self, task_distribution: T, noise_variance=0.25, device='cpu'):

        self.task_distribution = task_distribution
        self.noise_variance = noise_variance
        self.std = noise_variance**0.5

    def get_ws(self, B: int, D: int):
        """
        Get the true parameter vectors `w` underlying a batch of sequences
        of synthetic data (token sequences) for in-context regression.

        Parameters:

        * `batch_size : int >= 0`
            number of sequences to generate for this batch.
        
        Returns:

        * `ws : tensor(batch_size, task_size, device=device)`
            batch of sequences of true parameter vectors.
        """

        # sample a batch of random tasks
        ws = self.task_distribution.sample_tasks(B).view(B, D, 1) # B D -> B D 1
        return ws
    
    def get_xs(self, B: int, K: int, D: int, device='cpu'):
        # sample i.i.d. inputs 
        xs = torch.normal(
            mean=0.,
            std=1.,
            size=(B, K, D,),
            device=device,
        )
        return xs
    
    def get_errors(self, B: int, K: int, device='cpu'):
        # sample Gaussian errors
        errors = torch.normal(
            mean=0.,
            std=self.std,
            size=(B, K, 1,),
            device=device,
        )
        return errors 
    
    def get_ys(self, xs, ws, errors):
        # sample i.i.d. outputs
        ys = xs @ ws + errors
        return ys


    def get_batch(self, num_examples: int, batch_size: int):
        """
        Generate a batch of synthetic data (token sequences) for in-context
        regression.

        Parameters:

        * `num_examples : int >= 0`
            number of in-context examples to generate for each sequence in
            the batch. Note that the number of tokens will be twice as many,
            because each example has one input token followed by one output
            token.
        * `batch_size : int >= 0`
            number of sequences to generate for this batch.
        
        Returns:

        * `xs : tensor(batch_size, num_examples, task_size, device=device)`
            batch of sequences of input vectors.
        * `ys : tensor(batch_size, num_examples, 1, device=device)`
            batch of corresponding sequences of input vectors.
        """
        # shorthands
        B = batch_size
        K = num_examples
        D = self.task_distribution.task_size
        device = self.task_distribution.device

        # sample a batch of random tasks
        ws = self.get_ws(B, D) # -> B D 1
        xs = self.get_xs(B, K, D, device=device) # -> B K D
        errors = self.get_errors(B, K, device=device) # -> B K 1
        ys = self.get_ys(xs, ws, errors) # -> B K 1

        ys = apply_transformations(ws, xs, self.std, device)
        return xs, ys
        

    def loop_batches(self, num_examples: int, batch_size: int):
        """
        Iterate over batches of synthetic data (token sequences) for
        in-context regression.

        Yields:

        * `xs : tensor(batch_size, num_examples, task_size, device=device)`
            batch of sequences of input vectors.
        * `ys : tensor(batch_size, num_examples, 1, device=device)`
            batch of corresponding sequences of input vectors.
        """
        while True:
            yield self.get_batch(
                num_examples=num_examples,
                batch_size=batch_size,
            )


    def to(self, device: str):
        self.task_distribution.to(device)
        return self


class TaskDistribution:
    """
    Abstract base class: an abstract distribution over regression tasks, each
    task is a vector of size `task_size`, namely the vector of coefficients
    for the regression problem.
    The specific details of the distribution over tasks is to be provided by
    the subclass.

    Constructor parameters:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task_size, device='cpu'):
        self.task_size = task_size
        self.device = device


    def sample_tasks(self, n: int):
        """
        produce a sample of `n` tasks from the concrete task distribution

        parameters:

        * `n : int > 0`
            number of tasks to sample.
        
        returns:

        * `tasks : tensor(n, self.task_size, device=self.device)`
            the `n` tasks, each as one row of a 2d tensor.
        """
        return NotImplemented


    def to(self, device: str):
        """
        Move this task distribution to a different device

        parameters:

        * `device : str(device name)`
            which device to move to
        
        returns:

        * `self`
        """
        self.device = device
        return self


class DiscreteTaskDistribution(TaskDistribution):
    """
    Represent a fixed finite number of regression tasks `num_tasks` each of
    dimensionality `task_size`. 

    When method=='normal', the tasks are sampledi.i.d. from a `task_size`-dimensional 
    standard normal when this instance is constructed. 

    When method=='basis_vector_combinations', the tasks are generated according to the following rule:
    Define a discrete task distribution using basis vectors {e_1, ..., e_D} in R^D. 
    If M=1 then T={0} (control)
    If M=2 then T={0, e_1}
    If M=D+1 then T={0, e_1, e_2, ..., e_D}
    For M>D+1 the basis vectors are successively added together as different combinations. 
    e.g. M=D+3 then T={0, e_1, e_2, ..., e_D, e_1+e_2, e_1+e_3}. 
    e.g. M=1+D+nCr(D,2)+1 then T={0,e_1,e_2, ..., e_D, e_1+e_2, e_1+e_3, ..., e_{D-1}+e_D, e_1+e_2+e_3}.
    With this construction we (currently) require M <= 1+D+nCr(D,2)+nCr(D,3)+...+nCr(D,D) = 2^D (e.g. M<=256 if D=8). 
    

    Constructor parameters:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `num_tasks : int > 0`
        number of tasks in the discrete set.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    * `task_init_method : str`
        method for initialising tasks. Options are 'normal' or 'basis_vector_combinations'.
    * `method_params : dict`
        parameters for initialising tasks. For 'normal' method, method_params is empty. 
        Dor 'basis_vector_combinations' method, normal is a dictionary with keys 'scale_factor' and 'include_zero'.

    Fields:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `num_tasks : int > 0`
        number of tasks in the discrete set.
    * `method_params : dict`
        parameters for initialising tasks. For 'normal' method, method_params is empty.
    * `tasks : tensor(self.num_tasks, self.task_size, device=self.device)`
        the tasks, each as one row of a 2d tensor.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task_size: int, num_tasks: int, task_init_method='normal',
                 method_params=dict(), device='cpu'):
        # task_init_method = 'normal' or 'basis_vector_combinations'
        # method_params = dict() e.g. {'scale_factor':1, 'include_zero':True} for 'basis_vector_combinations'
        # basis_scale_factor=1, basis_include_zero=True
        super().__init__(task_size=task_size, device=device)

        self.num_tasks = num_tasks
        self.task_size = task_size
        self.method_params = method_params

        required_basis_params = {'scale_factor', 'include_zero'}
        if task_init_method == 'basis_vector_combinations' and not required_basis_params.issubset(method_params.keys()):
            raise ValueError("Not all required parameters are provided in 'method_params'.")

        if task_init_method == 'normal':
            # could optionally include mean, stdev data in method_params here
            self.tasks = torch.normal(
                mean=0.,
                std=1.,
                size=(self.num_tasks, self.task_size,),
                device=self.device,
            )
        elif task_init_method == 'basis_vector_combinations':
            self.tasks = self.generate_basis_vector_tasks()

    def generate_basis_vector_tasks(self):
        # Creates a M D sized tensor of tasks according to the rule outlined in class docstring. 

        M, D = self.num_tasks, self.task_size
        if M > 2**D:
            raise ValueError(f"num_tasks {M} must be less than or equal to 2^task_size {2**D} at the present time.")

        scale_factor, include_zero = self.method_params['scale_factor'], self.method_params['include_zero']
        
        # Create the identity matrix of size D x D, the basis vectors
        identity = torch.eye(D)
        # Initialize the tensor T with the zero vector if include_zero is True, otherwise as an empty tensor
        T = torch.zeros(1, D) if include_zero else torch.empty(0, D)
        
        # Add basis vectors
        if M > 1:
            basis_to_take = min(M-1, D)
            T = torch.cat((T, identity[:basis_to_take]))
            
        # Add combinations of basis vectors
        combination_count = M - D - 1
        if combination_count > 0:
            current_combination_length = 2
            while combination_count > 0:
                for comb in combinations(range(D), current_combination_length):
                    # comb = (0,1) for current_combination_length=2, (0,1,2) for current_combination_length=3 etc.
                    # produces e.g. e_1 + e_2 = [1,1,0,...,0]
                    combined_vector = torch.sum(identity[list(comb)], axis=0, keepdim=True)

                    # appends to T
                    T = torch.cat((T, combined_vector))

                    # if all excess combinations (beyond M-D-1) have been filled
                    combination_count -= 1
                    if combination_count == 0:
                        break
                current_combination_length += 1

        # scale_factor gives the option of scaling the task regression weights (in case noise_var=0.25^2 is too great). 
        # scale_factor defaults to 1.
        scaled_T = scale_factor*T
        
        return scaled_T.to(self.device)

    def sample_tasks(self, n: int):
        """
        Produce a uniformly random sample (with replacement) of `n` of
        the task distribution's tasks that were generated at construction
        time.

        Parameters:

        * `n : int > 0`
            number of tasks to sample
        
        Returns:

        * `tasks : tensor(n, self.task_size, device=self.device)`
            the `n` tasks, each as one row of a 2d tensor
        """
        task_selection = torch.randint(
            high=self.num_tasks,
            size=(n,),
            device=self.device,
        )
        return self.tasks[task_selection]

    def to(self, device: str):
        """
        Move this task distribution to a different device

        parameters:

        * `device : str(device name)`
            which device to move to
        
        returns:

        * `self`
        """
        self.tasks = self.tasks.to(device)
        return super().to(device)
    

class GaussianTaskDistribution(TaskDistribution):
    """
    Represent a gaussian distribution over of all possible regression tasks
    of dimensionality `task_size` (the tasks are selected i.i.d. from a
    `task_size`-dimensional standard normal when this task distribution is
    sampled from).

    Constructor parameters:
    
    * `task_size : int > 0`
        number of dimensions for all tasks
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task_size : int > 0`
        number of dimensions for all tasks
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """


    def sample_tasks(self, n: int):
        """
        produce a sample of `n` tasks drawn i.i.d. from a standard gaussian
        distribution in `self.task_size` dimensions

        parameters:

        * `n : int > 0`
            number of tasks to sample
        
        returns:

        * `tasks : tensor(n, self.task_size, device=self.device)`
            the `n` tasks, each as one row of a 2d tensor
        """
        tasks = torch.normal(
            mean=0.,
            std=1.,
            size=(n, self.task_size),
            device=self.device,
        )
        return tasks
    

class SingletonTaskDistribution(TaskDistribution):
    """
    An even simpler task distribution, for testing purposes. Just a single
    regression task with a fixed solution vector added at construction time.
    
    Constructor parameters:
    
    * `task : tensor(task_size)`
        fixed single task
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task : tensor(task_size)` (array-like ok)
        fixed single task.
    * `task_size : int > 0`
        number of dimensions for all tasks
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task, device='cpu'):
        self.task = torch.asarray(task, device=device)
        task_size, = self.task.shape
        super().__init__(task_size=task_size, device=device)


    def sample_tasks(self, n: int, device='cpu'):
        """
        Produce a batch of the singleton task repeated `n` times in a 2d
        tensor.

        Parameters:

        * `n : int > 0`
            number of tasks to sample
        
        Returns:

        * `tasks : tensor(n, self.task_size, device=device)`
            the task repeated `n` times, each as one row of a 2d tensor
        """
        return self.task.expand(n, -1)


    def to(self, device: str):
        """
        Move this task distribution to a different device

        parameters:

        * `device : str(device name)`
            which device to move to
        
        returns:

        * `self`
        """
        self.task = self.task.to(device)
        return super().to(device)



