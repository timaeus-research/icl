from icl.tasks import RegressionSequenceDistribution
from icl.tasks import RegressionSequenceDistribution, DiscreteTaskDistribution, SingletonTaskDistribution
import torch

class InductionHeadsTask():
    """
    Construct batches of examples to assess Induction heads. 

    Constructor parameters:

    * `task_distribution : TaskDistribution`
        A task distribution used to construct induction head batch.
        If measuring performance on T_pretrain then task_distribution should be equal to the 
        DiscreteTaskDistribution object that was used to train the transformer. 

    """
    def __init__(self, task_distribution):
        
        self.regression_dist = RegressionSequenceDistribution(
                                task_distribution,
                                noise_variance=0.25**2,
                                )
        self.tasks = task_distribution.tasks # size M D
        
    def get_batch(self, batch_size, num_examples, kind='duplicate', fuzzy_std=0.01, device='cpu'):
        """
        Creates Induction batch from a drawn batch (B K-1 D) from instantiated regression distribution. 
        For each batch element of size (K-1 D) we create a new batch element of size (K D) of the (transposed) form 
        X_1 X_2 ... X_{K-1} X_1
        Y_1 Y_2 ... Y_{K-1} Y_1

        The returned batch therefore has size B*(K-1) K D where the first (K-1) batch elements are of the form
        X_1 X_2 ... X_{K-1} X_1
        X_1 X_2 ... X_{K-1} X_2
        X_1 X_2 ... X_{K-1} X_3
        ...
        X_1 X_2 ... X_{K-1} X_{K-1}
        and ditto for the Y's. This sequence is this repeated on each batch element of the original batch.

        If kind == 'duplicate' then the X's and Y's are duplicated exactly.
        If kind == 'fuzzy' then the X's are duplicated with Gaussian noise added to each element (X* = X + N(0,fuzzy_std)), 
        and the Y's are reproduced with the original w vectors (Y* = X* w + noise). 


        Parameters:

        * `kind : string`
            either 'duplicate' or 'fuzzy' as above. 
        * `batch_size : int`
        * `num_examples : int`
            K, the max number of examples in the transformer's context.
        * `fuzzy_std : float`
            standard deviation of Gaussian noise added to X's if kind == 'fuzzy'.

        Returns:

        * `final_xs : tensor(B*(K-1) K D)`
            batch of `B*(K-1)` sequences of `K` input vectors of `D` dims, as described above.
        * `final_ys : tensor(B*(K-1) K 1)`
            batch of `B*(K-1)` sequences of `K` output scalars, as described above.
        """

        if kind != 'duplicate' and kind != 'fuzzy': raise ValueError(f"Invalid value for 'kind': {kind}. Expected 'duplicate' or 'fuzzy'.")

        B, K, D = batch_size, num_examples, self.regression_dist.task_distribution.task_size 
        regression_noise_variance = self.regression_dist.noise_variance

        # num_examples = K i.e. 16 i.e. max context length of transformer (/2), NOT K-1 as in IH setup below

        Km1 = K-1

        ws = self.regression_dist.get_ws(B, D) # -> B D 1
        xs = self.regression_dist.get_xs(B, Km1, D, device=device) #    -> B K-1 D
        errors = self.regression_dist.get_errors(B, Km1, device=device) # -> B K-1 1
        ys = self.regression_dist.get_ys(xs, ws, errors) # -> B K 1

        # Repeat context sequence Km1 times along batch dimension 
        xs_rep = xs.repeat_interleave(repeats=Km1, dim=0) # B K-1 D -> B*(K-1) K-1 D
        ys_rep = ys.repeat_interleave(repeats=Km1, dim=0) # B K-1 1 -> B*(K-1) K-1 D
        ws_rep = ws.repeat_interleave(repeats=Km1, dim=0) # B D   1 -> B*(K-1) D   1

        ys_index_tensor = (torch
                        .arange(Km1)       # Sequence [0,...,K-1] shape K-1
                        .repeat(B)         # Repeat B times along dim=0 e.g. [0,1,2] -> [0,1,2,0,1,2]
                        .unsqueeze(-1)     # B*(K-1)     -> B*(K-1) 1  
                        .unsqueeze(-1)     # B*(K-1) 1   -> B*(K-1) 1 1
                        ) 
        xs_index_tensor = ys_index_tensor.expand(-1, -1, D) # Repeat D times, B*(K-1) 1 1 -> B*(K-1) 1 D
                        
        xs_copied = torch.gather(xs_rep, 1, xs_index_tensor) # Get X_1 X_2 etc. on right tensor axes, size B*(K-1) 1 D 
        ys_copied = torch.gather(ys_rep, 1, ys_index_tensor) # Get Y_1 Y_2 etc. on right tensor axes, size B*(K-1) 1 1
        
        if kind=='duplicate':
           xs_new, ys_new = xs_copied, ys_copied
        elif kind=='fuzzy':
            xs_new = xs_copied + fuzzy_std*torch.randn_like(xs_copied) # B*(K-1) 1 D
            errors = torch.normal(
                mean=0.,
                std=regression_noise_variance**0.5,
                size=(B*(K-1), 1, 1,),
            )
            ys_new = xs_new @ ws_rep + errors # B*(K-1) 1 D @ B*(K-1) D 1 + B*(K-1) 1 1 -> B*(K-1) 1 1
            
        final_xs = torch.cat((xs_rep, xs_new), dim=1) # Size B*(K-1) K D
        final_ys = torch.cat((ys_rep, ys_new), dim=1) # Size B*(K-1) K 1

        return final_xs, final_ys

class RegressionTask():
    """
    Create batches of examples drawn from a chosen (or randomised) regression task w in task_distribution. 
    
    Constructor parameters:

    * `task_distribution : TaskDistribution`
        A task distribution. If measuring performance on T_pretrain then task_distribution should be equal to the 
        DiscreteTaskDistribution object that was used to train the transformer. 
    """

    def __init__(self, task_distribution):
        self.tasks = task_distribution.tasks # size M D

    def get_batch(self, batch_size, num_examples, randomise_task=False, rand_task_idx=0, device='cpu'):
        """
        Get a batch on a fixed regression task w, or a randomly sampled regression task w.

        Parameters:
            * `batch_size : int`
            * `num_examples : int`
            * `randomise_task : boolean`
                If true, random task will be picked from task distribution. 
                If false, task will be picked from task distribution at index rand_task_idx.
            * `rand_task_idx : int`
            * `device : string`

        Returns: 
            * `xs : tensor(B K D)`
            * `ys : tensor(B K 1)`
        """
        if rand_task_idx >= self.tasks.size()[0]:
            raise ValueError(f"rand_task_idx {rand_task_idx} is out of bounds for tasks with size {self.tasks.size()[0]}")


        # if randomise_task then randomly sample a task from the task distribution, otherwise use task index specified by rand_task
        if randomise_task:
            # sample a random row from self.tasks tensor of size M D
            rand_task_idx = torch.randint(low=0, high=self.tasks.size()[0], size=(1,)).item()
        rand_task = self.tasks[rand_task_idx] # size D, a single regression weight w

        singleton_task_dist = RegressionSequenceDistribution(
                                SingletonTaskDistribution(rand_task),
                                noise_variance=0.25**2,
                                )
        xs, ys = singleton_task_dist.get_batch(num_examples=num_examples, batch_size=batch_size)
        return xs, ys
