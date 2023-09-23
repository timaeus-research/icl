from icl.tasks import RegressionSequenceDistribution
from icl.tasks import RegressionSequenceDistribution, DiscreteTaskDistribution
import torch

class InductionHeadsTask():
    """
    Construct batches of examples to assess Induction heads. 

    Constructor parameters:

    * `task_distribution : TaskDistribution`
        A task distribution used to construct induction head batch.
        If measuring performance on T_pretrain then task_distribution should be equal to 
        DiscreteTaskDistribution that was used to train the transformer. 
    * `num_tasks : int > 0`
        number of tasks in the discrete set.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on

    Fields:

    * `task_size : int > 0`
        number of dimensions for the tasks.
    * `num_tasks : int > 0`
        number of tasks in the discrete set.
    * `tasks : tensor(self.num_tasks, self.task_size, device=self.device)`
        the tasks, each as one row of a 2d tensor.
    * `device='cpu' : str(device name)`
        which device to initialise tasks on
    """
    def __init__(self, task_distribution):
        self.regression_dist = RegressionSequenceDistribution(
                                task_distribution,
                                noise_variance=0.25**2,
                                )
        
    def get_batch(self, kind, batch_size, num_examples):
        # kind = 'duplicate' or 'fuzzy' 
        # num_examples = K i.e. 16 i.e. max context length of transformer (/2), NOT K-1 as in IH setup below
        Km1 = num_examples-1
        xs, ys = self.regression_dist.get_batch(num_examples = Km1, 
                                               batch_size=batch_size,
                                               ) 
        
        B, K, D = batch_size, num_examples, xs.size()[2]
        # Repeat context sequence Km1 times along batch dimension 
        xs_rep = xs.repeat_interleave(repeats=Km1, dim=0) # B K-1 D -> B*(K-1) K-1 D
        ys_rep = ys.repeat_interleave(repeats=Km1, dim=0) # B K-1 1 -> B*(K-1) K-1 D
        # * `xs : tensor(batch_size, num_examples, task_size, device=device)`
        #    batch of sequences of input vectors.
        # * `ys : tensor(batch_size, num_examples, 1, device=device)`
        #    batch of corresponding sequences of input vectors.
        
        ys_index_tensor = (torch
                        .arange(Km1)       # Sequence [0,...,K-1] shape K-1
                        .repeat(B)         # Repeat B times along dim=0 e.g. [0,1,2] -> [0,1,2,0,1,2]
                        .unsqueeze(-1)     # B*(K-1)     -> B*(K-1) 1  
                        .unsqueeze(-1)     # B*(K-1) 1   -> B*(K-1) 1 1
                        ) 
        xs_index_tensor = ys_index_tensor.expand(-1, -1, D) # Repeat D times, B*(K-1) 1 1 -> B*(K-1) 1 D
                        
        
        xs_copied = torch.gather(xs_rep, 1, xs_index_tensor) # Get X_1 X_2 etc. on right tensor axes, size B*(K-1) 1 D 
        ys_copied = torch.gather(ys_rep, 1, ys_index_tensor) # Get Y_1 Y_2 etc. on right tensor axex
        
        final_xs = torch.cat((xs_rep, xs_copied), dim=1) # Size B*(K-1) K D
        final_ys = torch.cat((ys_rep, ys_copied), dim=1) # Size B*(K-1) K 1

        # Use mechinterp to flag to return associated weight matrix for each task (batch element) in distribution in order to do fuzzy estimation



    pretrain_dist = RegressionSequenceDistribution(
                    task_distribution=DiscreteTaskDistribution(
                        num_tasks=2,
                        task_size=8,
                    ),
                    noise_variance = 0.25**2,
                )

    