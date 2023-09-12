How to train on TPUs for free
=============================

TPU (Tensor Processing Unit) is a specialised computer chip for deep learning
training that can run faster than GPUs in some cases
(more information: https://en.wikipedia.org/wiki/Tensor_Processing_Unit).

The Google Cloud Platform (GCP) TPU Research Cloud offers researchers
reasonably generous periods of free access to TPU Virtual Machines (VMs)
for the purpose of promoting research and ingratiating researchers[^trc]
(more information: https://sites.research.google/trc/about/).

This is a guide to acquiring and setting up one or more TPU VMs and running
experiments for the ICL project on the attached TPUs.

Contents:

* Part 1: Creating a project with TPU Access
* Part 2: Creating and connecting to a TPU Virtual Machine
* Part 3: Configuring the VM environment
* TODO: more.


Other resources

* Another tutorial (JAX, but helpful general info) is here:
  https://github.com/ayaka14732/tpu-starter
* There is also a `#tpu-research-cloud` channel in the Google Developers
  Discord---join here: https://discord.com/invite/ca5gdvNF5n
* Some more info about the TRC program is here (fun read):
  https://github.com/google/jax/issues/2108#issuecomment-866238579

The reason most tutorials use JAX is because apparently torch support for
TPUs is not that mature. To me it seems like it works OK but is not as simple
as `device='tpu'`. I would like to learn JAX sooner rather than later,
but for this project, we are using torch, so here we go.

[^trc]:
  Apparently the TPU Research Cloud people are very awesome and generous with
  their time, I mean only to besmirch their broader corporation.

Part 1: Creating a Google Cloud Platform project with TPU allocation
--------------------------------------------------------------------

We take advantage of the offer from Google Research Cloud to give researchers
30 days free access to some TPU VMs.

### Part 1.1: Apply for a TPU allocation

As follows:

1. Submit an application

   * Go to https://sites.research.google/trc/about/.
   * Click 'apply now' and fill in the form.

2. Then, wait to hear back (took ~30 minutes for me).

Actually, you don't have to wait before starting parts 1.2 and 1.3 below.
You can do those while you wait for the email, and then wait for the email
before starting part 1.4.

### Part 1.2: Set up Google Cloud Platform project for the TPUs

Next, set up a GCP project to use the TPUs. These are based on the
instructions from the response email from step 1.1.2 above.

1. (If you don't already have a GCP account) create a GCP account:

   * I think go to https://cloud.google.com/
   * You will need a Google account
   * You will need a credit or debit card, which must have a positive balance
     but will not be charged and (even when the free initial credits expire,
     unless you deliberately upgrade to a 'full account')

2. Create a GCP project for the experiments:

   * Follow the steps to create a project here:
     https://cloud.google.com/resource-manager/docs/creating-managing-projects
   * You can use whatever valid name and ID you like.
   * Note the 'project number' for step (6).

3. Turn on the Cloud TPU API for that project:

   * This link should work
     https://console.cloud.google.com/apis/library/tpu.googleapis.com
   * Make sure you have the correct project selected (the one from the
     previous step / the one you want the TPUs for).

### Part 1.3: Configure SSH access

Assuming you want to use a proper terminal emulator to ssh into your TPU VMs
(not the laggy google cloud browser version), you will want to add an ssh key
to your project.

1. Generate an SSH key on your local device, for example as follows:

   * Run the command:
     ```
     ssh-keygen -t ed25519 -f ~/.ssh/gcp -C <arbitrary_username>
     ```
     For example:
     ```
     ssh-keygen -t ed25519 -f ~/.ssh/gcp -C matt
     ```
     (and note the username for later).
   * The command will prompt you for a passcode, you can skip this (leave it
     blank) if you like.

2. Add the public key to your GCP project metadata:

   * Copy the contents of the new file `~/.ssh/gcp.pub`
   * Go to GCP console for your project and navigate to Navigation menu >
     Compute engine > Metadata > SSH Keys > Edit
   * Paste the contents into an empty SSH key field.
   * Press 'Save'.

Alternatives to the above:

* You could just use the in-browser gcloud SSH, but it's slow and clunky and
  you have to use a browser, which is an unnecessary drain on your system
  resources.
* You could install `gcloud` tool locally and use it to start the ssh
  connection, but the command to do so is way longer than `ssh tpu` enabled
  by adding your keys and using plain SSH.
* One advantage of native `gcloud` is that it manages the IP address for you,
  whereas with SSH you have to take note of the IP address every time the
  TRC reboots your TPU VMs which apparently happens from time to time.

### Part 1.4: Link the project to your TPU allocation

Before proceeding, you need the email from part 1.1 and the project number
from part 1.2.

1. Send the project number of your GCP project to the TPU Research Cloud
   people using the form linked in the response email from part 1.1.

2. Wait for the confirmation email to say you have TPU access.

Notes:

* This email will come to your gmail associated with GCP, even if you
  used a different email (e.g. unimelb email) to apply for the TPUs.

* The email (dubbed the 'allocation email') contains important
  information about your TPU allocation needed for part 2.
  Keep it handy.

### Part 1.5: Extend your access?

The TPUs are free for your project for 30 days after they are granted, but
apparently it is possible to prostrate yourself before the big G I mean email
the TPU research cloud folks to request more time after that.

I have heard that they are typically very kind and very willing to offer such
extended access on request.


Part 2: Creating and connecting to a TPU Virtual Machine
--------------------------------------------------------

This section contains the steps to create a single TPU Virtual Machine (which
hosts a single TPU). If you want to run your experiments on multiple TPUs, as
far as I am aware this requires following these steps once for each TPU.

### Part 2.1: Creating a TPU VM

First, create a TPU VM.

1.  Within the GCP project, go to Navigation menu > Compute engine > TPUs
    (or go to https://console.cloud.google.com/compute/tpus and then make
    sure you are on the right project).

2.  Press 'Create a TPU' and configure the TPU
    * pick some name.
    * use the 'TPU type' and 'zone' from your allocation email (e.g.,
      'v2-8' and 'us-central1-f').
    * if you want to use your preemptible TPUs you can flag that under
      'management'.
    * pick as software version 'tpu-vm-pt-2.0' (pytorch 2).
    * default values for the rest should be fine.
    
    > **Important note:** Be careful to get this config right because if you
    > spin up a TPU outside of your allocation it will eat your intro credits
    > (or your actual money). For example be sure to get the zone right!

3.  Wait for the TPU to spin up (takes < 1 min), and note the External IP
    address for next steps.

    If the allocation fails with an 'unknown error', try again tomorrow.
    
    If the allocation fails with an error about there being no available
    TPUs of that kind in the given zone, try again immediately, and after a
    while you will probably get one.

Alternatives: If you are setting up multiple TPU VMs, or if you are having
trouble getting TPU VMs allocated due to availability, it may be worth
investing in streamlining the above steps, as follows:

* A command in the cloud shell (in the browser) will take care of step 2.
  An example command is as follows:
  ```
  gcloud compute tpus tpu-vm create tpu2a --zone=us-central1-f --accelerator-type=v2-8 --version=tpu-vm-pt-2.0
  ```
  In this case the command creates a TPU VM with a v2-8 TPU and the name is
  set to `tpu2a`, but these can obviously be changed as needed.

  Do note the important note above about getting the config right to avoid
  excessive charges.

* You can also run such commands from a native terminal if you go through a
  few steps to download and install the google cloud SDK.


### Part 2.2: Delete your TPUs when you are done with them!

**Important note:** If you have TPU VMs active at the end of your 30 day trial I
then they will quickly eat through your free starter-credits (or your
actual money if you have a full account) to the tune of hundreds of US
dollars per day per TPU VM.
So ***make sure you delete your TPU VMs after you are done with them.***

You can delete the TPUs via the console page from step 2.1.1.


### Part 2.3: Establish SSH connection to the TPU VM

I continue to assume you want to use a local terminal emulator instead of the
laggy cloud shell in the GCP webapp. If you do want to use the webapp you can
skip these steps and load that instead, but each time you do so it will cost
you 3 computer science credibility tokens.[^ssh]

1. Add a hosts entry to your SSH config:
   
   * Open `~/.ssh/config`
   * Add the following entry:
     ```
     Host <tpu name chosen in step 2.1.2>
       HostName <External IP address noted in step 2.1.3>
       IdentityFile <path to key created in part 1.3>
       User <arbitrary_username used during key creation in part 1.3>
     ```
     For example:
     ```
     Host tpu2a
       HostName 34.69.74.175
       IdentityFile ~/.ssh/gcp
       User matt
     ```
     (Actually, the name doesn't have the be the same as the TPU name from
     step 2.1.2, which is immutable, it can be any name you like).

Now, every time you want to run commands on the TPU VM, you can SSH into
the TPU VM from a native terminal with a command like `ssh <tpu-name>`
(like `ssh tpu2a`, but using the name you configured in step 2.3.1).

[^ssh]: You don't need to create an account to spend these computer science
    credibility tokens (an account was automatically created for you when
    I met you).


Part 3: Configuring the VM environment
--------------------------------------

The VM runs Ubuntu 20.04 LTS with some default packages installed such as
Python (version 3.8), git, pytorch, and pytorch/XLA (the TPU operation
compiler).
On the hardware side, apparently it has a 96-core CPU and 335 GiB of memory
attached to one TPU with 8 cores and 128 GiB TPU memory (32 GiB for each of
four core pairs I think).
We have root access to this VM... Yum!

### Part 3.1: Navigate a version conflict!

*(no actions required---just for your information)*

The provision of Python 3.8 presents a conflict:

* Our code (incl. the devinterp library), developed in the era of Python
  3.11, currently has some minor dependencies on some features that are
  missing in Python 3.8. This means our code can't run unless we install
  a newer Python, or change the code to remove these dependencies.

* Python 3.8 isn't there just because nobody remembered to update the Ubuntu
  version---Pytorch/XLA's main release is stuck on Python 3.8, so we can't
  use TPUs if we move to a newer Python (which is easy enough otherwise).

Actually, there is a nightly release of Pytorch/XLA that supposedly runs on
Python 3.10, which would be high enough for our code. However, I couldn't get
this nightly release to run.

Going forward, we will have to clean up our libraries so that they don't
depend on Python >3.8. Until then, we will have to make a bit of a mess as we
hack out the non-essential dependencies.

In case we did want to install a newer version of Python on the VM,
instructions are in the appendix.
  
### Part 3.2: Authorize VM for GitHub

We need to configure the VM so that we are authorised to clone our private
GitHub repositories.

1. Generate an SSH key.
   Full instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
  
   In short, on the VM, run the following command:
   ```
   ssh-keygen -t ed25519 -C "your_github_email@example.com"
   ```
   where the email is obviously your actual GitHub email.
   
2. Add that SSH key to the SSH agent on the VM.
   Full instructions at the above link.
   
   In short, on the VM, run the following commands:
   ```
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519 # or wherever you put the key
   ```

3. Add the generated SSH key to your GitHub account.
   Full instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

The key generated in this step is a separate key to the GCP key we generated
in part 1. It is probably best to use two separate keys for this.

However, if you are creating multiple VMs, you don't necessarily have to
generate a new key for every VM and add it to GitHub. There's no reason the
VMs can't all use the same key.

Once you have one VM set up following the above instructions (called `tpu1`
in your SSH config), you can run the following command to copy the keys from
that VM to a new VM (called `tpu2`), assuming the keys were named
`id_ed25519` like in the example above.

NOTE: This command should be run from your local machine, i.e. do not SSH
into either VM.

```
scp tpu1:.ssh/id_ed25519{,.pub} tpu2:.ssh/
```

### Part 3.3: Configure git itself

Configure git itself so you can make new (merge) commits:

1. You need a username and email to make commits:
   ```
   git config --global user.email "your_git_email@example.com"
   git config --global user.name "your name"
   ```
   For example:
   ```
   git config --global user.email "m@far.in.net"
   git config --global user.name "Matthew Farrugia-Roberts"
   ```
   
2. You need to specify a merge strategy to do merges. There are several
   options, here's one:
   ```
   git config --global pull.rebase false
   ```

### Part 3.4: Optional: Set up personal tools

For example, I like to use neovim as an editor and zsh as the shell. Other
tools are possible for example it should be possible to link vs code and the
TPU VM, but I don't have instructions for that here.

1. Install personal favourite tools such as neovim and zsh
   ```
   sudo apt install neovim zsh
   ```

2. Change the default shell to zsh:
   
   * Configure a user password for your account on the VM:[^evil]
     ```
     sudo passwd matt
     ```
     (Changing a user's shell requires a password, otherwise we don't really
     need it).
   * Change default shell:
     ```
     chsh -s /bin/zsh
     ```
     (This requires entering the same password from the previous step.)

3. Configure zsh (e.g., put this in `.zshrc`):
   ```
   PS1='%(?.%F{green}%n@tpu1 :).%F{red}%n@tpu1 :( %?)%f %. $ '

   export VISUAL=nvim
   export EDITOR="$VISUAL"

   HISTSIZE=1000
   SAVEHIST=1000
   HISTFILE=~/.zsh_history
   setopt histignorealldups sharehistory

   alias gs='git status'
   alias vim=nvim
   ```
   The config will activate next time you log in or immediately if you run
   `source ~/.zshrc`.

[^evil]:
  This password shouldn't need to be that secure, since we are authenticating
  to the VM using keys rather than passwords, and the contents are not that
  sensitive.
  So I suggest making the password `evil`, as a playful throwback to Google's
  old motto.

Part 4: Installing our python code and dependencies
---------------------------------------------------

With the OS environment and supporting tools configured, the next step is to
install our python code and all of its dependencies. 

1. Clone the icl and devinterp repositories to get the experiment code:

   ```
   git clone git@github.com:timaeus-research/icl.git
   git clone git@github.com:timaeus-research/devinterp.git
   ```

   Note: this step relies on the github keys being configured from back in
   part 3. If you get a permission error, revisit that step.

2. Install the public python dependencies for the icl project.
   
   ```
   cd ~/icl
   pip install -r requirements.txt
   # Optional: Install dev dependencies:
   pip install pytest torch_testing
   ```

   There are some conflicts due to old Python version. TODO: Document or
   resolve in requirements.txt.

   <!--
   On Python 3.8 I resolved the conflicts as follows:
   ```
   pip install urllib3==1.26.11 typing-extensions==4.7.1
   ```

   And comment out the entry points in train script and the import of
   `Annotation` or whatever (requires 3.9).
   ```

   ```
   -->

3. Locally install the `devinterp` library and its dependencies:
   
   ```
   pip install --editable ~/devinterp
   ```

   There are some conflicts due to old Python version. TODO: Document or
   resolve in requirements.txt.
   <!--
   On Python 3.8 I resolved the conflicts by dropping some dependencies in
   the requirements file as follows:
   ```
   ipython==8.14.0 -> 8.13.1
   numpy==1.25.1 -> 1.24.4
   ```

   Then there were more errors:
   ```
   ERROR: launchpadlib 1.10.13 requires testresources, which is not installed.
   ERROR: virtualenv 20.14.1 has requirement platformdirs<3,>=2, but you'll have platformdirs 3.8.1 which is incompatible.
   ERROR: google-api-core 1.34.0 has requirement protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<4.0.0dev,>=3.19.5, but you'll have protobuf 4.23.4 which is incompatible.
   ERROR: google-auth-httplib2 0.1.0 has requirement httplib2>=0.15.0, but you'll have httplib2 0.14.0 which is incompatible.
   ERROR: importlib-resources 6.0.1 has requirement zipp>=3.1.0; python_version < "3.10", but you'll have zipp 1.0.0 which is incompatible.
   ```
   -->
    
   Note: The icl repo is actually currently depending on the `add/slt-1` branch
   of devinterp library, so it's currently necessary to check out that branch:

   ```
   cd ~/devinterp
   git checkout -t origin/add/slt-1
   ```

   In general, `pip install --editable` will make it so that we don't have to
   reinstall after changing branches like this (unless new dependencies are
   added, then these should be installed).

TODO: Document installing Pytorch/XLA. Not required if using system Python
where it was part of the image already.

<!--

   Note on base also requires numpy and pyyaml.

   Note this error:

   ```
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
  devinterp 0.0.0 requires protobuf==4.23.4, but you have protobuf 3.20.3 which is incompatible.
  ```

Congratulations! This TPU VM is now ready to run experiments!
-->


Part 5: Running the experiments
-------------------------------

It's as simple as configuring an experiment (or a sweep) as usual and then
running `python -m icl` from the icl repository, I think:

* Configure wandb (see README)
* Configure AWS (see README)
* Run `python -m icl`?

And should probably detach it so that it's possible to log out etc.

Appendix A: Upgrading Python
----------------------------

If you want Python >3.8 on the TPU VM, three broad options come to mind:

1. Upgrade Ubuntu from 20.04 LTS to 22.04 LTS, which gets you Python 3.10
   as system Python.
   * seems like overkill?
2. Install Python 3.xx via the 'deadsnakes' package repository and switch
   the system Python to 3.10.
   * it's probably a bad idea to change system Python.
3. Install Python 3.xx via the 'deadsnakes' package repisitory and enable
   it using a virtual environment.
   * Seems like the right way to do it.

Note: Pytorch/XLA only supports Python 3.8 right now.

### Option A.1: Upgrade to Ubuntu 22.04 LTS

Here are the instructions for bumping Ubuntu release to 22.04 LTS

1. Run these commands to sync up Ubuntu 20.04 (follow the prompts saying
   'yes' as required).
   ```
   sudo apt upgrade
   sudo apt dist-upgrade
   sudo apt upgrade linux-gcp linux-headers-gcp linux-image-gcp
   ```

2. Reboot:
  
   * Run the command:
     ```
     sudo reboot
     ```
   * After that you will be disconnected from SSH.
   * Wait a minute while the VM reboots, and then SSH back in.
     * If SSH doesn't work, wait another minute.

3. Now you can do the actual update:
   ```
   sudo do-release-upgrade
   ```

   Follow the prompts (which appear irregularly throughout about 15 minutes
   of operations---keep an eye on it).

   The final step of the upgrade is a reboot again: say yes, wait a minute,
   and then SSH back in. If SSH doesn't work, wait another minute.

This brings system python up to Python 3.10.

4. At this point just do yourself a favour and install one more package,
   to create an alias for python3 and all related binaries:
   ```
   sudo apt install python-is-python3
   ```

### Option A.2: Install additional Python and make it system Python

Follow steps for option A.3 below and then look up how to make it system
Python. But this is probably not a great idea, because some system stuff
might depend on the Python version that exists.

### Option A.3: Install additional Pythons and use a virtual environment

Install an additional Python from the 'deadsnakes' package repository.

0. I have seen some people recommend an additional command here,
   `sudo apt install software-properties-common`, but I didn't find it
   was necessary.

1. Add the package repository:
   ```
   sudo add-apt-repository ppa:deadsnakes/ppa
   ```

2. Install new Python (e.g. 3.10) and supporting tools:
   ```
   sudo apt install python3.10 python3.10-dev python3.10-venv python3.10-distutils
   ```

Set up a virtual environment with Python 3.10. You can do it in any
directory you like, I just used the home directory for this project.

3. Create the virtual environment.
   ```
   virtualenv venv-python3.10 -p python3.10
   ```
   where `venv-python3.10` can be any name for your virtual environment.

4. Activate the virtual environment and upgrade its `pip` (which will be
   slightly out of date).
   ```
   source venv-python3.10/bin/activate
   pip install --upgrade pip
   ```

It is necessary to activate this environment every time we log in. To avoid
doing this manually, you can add e.g., `source venv-python3.10/bin/activate`
to your shell startup script.

### Note A.4: Pytorch/XLA Nightly for Python 3.10

The command to install the 3.10 nightly build is:
```
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl
```

More information here: https://github.com/pytorch/xla

Appendix B: Stuff to do
-----------------------

### Mosh

Potentially worth installing `mosh` as a less laggy ssh client/server.

### Jupyter hub?

Could set up a jupyter notebook or jupyter hub server on one of the TPU VM
and give access to devinterp folks.

### Sharing files across TPU VMs

From ayaka's guide:

> TPU VM instances in the same zone are connected with internal IPs, so you
> can [create a shared file system using NFS](https://tecadmin.net/how-to-install-and-configure-an-nfs-server-on-ubuntu-20-04/).

