How to run an experiment on Free GCP TPUs
=========================================


Part 1: Creating a project with TPU access
------------------------------------------

We take advantage of the offer from Google to give researchers 30 days free
access to some TPUs.

First, apply for the TPUs:

1. Submit the application

   * Go to https://sites.research.google/trc/about/,
   * Click 'apply now' and fill in the form.

2. Then, wait to hear back (took ~30 minutes for me).

   (Actually, you don't have to wait except for step 6 and 7, you can
   start steps 3--5 and 8--9 now).

   Note: The TPUs are available for 30 days after they are granted.
   Apparently it is possible to email them to request more time after that.

Next, set up a GCP project to use the TPUs. These are based on the
instructions from the response email from step (2) above.

3. (If you don't already have a GCP account) create a GCP account:

   * I think go to https://cloud.google.com/
   * You will need a Google account
   * You will need a credit or debit card, which must have a positive balance
     but will not be charged and (even when the free initial credits expire,
     unless you deliberately upgrade to a 'full account')

4. Create a GCP project for the experiments:

   * Follow the steps to create a project here:
     https://cloud.google.com/resource-manager/docs/creating-managing-projects
   * You can use whatever valid name and ID you like.
   * Note the 'project number' for step (6).

5. Turn on the Cloud TPU API for that project:

   * This link should work
     https://console.cloud.google.com/apis/library/tpu.googleapis.com
   * Make sure you have the correct project selected (the one from the
     previous step / the one you want the TPUs for).

6. (Once step 2 is complete) get the 'project number' from step (4) or find
   it in the GCP console and send it to the TPU people using the form linked
   in the response email from step (2).

7. Wait for the confirmation email to say you have TPU access.

   * This email will come to your gmail associated with GCP, even if you
     used a different email (e.g. unimelb email) to apply for the TPUs.
   * The email (dubbed the 'allocation email') contains important
     information about your TPU allocation needed for part 2. Keep it handy.


Assuming you want to use a proper terminal emulator to ssh into your TPU VMs
(not the laggy AF google browser version), you will need to add an ssh key to
your project.

8. Generate an SSH key on your local device, for example as follows:

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

9. Add the public key to your GCP project metadata:

   * Copy the contents of the new file `~/.ssh/gcp.pub`
   * Go to GCP console for your project and navigate to Navigation menu >
     Compute engine > Metadata > SSH Keys > Edit
   * Paste the contents into an empty SSH key field.
   * Press 'Save'.



Part 2: Creating and connecting to a TPU Virtual Machine
--------------------------------------------------------

To take full advantage of the TPU allocation, which includes multiple TPUs,
we'll want to be running our experiments on multiple TPUs.

I don't really know about coordinating training of a single model across
multple TPUs (this should be possible but probably a bigger headache) but a
simple way to parallelise our science is to have different TPUs running
different experiments (e.g. different slices of the hyperparameter sweep).

Unless I am mistaken a single TPU VM is host to a single TPU, so if you want
to run experiments across multiple TPUs then you will need to create and
configure multiple VMs. The steps in this section need to be carried out once
per such VM.

> **Note:** If you have TPU VMs active at the end of your 30 day trial I
> then they will quickly eat through your free starter-credits (or your
> actual money if you have a full account) to the tune of hundreds of US
> dollars per day per TPU VM.
> So ***make sure you delete your TPU VMs after you are done with them***
> (instructions at the end of this section).

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

    Be careful to get this config right because if you spin up a TPU outside
    of your allocation it will eat your intro credits (or your actual money).
    For example be sure to get the zone right!

    By the way, so far I have been unable to spin up v3-8 TPUs, it just gives
    an 'unknown error' every time I try; but v2-8 TPUs are working OK.

3.  Wait for the TPU to spin up (takes < 1 min), and note the External IP
    address for next steps.

Note: Steps 1 and 2 can be replaced by the following gcloud command entered
into the gcp cloud shell.
In this case the command creates a TPU VM with a v2-8 TPU and the name is set
to `tpu1`, but these can obviously be changed as needed.
```
gcloud compute tpus tpu-vm create tpu5 --zone=us-central1-f --accelerator-type=v2-8 --version=tpu-vm-pt-2.0
```

Anyway, next, establish an SSH connection to the TPU VM. I continue to assume
you want to use a local terminal emulator instead of the laggy cloud shell in
the GCP webapp. If you do want to use the webapp you can skip these steps and
load that instead, but each time you do so it will cost you 3 computer
science credibility tokens.[^1]

4. (Assuming you are using SSH from a native terminal emulator) you will
   want to add a hosts entry to your SSH config to easily ssh in with a
   command as simple as `ssh tpu-name` (rather than typing out the IP
   address every time).
   
   * Open `~/.ssh/config`
   * Add the following entry:
     ```
     Host <the name chosen during TPU creation in step 2>
       HostName <the External IP address noted after TPU creation in step 2>
       IdentityFile <the path to your private key created in part 1>
       User <the arbitrary_username used during key creation in part 1>
     ```
     For example:
     ```
     Host tpu1
       HostName 34.69.74.175
       IdentityFile ~/.ssh/gcp
       User matt
     ```

5. (Every time you want to run commands on the TPU VM) you can now SSH into
   the TPU VM with a command like `ssh <tpu-name>` (e.g., `ssh tpu1`, using
   the name you configured in step 4).

Part 3: Configuring the VM environment
--------------------------------------

The VM runs Ubuntu 20.04 LTS with Python 3.8, git, pytorch, and XLA (TPU
compiler) installed. This presents a conflict:

* Our code (incl. the devinterp library), developed on Python 3.11, depends
  on some features that are missing in Python 3.8, so we can't run our code
  without installing a newer Python or changing our code.
* Pytorch/XLA's main release only runs on Python 3.8, so we can't use it for
  TPUs if we move to Python 3.11.

There is a nightly release of Pytorch/XLA that supposedly runs on Python 3.10
(which should be high enough for our code), but I couldn't get it to run. So
the plan will be to go forward with Python 3.8 and hack our code and
requirements to match.

<!--
A compromise that should allow us to run our code on the TPUs is to use the
nightly XLA build for Python 3.10, Python 3.10 being just new enough for our
code too. Let's try that! What are our options for running Python 3.10 on the
VM?

1. Upgrade Ubuntu from 20.04 LTS to 22.04 LTS, which will get us Python 3.10.
   * I got this working first, but the update was long-ish and overall it
     seemed like overkill.
2. Install Python 3.10 via the 'deadsnakes' package repository and switch
   the system Python to 3.10.
   * But it's probably a bad idea to change system Python.
3. Install Python 3.10 via the 'deadsnakes' package repisitory and enable
   it using a virtual environment.
   * This seems like the *right* way to do it, and was the second thing I
     tried.

Here are the instructions for bumping Ubuntu release to 22.04 LTS

1. Upgrade Ubuntu.
   * Run these commands to sync up Ubuntu 20.04 (follow the prompts saying
     'yes' as required).
     ```
     sudo apt upgrade
     sudo apt dist-upgrade
     sudo apt upgrade linux-gcp linux-headers-gcp linux-image-gcp
     ```
   * Reboot: `sudo reboot`, wait a minute, and then SSH back in (follow step
     5, above). If SSH doesn't work, wait another minute.
   * Upgrade to Ubuntu 22.04 LTS: `sudo do-release-upgrade` and follow the
     prompts (which appear irregularly throughout about 15 minutes of
     operations---keep an eye on it).
   * The final step of the upgrade is a reboot again: say yes, wait a minute,
     and then SSH back in (follow step 5, above). If SSH doesn't work, wait
     another minute.

   This brings system python up to Python 3.10 which should be enough.

2. At this point just do yourself a favour and install another package,
   to create an alias for python3 and all related binaries:
   ```
   sudo apt install python-is-python3
   ```

So here are the instructions for setting up the virtual environment:

1. Install Python 3.10 from 'deadsnakes' package repository.
   * Add the package repository:
     ```
     sudo add-apt-repository ppa:deadsnakes/ppa
     ```
   * Install Python 3.10 and supporting tools:
     ```
     sudo apt install python3.10 python3.10-dev python3.10-venv python3.10-distutils
     ```

2. Set up a virtual environment with Python 3.10. We can just do so in the
   home directory since this VM is going to be used only for running
   experiments with Python 3.10 anyway:
   ```
   virtualenv venv-python3.10 -p python3.10
   source venv-python3.10/bin/activate
   pip install --upgrade pip
   ```
   These commands create the virtual environment, activate it, and then
   upgrade pip inside the virtual environment (because by default it's a
   little dated).
   
   It is necessary to activate this environment every time we log in. To
   avoid doing this manually, in step 7 below, we add the activation command
   to the shell startup file.
-->

We'll also need to configure git on our VM so that we are authorised to clone
our private GitHub repositories and so that we can create commits and so on.

3. Authorise the VM on your GitHub account so you can clone our private
   repositories on the VM.
  
   * Follow the steps
     [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
     to generate an SSH key.
     In short, on the VM, run the following commands:
     ```
     ssh-keygen -t ed25519 -C "your_github_email@example.com"
     ```
     
   * Continue to follow the steps at that link to add the key to the SSH
     agent. In short, on the VM, run the following commands:
     ```
     eval "$(ssh-agent -s)"
     ssh-add ~/.ssh/id_ed25519 # or wherever you put the key
     ```

   * Follow the steps
     [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
     to add the SSH key to your GitHub account.

   **Note:** The key generated in this step is a separate key to the GCP key
   we generated in part 1. It is probably best to use two separate keys here.
   However, you don't necessarily need to generate a new key for every VM,
   they could all use the same key no problem. This would save you adding
   extra keys to GitHub, but you would still have to copy the public and
   private key files to each VM and add them to the SSH agent.

   * To copy key files from one VM (`tpu1` in your ssh config) to another
     (`tpu2`), assuming the key was named `id_ed25519`.
     ```
     scp tpu1:.ssh/id_ed25519{,.pub} tpu2:.ssh/
     ```

   * Do not check the keys into git.

4. While you're at it you'll probably have to do some git configuration so
   you can do merges and commits:
   ```
   git config --global user.email "your_git_email@example.com"
   git config --global user.name "your name"
   git config --global pull.rebase false
   ```
   For example:
   ```
   git config --global user.email "m@far.in.net"
   git config --global user.name "Matthew Farrugia-Roberts"
   git config --global pull.rebase false
   ```

Optional: Set up your personal user account with some familiar tools.

5. Install personal favourite tools such as neovim and zsh
   ```
   sudo apt install neovim zsh
   ```

6. Change the default shell to zsh:
   
   * Configure a user password for the VM:
     ```
     sudo passwd matt
     ```
     (I suggest `evil`, as in Google's original motto...)
   * Change default shell:
     ```
     chsh -s /bin/zsh
     ```
     (This requires entering the same password from the previous step.)

7. Configure zsh (e.g., put this in `.zshrc`):
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

   <!--
   This last line causes the virtual environment from earlier step to
   activate upon login:
   ```
   source ~/venv-python3.10/bin/activate
   ```
   -->


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
   ipython==8.14.0 -> 8.13.0
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
4. TODO: install Pytorch/XLA nightly build.

   ```
   pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl
   ```

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


-----

[^1]: You don't need to create an account to spend these computer science
    credibility tokens (an account was automatically created for you when
    I met you).
