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

By default, the VM runs Ubuntu 20.04 LTS with Python 3.8, git, pytorch, and
XLA (TPU compiler) installed. This is OK, except for Python 3.8, which is
missing some features used by the devinterp library and our code.

We have the following options for getting Python 3.11 on the VM:

1. Upgrade Ubuntu from 20.04 LTS to 22.04 LTS, which will get us Python 3.10.
   * I got this working first, but it was overkill, and only got us to 3.10.
2. Install Python 3.11 via third-party deadsnakes package repository and
   change it to system Python.
   * But it's probably a bad idea to change system Python.
3. Install Python 3.11 via third-party deadsnakes package repisitory and
   activate it using a virtual environment.
   * This is the right way to do it, instructions below.
4. Changing our code to work with Python 3.8.
   * Seems not worthwhile.


<!--
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
   At this point just do yourself a favour and install one final package,
   an alias for python3:
   ```
   sudo apt install python-is-python3
   ```
-->

1. Install Python 3.11 from 'deadsnakes' package repository.
   * Add the package repository:
     ```
     sudo add-apt-repository ppa:deadsnakes/ppa
     ```
   * Install Python 3.11 and supporting tools:
     ```
     sudo apt install python3.11 python3.11-dev python3.11-venv python3.11-distutils
     ```

2. Set up a virtual environment with Pyhton 3.11
   * Create virtual environment (in home directory).
     ```
     virtualenv venv -p python3.11
     ```
   * activate the venv
     ```
     source venv/bin/activate
     ```
   * update pip inside the venv
     ```
     pip install --upgrade pip
     ```

   It becomes necessary to activate this environment every time we log in,
   or, as in step 7 below, we could add that to the shell config (since we're
   only using this VM for work within this venv).

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

Optional: Set up my personal user account with some favourite apps.

5. Install neovim. `sudo apt install neovim`.

6. Install and change the default shell to zsh.
   
   * Configure a user password for the vm (I suggest `evil`, as in, Google's
     original motto): `sudo passwd matt`.
   * Install `zsh`: `sudo apt install zsh`.
   * Change default shell: `chsh` (to `/bin/zsh`).

   Zsh will activate next time you SSH in.

7. Configure zsh with some basics (put this in `.zshrc`):
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

   source ~/venv/bin/activate
   ```
   The config will activate next time you log in or immediately if you run
   `source ~/.zshrc`.


Part 4: Installing our python code and dependencies
---------------------------------------------------

With the OS environment and supporting tools configured, the next step is to
install our python code and all of its dependencies. 

1. Clone the icl and devinterp repositories to get the experiment code.
   
   * `git clone git@github.com:timaeus-research/icl.git`
   * `git clone git@github.com:timaeus-research/devinterp.git`

2. Install the public python dependencies for the icl project.
   
   * Change into icl repo:
     ```
     cd ~/icl
     ```
   * Install dependencies:
     ```
     pip3 install -r requirements.txt
     ```
   * Optional: Install dev dependencies:
     ```
     pip3 install pytest torch_testing
     ```

3. Locally install the `devinterp` library and its dependencies:
   
   * Change into devinterp repo:
     ```
     cd ~/devinterp
     ```
   * ICL repo is actually currently depending on the `add/slt-1` branch, so
     it's important to check out that branch:
     ```
     git checkout -t origin/add/slt-1
     ```
     (This information may become outdated soon.)
   * Install dependencies and install package as a locally editable library
     with:
     ```
     pip install --editable .
     ```

4. TODO: install XLA.

Congratulations! This TPU VM is now ready to run experiments!


Part 5: Running the experiments
-------------------------------

It's as simple as configuring an experiment (or a sweep) as usual and then
running `python -m icl` from the icl repository, I think.

And should probably detach it so that it's possible to log out etc.


-----

[^1]: You don't need to create an account to spend these computer science
    credibility tokens (an account was automatically created for you when
    I met you).
