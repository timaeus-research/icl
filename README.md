Singular Learning Theory for In-Context Learning
================================================

Understand the emergence/development of in-context learning in transformers
trained on in-context linear regression tasks.

The main focus is on the phase transitions ehibited in the
[**task diversity paper**](https://arxiv.org/abs/2306.15063).

Project admin
-------------

For discussion and managing the project:

* Project discussion and meetings take place on the DevInterp Discord server.
  (some prior discussion in the group chat).
* A GitHub project tracks TODO items:
  https://github.com/orgs/timaeus-research/projects/2/views/3
* There are two google docs associated to the project:
  * Jesse is maintaining meeting notes:
    https://docs.google.com/document/d/1yQmCNlIql18TYX--9CAgmI1kxcnVe1kpymadifK6wy4/edit
  * Dan's original note appears to have been abandoned:
    https://docs.google.com/document/d/1S4kBVFhlQBVRrdMrhRZz_4BS5N2_ef9HA051rKM7nCE/edit

For code and experiments:

* This GitHub repository stores the experiment code, issues, etc.
* W&B is used to track metrics from training runs:
    https://wandb.ai/devinterp/icl
* AWS is used to store snapshots: (unsure what to link here...)

For writing up:

* Regrettably we are using overleaf to draft a preprint:
  https://www.overleaf.com/project/64ee6fc2297aa3dfc799310a

Tentative target venues:

* AISTATS 2024 (http://aistats.org/aistats2024/). Key dates (subject to change? there is no cfp yet):
  * Abstract deadline:               6 October 2023 (Anywhere on Earth)
  * Paper submission deadline:      13 October 2023 (Anywhere on Earth)
  * Paper decision notifications:   19 January 2024
  * Conference dates:                2 May - 4 May 2024 (Valencia 💪)
* Present preliminary results at DevInterp confernece in November
  (no proceedings?)


Set-up environment
------------------

Clone this repository locally:

```
git clone git@github.com:timaeus-research/icl.git
```

Install standard dependencies (`torch`, `wandb`, `tqdm`, etc.):

```
# inside icl repository
pip install -r requirements.txt
```

The `devinterp` library is another dependency, but it's not yet available
through `pip` index.
For now, install this library from source using the command
  `pip install --editable`
(`--editable` means any changes to the library will not require
reinstallation):

* IF you already have the repository on your machine (e.g. you are a
  devinterp developer), run:
  ```
  pip install --editable /path/to/devinterp
  ```

* IF you do not already have the repository, step outside the icl repository
  and run:
  ```
  # OUTSIDE icl repository
  git clone git@github.com:timaeus-research/devinterp.git
  pip install --editable devinterp
  ```
  Cloning devinterp *inside* the icl repository is not ideal as (1) you will
  have to deal with nested git repositories and (2) you will have to deal
  with the name conflict between the repository/folder named `devinterp` and
  the pip-installed library `devinterp`.
  If you want to do this I suggest (1) using submodules and (2) renaming the
  repository folder.

* IF you already have cloned the devinterp library and installed with
  `pip install --editable`, but now you want to update to the latest source,
  go into the devinterp repository on your machine and run:
  ```
  git pull
  ```
  The changes to the source should be reflected next time you
  `import devinterp`.

To run code that reads or writes snapshots you will need your AWS API keys.

```
TODO.
```

To run experiments with W&B logging you will need your associated API keys.

```
TODO: instructions
```

Configuring and running experiments
-----------------------------------

Configure and run a single experiment locally:

* TODO.

Configure and run a single experiment on Spartan:

* TODO.

Configure and run a sweep:

* TODO: Sweeps are defined using YAML etc.



Notes
-----

Work in this repository initially started in the devinterp repo. There may be
some atefacts such as commit messages referencing unrelated files and
projects.

Furthermore, for posterity, there is some discussion around metrics in this
PR:

* https://github.com/timaeus-research/devinterp/pull/2
