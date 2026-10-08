# Security

train-doctor runs the command you give it, on your machine, with your permissions. Only point it at training code you would run yourself.

It makes no network requests. It writes only to the run directory (default `.train-doctor/runs/`) and, for `setup --yes`, to the agent config files it lists before changing them (with backups).

To report a vulnerability, open a private security advisory on the GitHub repository or email the maintainer. Please don't file public issues for security problems.
