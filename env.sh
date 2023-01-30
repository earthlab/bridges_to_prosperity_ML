#!/usr/bin/env bash

source start-notebook.sh

## Optain top level directory: PATH/TO/bridges_to_prosperity_ML
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

## Set python path
export PYTHONPATH=$DIR:$PYTHONPATH

## set bash shortcuts 
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# up and down arrows search 
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'

#PS1 modification
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
export PS1="\u@\h \[\033[32m\]\w\[\033[33m\]\$(parse_git_branch)\[\033[00m\] $ "

# git auto complete
if [ -f ~/.git-completion.bash ]; then
  . ~/.git-completion.bash
fi

# aliases
alias gc="git commit -m"
alias gcu="git add -u; git commit -m"
alias gp="git push origin HEAD"
alias gk="gitk &"
alias gka="gitk --all &"
alias gf="git fetch"
alias gfa="git fetch --all"

ipython profile create
echo "c.InteractiveShellApp.extensions = ['autoreload']" >> ~/.ipython/profile_default/ipython_config.py 
echo "c.InteractiveShellApp.exec_lines = ['%autoreload 2']"  >> ~/.ipython/profile_default/ipython_config.py 
# git config --global user.email "nicrummel@gmail.com"
# git config --global user.name "nrummel"
# token: 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

