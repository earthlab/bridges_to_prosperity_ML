#!/usr/bin/env bash

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
if pip -vvv freeze -r requirements.txt | grep "not installed"
then 
  echo "Pip install requirements..."
  pip install -r requirements.txt > /dev/null 2>&1
else
  echo "All python requirements met"
fi
## Set up ipython with readload
IPYTHON_CONFIG=~/.ipython/profile_default/ipython_config.py
if [ ! -f $IPYTHON_CONFIG ]; then 
  ipython profile create
fi
RELOAD1="c.InteractiveShellApp.extensions = ['autoreload']"
RELOAD2="c.InteractiveShellApp.exec_lines = ['%autoreload 2']"
if ! grep -Fxq "$READLOAD1" $IPYTHON_CONFIG
then
  echo "$READLOAD1" >> $IPYTHON_CONFIG
fi
if ! grep -Fxq "$READLOAD2" $IPYTHON_CONFIG 
then
  echo "$READLOAD2"  >> $IPYTHON_CONFIG 
fi

# if [ ! -f ~/work/b2p ]; then
#   echo "Creating link in ~/work/b2p"
#   ln -s /b2p/ ~/work/b2p
# fi
# git config --global user.email "nicrummel@gmail.com"
# git config --global user.name "nrummel"
cwd=$PWD
cd $DIR
echo "Activating conda env"
conda activate .
cd $cwd 
export BASE_DIR=$DIR
export TORCH_PARAMS="$DIR/data/torch.yaml"