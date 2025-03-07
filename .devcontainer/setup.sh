#!/bin/bash

cargo install gitui

chezmoi init https://github.com/john-1127/dotfiles.git
chezmoi apply

if [ ! -d "$HOME/.config/nvim" ]; then
  git clone https://github.com/LazyVim/starter $HOME/.config/nvim
fi

nvim --headless "+Lazy sync" +qall

echo "Setup completed!"
