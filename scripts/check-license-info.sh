#!/usr/bin/env zsh

##########################################################################
# Copyright (c) 2024, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

# This is used for the ~*tpl* line to ignore files in bundled tpls
setopt extended_glob

autoload colors

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

files_no_license=$(grep -rL "the MPIDiff LICENSE file" . \
   --exclude-dir=.git \
   --exclude-dir=tpl \
   --exclude=LICENSE \
   --exclude=NOTICE \
   --exclude=taskfile.txt \
   --exclude=.gitignore \
   --exclude=.gitmodules)

if [ $files_no_license ]; then
  print "${RED} [!] Some files are missing license text: ${NOCOLOR}"
  echo "${files_no_license}"
  exit 255
else
  print "${GREEN} [Ok] All files have required license info. ${NOCOLOR}"
  exit 0
fi
