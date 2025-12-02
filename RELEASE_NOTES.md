[comment]: # (#################################################################)
[comment]: # (Copyright 2024, Lawrence Livermore National Security, LLC and)
[comment]: # (MPIDiff project contributors. See the MPIDiff LICENSE file for details.)
[comment]: # 
[comment]: # (SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# MPIDiff Software Release Notes

Notes describing significant changes in each MPIDiff release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Version 0.2.1] - Release date 2025-12-02

### Changed
- MPIDiff::Diff now returns a boolean based off whether there was a difference, similar to MPIDiff::DiffUpdate

## [Version 0.2.0] - Release date 2024-09-13

### Added
- Annotations to the MPIDiff output files for improved diff context.
- DiffUpdate method that prevents differences from compounding over time.
- BLT submodule and host configs for easier standalone builds.

### Changed
- CMake build system to export targets.
- Removed noisy output message if MPIDiff has not been initialized.
