# Open SMART-PMI


-------------------------------------------------------------------------------

This package contains the model, application, and copy of the training data used for the Open SMART-PMI project.

The open-smart-pmi package features:

* Scientific application for predicting PMI.
* Scripts for training the model and testing on new data 

-------------------------------------------------------------------------------

Table of Contents
-----------------

1. [Overview][#1]

2. [Getting Started][#2]

3. [Known Issues][#3]

4. [Contributor Notes][#4]

   4.1. [License][#4.1]

   4.2. [Package Contents][#4.2]

   4.3. [Setting Up a Development Environment][#4.3]

-------------------------------------------------------------------------------

## 1. Overview

The application of green chemistry is critical for cultivating environmental responsibility and sustainable practices in pharmaceutical manufacturing. Process Mass Intensity (PMI) is a key metric which quantifies the resource efficiency of a manufacturing process, but determining what constitutes a successful PMI of a specific molecule is challenging. A recent approach utilized the inherent complexity of the molecule to determine PMI targets from the chemical structure alone. While recent machine learning tools show promise in predicting molecular complexity, a more extensive application could significantly optimize manufacturing processes. To this end, we refine and expand upon the SMART-PMI tool by Sheridan et al. to create an open-source model and application. Our solution emphasizes explainability and parsimony to facilitate a nuanced understanding of prediction and ensure informed decision-making. The resulting model uses only two 0D and two 2D topological descriptors to compute molecular complexity. We develop a corresponding user-friendly app that takes in structured data files (SDF) files to rapidly quantify molecular complexity and provide a PMI target that can be used to drive process development activities. By integrating machine learning explainability and open-source accessibility, we provide flexible tools to advance the field of green chemistry and sustainable pharmaceutical manufacturing.

-------------------------------------------------------------------------------

## 2. Getting Started

USER GUIDE PENDING. Instructions to set up and run the local application will be written here. 
For modeling applications, refer to Section 4.3 in Setting Up a Development Environment.

-------------------------------------------------------------------------------

## 3. Known Issues

No currently known issues.

-------------------------------------------------------------------------------

## 4. Contributor Notes

### 4.1. License

The contents of this package are covered under the MIT License (included
in the `LICENSE` file). The copyright for this package is contained in the
`NOTICE` file.

### 4.2. Package Contents

```
├── README.md          <- this file
├── LICENSE            <- package license
├── NOTICE             <- package copyright notice
├── pyproject.toml     <- Python project metadata file
├── poetry.lock        <- Poetry lockfile
├── bin/               <- scripts and programs (e.g., CLI tools)
└── src/               <- package source code
```

### 4.3. Setting Up a Development Environment

<strong><em>Note</em></strong>: this project uses `poetry` to manage Python
package dependencies.

1. Prerequisites

   * Install [Git][git].

   * Install [Python][python] 3.9 (or greater).
     <strong><em>Recommendation</em></strong>: use `pyenv` to configure the
     project to use a specific version of Python.

   * Install [Poetry][poetry] 1.2 (or greater).

   * <em>Optional</em>. Install [direnv][direnv].

2. Set up a dedicated virtual environment for the project. Any of the common
   virtual environment options (e.g., `venv`, `direnv`, `conda`) should work.
   Below are instructions for setting up a `direnv` or `poetry` environment.

   <strong><em>Note</em></strong>: to avoid conflicts between virtual
   environments, only one method should be used to manage the virtual
   environment.

   * <strong>`direnv` Environment</strong>. <em>Note</em>: `direnv` manages the
     environment for both Python and the shell.

     * Prerequisite. Install `direnv`.

     * Copy `extras/dot-envrc` to the project root directory, and rename it to
       `.envrc`.

       ```shell
       $ cd $PROJECT_ROOT_DIR
       $ cp extras/dot-envrc .envrc
       ```

     * Grant permission to direnv to execute the .envrc file.

       ```shell
       $ direnv allow
       ```

   * <strong>`poetry` Environment</strong>. <em>Note</em>: `poetry` only
     manages the Python environment (it does not manage the shell environment).

     * Create a `poetry` environment that uses a specific Python executable.
       For instance, if `python3` is on your `PATH`, the following command
       creates (or activates if it already exists) a Python virtual environment
       that uses `python3`.

       ```shell
       $ poetry env use python3
       ```

       For commands to use other Python executables for the virtual environment,
       see the [Poetry Quick Reference][poetry-quick-reference].

3. Install the Python package dependencies (including `dev`, `docs`, and `test`
   dependencies).

   ```shell
   $ poetry install --with dev,docs,test
   ```

4. Install the git pre-commit hooks.

   ```shell
   $ pre-commit install
   ```
