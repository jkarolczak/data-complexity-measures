# Data complexity measures

## Common api

All metrics in this repository should be implemented following functional programming paradigm. Methods signatures
should follow the `prototype` defined in `data-complexity/prototype.py`. It means that the first argument should be of
type `pandas.DataFrame` and contain features. The second argument also should be of type `pandas.DataFrame` but should
contain target classes. Other arguments should be passed afterwards and should have default values. Returned datatype
should be of type `float`.

## Python environment

Below you can find instructions how to create environment solving necessary dependencies. If during implementation you
decide to use dependency not included in `requirements.txt` and `environment.yml` add appropriate name to mentioned
file.

### Conda

To create conda environment execute:

```shell
conda env create -f environment.yml
```

To activate created environment execute:

```shell
conda activate data-complexity
```

### Virtual environment

To create virtual environment execute:

```shell
python -m venv data-complexity
source data-complexity/bin/activate
pip install -r requirements.txt
```

To activate created environment execute:

```shell
source data-complexity/bin/activate
```