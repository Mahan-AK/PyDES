# PyDes

Implementation of DES Encryption/Decryption Algorithm written in python3 with Numpy and optimized by Numba.

## Requirements:

You will need to install numpy and numba libraries to run this script:

```shell
$ pip install numpy numba
```

## Usage:
```shell
$ python DES.py
Usage: DES.py input_file -e/-d -k path_to_key

Options:
  -h, --help         show this help message and exit
  -e, --encrypt      
  -d, --decrypt      
  -k KEY, --key=KEY
```