#!/usr/bin/env python
import numpy as np
import numba as nb
from optparse import OptionParser


"""
First section of the code consists of all of the static substitution tables and sboxes used in DES cipher
All of the data is copied from: http://orion.towson.edu/~mzimand/cryptostuff/DES-tables.pdf
"""

# initial permutation sub_table
IP_table = np.array([
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
])

# permuted choice keys
# permutates initial 64 bit key to 56 bit key by removing the parity bits
PC_table = np.array([
    57, 49, 41, 33, 25, 17,  9,
     1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
])

# permutates 56 bit key to 48 bits (key compression)
KC_table = np.array([
    14, 17, 11, 24,  1,  5,  3, 28,
    15,  6, 21, 10, 23, 19, 12,  4,
    26,  8, 16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
])

# expansion table
E_table = np.array([
    32,  1,  2,  3,  4,  5,
     4,  5,  6,  7,  8,  9,
     8,  9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32,  1
])

# permutation table
P_table = np.array([
    16,  7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26,  5, 18, 31, 10,
     2,  8, 24, 14, 32, 27,  3,  9,
    19, 13, 30,  6, 22, 11,  4, 25
])

# final permutation table
FP_table = np.array([
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
])

# all 8 sboxes
S_boxes = np.array([
        # S1
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
         0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
         4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
         15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

        # S2
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
         3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
         0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
         13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

        # S3
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
         13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
         13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
         1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

        # S4
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
         13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
         10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
         3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

        # S5
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
         14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
         4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
         11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

        # S6
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
         10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
         9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
         4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

        # S7
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
         13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
         1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
         6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

        # S8
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
         1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
         7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
         2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
	], dtype=np.uint8)

# we will use the subtables as array indices so they need to start from zero
IP_table -= 1
PC_table -= 1
KC_table -= 1
E_table -= 1
P_table -= 1
FP_table -= 1


"""
Here we will define some helper function that we are going to use in DES functions

* frequently used calculational functions are optimized with numba library
"""

@nb.njit
def permutate(X, sub_table):
    y = np.zeros(sub_table.shape, dtype=X.dtype)
    
    for i in range(sub_table.shape[0]):
        y[i] = X[sub_table[i]]
        
    return y


@nb.njit
def xor(x1, x2):
    return np.logical_xor(x1,x2).astype(np.uint8)


def read_file_bits(filename):
    return np.unpackbits(np.fromfile(filename, dtype=np.uint8))


def write_bits_to_file(filename, buffer):
    np.packbits(buffer).tofile(filename)


def pad_buffer(buffer):
    # PKCS5 Padding
    N = int(8 - np.ceil((buffer.size%64)/8))
    pad = np.unpackbits(np.full((N,), N, dtype=np.uint8))
    
    return np.append(buffer, pad)


def unpad_buffer(buffer):
    # PKCS5 Padding
    N = np.packbits(buffer[-8:])[0]
    
    return buffer[:-8*N]


"""
Core DES functions
"""

def initial_permutation(X):
    # initial permutation subtable
    return permutate(X, IP_table)


@nb.njit
def expand_key(init_key):
    K = np.zeros((16,48), dtype=np.uint8)
    
    # removing the parity bits
    K56 = permutate(init_key, PC_table)
    
    rotations = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    for i in range(16):
        K56[:28], K56[28:] = np.roll(K56[:28],-rotations[i]), np.roll(K56[28:],-rotations[i])
        K[i] = permutate(K56, KC_table)
        
    return K


@nb.njit
def S(x, i):
    res = np.zeros((4,), dtype=np.uint8)
    
    row = 2*x[0] + x[5]
    column = 8*x[1] + 4*x[2] + 2*x[3] + x[4]
    dec = S_boxes[i][16*row + column]
    
    for i in range(4):
        res[i] = dec // 2**(3-i)
        dec = dec % 2**(3-i)
    
    return res


@nb.njit
def P(X):
    # permutation sub_table
    # used in f-function
    return permutate(X, P_table)


@nb.njit
def E(X):
    # expansion sub_table
    # used in f-function
    return permutate(X, E_table)


@nb.njit
def f(X, k):
    res = np.zeros((32,), dtype=X.dtype)
    
    exp = E(X)
    exp = xor(exp, k)
    
    # taking 8 6-bit slices for 8 S-boxes
    for i in range(0, 48, 6):
        S_idx = i//6
        res[4*S_idx : 4*S_idx+4] = S(exp[6*S_idx:6*S_idx+6], S_idx)
        
    return P(res)


@nb.njit
def feistel_network(X, K, mode=0):
    rounds = np.arange(16) if not mode else np.arange(15, -1, -1)
    for i in rounds:
        X[:32], X[32:] = X[32:], xor(X[:32], f(X[32:], K[i]))
        
    X[:32], X[32:] = X[32:].copy(), X[:32].copy()


def final_permutation(X):
    # final permutation subtable
    return permutate(X, FP_table)


def encrypt_block(X, key):
    X = initial_permutation(X)
    K = expand_key(key)
    feistel_network(X, K, 0)
    X = final_permutation(X)
    
    return X


def decrypt_block(X, key):
    X = initial_permutation(X)
    K = expand_key(key)
    feistel_network(X, K, 1)
    X = final_permutation(X)
    
    return X


def encrypt_buffer(buffer, key):
    buffer = pad_buffer(buffer)
    enc = np.zeros(buffer.shape, dtype=np.uint8)
    
    for i in range(0, buffer.size, 64):
        enc[i:i+64] = encrypt_block(buffer[i:i+64], key)
        
    return enc


def decrypt_buffer(buffer, key):
    dec = np.zeros(buffer.shape, dtype=np.uint8)
    
    for i in range(0, buffer.size, 64):
        dec[i:i+64] = decrypt_block(buffer[i:i+64], key)
        
    return unpad_buffer(dec)


def encrypt_file(filename, key):
    buffer = read_file_bits(filename)
    enc = encrypt_buffer(buffer, key)
    write_bits_to_file(filename + ".des_crypt", enc)


def decrypt_file(filename, key):
    buffer = read_file_bits(filename)
    dec = decrypt_buffer(buffer, key)
    write_bits_to_file("dec_" + filename.strip(".des_crypt"), dec)


"""
Driver Code
"""

def main():
    parser = OptionParser(usage="usage: %prog input_file -e/-d -k path_to_key")
    parser.add_option('-e', '--encrypt', action="store_true", dest="en", default=False)
    parser.add_option('-d', '--decrypt', action="store_true", dest="de", default=False)
    parser.add_option('-k', '--key', action="store", dest="key")
    opts, args = parser.parse_args()

    if not len(args) or not (opts.en or opts.de) or not opts.key:
        print(parser.print_help())
        raise Exception("Provide all needed arguments.")

    path_in = args[0]

    with open(opts.key, 'rb') as kf:
        k = kf.read()

    if len(k) != 8:
        raise Exception("Key length must be 64 bits long (8 bytes).")

    k = np.unpackbits(np.array(list(k), dtype=np.uint8))

    if opts.en:
        encrypt_file(path_in, k)
    else:
        decrypt_file(path_in, k)


if __name__ == '__main__':
    main()