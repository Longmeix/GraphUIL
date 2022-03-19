import pickle


def write_pickle(obj, outfile, protocol=-1):
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(infile):
    with open(infile, 'rb') as f:
        return pickle.load(f)


