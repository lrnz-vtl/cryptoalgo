from algo.definitions import ROOT_DIR


def get_pub_key():
    keyfname = ROOT_DIR / 'keys' / 'bybit-pub.key'
    with open(keyfname, 'r') as f:
        return f.read().strip()


def get_key():
    keyfname = ROOT_DIR / 'keys' / 'bybit.key'
    with open(keyfname, 'r') as f:
        return f.read().strip()
