import sys
sys.dont_write_bytecode = True
import numpy as np

def clz_to_prob(clz):
	"""
    目的変数ラベルを確率に変換
    
    Parameters
    ----------
    clz : pd.series(1*データ数)
        目的変数

    Returns
    -------
    z : np.2Darray(目的変数のクラス×データ数)
		目的変数ラベルごとの所属確率
	l_str : list
		データセットに含まれる目的変数ラベルの文字列
    """
	l = sorted(set(clz))
	m = [l.index(c) for c in clz]
	z = np.zeros((len(clz), len(l)))
	for i, j in enumerate(m):
		z[i,j] = 1.0
	l_str = list(map(str, l))
	return z, l_str