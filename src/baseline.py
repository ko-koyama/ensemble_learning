import sys
sys.dont_write_bytecode = True
import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

if __name__ == '__main__':
	# ベンチマークとして利用するアルゴリズムのlist
	models = [ 
		("SVM", SVC(random_state=1), SVR()),  # SVM
		("GaussianProcess", GaussianProcessClassifier(random_state=1),
			GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1)),  # ガウス過程 
		("KNeighbors", KNeighborsClassifier(), KNeighborsRegressor()),  # K-近傍法 
		("MLP", MLPClassifier(random_state=1),
			MLPRegressor(hidden_layer_sizes=(5), solver='lbfgs', random_state=1)),  # NN 
		]
	
	# データセット、区切り文字、ヘッダー有無、インデックス有無のlist
	input_path = "../data/input/"
	classifier_files = ["iris.data", "sonar.all-data", "glass.data"]
	classifier_params = [(",", None, None), (",", None, None), (",", None, 0)]
	regressor_files = ["airfoil_self_noise.dat", "winequality-red.csv", "winequality-white.csv"]
	regressor_params = [(r"\t", None, None), (";", 0, None), (";", 0, None)]

	# スコアを格納するDataframe
	result = pd.DataFrame(columns=["target", "function"] + [m[0] for m in models],
							index=range(len(classifier_files+regressor_files)*2))

	# 分類
	ncol = 0
	for i, (f, p) in enumerate(zip(classifier_files, classifier_params)):
		# ファイル読み込み
		df = pd.read_csv(input_path+f, sep=p[0], header=p[1], index_col=p[2])
		
		# 説明変数
		x = df[df.columns[:-1]].values
		
		# 目的変数と全ラベルのlist
		y, clz = support.clz_to_prob(df[df.columns[-1]])	
		
		# スコアを格納するDataframeに対して、データセット名と評価関数名を追記
		result.loc[ncol, "target"] = re.split(r"[._]", f)[0]
		result.loc[ncol+1, "target"] = ""
		result.loc[ncol, "function"] = "F1Score"
		result.loc[ncol + 1, "function"] = "Accuracy"
		
		
		# 全てのアルゴリズムで評価
		for l, c_m, r_m in models:
			kf = KFold(n_splits=5, random_state=1, shuffle=True)
			s = cross_validate(c_m, x, y.argmax(axis=1), cv=kf, scoring=("f1_weighted", "accuracy"))
			result.loc[ncol, l] = np.mean(s["test_f1_weighted"])
			result.loc[ncol+1, l] = np.mean(s["test_accuracy"])
		
		ncol += 2
	
	# 回帰
	for i, (f, p) in enumerate(zip(regressor_files, regressor_params)):
		# ファイル読み込み
		df = pd.read_csv(input_path+f, sep=p[0], header=p[1], index_col=p[2], engine="python")

		# 説明変数
		x = df[df.columns[:-1]].values

		# 目的変数
		y = df[df.columns[-1]].values.reshape((-1,))
	
		# スコアを格納するDataframeに対して、データセット名と評価関数名を追記
		result.loc[ncol, "target"] = re.split(r"[._]", f)[ 0 ]
		result.loc[ncol+1, "target"] = ""
		result.loc[ncol, "function"] = "R2Score"
		result.loc[ncol+1, "function"] = "MeanSquared"
		
		# 全てのアルゴリズムで評価
		for l, c_m, r_m in models:
			kf = KFold(n_splits=5, random_state=1, shuffle=True)
			s = cross_validate(r_m, x, y, cv=kf, scoring=("r2", "neg_mean_squared_error"))
			result.loc[ncol, l] = np.mean( s["test_r2"] )
			result.loc[ncol + 1, l] = -np.mean( s["test_neg_mean_squared_error"])
		
		ncol += 2

	# 結果保存
	# SVMに関しては結果が書籍と大きく異なる、scikit-learnのバージョンによるもの？
	print(result)
	result.to_csv("../data/output/baseline.csv", index=None)