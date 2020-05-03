import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from scipy.sparse import csr_matrix
import pandas as pd
import time
import os




class SubmodModel:

    def __init__(self, topics_file_name):
        self.X_train_counts, self.X_train_tf, self.tokenizer, self.bigram_vectorizer \
            = self.get_V(topics_file_name)
        self.V = range(self.X_train_counts.shape[0])
        self.alpha = 6 / len(list(self.V))
        self.cluster = self.get_cluster(int(0.2 * len(self.V)))
        self.gold_summ_count = self.get_gold_summ_count(topics_file_name)
        self.cosine_similarity_kernel = cosine_similarity(self.X_train_tf, self.X_train_tf)

    # initialization helper
    def get_V(self, topics_file_name):
        path = 'OpinosisDataset1.0_0/topics/{}'.format(topics_file_name)
        text = open(path, encoding="utf8", errors='ignore')
        text = text.read()

        # get the X_train_counts and X_train_tf
        trainer = PunktTrainer()
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.train(text)
        tokenizer = PunktSentenceTokenizer(trainer.get_params())
        X = tokenizer.tokenize(text)
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                            token_pattern=r'\b\w+\b', min_df=1)
        X_train_counts = bigram_vectorizer.fit_transform(X)

        tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        return X_train_counts, X_train_tf, tokenizer, bigram_vectorizer

    def get_cluster(self, n_clusters=4):
        # Clustering for div_R
        X = self.X_train_tf.toarray()
        # Initializing KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        # Fitting with inputs
        kmeans = kmeans.fit(X)
        # Predicting the clusters
        labels = kmeans.predict(X)
        # Getting the cluster centers
        C = kmeans.cluster_centers_
        cluster = {k: [] for k in range(n_clusters)}
        for i in range(len(labels)):
            cluster[labels[i]].append(i)
        return cluster

    def get_gold_summ_count(self, topics_file_name):
        dir_name = topics_file_name.split(".")[0]
        path = 'OpinosisDataset1.0_0/summaries-gold/{}'.format(dir_name)
        K = len([name for name in os.listdir(path)])
        # print(K)
        gold_summ_txt = []
        gold_summ_count = []
        for i in range(1, K + 1):
            text2 = open(path + '/{}.{}.gold'.format(dir_name, i), 'r')
            txt = text2.read()
            #     print(tokenizer.tokenize(txt))
            gold_summ_txt.append(txt)
            a = self.tokenizer.tokenize(txt)
            if len(a) != 0:
                gold_summ_count.append(self.bigram_vectorizer.transform((a)))
        return gold_summ_count

    # submodular function
    # helper
    def get_cost(self, S):
        """get the cost of Summary which is the number of words
        Args:
          S: a list of senetences in summary
          V_counts: sparse matrix to represent all sentences in grams counts
        """
        return self.X_train_counts[S].sum()

    # rouge-n
    def rouge_n(self, S):
        '''
        S (list) : list of the index of the selected sentences S

        X_count (sparse.matrix) : sparse matrix of the studied document.
            X_count[k,s] : number of times the bigram s occurs in the sentence k

        gold_sum_count (list of sparse matrix)  for each sparse matrix,
            gold_sum_count[k][i,j] : number of times the bigram j occurs in the sentence
            in the summary k
        '''
        X_count_S_1 = self.X_train_counts[S].sum(axis=0)  # delete the sentence structure
        # X_count_S_1[s] : number of times the bigram s occurs in the summary S
        gold_summ_count_1 = [k.sum(axis=0) for k in self.gold_summ_count]  # delete the sentence structure
        num = 0
        denom = 0
        for i in range(len(gold_summ_count_1)):
            for j in range(X_count_S_1.shape[1]):
                r_ei = gold_summ_count_1[i][0, j]
                if r_ei != 0:
                    c_es = X_count_S_1[0, j]
                    num += min(c_es, r_ei)
                    denom += r_ei
        res = num / denom
        return (res)

    # MMR
    def get_f_MMR(self, S, lmbda_MMR=4):
        """get the MMR value of summary S

        Args:
          S: a list of senetences in summary
          V: sparse matrix to represent all sentences in grams
          lmbda_MMR
        Returns:
          res: The MMR value of summary S
        """
        res = 0
        if len(S) == 0: return 0

        U = list(range(self.X_train_tf.shape[0]))
        S_c = U[:]  # complement of S
        for s in S:
            S_c.remove(s)

        for i in S_c:
            res += self.cosine_similarity_kernel[i,S].sum() #cosine_similarity(self.X_train_tf[i], self.X_train_tf[S]).sum()
        if len(S) == 1: return res
        for i in S:
            S_without_i = S[:]
            S_without_i.remove(i)
            res -= lmbda_MMR * self.cosine_similarity_kernel[i, S_without_i].sum() / 2

        return res


    # f_sub
    def get_f_sub(self, S):
        return self.covdiv_F(S)

    def covdiv_F(self, S, lambda1=4):
        if len(S)==0:
            return(0)
        return self.cov_L(S) + lambda1 * self.div_R(S)

    def cov_L(self, S):
        # V = self.X_train_tf
        res = 0
        # y= V[S]
        for x in self.V:
            res1 = self.cosine_similarity_kernel[x,S].sum()#cosine_similarity(x.reshape(1, -1), y).sum()
            res2 = self.alpha*self.cosine_similarity_kernel[x, self.V].sum()#cosine_similarity(x.reshape(1, -1), V).sum()
            res += min(res1, res2)
        return res

    def div_R(self, S):
        V = self.X_train_tf
        K = len(self.cluster)
        res = 0
        for k in range(K):
            S_inter_Pk = list(set(S) & set(self.cluster[k]))
            res1 = 0
            for j in S_inter_Pk:
                res1 += self.cosine_similarity_kernel[j, self.V].sum()
            res += np.sqrt(res1)
        return(res)

    # greedy algorithm
    def greedy_submodular(self, fun, cost_fun, budget=50, r=1, lazy=False):
        """get the Summary using greedy

        Args:
          fun: the function to maximize
          V: sparse matrix to represent all sentences in grams tf-idf
          cost_fun: the cost function for summary
          V_counts: sparse matrix to represent all sentences in grams counts
          budget: the upper bound for the cost of summary
          r: the parameter for scalability
        Returns:
            res: the index for the best sentences to reprensent the article
        """
        V = self.X_train_tf
        # V_counts = self.X_train_counts
        G = []
        U = list(range(V.shape[0]))
        lowest_cost = np.min([cost_fun([u]) for u in U])
        #     print(lowest_cost)
        cost = 0
        if lazy == True:
            Delta = [fun([u])/ (cost_fun([u])) ** r for u in U]
        while len(U) != 0 and budget - cost >= 0.1 * budget:  # stop when the cost is 90% of budget
            #         print("****")
            if lazy==True:
                max_index = np.argmax(Delta)
                delta = (fun(G + [U[max_index]]) - fun(G))/ cost_fun([U[max_index]]) ** r
                Delta[max_index] = delta
                idx =[]
                while (max_index not in idx) and (delta < np.amax(Delta)):
                    idx.append(max_index)
                    max_index = np.argmax(Delta)
                    delta = (fun(G + [U[max_index]]) - fun(G))/ (cost_fun([U[max_index]]) ** r)
                    Delta[max_index] = delta
                k = U[max_index]
                del Delta[max_index]
            else:
                L = [(fun(G + [u]) - fun(G)) / (cost_fun([u])) ** r for u in U]
                k = U[np.array(L).argmax()]
            cur_cost = cost_fun(G + [k])
            if cur_cost <= budget:  # and f(G + [k])- f(G) >= 0:
                G += [k]
                cost = cur_cost
            U.remove(k)
        L = [fun([u]) for u in self.V if cost_fun([u])<budget]
        v = np.array(L).argmax()
        if fun(G) > fun([v]):
            res = G
        else:
            res = v
        return res

    def double_greedy(self, fun, cost_fun, budget=50):
        """get the Summary using double greedy

        Args:
          fun: the function to maximize
          V: sparse matrix to represent all sentences in grams
          cost_fun: the cost function for summary
          V_counts: sparse matrix to represent all sentences in grams counts
          budget: the upper bound for the cost of summary
        Returns:
            X0: the index for the best sentences to reprensent the article
        """
        V = self.X_train_tf
        V_counts = self.X_train_counts
        U = list(range(V.shape[0]))
        X0 = []
        X1 = U[:]
        for e in U:
            improve1 = fun(X0 + [e]) - fun(X0)
            X1_without_e = X1[:]
            X1_without_e.remove(e)
            improve2 = fun(X1_without_e) - fun(X1)
            if improve1 >= improve2 and cost_fun(X0 + [e]) <= budget:
                X0 += [e]
            else:
                X1.remove(e)
        return X0

    def compare(self, budget = 50, r = 1):
        t_start = time.time()
        S_MMR = self.greedy_submodular(self.get_f_MMR, self.get_cost, budget, r)
        t_MMR = time.time() - t_start
        
        t_start = time.time()
        S_MMR_lazy = self.greedy_submodular(self.get_f_MMR, self.get_cost, budget, r, lazy = True)
        t_MMR_lazy = time.time() - t_start

        t_start = time.time()
        S_MMR_double = self.double_greedy(self.get_f_MMR, self.get_cost, budget)
        t_MMR_double = time.time() - t_start

        t_start = time.time()
        S_sub = self.greedy_submodular(self.get_f_sub, self.get_cost, budget, r)
        t_sub = time.time() - t_start

        rouge_MMR = self.rouge_n(S_MMR)
        rouge_MMR_lazy = self.rouge_n(S_MMR_lazy)
        rouge_MMR_double = self.rouge_n(S_MMR_double)
        rouge_sub = self.rouge_n(S_sub)

        return rouge_MMR, rouge_MMR_lazy, rouge_MMR_double, rouge_sub, t_MMR, t_MMR_lazy, t_MMR_double, t_sub





def main():
    path = 'OpinosisDataset1.0_0/topics/'
    topics_file_names = os.listdir(path)
    n1 = 0
    n2 = 0
    names = ['rouge_MMRs', 'rouge_MMR_lazy', 'rouge_MMR_doubles', 'rouge_subs', 't_MMRs', 't_MMR_lazy', 't_MMR_doubles', 't_subs']
    arr = np.zeros((len(topics_file_names), 6))
    results = pd.DataFrame(arr,columns=names)
    i = 0
    for topics_file_name in topics_file_names:
        print(topics_file_name)
        sub_model = SubmodModel(topics_file_name)
        res = sub_model.compare()
        for k in range(len(names)):
            results[names[k]].iloc[i] = res[k]
        i +=1
        # try:
        #     sub_model = SubmodModel(topics_file_name)
        
        #     n1 += 1
        # except:
        #     print(topics_file_name)
        #     n2 += 1
        #     continue
    print(results.mean())

    #print(n1, n2)

    # some other things useful later:
    # rouge_MMR, rouge_MMR_double, rouge_sub, t_MMR, t_MMR_double, t_sub = sub_model.compare()
    #
    # rouge_MMRs.append(rouge_MMR)
    # rouge_MMR_doubles.append(rouge_MMR_double)
    # rouge_subs.append(rouge_sub)
    # t_MMRs.append(t_MMR)
    # t_MMR_doubles.append(t_MMR_double)
    # t_subs.append(t_sub)




if __name__ == '__main__':
    main()



