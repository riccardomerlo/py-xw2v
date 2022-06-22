"""Contains all methods used to evaluate the perturbed model vs original model vs retrained model
    
    todo etc etc
"""

from spacy import spatial
from after_training_torch import get_emb_og, get_emb_pert


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """

    return 1 - spatial.distance.cosine(v1, v2)

def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """

    return np.mean([cos_sim(W, a) for a in A]) - np.mean([cos_sim(W, b) for b in B])


def weat_score(X, Y, A, B):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = [weat_association(x, A, B) for x in X]
    y_association = [weat_association(y, A, B) for y in Y]

    tmp1 = np.mean(x_association) - np.mean(y_association)

    denom = np.std(np.concatenate((x_association, y_association), axis=0), ddof= 1)
  
    return tmp1/denom


def get_full_effectsize(wv, S, T, A, B):
    """
    Returns effect size calculated on full corpus

    """
    t1 = np.array([get_emb_og(wv, word) for word in S])
    t2 = np.array([get_emb_og(wv, word) for word in T])
    att1 = np.array([get_emb_og(wv, word) for word in A])
    att2 = np.array([get_emb_og(wv, word) for word in B])
    ef_full = weat_score(t1, t2, att1, att2)

    return et_full

def get_perturbated_effectsize(perturbed_emb, S, T, A, B):
    """
    Returns effect size calculated on perturbed corpus

    """
    t1 = np.array([get_emb_pert(perturbed_emb, word) for word in S])
    t2 = np.array([get_emb_pert(perturbed_emb, word) for word in T])
    att1 = np.array([get_emb_pert(perturbed_emb, word) for word in A])
    att2 = np.array([get_emb_pert(perturbed_emb, word) for word in B])
    ef_perturbed = weat_score(t1, t2, att1, att2)

    return ef_perturbed