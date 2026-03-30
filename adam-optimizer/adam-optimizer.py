import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    mt = np.zeros(len(param))
    vt = np.zeros(len(param))
    mt_unbiased = np.zeros(len(param))
    vt_unbiased = np.zeros(len(param))
    param_new = np.zeros(len(param))
    for i in range(len(param)):
        mt[i] = beta1*m[i] + (1-beta1)*grad[i]
        vt[i] = beta2*v[i] +(1-beta2)*grad[i]**2
        mt_unbiased[i] = mt[i]/(1-beta1**t)
        vt_unbiased[i] = vt[i]/(1-beta2**t)
        param_new[i] = param[i] - lr*mt_unbiased[i]/(np.sqrt(vt_unbiased[i])+eps)
    return param_new, mt, vt