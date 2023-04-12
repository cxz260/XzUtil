import matplotlib.pyplot as plt
import numpy as np

def create_dataset(data, n_input, n_output, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :n_input])
        Y.append(data[i + seq_len, :n_output])
    return np.array(X), np.array(Y)

def pred_eval(p0,p): # plot actual(p0-2D ndarray) vs prediction(p); cal correlation corr[dp;dp0] and sign accuracy
    n=len(p0)
    m=p0.shape[1]
    dp=(p[1:]-p0[:n-1])/p0[:n-1]
    dp0=(p0[1:]-p0[:n-1])/p0[:n-1]
    ps,ps0=np.sign(dp),np.sign(dp0)
    plt.plot(p0, label='actual')
    plt.plot(p, label='predicted')
    plt.legend()
    plt.show()
    #prediction delta correlation, similar to R-squared)
    print('\n predicted return dp  vs real return dp0 correlation and sign accuracy:\n')
    for i in range(m):
      crr=np.corrcoef(dp[:,i],dp0[:,i])
      acc=np.dot(ps[:,i],ps0[:,i])/(n-1)
      print(f"({i}) crr: {crr[0,1]:.4f}; sign acc:{acc:0.4f}")
    
def plot_cmp_scat(p0,p):  #p/p0, 2-D ndarray
  # Plot the predictions(p) vs. actual values(p0)
  m=p0.shape[1]
  fig, ax = plt.subplots(2, m, figsize=(3*2*m, 8))  # 2-rows,m-# of plot each row, each plot HxW=3x8
  fig.tight_layout(h_pad=4)
  for i in range(m):
      # Line plot
      i1=i+1
      ax[0, i].plot(p0[:, i], label="actual{i1}")
      ax[0, i].plot(p[:, i], label="pred{i1}")
      ax[0, i].set_title(f"Output variable {i1}")
      ax[0, i].set_xlabel("Sequence index")
      ax[0, i].set_ylabel("Value")
      ax[0, i].legend()
      # Scatter plot
      ax[1, i].scatter(p0[:, i], p[:, i], alpha=0.5)
      ax[1, i].set_title(f"Output variable {i1}")
      ax[1, i].set_xlabel("actual{i1}")
      ax[1, i].set_ylabel("pred{i1}")
      fig.suptitle('Overall Title')
      plt.subplots_adjust(top=0.9) #plot(occupy 90%), Overall Title(10%)