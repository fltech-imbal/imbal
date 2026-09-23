import os, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import imbal

TARGET_COLUMN='ln_peak_intensity'; RARE_THRESHOLD=np.log(10)
TRAIN_DATA_PATH='../../../tutorials/data/SEP-C/sep_10mev_training.csv'; TEST_DATA_PATH='../../../tutorials/data/SEP-C/sep_10mev_testing.csv'
EXPECTED_INPUT_COUNT=22; MAX_EPOCHS=500; PATIENCE=100; BATCH_SIZE=32; FOLD_COUNT=5
SCRIPT_DIRECTORY=os.path.dirname(os.path.abspath(__file__)); LOAD_SAVED_MODEL=False
MODEL_DIRECTORY=os.path.join(SCRIPT_DIRECTORY,'saved_models','normal_regression_sep_c'); MODEL_PATH=os.path.join(MODEL_DIRECTORY,'model.keras')

def set_seed(s):
 os.environ['PYTHONHASHSEED']=str(s); random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)

def load_data():
 tr=pd.read_csv(TRAIN_DATA_PATH); te=pd.read_csv(TEST_DATA_PATH)
 Xtr=tr.drop(columns=[TARGET_COLUMN]).to_numpy(dtype='float32'); Xte=te.drop(columns=[TARGET_COLUMN]).to_numpy(dtype='float32')
 ytr=tr[TARGET_COLUMN].to_numpy(dtype='float32').reshape(-1,1); yte=te[TARGET_COLUMN].to_numpy(dtype='float32').reshape(-1,1)
 if Xtr.shape[1]!=EXPECTED_INPUT_COUNT or Xte.shape[1]!=EXPECTED_INPUT_COUNT: raise ValueError('Unexpected feature count.')
 ltr=(ytr>=RARE_THRESHOLD).astype('int32'); lte=(yte>=RARE_THRESHOLD).astype('int32')
 return Xtr,Xte,ytr,yte,ltr,lte,np.concatenate([Xtr,Xte]),np.concatenate([ytr,yte])

def build_model(n=EXPECTED_INPUT_COUNT):
 x=keras.Input((n,),name='model_input'); h=layers.Dense(18,activation='relu',name='dense_1')(x); h=layers.Dense(12,activation='relu',name='dense_2')(h); h=layers.Dense(8,activation='relu',name='dense_3')(h); r=layers.Dense(6,activation='relu',name='representation')(h); y=layers.Dense(1,name='prediction')(r); return keras.Model(x,y,name='normal_regression_model')

def compile_model(m): m.compile(optimizer=keras.optimizers.Adam(1e-3),loss='mse',metrics=['mae'])

def kfold_epochs(X,y,seed=42):
 vals=[]; k=KFold(FOLD_COUNT,shuffle=True,random_state=seed); print('\nK-fold Epoch Estimation')
 for f,(ti,vi) in enumerate(k.split(X),1):
  tf.keras.backend.clear_session(); set_seed(seed+f); m=build_model(X.shape[1]); compile_model(m); es=keras.callbacks.EarlyStopping(monitor='val_loss',patience=PATIENCE,mode='min',restore_best_weights=True)
  h=m.fit(X[ti],y[ti],validation_data=(X[vi],y[vi]),epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,callbacks=[es],shuffle=False,verbose=0)
  e=int(np.argmin(h.history['val_loss'])+1); vals.append(e); print(f'Fold {f}: best epoch = {e}')
 out=max(1,int(np.round(np.mean(vals)))); print(f'Fold best epochs: {vals}\nMean/rounded epoch count used for final training: {out}'); return out

def metrics(m,X,y,l):
 p=np.asarray(m(X,training=False)).reshape(-1); t=np.asarray(y).reshape(-1); mask=np.asarray(l).reshape(-1).astype(bool); o=mean_absolute_error(t,p); r=mean_absolute_error(t[mask],p[mask]); return o,r,(o+r)/2

def table(m,X,y):
 t=np.asarray(y).reshape(-1); p=np.asarray(m(X,training=False)).reshape(-1)
 for name,idx in [('ELEVATED SAMPLES (0 <= y < ln(10))',np.flatnonzero((t>=0)&(t<RARE_THRESHOLD))),('RARE SAMPLES (y >= ln(10))',np.flatnonzero(t>=RARE_THRESHOLD))]:
  print('\nTest '+name); head=f"{'Idx':>5} {'True y':>10} {'Prediction':>12} {'Delta':>12}"; print(head); print('-'*len(head))
  for i in idx: print(f'{i:5d} {t[i]:10.4f} {p[i]:12.4f} {p[i]-t[i]:12.4f}')

def run(seed=42):
 set_seed(seed); Xtr,Xte,ytr,yte,ltr,lte,Xall,yall=load_data(); print(f'Training samples: {len(Xtr)}\nRare training samples: {int(ltr.sum())}\nTesting samples: {len(Xte)}\nRare testing samples: {int(lte.sum())}')
 if LOAD_SAVED_MODEL:
  if not os.path.exists(MODEL_PATH): raise FileNotFoundError(MODEL_PATH)
  m=keras.models.load_model(MODEL_PATH); print(f'\nLoaded saved model from: {MODEL_PATH}')
 else:
  e=kfold_epochs(Xtr,ytr,seed); tf.keras.backend.clear_session(); set_seed(seed+1000); m=build_model(Xtr.shape[1]); compile_model(m); m.fit(Xtr,ytr,epochs=e,batch_size=BATCH_SIZE,shuffle=False,verbose=0); os.makedirs(MODEL_DIRECTORY,exist_ok=True); m.save(MODEL_PATH); print(f'\nSaved model to: {MODEL_PATH}')
 a=metrics(m,Xtr,ytr,ltr); b=metrics(m,Xte,yte,lte); print(f'\nTraining Normal Regression Results\nTraining overall MAE: {a[0]:.4f}\nTraining rare MAE: {a[1]:.4f}\nTraining AORE: {a[2]:.4f}'); print(f'\nFinal Test Set Results\nTest overall MAE: {b[0]:.4f}\nTest rare MAE: {b[1]:.4f}\nTest AORE: {b[2]:.4f}'); table(m,Xte,yte)
 rep=keras.Model(m.input,m.get_layer('representation').output); imbal.regression.tsne_visualization(rep,Xall,yall); pred=m.predict(Xte); imbal.regression.plot_true_vs_predictions(yte,pred); return m
if __name__=='__main__': final_model=run(42)
