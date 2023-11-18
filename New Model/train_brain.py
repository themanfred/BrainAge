# Note: Code needs to be run on Training


import numpy as np
import nibabel as nib
import os, glob, sys, time, pickle, csv
from tqdm import tqdm
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import log_cosh

from ST_utils import build_ST

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser('   ==========   Fetal brain age prediction with Swin transformer, made by Sungmin You   ==========   ')
parser.add_argument('-train_csv',action='store',dest='train',type=str, default='BAST_train_all_files.csv', help='input csv table')
parser.add_argument('-val_csv',action='store',dest='valid',type=str, default='BAST_valid_all_files.csv', help='input csv table')
parser.add_argument('-batch_size',action='store',default=64,dest='num_batch',type=int, help='Number of batch')
parser.add_argument('-n_slice',action='store',dest='num_slice', default=4,type=int, help='Number of training slice from a volume')
parser.add_argument('-n_epoch',action='store',dest='n_epoch', default=1000,type=int, help='Number of training epoch')
parser.add_argument('-learning_rate',action='store',dest='learning_rate', default=1e-4, type=float, help='Learning rate')
parser.add_argument('-red_factor',action='store',dest='red_factor', default=0.9 ,type=float, help='Reduction factor for ReduceLROnPlateau')
parser.add_argument('-threshold',action='store',dest='threshold',type=float, default=0.4, help='IQA threshold for slice quality --deprecated')
parser.add_argument('-tta',action='store',dest='num_tta',default=20, type=int, help='Number of tta')
parser.add_argument('-d_huber', action='store',dest='delta_huber', default=0.9, type=float, help='delta value of huber loss')
parser.add_argument('-gpu',action='store',dest='num_gpu',default='1', type=str, help='GPU selection')
parser.add_argument('-rl', '--result_save_location', action='store', default='./result', dest='result_loc', type=str, help='Output folder name, default: ./')
parser.add_argument('-wl', '--weight_save_location', action='store', default='./weight', dest='weight_loc', type=str, help='Output folder name, default: ./')
parser.add_argument('-hl', '--history_save_location', action='store', default='./hist', dest='hist_loc',  type=str, help='Output folder name, default: ./')
parser.add_argument('-output',action='store',dest='output',type=str, default='output', help='name for csv logger')
args = parser.parse_args()

result_loc=args.result_loc + '_opt_longer_tiny_bs_{0}_lr_{1}_rf_{2}'.format(args.num_batch, args.learning_rate, args.red_factor)
weight_loc=args.weight_loc + '_opt_longer_tiny_bs_{0}_lr_{1}_rf_{2}'.format(args.num_batch, args.learning_rate, args.red_factor)
hist_loc=args.hist_loc + '_opt_longer_tiny_bs_{0}_lr_{1}_rf_{2}'.format(args.num_batch, args.learning_rate, args.red_factor)
output_file=args.output + '_opt_longer_tiny_bs_{0}_lr_{1}_rf_{2}'.format(args.num_batch, args.learning_rate, args.red_factor)

if os.path.exists(result_loc)==False:
    os.makedirs(result_loc,exist_ok=True)
if os.path.exists(weight_loc)==False:
    os.makedirs(weight_loc, exist_ok=True)
if os.path.exists(hist_loc)==False:
    os.makedirs(hist_loc, exist_ok=True)

print('\n\n')
print('\t\t Prediction result save location: \t\t\t'+os.path.realpath(result_loc))
print('\t\t Prediction weights save location: \t\t\t'+os.path.realpath(weight_loc))
print('\t\t Prediction history save location: \t\t\t'+os.path.realpath(hist_loc))
print('\t\t number training slice: \t\t\t\t'+str(args.num_slice))
print('\t\t TTA times: \t\t\t\t\t\t'+str(args.num_tta))
print('\t\t batch_size: \t\t\t\t\t\t'+str(args.num_batch))
print('\t\t delta of Huber loss: \t\t\t\t\t'+str(args.delta_huber))
print('\t\t GPU number: \t\t\t\t\t\t'+str(args.num_gpu))
print('\n\n')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.num_gpu

#print(tf.__version__)
print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

batch_size = args.num_batch
num_slice=args.num_slice
threshold=args.threshold

tf.random.set_seed(1234)
np.random.seed(1234)

# functions

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 200:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 400:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 600:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 800:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 1000:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 1200:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 1400:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))
        if epoch == 1600:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights(weight_loc+"/weights{}.h5".format(epoch))


def crop_pad_ND(img, target_shape):
    import operator, numpy as np
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result

def huber_loss(y_true, y_pred, delta=args.delta_huber):
  error = y_pred - y_true
  abs_error = K.abs(error)
  quadratic = K.minimum(abs_error, delta)
  linear = abs_error - quadratic
  return 0.5 * K.square(quadratic) + delta * linear

datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    #rotation_range=360,
    #width_shift_range=0.15,
    #height_shift_range=0.1,
    #brightness_range=[0.5,1],
    vertical_flip=True,
    horizontal_flip=True)

def tta_prediction(datagen, model, dic, n_example):
    preds=np.zeros([len(dic),])
    for i in range(len(dic)):
        image = np.expand_dims(dic[i],0)
        pred = model.predict_generator(datagen.flow(image, batch_size=n_example),workers=4,steps=n_example, verbose=0)
        preds[i]=np.mean(pred)
    return preds

def make_dic(img_list, num_slice, slice_mode=0, desc=''):
    max_size = [224, 224, 1]
    if slice_mode:
        dic = np.zeros([len(img_list), max_size[1], max_size[0], num_slice],dtype=np.float16)
    else:
        dic = np.zeros([len(img_list)*num_slice, max_size[1], max_size[0], 1],dtype=np.float16)
    for i in tqdm(range(0, len(img_list)),desc=desc):
        img = np.squeeze(nib.load(img_list[i]).get_fdata())
        img = crop_pad_ND(img, np.max(np.vstack((max_size, img.shape)),axis=0))       
        if slice_mode:
            dic[i,:,:,:]=np.swapaxes(img[:,:,int(img.shape[-1]/2-1-int(num_slice/2)):int(img.shape[-1]/2+int(num_slice/2))],0,1)
        else:
            try:
                dic[i*num_slice:i*num_slice+num_slice,:,:,0]=np.swapaxes(img[:,:,int(img.shape[-1]/2-int(num_slice/2)):int(img.shape[-1]/2+int(num_slice/2))],0,2)
            except:
                print("Discarded image : " + img_list[i])
    return dic

train_df = pd.read_csv(args.train)
valid_df = pd.read_csv(args.valid)

train_dic = make_dic(train_df.MR.values, num_slice, slice_mode=0, desc='make train dic')
val_dic = make_dic(valid_df.MR.values, num_slice, slice_mode=0, desc='make val dic')

train_GW = train_df.GA.values
b_train_GW = np.zeros([len(train_GW)*num_slice,])

for tt in range(len(train_GW)):
    b_train_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(train_GW[tt],(num_slice,))

val_GW = valid_df.GA.values
b_val_GW = np.zeros([len(val_GW)*num_slice,])
val_ID = valid_df.ID.values
b_val_ID = []
for tt in range(len(val_GW)):
    b_val_ID = np.concatenate((b_val_ID,np.tile(val_ID[tt],(num_slice,))),axis=0)
for tt in range(len(val_GW)):
    b_val_GW[tt*num_slice:tt*num_slice+num_slice,]=np.tile(val_GW[tt],(num_slice,))

#model = age_predic_network([138,176,1])
model = build_ST()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, decay=0.0001), loss=huber_loss, metrics=['mae','mse'])

cb_reduceLR = ReduceLROnPlateau(monitor='mae', factor=args.red_factor, patience=100, verbose=1)
cb_earlyStop = EarlyStopping(monitor='mae', patience=500, verbose=1, mode='min',restore_best_weights=True)
cb_modelCheckpoint = ModelCheckpoint(filepath=weight_loc+'/best_fold_rsl.h5', monitor='mae', save_best_only=True, mode='min', save_weights_only=True, verbose=0)
cb_CSVLogger = CSVLogger('log_'+str(output_file)+'.csv', separator=",", append=True)


callbacks = [cb_reduceLR, cb_earlyStop, cb_modelCheckpoint, cb_CSVLogger]


histo = model.fit(datagen.flow(train_dic,b_train_GW,batch_size=batch_size,shuffle=True),steps_per_epoch=len(train_dic)/batch_size,epochs=args.n_epoch, validation_data=datagen.flow(val_dic, b_val_GW, batch_size=batch_size,shuffle=True),validation_steps=len(val_dic),workers=8,callbacks=callbacks, verbose=2)

with open(hist_loc+'/history_fold_rsl.pkl', 'wb') as file_pi:
        pickle.dump(histo.history, file_pi)
model.load_weights(weight_loc+'/best_fold_rsl.h5')

# save important data
#c_path = os.getcwd()
#os.chdir(hist_loc)
#with open('Data'+str(output_file)+'.csv', 'w') as out:
#    csv_out=csv.writer(out)
#    csv_out.writerow(['Number of slices used: '+ str(len(b_train_GW))])
#os.chdir(c_path)

del model, histo, #p_age2

K.clear_session()
tf.compat.v1.reset_default_graph()
