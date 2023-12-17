# from keras_tuner import HyperModel
from keras.layers import Dense,LSTM, Dropout,Input
from keras.models import Model
from keras.optimizers import RMSprop
# from metrics.evaluate import R2_SCORES


def model_fn(self,hp):

    input_layer=Input(self.input_shape)
    lstm1=LSTM(units=hp.Int(f"LSTM_1_units",min_value=100,max_value=256), return_sequences=True)(input_layer) #128
    dropout1=Dropout(rate=hp.Float(f"Dropout_1_rate",min_value=0.2,max_value=0.7))(lstm1)   #0.55
    #batch_normalization=BatchNormalization()(lstm1)
    #BatchNormalization
    lstm2 = LSTM(units=hp.Int(f"LSTM_2_units",min_value=50,max_value=256), return_sequences=False)(dropout1) #256
    dropout2=Dropout(rate=hp.Float(f"Dropout_2_rate",min_value=0.2,max_value=0.7))(lstm2)   #031
    # dense=Dense(units=hp.Choice(f"Dense_output",values=[self.nb_output]),activation=hp.Choice(f"activation_function",values=['linear','sigmoid']))(dropout2)
    #output_layer=Dense(units=hp.Choice(f"Dense_output",values=[self.nb_output]),activation=hp.Choice(f"activation_function",values=['linear','sigmoid']))(dropout2)
    output_layer=Dense(units=1,activation='linear')(dropout2)
    lr=hp.Float(f"learning_rate",min_value=1e-5,max_value=1e-2)
    model=Model(inputs=input_layer,outputs=output_layer)
    model.compile(loss='mean_squared_error',optimizer=RMSprop(learning_rate=lr),metrics=['mae',self.r2_scores.r2_keras,self.r2_scores.adjusted_r2])
    model.summary()
    return model




# class LSTM_RegNetwork(HyperModel):
#     def __init__(self,input_shape:list,nb_output:int):
#         self.input_shape=input_shape
#         self.nb_output=nb_output
#         self.r2_scores =R2_SCORES(self.input_shape)
#     def build(self,hp):
#
#         input_layer=Input(self.input_shape)
#         lstm1=LSTM(units=hp.Int(f"LSTM_1_units",min_value=100,max_value=256), return_sequences=True)(input_layer) #128
#         dropout1=Dropout(rate=hp.Float(f"Dropout_1_rate",min_value=0.2,max_value=0.7))(lstm1)   #0.55
#         #batch_normalization=BatchNormalization()(lstm1)
#         #BatchNormalization
#         lstm2 = LSTM(units=hp.Int(f"LSTM_2_units",min_value=50,max_value=256), return_sequences=False)(dropout1) #256
#         dropout2=Dropout(rate=hp.Float(f"Dropout_2_rate",min_value=0.2,max_value=0.7))(lstm2)   #031
#         # dense=Dense(units=hp.Choice(f"Dense_output",values=[self.nb_output]),activation=hp.Choice(f"activation_function",values=['linear','sigmoid']))(dropout2)
#         #output_layer=Dense(units=hp.Choice(f"Dense_output",values=[self.nb_output]),activation=hp.Choice(f"activation_function",values=['linear','sigmoid']))(dropout2)
#         output_layer=Dense(units=1,activation='linear')(dropout2)
#         lr=hp.Float(f"learning_rate",min_value=1e-5,max_value=1e-2)
#         model=Model(inputs=input_layer,outputs=output_layer)
#         model.compile(loss='mean_squared_error',optimizer=RMSprop(learning_rate=lr),metrics=['mae',self.r2_scores.r2_keras,self.r2_scores.adjusted_r2])
#         model.summary()
#         return model



