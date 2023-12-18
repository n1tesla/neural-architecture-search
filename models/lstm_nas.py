from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D,LSTM

def model_fn(actions):

    input_layer=Input(shape=(30,1))
    units_1, units_2, units_3 = actions
    x=LSTM(units=units_1, return_sequences=True)(input_layer) #128
    x=LSTM(units=units_2,return_sequences=True)(x)
    x=LSTM(units=units_3)(x)

    output_layer=Dense(1)(x)
    model=Model(inputs=input_layer,outputs=output_layer)

    return model


