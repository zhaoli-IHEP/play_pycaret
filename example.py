from pylab import *
import sys
import torch
import numpy as np

import pandas
from collections import OrderedDict


from pycaret.regression import *
from sklearn.metrics import make_scorer


#------------------------------------------------
# START
#------------------------------------------------
n_arg = len(sys.argv)
if n_arg != 4:
    print( "We need one input data file and test data file and set index (1/2) " )
    quit()


setnumber = int(sys.argv[3])
print( "set number:", setnumber )






#---------------------------------------------------
print( "Loading data file:", sys.argv[1] )
x_yk_data = torch.load( sys.argv[1] )
if setnumber == 1:
    x_yk_data = x_yk_data[:,0:80]
elif setnumber == 2:
    x_yk_data = x_yk_data[:,80:120]
else:
    print( "set number is unexpected!" )
    quit()
n_point = x_yk_data.size()[1]
x_data = x_yk_data[0,:]
yk_data = x_yk_data[1,:]
delta_yk_data = x_yk_data[2,:]
#---------------------------------------------------



print( "# of points:", n_point )
print( "x:", x_data )
print( "yk:", yk_data )
print( "delta_yk:", delta_yk_data )

nx = x_data.size()[0]
print( "nx:", nx )


#---------------------------------------------------
print( "Loading test data file:", sys.argv[2] )
test_x_yk_data = torch.load( sys.argv[2] )
if setnumber == 1:
    test_x_yk_data = test_x_yk_data[:,0:80]
elif setnumber == 2:
    test_x_yk_data = test_x_yk_data[:,80:120]
else:
    print( "set number is unexpected!" )
    quit()
test_n_point = test_x_yk_data.size()[1]
test_x_data = test_x_yk_data[0,:]
test_yk_data = test_x_yk_data[1,:]
test_delta_yk_data = test_x_yk_data[2,:]
#---------------------------------------------------


double_x_data = torch.cat( [ x_data, test_x_data ] )
double_yk_data = torch.cat( [ yk_data, test_yk_data ] )


test_dict = {
    'x':double_x_data,
    'lnx':torch.log(double_x_data),
    'y':double_yk_data
}

test_df = pandas.DataFrame( OrderedDict( test_dict ) )

test_reg = setup( data = test_df, train_size=0.5, target = 'y', data_split_shuffle = False )
print( test_reg )




#--------------------------------
dt = create_model('dt', verbose=False)
#print( dt )

tuned_dt = tune_model(dt, verbose=False)

results = pull()
print( results )

exit()
#--------------------------------


