лк<
бЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Іў9
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:&'*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:&'*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:'*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:'N* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:'N*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:N*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:N'*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:N'*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:'*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:''*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:''*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:'*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:'*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	NИ*&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	NИ*
dtype0

lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	NИ*0
shared_name!lstm/lstm_cell/recurrent_kernel

3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes
:	NИ*
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:И*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&'*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:&'*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:'*
dtype0

Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:'N*'
shared_nameAdam/conv1d_1/kernel/m

*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:'N*
dtype0

Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:N*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:N'*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:N'*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:'*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:''*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:''*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:'*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:'*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	NИ*-
shared_nameAdam/lstm/lstm_cell/kernel/m

0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m*
_output_shapes
:	NИ*
dtype0
Љ
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	NИ*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
Ђ
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes
:	NИ*
dtype0

Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*+
shared_nameAdam/lstm/lstm_cell/bias/m

.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes	
:И*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&'*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:&'*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:'*
dtype0

Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:'N*'
shared_nameAdam/conv1d_1/kernel/v

*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:'N*
dtype0

Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:N*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:N*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:N'*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:N'*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:'*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:''*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:''*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:'*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:'*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	NИ*-
shared_nameAdam/lstm/lstm_cell/kernel/v

0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v*
_output_shapes
:	NИ*
dtype0
Љ
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	NИ*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
Ђ
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes
:	NИ*
dtype0

Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*+
shared_nameAdam/lstm/lstm_cell/bias/v

.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes	
:И*
dtype0

NoOpNoOp
ЖL
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ёK
valueчKBфK BнK

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
R
%	variables
&regularization_losses
'trainable_variables
(	keras_api
l
)cell
*
state_spec
+	variables
,regularization_losses
-trainable_variables
.	keras_api
h

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
h

5kernel
6bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
Ф
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemmmm/m0m5m6m;m<mFmGmHmvvvv /vЁ0vЂ5vЃ6vЄ;vЅ<vІFvЇGvЈHvЉ
^
0
1
2
3
F4
G5
H6
/7
08
59
610
;11
<12
 
^
0
1
2
3
F4
G5
H6
/7
08
59
610
;11
<12
­
	variables
Ilayer_regularization_losses
regularization_losses
Jlayer_metrics
trainable_variables
Kmetrics
Lnon_trainable_variables

Mlayers
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
Nlayer_regularization_losses
regularization_losses
Olayer_metrics
trainable_variables
Pmetrics
Qnon_trainable_variables

Rlayers
 
 
 
­
	variables
Slayer_regularization_losses
regularization_losses
Tlayer_metrics
trainable_variables
Umetrics
Vnon_trainable_variables

Wlayers
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
	variables
Xlayer_regularization_losses
regularization_losses
Ylayer_metrics
trainable_variables
Zmetrics
[non_trainable_variables

\layers
 
 
 
­
!	variables
]layer_regularization_losses
"regularization_losses
^layer_metrics
#trainable_variables
_metrics
`non_trainable_variables

alayers
 
 
 
­
%	variables
blayer_regularization_losses
&regularization_losses
clayer_metrics
'trainable_variables
dmetrics
enon_trainable_variables

flayers
~

Fkernel
Grecurrent_kernel
Hbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
 

F0
G1
H2
 

F0
G1
H2
Й
+	variables
klayer_regularization_losses
,regularization_losses
llayer_metrics
-trainable_variables
mmetrics

nstates
onon_trainable_variables

players
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01
 

/0
01
­
1	variables
qlayer_regularization_losses
2regularization_losses
rlayer_metrics
3trainable_variables
smetrics
tnon_trainable_variables

ulayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
­
7	variables
vlayer_regularization_losses
8regularization_losses
wlayer_metrics
9trainable_variables
xmetrics
ynon_trainable_variables

zlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
­
=	variables
{layer_regularization_losses
>regularization_losses
|layer_metrics
?trainable_variables
}metrics
~non_trainable_variables

layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

F0
G1
H2
 

F0
G1
H2
В
g	variables
 layer_regularization_losses
hregularization_losses
layer_metrics
itrainable_variables
metrics
non_trainable_variables
layers
 
 
 
 
 

)0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
|z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџ
&*
dtype0* 
shape:џџџџџџџџџ
&
Ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_488606
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*=
Tin6
422	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_491885


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastotalcounttotal_1count_1Adam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_492039юВ8
ТV
Џ
'__forward_gpu_lstm_with_fallback_489120

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_14986aff-0904-487d-8a55-1f6d987dc954*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_488945_489121*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ТV
Џ
'__forward_gpu_lstm_with_fallback_487828

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_51040e20-c121-47c2-a81a-cc07c79c973e*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_487653_487829*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ТV
Џ
'__forward_gpu_lstm_with_fallback_490249

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_eca0e142-bba6-46cb-a6a2-75c11068f1a8*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_490074_490250*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias

N
2__inference_average_pooling1d_layer_call_fn_485386

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_4853802
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќG
Ы
@__inference_lstm_layer_call_and_return_conditional_losses_487831

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeз
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2и2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeн
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЌМ2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_1/GreaterEqual/yЪ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeн
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2УмЎ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_2/GreaterEqual/yЪ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeн
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2уу2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_3/GreaterEqual/yЪ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_3/Mul_1b
mulMulinputsdropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2Щ
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4875552
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:џџџџџџџџџN::::S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
яV
Џ
'__forward_gpu_lstm_with_fallback_487253

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1у
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_82f38c3f-3df0-4c08-820e-16668ab41d21*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_487078_487254*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ТV
Џ
'__forward_gpu_lstm_with_fallback_485349

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_2838e972-21e6-4da1-bb46-64b835246701*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_485174_485350*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias


%__inference_lstm_layer_call_fn_490707

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4878312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџN:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
Є-
Я
while_body_488761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
д	
Љ
-__inference_functional_1_layer_call_fn_489677

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_4884662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
ЏA
Р
 __inference_standard_lstm_487555

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_487469*
condR
while_cond_487468*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_51040e20-c121-47c2-a81a-cc07c79c973e*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ТV
Џ
'__forward_gpu_lstm_with_fallback_489624

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_07d4d0ac-fa9a-4d34-8b01-17737d2a1756*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_489449_489625*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ч

;__inference___backward_gpu_lstm_with_fallback_490074_490250
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_eca0e142-bba6-46cb-a6a2-75c11068f1a8*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_490249*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч

;__inference___backward_gpu_lstm_with_fallback_488097_488273
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_1528d0fd-cd6e-4d3e-ad64-8d9d6bf3dec7*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_488272*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч

;__inference___backward_gpu_lstm_with_fallback_488945_489121
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_14986aff-0904-487d-8a55-1f6d987dc954*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_489120*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
э
_
C__inference_reshape_layer_call_and_return_conditional_losses_487348

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :N2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
ч

;__inference___backward_gpu_lstm_with_fallback_490518_490694
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_8d72fd4b-afec-44d5-9196-f133f476a647*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_490693*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
яV
Џ
'__forward_gpu_lstm_with_fallback_491635

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1у
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_92ecc2c0-3dbb-42ff-a893-dbf861d5794c*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_491460_491636*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
Ь
Ћ
C__inference_dense_1_layer_call_and_return_conditional_losses_491689

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:''*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ'2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ':::O K
'
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
ы
|
'__inference_conv1d_layer_call_fn_489733

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_4872852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
'2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ
&::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
ш

;__inference___backward_gpu_lstm_with_fallback_491460_491636
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesл
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeЖ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationџ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1В
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapesђ
я:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџџџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_92ecc2c0-3dbb-42ff-a893-dbf861d5794c*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_491635*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
:::
6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
уJ
з
)__inference_gpu_lstm_with_fallback_488944

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_14986aff-0904-487d-8a55-1f6d987dc954*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
K
з
)__inference_gpu_lstm_with_fallback_486622

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1п
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_b8a246ee-7727-431c-a314-5df23aca4023*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ъ
Г
H__inference_functional_1_layer_call_and_return_conditional_losses_489142

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource%
!lstm_read_readvariableop_resource'
#lstm_read_1_readvariableop_resource'
#lstm_read_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЋ
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
&2
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&'*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&'2
conv1d/conv1d/ExpandDims_1в
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'*
paddingSAME*
strides
2
conv1d/conv1dЇ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/SqueezeЁ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
conv1d/BiasAdd/ReadVariableOpЈ
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
conv1d/Relu
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dimЪ
average_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'2
average_pooling1d/ExpandDimsо
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'*
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPoolВ
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ'*
squeeze_dims
2
average_pooling1d/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_1/conv1d/ExpandDims/dimЭ
conv1d_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'2
conv1d_1/conv1d/ExpandDimsг
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:'N*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimл
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:'N2
conv1d_1/conv1d/ExpandDims_1к
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
paddingSAME*
strides
2
conv1d_1/conv1d­
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims

§џџџџџџџџ2
conv1d_1/conv1d/SqueezeЇ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype02!
conv1d_1/BiasAdd/ReadVariableOpА
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџN2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
conv1d_1/Relu
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dimв
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџN2 
average_pooling1d_1/ExpandDimsф
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
ksize
*
paddingVALID*
strides
2
average_pooling1d_1/AvgPoolИ
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims
2
average_pooling1d_1/Squeezer
reshape/ShapeShape$average_pooling1d_1/Squeeze:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :N2
reshape/Reshape/shape/2Ш
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЉ
reshape/ReshapeReshape$average_pooling1d_1/Squeeze:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
reshape/Reshape`

lstm/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
lstm/zeros_1t
lstm/ones_like/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
lstm/ones_like/Shapeq
lstm/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm/ones_like/Const
lstm/ones_likeFilllstm/ones_like/Shape:output:0lstm/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/ones_likem
lstm/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/dropout/Const
lstm/dropout/MulMullstm/ones_like:output:0lstm/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout/Mulo
lstm/dropout/ShapeShapelstm/ones_like:output:0*
T0*
_output_shapes
:2
lstm/dropout/Shapeц
)lstm/dropout/random_uniform/RandomUniformRandomUniformlstm/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2яњ2+
)lstm/dropout/random_uniform/RandomUniform
lstm/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
lstm/dropout/GreaterEqual/yж
lstm/dropout/GreaterEqualGreaterEqual2lstm/dropout/random_uniform/RandomUniform:output:0$lstm/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout/GreaterEqual
lstm/dropout/CastCastlstm/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout/Cast
lstm/dropout/Mul_1Mullstm/dropout/Mul:z:0lstm/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout/Mul_1q
lstm/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/dropout_1/Const
lstm/dropout_1/MulMullstm/ones_like:output:0lstm/dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_1/Muls
lstm/dropout_1/ShapeShapelstm/ones_like:output:0*
T0*
_output_shapes
:2
lstm/dropout_1/Shapeь
+lstm/dropout_1/random_uniform/RandomUniformRandomUniformlstm/dropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2ВыЗ2-
+lstm/dropout_1/random_uniform/RandomUniform
lstm/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
lstm/dropout_1/GreaterEqual/yо
lstm/dropout_1/GreaterEqualGreaterEqual4lstm/dropout_1/random_uniform/RandomUniform:output:0&lstm/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_1/GreaterEqual
lstm/dropout_1/CastCastlstm/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_1/Cast
lstm/dropout_1/Mul_1Mullstm/dropout_1/Mul:z:0lstm/dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_1/Mul_1q
lstm/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/dropout_2/Const
lstm/dropout_2/MulMullstm/ones_like:output:0lstm/dropout_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_2/Muls
lstm/dropout_2/ShapeShapelstm/ones_like:output:0*
T0*
_output_shapes
:2
lstm/dropout_2/Shapeы
+lstm/dropout_2/random_uniform/RandomUniformRandomUniformlstm/dropout_2/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2л2-
+lstm/dropout_2/random_uniform/RandomUniform
lstm/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
lstm/dropout_2/GreaterEqual/yо
lstm/dropout_2/GreaterEqualGreaterEqual4lstm/dropout_2/random_uniform/RandomUniform:output:0&lstm/dropout_2/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_2/GreaterEqual
lstm/dropout_2/CastCastlstm/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_2/Cast
lstm/dropout_2/Mul_1Mullstm/dropout_2/Mul:z:0lstm/dropout_2/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_2/Mul_1q
lstm/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/dropout_3/Const
lstm/dropout_3/MulMullstm/ones_like:output:0lstm/dropout_3/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_3/Muls
lstm/dropout_3/ShapeShapelstm/ones_like:output:0*
T0*
_output_shapes
:2
lstm/dropout_3/Shapeь
+lstm/dropout_3/random_uniform/RandomUniformRandomUniformlstm/dropout_3/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2кЕЏ2-
+lstm/dropout_3/random_uniform/RandomUniform
lstm/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
lstm/dropout_3/GreaterEqual/yо
lstm/dropout_3/GreaterEqualGreaterEqual4lstm/dropout_3/random_uniform/RandomUniform:output:0&lstm/dropout_3/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_3/GreaterEqual
lstm/dropout_3/CastCastlstm/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_3/Cast
lstm/dropout_3/Mul_1Mullstm/dropout_3/Mul:z:0lstm/dropout_3/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/dropout_3/Mul_1
lstm/mulMulreshape/Reshape:output:0lstm/dropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџN2

lstm/mul
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
lstm/Read/ReadVariableOpv
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2
lstm/Identity
lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
lstm/Read_1/ReadVariableOp|
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2
lstm/Identity_1
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
lstm/Read_2/ReadVariableOpx
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2
lstm/Identity_2ё
lstm/PartitionedCallPartitionedCalllstm/mul:z:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4888472
lstm/PartitionedCall
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:N'*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMullstm/PartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense/BiasAddЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:''*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense_1/BiasAddЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&::::::::::::::S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
ЏA
Р
 __inference_standard_lstm_487999

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_487913*
condR
while_cond_487912*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_1528d0fd-cd6e-4d3e-ad64-8d9d6bf3dec7*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ч

;__inference___backward_gpu_lstm_with_fallback_489449_489625
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_07d4d0ac-fa9a-4d34-8b01-17737d2a1756*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_489624*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
С
while_cond_491275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_491275___redundant_placeholder04
0while_while_cond_491275___redundant_placeholder14
0while_while_cond_491275___redundant_placeholder24
0while_while_cond_491275___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Й
Ж
!__inference__wrapped_model_485371
input_1C
?functional_1_conv1d_conv1d_expanddims_1_readvariableop_resource7
3functional_1_conv1d_biasadd_readvariableop_resourceE
Afunctional_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource9
5functional_1_conv1d_1_biasadd_readvariableop_resource2
.functional_1_lstm_read_readvariableop_resource4
0functional_1_lstm_read_1_readvariableop_resource4
0functional_1_lstm_read_2_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource7
3functional_1_dense_2_matmul_readvariableop_resource8
4functional_1_dense_2_biasadd_readvariableop_resource
identityЁ
)functional_1/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2+
)functional_1/conv1d/conv1d/ExpandDims/dimг
%functional_1/conv1d/conv1d/ExpandDims
ExpandDimsinput_12functional_1/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
&2'
%functional_1/conv1d/conv1d/ExpandDimsє
6functional_1/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?functional_1_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&'*
dtype028
6functional_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp
+functional_1/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+functional_1/conv1d/conv1d/ExpandDims_1/dim
'functional_1/conv1d/conv1d/ExpandDims_1
ExpandDims>functional_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:04functional_1/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&'2)
'functional_1/conv1d/conv1d/ExpandDims_1
functional_1/conv1d/conv1dConv2D.functional_1/conv1d/conv1d/ExpandDims:output:00functional_1/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'*
paddingSAME*
strides
2
functional_1/conv1d/conv1dЮ
"functional_1/conv1d/conv1d/SqueezeSqueeze#functional_1/conv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'*
squeeze_dims

§џџџџџџџџ2$
"functional_1/conv1d/conv1d/SqueezeШ
*functional_1/conv1d/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv1d_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02,
*functional_1/conv1d/BiasAdd/ReadVariableOpм
functional_1/conv1d/BiasAddBiasAdd+functional_1/conv1d/conv1d/Squeeze:output:02functional_1/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
functional_1/conv1d/BiasAdd
functional_1/conv1d/ReluRelu$functional_1/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
functional_1/conv1d/Relu 
-functional_1/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/average_pooling1d/ExpandDims/dimў
)functional_1/average_pooling1d/ExpandDims
ExpandDims&functional_1/conv1d/Relu:activations:06functional_1/average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'2+
)functional_1/average_pooling1d/ExpandDims
&functional_1/average_pooling1d/AvgPoolAvgPool2functional_1/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'*
ksize
*
paddingVALID*
strides
2(
&functional_1/average_pooling1d/AvgPoolй
&functional_1/average_pooling1d/SqueezeSqueeze/functional_1/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ'*
squeeze_dims
2(
&functional_1/average_pooling1d/SqueezeЅ
+functional_1/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2-
+functional_1/conv1d_1/conv1d/ExpandDims/dim
'functional_1/conv1d_1/conv1d/ExpandDims
ExpandDims/functional_1/average_pooling1d/Squeeze:output:04functional_1/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'2)
'functional_1/conv1d_1/conv1d/ExpandDimsњ
8functional_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAfunctional_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:'N*
dtype02:
8functional_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp 
-functional_1/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-functional_1/conv1d_1/conv1d/ExpandDims_1/dim
)functional_1/conv1d_1/conv1d/ExpandDims_1
ExpandDims@functional_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:06functional_1/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:'N2+
)functional_1/conv1d_1/conv1d/ExpandDims_1
functional_1/conv1d_1/conv1dConv2D0functional_1/conv1d_1/conv1d/ExpandDims:output:02functional_1/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
paddingSAME*
strides
2
functional_1/conv1d_1/conv1dд
$functional_1/conv1d_1/conv1d/SqueezeSqueeze%functional_1/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims

§џџџџџџџџ2&
$functional_1/conv1d_1/conv1d/SqueezeЮ
,functional_1/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype02.
,functional_1/conv1d_1/BiasAdd/ReadVariableOpф
functional_1/conv1d_1/BiasAddBiasAdd-functional_1/conv1d_1/conv1d/Squeeze:output:04functional_1/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџN2
functional_1/conv1d_1/BiasAdd
functional_1/conv1d_1/ReluRelu&functional_1/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
functional_1/conv1d_1/ReluЄ
/functional_1/average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/average_pooling1d_1/ExpandDims/dim
+functional_1/average_pooling1d_1/ExpandDims
ExpandDims(functional_1/conv1d_1/Relu:activations:08functional_1/average_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџN2-
+functional_1/average_pooling1d_1/ExpandDims
(functional_1/average_pooling1d_1/AvgPoolAvgPool4functional_1/average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
ksize
*
paddingVALID*
strides
2*
(functional_1/average_pooling1d_1/AvgPoolп
(functional_1/average_pooling1d_1/SqueezeSqueeze1functional_1/average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims
2*
(functional_1/average_pooling1d_1/Squeeze
functional_1/reshape/ShapeShape1functional_1/average_pooling1d_1/Squeeze:output:0*
T0*
_output_shapes
:2
functional_1/reshape/Shape
(functional_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_1/reshape/strided_slice/stackЂ
*functional_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_1/reshape/strided_slice/stack_1Ђ
*functional_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_1/reshape/strided_slice/stack_2р
"functional_1/reshape/strided_sliceStridedSlice#functional_1/reshape/Shape:output:01functional_1/reshape/strided_slice/stack:output:03functional_1/reshape/strided_slice/stack_1:output:03functional_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"functional_1/reshape/strided_slice
$functional_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$functional_1/reshape/Reshape/shape/1
$functional_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :N2&
$functional_1/reshape/Reshape/shape/2
"functional_1/reshape/Reshape/shapePack+functional_1/reshape/strided_slice:output:0-functional_1/reshape/Reshape/shape/1:output:0-functional_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"functional_1/reshape/Reshape/shapeн
functional_1/reshape/ReshapeReshape1functional_1/average_pooling1d_1/Squeeze:output:0+functional_1/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
functional_1/reshape/Reshape
functional_1/lstm/ShapeShape%functional_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2
functional_1/lstm/Shape
%functional_1/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%functional_1/lstm/strided_slice/stack
'functional_1/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'functional_1/lstm/strided_slice/stack_1
'functional_1/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'functional_1/lstm/strided_slice/stack_2Ю
functional_1/lstm/strided_sliceStridedSlice functional_1/lstm/Shape:output:0.functional_1/lstm/strided_slice/stack:output:00functional_1/lstm/strided_slice/stack_1:output:00functional_1/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
functional_1/lstm/strided_slice
functional_1/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
functional_1/lstm/zeros/mul/yД
functional_1/lstm/zeros/mulMul(functional_1/lstm/strided_slice:output:0&functional_1/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
functional_1/lstm/zeros/mul
functional_1/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
functional_1/lstm/zeros/Less/yЏ
functional_1/lstm/zeros/LessLessfunctional_1/lstm/zeros/mul:z:0'functional_1/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
functional_1/lstm/zeros/Less
 functional_1/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2"
 functional_1/lstm/zeros/packed/1Ы
functional_1/lstm/zeros/packedPack(functional_1/lstm/strided_slice:output:0)functional_1/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2 
functional_1/lstm/zeros/packed
functional_1/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
functional_1/lstm/zeros/ConstН
functional_1/lstm/zerosFill'functional_1/lstm/zeros/packed:output:0&functional_1/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
functional_1/lstm/zeros
functional_1/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2!
functional_1/lstm/zeros_1/mul/yК
functional_1/lstm/zeros_1/mulMul(functional_1/lstm/strided_slice:output:0(functional_1/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
functional_1/lstm/zeros_1/mul
 functional_1/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 functional_1/lstm/zeros_1/Less/yЗ
functional_1/lstm/zeros_1/LessLess!functional_1/lstm/zeros_1/mul:z:0)functional_1/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2 
functional_1/lstm/zeros_1/Less
"functional_1/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2$
"functional_1/lstm/zeros_1/packed/1б
 functional_1/lstm/zeros_1/packedPack(functional_1/lstm/strided_slice:output:0+functional_1/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 functional_1/lstm/zeros_1/packed
functional_1/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
functional_1/lstm/zeros_1/ConstХ
functional_1/lstm/zeros_1Fill)functional_1/lstm/zeros_1/packed:output:0(functional_1/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
functional_1/lstm/zeros_1
!functional_1/lstm/ones_like/ShapeShape%functional_1/reshape/Reshape:output:0*
T0*
_output_shapes
:2#
!functional_1/lstm/ones_like/Shape
!functional_1/lstm/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!functional_1/lstm/ones_like/Constа
functional_1/lstm/ones_likeFill*functional_1/lstm/ones_like/Shape:output:0*functional_1/lstm/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
functional_1/lstm/ones_likeИ
functional_1/lstm/mulMul%functional_1/reshape/Reshape:output:0$functional_1/lstm/ones_like:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
functional_1/lstm/mulО
%functional_1/lstm/Read/ReadVariableOpReadVariableOp.functional_1_lstm_read_readvariableop_resource*
_output_shapes
:	NИ*
dtype02'
%functional_1/lstm/Read/ReadVariableOp
functional_1/lstm/IdentityIdentity-functional_1/lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2
functional_1/lstm/IdentityФ
'functional_1/lstm/Read_1/ReadVariableOpReadVariableOp0functional_1_lstm_read_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02)
'functional_1/lstm/Read_1/ReadVariableOpЃ
functional_1/lstm/Identity_1Identity/functional_1/lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2
functional_1/lstm/Identity_1Р
'functional_1/lstm/Read_2/ReadVariableOpReadVariableOp0functional_1_lstm_read_2_readvariableop_resource*
_output_shapes	
:И*
dtype02)
'functional_1/lstm/Read_2/ReadVariableOp
functional_1/lstm/Identity_2Identity/functional_1/lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2
functional_1/lstm/Identity_2й
!functional_1/lstm/PartitionedCallPartitionedCallfunctional_1/lstm/mul:z:0 functional_1/lstm/zeros:output:0"functional_1/lstm/zeros_1:output:0#functional_1/lstm/Identity:output:0%functional_1/lstm/Identity_1:output:0%functional_1/lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4850762#
!functional_1/lstm/PartitionedCallЦ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes

:N'*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpа
functional_1/dense/MatMulMatMul*functional_1/lstm/PartitionedCall:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
functional_1/dense/MatMulХ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpЭ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
functional_1/dense/BiasAddЬ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:''*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpЯ
functional_1/dense_1/MatMulMatMul#functional_1/dense/BiasAdd:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
functional_1/dense_1/MatMulЫ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpе
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
functional_1/dense_1/BiasAddЬ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpб
functional_1/dense_2/MatMulMatMul%functional_1/dense_1/BiasAdd:output:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/dense_2/MatMulЫ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpе
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/dense_2/BiasAdd 
functional_1/dense_2/SoftmaxSoftmax%functional_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/dense_2/Softmaxz
IdentityIdentity&functional_1/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&::::::::::::::T P
+
_output_shapes
:џџџџџџџџџ
&
!
_user_specified_name	input_1
Є-
Я
while_body_486894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
$
Э
@__inference_lstm_layer_call_and_return_conditional_losses_491638
inputs_0 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1Z
ones_like/ShapeShapeinputs_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	ones_liken
mulMulinputs_0ones_like:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2в
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4913622
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџN::::^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
"
_user_specified_name
inputs/0
уJ
з
)__inference_gpu_lstm_with_fallback_487652

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_51040e20-c121-47c2-a81a-cc07c79c973e*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias

З
B__inference_conv1d_layer_call_and_return_conditional_losses_489724

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
&2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&'*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&'2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ
&:::S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
ѓ
i
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_485380

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsК
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ)
І
H__inference_functional_1_layer_call_and_return_conditional_losses_488385
input_1
conv1d_487296
conv1d_487298
conv1d_1_487329
conv1d_1_487331
lstm_488298
lstm_488300
lstm_488302
dense_488326
dense_488328
dense_1_488352
dense_1_488354
dense_2_488379
dense_2_488381
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_487296conv1d_487298*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_4872852 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_4853802#
!average_pooling1d/PartitionedCallП
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_487329conv1d_1_487331*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_4873182"
 conv1d_1/StatefulPartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_4853952%
#average_pooling1d_1/PartitionedCallў
reshape/PartitionedCallPartitionedCall,average_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_4873482
reshape/PartitionedCallЌ
lstm/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0lstm_488298lstm_488300lstm_488302*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4878312
lstm/StatefulPartitionedCallЇ
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_488326dense_488328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4883152
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_488352dense_1_488354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4883412!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_488379dense_2_488381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4883682!
dense_2/StatefulPartitionedCallУ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
&
!
_user_specified_name	input_1
д	
Љ
-__inference_functional_1_layer_call_fn_489708

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_4885362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
уJ
з
)__inference_gpu_lstm_with_fallback_488096

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_1528d0fd-cd6e-4d3e-ad64-8d9d6bf3dec7*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ќG
Ы
@__inference_lstm_layer_call_and_return_conditional_losses_490252

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeз
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЖЏ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeн
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2ОФд2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_1/GreaterEqual/yЪ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeн
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЗщЙ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_2/GreaterEqual/yЪ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeн
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
dtype0*
seedБџх)*
seed2іІ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_3/GreaterEqual/yЪ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџN2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2
dropout_3/Mul_1b
mulMulinputsdropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2Щ
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4899762
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:џџџџџџџџџN::::S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
Є-
Я
while_body_486439
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
й
{
&__inference_dense_layer_call_fn_491679

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4883152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ'2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџN::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
з	
Њ
-__inference_functional_1_layer_call_fn_488565
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_4885362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
&
!
_user_specified_name	input_1
ш

;__inference___backward_gpu_lstm_with_fallback_491016_491192
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesл
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeЖ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationџ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1В
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapesђ
я:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџџџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_e8b5dda6-b1cd-46e6-87db-d795eb08e159*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_491191*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
:::
6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Є-
Я
while_body_487913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
уJ
з
)__inference_gpu_lstm_with_fallback_490073

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_eca0e142-bba6-46cb-a6a2-75c11068f1a8*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
цA
Р
 __inference_standard_lstm_491362

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_491276*
condR
while_cond_491275*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_92ecc2c0-3dbb-42ff-a893-dbf861d5794c*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
	
С
while_cond_489264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_489264___redundant_placeholder04
0while_while_cond_489264___redundant_placeholder14
0while_while_cond_489264___redundant_placeholder24
0while_while_cond_489264___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

З
B__inference_conv1d_layer_call_and_return_conditional_losses_487285

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
&2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&'*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&'2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ
&:::S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
уJ
з
)__inference_gpu_lstm_with_fallback_490517

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_8d72fd4b-afec-44d5-9196-f133f476a647*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
цA
Р
 __inference_standard_lstm_486525

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_486439*
condR
while_cond_486438*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_b8a246ee-7727-431c-a314-5df23aca4023*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ЏA
Р
 __inference_standard_lstm_489351

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_489265*
condR
while_cond_489264*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_07d4d0ac-fa9a-4d34-8b01-17737d2a1756*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
з	
Њ
-__inference_functional_1_layer_call_fn_488495
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_4884662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
&
!
_user_specified_name	input_1
Є-
Я
while_body_490334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
ТV
Џ
'__forward_gpu_lstm_with_fallback_490693

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_8d72fd4b-afec-44d5-9196-f133f476a647*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_490518_490694*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
Ь#
Ы
@__inference_lstm_layer_call_and_return_conditional_losses_488275

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	ones_likec
mulMulinputsones_like:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2Щ
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4879992
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:џџџџџџџџџN::::S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
Ј)
Ѕ
H__inference_functional_1_layer_call_and_return_conditional_losses_488536

inputs
conv1d_488500
conv1d_488502
conv1d_1_488506
conv1d_1_488508
lstm_488513
lstm_488515
lstm_488517
dense_488520
dense_488522
dense_1_488525
dense_1_488527
dense_2_488530
dense_2_488532
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_488500conv1d_488502*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_4872852 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_4853802#
!average_pooling1d/PartitionedCallП
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_488506conv1d_1_488508*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_4873182"
 conv1d_1/StatefulPartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_4853952%
#average_pooling1d_1/PartitionedCallў
reshape/PartitionedCallPartitionedCall,average_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_4873482
reshape/PartitionedCallЌ
lstm/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0lstm_488513lstm_488515lstm_488517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4882752
lstm/StatefulPartitionedCallЇ
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_488520dense_488522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4883152
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_488525dense_1_488527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4883412!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_488530dense_2_488532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4883682!
dense_2/StatefulPartitionedCallУ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
K
з
)__inference_gpu_lstm_with_fallback_491459

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1п
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_92ecc2c0-3dbb-42ff-a893-dbf861d5794c*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
яV
Џ
'__forward_gpu_lstm_with_fallback_486798

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1у
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_b8a246ee-7727-431c-a314-5df23aca4023*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_486623_486799*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
Є-
Я
while_body_490832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
Є-
Я
while_body_489890
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
Ћ)
І
H__inference_functional_1_layer_call_and_return_conditional_losses_488424
input_1
conv1d_488388
conv1d_488390
conv1d_1_488394
conv1d_1_488396
lstm_488401
lstm_488403
lstm_488405
dense_488408
dense_488410
dense_1_488413
dense_1_488415
dense_2_488418
dense_2_488420
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_488388conv1d_488390*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_4872852 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_4853802#
!average_pooling1d/PartitionedCallП
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_488394conv1d_1_488396*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_4873182"
 conv1d_1/StatefulPartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_4853952%
#average_pooling1d_1/PartitionedCallў
reshape/PartitionedCallPartitionedCall,average_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_4873482
reshape/PartitionedCallЌ
lstm/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0lstm_488401lstm_488403lstm_488405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4882752
lstm/StatefulPartitionedCallЇ
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_488408dense_488410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4883152
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_488413dense_1_488415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4883412!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_488418dense_2_488420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4883682!
dense_2/StatefulPartitionedCallУ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
&
!
_user_specified_name	input_1
ЏA
Р
 __inference_standard_lstm_490420

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_490334*
condR
while_cond_490333*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_8d72fd4b-afec-44d5-9196-f133f476a647*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
	
С
while_cond_486893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_486893___redundant_placeholder04
0while_while_cond_486893___redundant_placeholder14
0while_while_cond_486893___redundant_placeholder24
0while_while_cond_486893___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

P
4__inference_average_pooling1d_1_layer_call_fn_485401

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_4853952
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
уJ
з
)__inference_gpu_lstm_with_fallback_489448

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_07d4d0ac-fa9a-4d34-8b01-17737d2a1756*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
	
С
while_cond_487912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_487912___redundant_placeholder04
0while_while_cond_487912___redundant_placeholder14
0while_while_cond_487912___redundant_placeholder24
0while_while_cond_487912___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ЏA
Р
 __inference_standard_lstm_488847

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_488761*
condR
while_cond_488760*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_14986aff-0904-487d-8a55-1f6d987dc954*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
чI
Э
@__inference_lstm_layer_call_and_return_conditional_losses_491194
inputs_0 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1Z
ones_like/ShapeShapeinputs_0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeр
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2ѓг2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЫ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeх
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЮЛk2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_1/GreaterEqual/yг
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeц
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЊЕ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_2/GreaterEqual/yг
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeц
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЪЛЏ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_3/GreaterEqual/yг
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/Mul_1m
mulMulinputs_0dropout/Mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2в
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4909182
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџN::::^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
"
_user_specified_name
inputs/0
ч

;__inference___backward_gpu_lstm_with_fallback_485174_485350
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_2838e972-21e6-4da1-bb46-64b835246701*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_485349*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
яa
є
__inference__traced_save_491885
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_28f1e839fc294790a02eb942297ba0e9/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*Ђ
valueB1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesъ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesВ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesј
ѕ: :&':':'N:N:N':':'':':':: : : : : :	NИ:	NИ:И: : : : :&':':'N:N:N':':'':':'::	NИ:	NИ:И:&':':'N:N:N':':'':':'::	NИ:	NИ:И: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:&': 

_output_shapes
:':($
"
_output_shapes
:'N: 

_output_shapes
:N:$ 

_output_shapes

:N': 

_output_shapes
:':$ 

_output_shapes

:'': 

_output_shapes
:':$	 

_output_shapes

:': 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%!

_output_shapes
:	NИ:!

_output_shapes	
:И:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:&': 

_output_shapes
:':($
"
_output_shapes
:'N: 

_output_shapes
:N:$ 

_output_shapes

:N': 

_output_shapes
:':$ 

_output_shapes

:'': 

_output_shapes
:':$ 

_output_shapes

:':  

_output_shapes
::%!!

_output_shapes
:	NИ:%"!

_output_shapes
:	NИ:!#

_output_shapes	
:И:($$
"
_output_shapes
:&': %

_output_shapes
:':(&$
"
_output_shapes
:'N: '

_output_shapes
:N:$( 

_output_shapes

:N': )

_output_shapes
:':$* 

_output_shapes

:'': +

_output_shapes
:':$, 

_output_shapes

:': -

_output_shapes
::%.!

_output_shapes
:	NИ:%/!

_output_shapes
:	NИ:!0

_output_shapes	
:И:1

_output_shapes
: 
Є-
Я
while_body_491276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
Є-
Я
while_body_487469
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
н
}
(__inference_dense_1_layer_call_fn_491698

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4883412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ'2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ'::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs


%__inference_lstm_layer_call_fn_491660
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4872562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџN:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
"
_user_specified_name
inputs/0

Й
D__inference_conv1d_1_layer_call_and_return_conditional_losses_487318

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:'N*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:'N2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:N*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџN2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ':::S O
+
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
Ѓ
D
(__inference_reshape_layer_call_fn_489776

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_4873482
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
э
_
C__inference_reshape_layer_call_and_return_conditional_losses_489771

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :N2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
	
С
while_cond_486438
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_486438___redundant_placeholder04
0while_while_cond_486438___redundant_placeholder14
0while_while_cond_486438___redundant_placeholder24
0while_while_cond_486438___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
уJ
з
)__inference_gpu_lstm_with_fallback_485173

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1ж
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_2838e972-21e6-4da1-bb46-64b835246701*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
Ј)
Ѕ
H__inference_functional_1_layer_call_and_return_conditional_losses_488466

inputs
conv1d_488430
conv1d_488432
conv1d_1_488436
conv1d_1_488438
lstm_488443
lstm_488445
lstm_488447
dense_488450
dense_488452
dense_1_488455
dense_1_488457
dense_2_488460
dense_2_488462
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂlstm/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_488430conv1d_488432*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_4872852 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ'* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_4853802#
!average_pooling1d/PartitionedCallП
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_488436conv1d_1_488438*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_4873182"
 conv1d_1/StatefulPartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_4853952%
#average_pooling1d_1/PartitionedCallў
reshape/PartitionedCallPartitionedCall,average_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_4873482
reshape/PartitionedCallЌ
lstm/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0lstm_488443lstm_488445lstm_488447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4878312
lstm/StatefulPartitionedCallЇ
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_488450dense_488452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4883152
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_488455dense_1_488457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4883412!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_488460dense_2_488462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4883682!
dense_2/StatefulPartitionedCallУ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
Ь
Ћ
C__inference_dense_1_layer_call_and_return_conditional_losses_488341

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:''*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ'2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ':::O K
'
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
	
С
while_cond_488760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_488760___redundant_placeholder04
0while_while_cond_488760___redundant_placeholder14
0while_while_cond_488760___redundant_placeholder24
0while_while_cond_488760___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
K
з
)__inference_gpu_lstm_with_fallback_491015

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1п
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_e8b5dda6-b1cd-46e6-87db-d795eb08e159*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias


%__inference_lstm_layer_call_fn_490718

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4882752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџN:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
я
~
)__inference_conv1d_1_layer_call_fn_489758

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџN*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_4873182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ'::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs


%__inference_lstm_layer_call_fn_491649
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџN*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_4868012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџN:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
"
_user_specified_name
inputs/0
яV
Џ
'__forward_gpu_lstm_with_fallback_491191

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1у
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_e8b5dda6-b1cd-46e6-87db-d795eb08e159*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_491016_491192*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
АЩ

"__inference__traced_restore_492039
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate-
)assignvariableop_15_lstm_lstm_cell_kernel7
3assignvariableop_16_lstm_lstm_cell_recurrent_kernel+
'assignvariableop_17_lstm_lstm_cell_bias
assignvariableop_18_total
assignvariableop_19_count
assignvariableop_20_total_1
assignvariableop_21_count_1,
(assignvariableop_22_adam_conv1d_kernel_m*
&assignvariableop_23_adam_conv1d_bias_m.
*assignvariableop_24_adam_conv1d_1_kernel_m,
(assignvariableop_25_adam_conv1d_1_bias_m+
'assignvariableop_26_adam_dense_kernel_m)
%assignvariableop_27_adam_dense_bias_m-
)assignvariableop_28_adam_dense_1_kernel_m+
'assignvariableop_29_adam_dense_1_bias_m-
)assignvariableop_30_adam_dense_2_kernel_m+
'assignvariableop_31_adam_dense_2_bias_m4
0assignvariableop_32_adam_lstm_lstm_cell_kernel_m>
:assignvariableop_33_adam_lstm_lstm_cell_recurrent_kernel_m2
.assignvariableop_34_adam_lstm_lstm_cell_bias_m,
(assignvariableop_35_adam_conv1d_kernel_v*
&assignvariableop_36_adam_conv1d_bias_v.
*assignvariableop_37_adam_conv1d_1_kernel_v,
(assignvariableop_38_adam_conv1d_1_bias_v+
'assignvariableop_39_adam_dense_kernel_v)
%assignvariableop_40_adam_dense_bias_v-
)assignvariableop_41_adam_dense_1_kernel_v+
'assignvariableop_42_adam_dense_1_bias_v-
)assignvariableop_43_adam_dense_2_kernel_v+
'assignvariableop_44_adam_dense_2_bias_v4
0assignvariableop_45_adam_lstm_lstm_cell_kernel_v>
:assignvariableop_46_adam_lstm_lstm_cell_recurrent_kernel_v2
.assignvariableop_47_adam_lstm_lstm_cell_bias_v
identity_49ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*Ђ
valueB1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names№
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЃ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѓ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Є
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ђ
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Є
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8І
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Є
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10Ѕ
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ї
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ї
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ў
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Б
AssignVariableOp_15AssignVariableOp)assignvariableop_15_lstm_lstm_cell_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Л
AssignVariableOp_16AssignVariableOp3assignvariableop_16_lstm_lstm_cell_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Џ
AssignVariableOp_17AssignVariableOp'assignvariableop_17_lstm_lstm_cell_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ё
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ё
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ѓ
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ў
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adam_conv1d_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24В
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv1d_1_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25А
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv1d_1_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Џ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27­
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Б
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Џ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Б
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Џ
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32И
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_lstm_lstm_cell_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Т
AssignVariableOp_33AssignVariableOp:assignvariableop_33_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ж
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_lstm_lstm_cell_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35А
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv1d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ў
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv1d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37В
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38А
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Џ
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40­
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Б
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Џ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Б
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Џ
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45И
AssignVariableOp_45AssignVariableOp0assignvariableop_45_adam_lstm_lstm_cell_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Т
AssignVariableOp_46AssignVariableOp:assignvariableop_46_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ж
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_lstm_lstm_cell_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_479
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpў
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48ё
Identity_49IdentityIdentity_48:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_49"#
identity_49Identity_49:output:0*з
_input_shapesХ
Т: ::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ъ
Љ
A__inference_dense_layer_call_and_return_conditional_losses_491670

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:N'*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ'2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџN:::O K
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
ш

;__inference___backward_gpu_lstm_with_fallback_486623_486799
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesл
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeЖ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationџ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1В
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapesђ
я:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџџџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_b8a246ee-7727-431c-a314-5df23aca4023*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_486798*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
:::
6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ъ
Љ
A__inference_dense_layer_call_and_return_conditional_losses_488315

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:N'*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ'2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџN:::O K
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
н
}
(__inference_dense_2_layer_call_fn_491718

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4883682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ'::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
љ#
Ы
@__inference_lstm_layer_call_and_return_conditional_losses_487256

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	ones_likel
mulMulinputsones_like:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2в
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4869802
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџN::::\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs
ч

;__inference___backward_gpu_lstm_with_fallback_487653_487829
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesв
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationп
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationі
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1Љ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*щ
_input_shapesз
д:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_51040e20-c121-47c2-a81a-cc07c79c973e*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_487828*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :1-
+
_output_shapes
:џџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
::1
-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЏA
Р
 __inference_standard_lstm_485076

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_484990*
condR
while_cond_484989*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_2838e972-21e6-4da1-bb46-64b835246701*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
А
Ћ
C__inference_dense_2_layer_call_and_return_conditional_losses_488368

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:'*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ':::O K
'
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
А
Ћ
C__inference_dense_2_layer_call_and_return_conditional_losses_491709

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:'*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ':::O K
'
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
	
С
while_cond_484989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_484989___redundant_placeholder04
0while_while_cond_484989___redundant_placeholder14
0while_while_cond_484989___redundant_placeholder24
0while_while_cond_484989___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
K
з
)__inference_gpu_lstm_with_fallback_487077

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim}

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axisЌ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:2

concat_1п
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_82f38c3f-3df0-4c08-820e-16668ab41d21*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
ЏA
Р
 __inference_standard_lstm_489976

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_489890*
condR
while_cond_489889*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityk

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_eca0e142-bba6-46cb-a6a2-75c11068f1a8*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
	
С
while_cond_487468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_487468___redundant_placeholder04
0while_while_cond_487468___redundant_placeholder14
0while_while_cond_487468___redundant_placeholder24
0while_while_cond_487468___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
цA
Р
 __inference_standard_lstm_490918

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_490832*
condR
while_cond_490831*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_e8b5dda6-b1cd-46e6-87db-d795eb08e159*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
Ь#
Ы
@__inference_lstm_layer_call_and_return_conditional_losses_490696

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
	ones_likec
mulMulinputsones_like:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2Щ
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4904202
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*6
_input_shapes%
#:џџџџџџџџџN::::S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs
ѕ
k
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_485395

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsК
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Й
D__inference_conv1d_1_layer_call_and_return_conditional_losses_489749

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:'N*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:'N2
conv1d/ExpandDims_1Ж
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:N*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџN2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ':::S O
+
_output_shapes
:џџџџџџџџџ'
 
_user_specified_nameinputs
	
С
while_cond_490333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_490333___redundant_placeholder04
0while_while_cond_490333___redundant_placeholder14
0while_while_cond_490333___redundant_placeholder24
0while_while_cond_490333___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
С
while_cond_490831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_490831___redundant_placeholder04
0while_while_cond_490831___redundant_placeholder14
0while_while_cond_490831___redundant_placeholder24
0while_while_cond_490831___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
С
while_cond_489889
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_489889___redundant_placeholder04
0while_while_cond_489889___redundant_placeholder14
0while_while_cond_489889___redundant_placeholder24
0while_while_cond_489889___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџN:џџџџџџџџџN: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ш

;__inference___backward_gpu_lstm_with_fallback_487078_487254
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђ(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4Ѓ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/ShapeН
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ25
3gradients/strided_slice_grad/StridedSliceGrad/beginА
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/endИ
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/stridesл
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2&
$gradients/transpose_9_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeЦ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2 
gradients/Squeeze_grad/Reshape
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/ShapeЬ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2"
 gradients/Squeeze_1_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeЖ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropФ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationџ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2#
!gradients/ExpandDims_grad/Reshape
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/Shape№
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/RankЙ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/mod
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ф/2
gradients/concat_1_grad/Shape
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_1
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_2
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_3
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_4
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_5
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_6
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Ф/2!
gradients/concat_1_grad/Shape_7
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_8
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/concat_1_grad/Shape_9
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_10
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_11
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_12
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_13
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_14
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:N2"
 gradients/concat_1_grad/Shape_15 
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffset
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ф/2
gradients/concat_1_grad/Slice
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_1
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_2
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_3
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_4
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_5
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_6
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ф/2!
gradients/concat_1_grad/Slice_7
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_8
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:N2!
gradients/concat_1_grad/Slice_9
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_10
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_11
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_12
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_13
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_14
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:N2"
 gradients/concat_1_grad/Slice_15
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2
gradients/Reshape_grad/ShapeУ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:NN2 
gradients/Reshape_grad/Reshape
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_1_grad/ShapeЫ
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_2_grad/ShapeЫ
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_3_grad/ShapeЫ
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_4_grad/ShapeЫ
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_5_grad/ShapeЫ
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_6_grad/ShapeЫ
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"N   N   2 
gradients/Reshape_7_grad/ShapeЫ
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:NN2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_8_grad/ShapeЧ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2 
gradients/Reshape_9_grad/ShapeЧ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:N2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_10_grad/ShapeЫ
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_11_grad/ShapeЫ
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_12_grad/ShapeЫ
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_12_grad/Reshape
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_13_grad/ShapeЫ
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_13_grad/Reshape
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_14_grad/ShapeЫ
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_14_grad/Reshape
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:N2!
gradients/Reshape_15_grad/ShapeЫ
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:N2#
!gradients/Reshape_15_grad/ReshapeЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationн
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_1_grad/transposeЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationп
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_2_grad/transposeЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationп
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_3_grad/transposeЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationп
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_4_grad/transposeЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationп
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_5_grad/transposeЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationп
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_6_grad/transposeЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationп
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_7_grad/transposeЬ
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutationп
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:NN2&
$gradients/transpose_8_grad/transpose
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:№2
gradients/split_2_grad/concatЮ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_grad/concatж
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	NИ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankЏ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:И2
gradients/concat_grad/Shape_1№
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Sliceї
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:И2
gradients/concat_grad/Slice_1В
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity­

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_1Џ

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_3Ё

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	NИ2

Identity_4

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:И2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapesђ
я:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: :џџџџџџџџџџџџџџџџџџN::џџџџџџџџџN:џџџџџџџџџN::џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:::џџџџџџџџџN:џџџџџџџџџN: ::::::::: : : : *=
api_implements+)lstm_82f38c3f-3df0-4c08-820e-16668ab41d21*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_487253*
go_backwards( *

time_major( 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:- )
'
_output_shapes
:џџџџџџџџџN::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN: 

_output_shapes
::1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:	

_output_shapes
:::
6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:1-
+
_output_shapes
:џџџџџџџџџN:"

_output_shapes

:: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ТV
Џ
'__forward_gpu_lstm_with_fallback_488272

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisЂCudnnRNNu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
ExpandDims_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЁ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:NN:NN:NN:NN*
	num_split2	
split_1g

zeros_likeConst*
_output_shapes	
:И*
dtype0*
valueBИ*    2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis|
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:№2
concatT
Const_2Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_2h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimЈ
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0:N:N:N:N:N:N:N:N*
	num_split2	
split_2e
Const_3Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2	
Const_3u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm{
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:NN2
transpose_1f
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm{
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:NN2
transpose_2j
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_1u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm{
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:NN2
transpose_3j
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_2u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm{
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:NN2
transpose_4j
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_3u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm}
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:NN2
transpose_5j
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_4u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm}
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:NN2
transpose_6j
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_5u
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_7/perm}
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:NN2
transpose_7j
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_6u
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_8/perm}
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:NN2
transpose_8j
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0*
_output_shapes	
:Ф/2
	Reshape_7j
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_8j
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0*
_output_shapes
:N2
	Reshape_9l

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_10l

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_11l

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_12l

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_13l

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_14l

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T0*
_output_shapes
:N2

Reshape_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1к
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2і
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slicey
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_9/perm
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
transpose_9{
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2	
Squeeze
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџN*
squeeze_dims
 2
	Squeeze_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimeu
IdentityIdentitystrided_slice:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityv

Identity_1Identitytranspose_9:y:0	^CudnnRNN*
T0*+
_output_shapes
:џџџџџџџџџN2

Identity_1s

Identity_2IdentitySqueeze:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2u

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3b

Identity_4Identityruntime:output:0	^CudnnRNN*
T0*
_output_shapes
: 2

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*m
_input_shapes\
Z:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_1528d0fd-cd6e-4d3e-ad64-8d9d6bf3dec7*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_488097_488273*
go_backwards( *

time_major( 2
CudnnRNNCudnnRNN:S O
+
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
Уn
Г
H__inference_functional_1_layer_call_and_return_conditional_losses_489646

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource%
!lstm_read_readvariableop_resource'
#lstm_read_1_readvariableop_resource'
#lstm_read_2_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimЋ
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
&2
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&'*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimг
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&'2
conv1d/conv1d/ExpandDims_1в
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'*
paddingSAME*
strides
2
conv1d/conv1dЇ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/SqueezeЁ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
conv1d/BiasAdd/ReadVariableOpЈ
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
'2
conv1d/Relu
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dimЪ
average_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
'2
average_pooling1d/ExpandDimsо
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'*
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPoolВ
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ'*
squeeze_dims
2
average_pooling1d/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_1/conv1d/ExpandDims/dimЭ
conv1d_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ'2
conv1d_1/conv1d/ExpandDimsг
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:'N*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimл
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:'N2
conv1d_1/conv1d/ExpandDims_1к
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
paddingSAME*
strides
2
conv1d_1/conv1d­
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims

§џџџџџџџџ2
conv1d_1/conv1d/SqueezeЇ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:N*
dtype02!
conv1d_1/BiasAdd/ReadVariableOpА
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџN2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
conv1d_1/Relu
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dimв
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџN2 
average_pooling1d_1/ExpandDimsф
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџN*
ksize
*
paddingVALID*
strides
2
average_pooling1d_1/AvgPoolИ
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџN*
squeeze_dims
2
average_pooling1d_1/Squeezer
reshape/ShapeShape$average_pooling1d_1/Squeeze:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :N2
reshape/Reshape/shape/2Ш
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЉ
reshape/ReshapeReshape$average_pooling1d_1/Squeeze:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
reshape/Reshape`

lstm/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
lstm/zeros_1t
lstm/ones_like/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
lstm/ones_like/Shapeq
lstm/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm/ones_like/Const
lstm/ones_likeFilllstm/ones_like/Shape:output:0lstm/ones_like/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2
lstm/ones_like
lstm/mulMulreshape/Reshape:output:0lstm/ones_like:output:0*
T0*+
_output_shapes
:џџџџџџџџџN2

lstm/mul
lstm/Read/ReadVariableOpReadVariableOp!lstm_read_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
lstm/Read/ReadVariableOpv
lstm/IdentityIdentity lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2
lstm/Identity
lstm/Read_1/ReadVariableOpReadVariableOp#lstm_read_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
lstm/Read_1/ReadVariableOp|
lstm/Identity_1Identity"lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2
lstm/Identity_1
lstm/Read_2/ReadVariableOpReadVariableOp#lstm_read_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
lstm/Read_2/ReadVariableOpx
lstm/Identity_2Identity"lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2
lstm/Identity_2ё
lstm/PartitionedCallPartitionedCalllstm/mul:z:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/Identity:output:0lstm/Identity_1:output:0lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4893512
lstm/PartitionedCall
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:N'*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMullstm/PartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense/BiasAddЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:''*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:'*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ'2
dense_1/BiasAddЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&::::::::::::::S O
+
_output_shapes
:џџџџџџџџџ
&
 
_user_specified_nameinputs
цA
Р
 __inference_standard_lstm_486980

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeА
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ќ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2
MatMulk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:џџџџџџџџџИ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
add_
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:џџџџџџџџџИ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   2
TensorArrayV2_1/element_shapeЖ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И* 
_read_only_resource_inputs
 *
bodyR
while_body_486894*
condR
while_cond_486893*b
output_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И*
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџN*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimel
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identityt

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2

Identity_1f

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_2f

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3W

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*v
_input_shapese
c:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:	NИ:	NИ:И*=
api_implements+)lstm_82f38c3f-3df0-4c08-820e-16668ab41d21*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџN
 
_user_specified_nameinit_c:GC

_output_shapes
:	NИ
 
_user_specified_namekernel:QM

_output_shapes
:	NИ
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:И

_user_specified_namebias
нI
Ы
@__inference_lstm_layer_call_and_return_conditional_losses_486801

inputs 
read_readvariableop_resource"
read_1_readvariableop_resource"
read_2_readvariableop_resource

identity_3D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :N2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2	
zeros_1X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shapeр
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2ђ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЫ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeц
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2Пу2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_1/GreaterEqual/yг
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeц
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2ЊМ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_2/GreaterEqual/yг
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeх
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN*
dtype0*
seedБџх)*
seed2Ко=2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout_3/GreaterEqual/yг
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
dropout_3/Mul_1k
mulMulinputsdropout/Mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN2
mul
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	NИ*
dtype02
Read_1/ReadVariableOpm

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	NИ2

Identity_1
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:И*
dtype02
Read_2/ReadVariableOpi

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:И2

Identity_2в
PartitionedCallPartitionedCallmul:z:0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:џџџџџџџџџN:џџџџџџџџџџџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference_standard_lstm_4865252
PartitionedCallp

Identity_3IdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2

Identity_3"!

identity_3Identity_3:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџN::::\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџN
 
_user_specified_nameinputs
Ї	
Ё
$__inference_signature_wrapper_488606
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_4853712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ
&:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
&
!
_user_specified_name	input_1
Є-
Я
while_body_489265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И
Є-
Я
while_body_484990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџN   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџN*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЂ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/MatMul_1
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџИ2
	while/add
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:џџџџџџџџџИ2
while/BiasAdd\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dimз
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN:џџџџџџџџџN*
	num_split2
while/splitq
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoidu
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_1y
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџN2
	while/mulh

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџN2

while/Tanhv
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_1u
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/add_1u
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Sigmoid_2g
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Tanh_1z
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/mul_2г
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_2/yo
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: 2
while/add_2`
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_3/yv
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: 2
while/add_3^
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3s
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_4s
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџN2
while/Identity_5"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*b
_input_shapesQ
O: : : : :џџџџџџџџџN:џџџџџџџџџN: : :	NИ:	NИ:И: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџN:-)
'
_output_shapes
:џџџџџџџџџN:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	NИ:%	!

_output_shapes
:	NИ:!


_output_shapes	
:И"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ў
serving_default
?
input_14
serving_default_input_1:0џџџџџџџџџ
&;
dense_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:љд
СY
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+Њ&call_and_return_all_conditional_losses
Ћ__call__
Ќ_default_save_signature"еU
_tf_keras_networkЙU{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 39, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 78, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 78]}}, "name": "reshape", "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 78, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 39, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 39, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 38]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 39, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 78, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 78]}}, "name": "reshape", "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 78, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 39, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["lstm", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 39, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ѓ"№
_tf_keras_input_layerа{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 38]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 38]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
у	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+­&call_and_return_all_conditional_losses
Ў__call__"М
_tf_keras_layerЂ{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 39, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 38]}}

	variables
regularization_losses
trainable_variables
	keras_api
+Џ&call_and_return_all_conditional_losses
А__call__"ђ
_tf_keras_layerи{"class_name": "AveragePooling1D", "name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ц	

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"П
_tf_keras_layerЅ{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 78, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 39}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 39]}}

!	variables
"regularization_losses
#trainable_variables
$	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"і
_tf_keras_layerм{"class_name": "AveragePooling1D", "name": "average_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
є
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"у
_tf_keras_layerЩ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 78]}}}
К
)cell
*
state_spec
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+З&call_and_return_all_conditional_losses
И__call__"

_tf_keras_rnn_layerё	{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 78, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 78]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 78]}}
№

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"Щ
_tf_keras_layerЏ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 39, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 78}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 78]}}
є

5kernel
6bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 39, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 39}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 39]}}
є

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 39}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 39]}}
з
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemmmm/m0m5m6m;m<mFmGmHmvvvv /vЁ0vЂ5vЃ6vЄ;vЅ<vІFvЇGvЈHvЉ"
	optimizer
~
0
1
2
3
F4
G5
H6
/7
08
59
610
;11
<12"
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
F4
G5
H6
/7
08
59
610
;11
<12"
trackable_list_wrapper
Ю
	variables
Ilayer_regularization_losses
regularization_losses
Jlayer_metrics
trainable_variables
Kmetrics
Lnon_trainable_variables

Mlayers
Ћ__call__
Ќ_default_save_signature
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
-
Пserving_default"
signature_map
#:!&'2conv1d/kernel
:'2conv1d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
	variables
Nlayer_regularization_losses
regularization_losses
Olayer_metrics
trainable_variables
Pmetrics
Qnon_trainable_variables

Rlayers
Ў__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
	variables
Slayer_regularization_losses
regularization_losses
Tlayer_metrics
trainable_variables
Umetrics
Vnon_trainable_variables

Wlayers
А__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
%:#'N2conv1d_1/kernel
:N2conv1d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
	variables
Xlayer_regularization_losses
regularization_losses
Ylayer_metrics
trainable_variables
Zmetrics
[non_trainable_variables

\layers
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
!	variables
]layer_regularization_losses
"regularization_losses
^layer_metrics
#trainable_variables
_metrics
`non_trainable_variables

alayers
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
%	variables
blayer_regularization_losses
&regularization_losses
clayer_metrics
'trainable_variables
dmetrics
enon_trainable_variables

flayers
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
І

Fkernel
Grecurrent_kernel
Hbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"щ
_tf_keras_layerЯ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 78, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
М
+	variables
klayer_regularization_losses
,regularization_losses
llayer_metrics
-trainable_variables
mmetrics

nstates
onon_trainable_variables

players
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
:N'2dense/kernel
:'2
dense/bias
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
А
1	variables
qlayer_regularization_losses
2regularization_losses
rlayer_metrics
3trainable_variables
smetrics
tnon_trainable_variables

ulayers
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 :''2dense_1/kernel
:'2dense_1/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
А
7	variables
vlayer_regularization_losses
8regularization_losses
wlayer_metrics
9trainable_variables
xmetrics
ynon_trainable_variables

zlayers
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 :'2dense_2/kernel
:2dense_2/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
А
=	variables
{layer_regularization_losses
>regularization_losses
|layer_metrics
?trainable_variables
}metrics
~non_trainable_variables

layers
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(:&	NИ2lstm/lstm_cell/kernel
2:0	NИ2lstm/lstm_cell/recurrent_kernel
": И2lstm/lstm_cell/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
Е
g	variables
 layer_regularization_losses
hregularization_losses
layer_metrics
itrainable_variables
metrics
non_trainable_variables
layers
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


total

count

_fn_kwargs
	variables
	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
(:&&'2Adam/conv1d/kernel/m
:'2Adam/conv1d/bias/m
*:('N2Adam/conv1d_1/kernel/m
 :N2Adam/conv1d_1/bias/m
#:!N'2Adam/dense/kernel/m
:'2Adam/dense/bias/m
%:#''2Adam/dense_1/kernel/m
:'2Adam/dense_1/bias/m
%:#'2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
-:+	NИ2Adam/lstm/lstm_cell/kernel/m
7:5	NИ2&Adam/lstm/lstm_cell/recurrent_kernel/m
':%И2Adam/lstm/lstm_cell/bias/m
(:&&'2Adam/conv1d/kernel/v
:'2Adam/conv1d/bias/v
*:('N2Adam/conv1d_1/kernel/v
 :N2Adam/conv1d_1/bias/v
#:!N'2Adam/dense/kernel/v
:'2Adam/dense/bias/v
%:#''2Adam/dense_1/kernel/v
:'2Adam/dense_1/bias/v
%:#'2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
-:+	NИ2Adam/lstm/lstm_cell/kernel/v
7:5	NИ2&Adam/lstm/lstm_cell/recurrent_kernel/v
':%И2Adam/lstm/lstm_cell/bias/v
ю2ы
H__inference_functional_1_layer_call_and_return_conditional_losses_488385
H__inference_functional_1_layer_call_and_return_conditional_losses_489646
H__inference_functional_1_layer_call_and_return_conditional_losses_488424
H__inference_functional_1_layer_call_and_return_conditional_losses_489142Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
-__inference_functional_1_layer_call_fn_488495
-__inference_functional_1_layer_call_fn_489708
-__inference_functional_1_layer_call_fn_488565
-__inference_functional_1_layer_call_fn_489677Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
у2р
!__inference__wrapped_model_485371К
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ
&
ь2щ
B__inference_conv1d_layer_call_and_return_conditional_losses_489724Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_conv1d_layer_call_fn_489733Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2Ѕ
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_485380г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
2__inference_average_pooling1d_layer_call_fn_485386г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
ю2ы
D__inference_conv1d_1_layer_call_and_return_conditional_losses_489749Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv1d_1_layer_call_fn_489758Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Њ2Ї
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_485395г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
4__inference_average_pooling1d_1_layer_call_fn_485401г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
э2ъ
C__inference_reshape_layer_call_and_return_conditional_losses_489771Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_reshape_layer_call_fn_489776Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
у2р
@__inference_lstm_layer_call_and_return_conditional_losses_490252
@__inference_lstm_layer_call_and_return_conditional_losses_491194
@__inference_lstm_layer_call_and_return_conditional_losses_491638
@__inference_lstm_layer_call_and_return_conditional_losses_490696е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
%__inference_lstm_layer_call_fn_490718
%__inference_lstm_layer_call_fn_491660
%__inference_lstm_layer_call_fn_490707
%__inference_lstm_layer_call_fn_491649е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_491670Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_dense_layer_call_fn_491679Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_491689Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_1_layer_call_fn_491698Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_491709Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_2_layer_call_fn_491718Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_488606input_1
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
!__inference__wrapped_model_485371xFGH/056;<4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ
&
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџи
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_485395EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Џ
4__inference_average_pooling1d_1_layer_call_fn_485401wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџж
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_485380EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ­
2__inference_average_pooling1d_layer_call_fn_485386wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџЌ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_489749d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ'
Њ ")Ђ&

0џџџџџџџџџN
 
)__inference_conv1d_1_layer_call_fn_489758W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ'
Њ "џџџџџџџџџNЊ
B__inference_conv1d_layer_call_and_return_conditional_losses_489724d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
&
Њ ")Ђ&

0џџџџџџџџџ
'
 
'__inference_conv1d_layer_call_fn_489733W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
&
Њ "џџџџџџџџџ
'Ѓ
C__inference_dense_1_layer_call_and_return_conditional_losses_491689\56/Ђ,
%Ђ"
 
inputsџџџџџџџџџ'
Њ "%Ђ"

0џџџџџџџџџ'
 {
(__inference_dense_1_layer_call_fn_491698O56/Ђ,
%Ђ"
 
inputsџџџџџџџџџ'
Њ "џџџџџџџџџ'Ѓ
C__inference_dense_2_layer_call_and_return_conditional_losses_491709\;</Ђ,
%Ђ"
 
inputsџџџџџџџџџ'
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_2_layer_call_fn_491718O;</Ђ,
%Ђ"
 
inputsџџџџџџџџџ'
Њ "џџџџџџџџџЁ
A__inference_dense_layer_call_and_return_conditional_losses_491670\/0/Ђ,
%Ђ"
 
inputsџџџџџџџџџN
Њ "%Ђ"

0џџџџџџџџџ'
 y
&__inference_dense_layer_call_fn_491679O/0/Ђ,
%Ђ"
 
inputsџџџџџџџџџN
Њ "џџџџџџџџџ'Р
H__inference_functional_1_layer_call_and_return_conditional_losses_488385tFGH/056;<<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
&
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Р
H__inference_functional_1_layer_call_and_return_conditional_losses_488424tFGH/056;<<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
&
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 П
H__inference_functional_1_layer_call_and_return_conditional_losses_489142sFGH/056;<;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
&
p

 
Њ "%Ђ"

0џџџџџџџџџ
 П
H__inference_functional_1_layer_call_and_return_conditional_losses_489646sFGH/056;<;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
&
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_functional_1_layer_call_fn_488495gFGH/056;<<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
&
p

 
Њ "џџџџџџџџџ
-__inference_functional_1_layer_call_fn_488565gFGH/056;<<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
&
p 

 
Њ "џџџџџџџџџ
-__inference_functional_1_layer_call_fn_489677fFGH/056;<;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
&
p

 
Њ "џџџџџџџџџ
-__inference_functional_1_layer_call_fn_489708fFGH/056;<;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
&
p 

 
Њ "џџџџџџџџџБ
@__inference_lstm_layer_call_and_return_conditional_losses_490252mFGH?Ђ<
5Ђ2
$!
inputsџџџџџџџџџN

 
p

 
Њ "%Ђ"

0џџџџџџџџџN
 Б
@__inference_lstm_layer_call_and_return_conditional_losses_490696mFGH?Ђ<
5Ђ2
$!
inputsџџџџџџџџџN

 
p 

 
Њ "%Ђ"

0џџџџџџџџџN
 С
@__inference_lstm_layer_call_and_return_conditional_losses_491194}FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџN

 
p

 
Њ "%Ђ"

0џџџџџџџџџN
 С
@__inference_lstm_layer_call_and_return_conditional_losses_491638}FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџN

 
p 

 
Њ "%Ђ"

0џџџџџџџџџN
 
%__inference_lstm_layer_call_fn_490707`FGH?Ђ<
5Ђ2
$!
inputsџџџџџџџџџN

 
p

 
Њ "џџџџџџџџџN
%__inference_lstm_layer_call_fn_490718`FGH?Ђ<
5Ђ2
$!
inputsџџџџџџџџџN

 
p 

 
Њ "џџџџџџџџџN
%__inference_lstm_layer_call_fn_491649pFGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџN

 
p

 
Њ "џџџџџџџџџN
%__inference_lstm_layer_call_fn_491660pFGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџN

 
p 

 
Њ "џџџџџџџџџNЇ
C__inference_reshape_layer_call_and_return_conditional_losses_489771`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџN
Њ ")Ђ&

0џџџџџџџџџN
 
(__inference_reshape_layer_call_fn_489776S3Ђ0
)Ђ&
$!
inputsџџџџџџџџџN
Њ "џџџџџџџџџNЌ
$__inference_signature_wrapper_488606FGH/056;<?Ђ<
Ђ 
5Њ2
0
input_1%"
input_1џџџџџџџџџ
&"1Њ.
,
dense_2!
dense_2џџџџџџџџџ