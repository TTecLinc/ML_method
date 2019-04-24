import numpy as np

#Kronecker Product
def Kronecker(A,B):
    #np.array()
    lengthA=A.shape[0]
    widthA=A.shape[1]
    lengthB=B.shape[0]
    widthB=B.shape[1]
    C=np.zeros((lengthA*lengthB,widthA*widthB))
    for i in range(lengthA):
        for j in range(widthA):
            for ib in range(lengthB):
                for jb in range(widthB):
                    C[i*lengthA+ib,j*widthA+jb]=A[i,j]*B[ib,jb]
    return C
#Dispose the matrix
def vec(A):
    V=[]
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            V.append(A[i][j])
    return np.array(V).T

def Slove_DSGE(A,B,C,D,F,J,K,L,G,H,M,N):
    #x_t=Px_{t-1}+Qz_t
    #y_t=Rx_{t-1}+Sz_t
    #P is an variable
    B_sq=-(J*C.I*B-G+K*C.I*A)
    A_sq=(F-J*C.I*A)
    C_sq=H-K*C.I*B
    P_pos=(-B_sq+np.sqrt(B_sq**2-4*A_sq*C_sq))/2/A_sq
    P_neg=(-B_sq-np.sqrt(B_sq**2-4*A_sq*C_sq))/2/A_sq
    P=[P_pos,P_neg]
    #==================================================#
    #R
    R=-C.I*(A*P+B)
    #==================================================#
    #Q
    I_k=np.ones(N.shape[0])
    K_A=Kronecker(N.T,F-J*C.I*A)+Kronecker(I_k,J*R+F*P+G-K*C.I*A)
    Q=(K_A).I*vec((J*C.I*D-L)*N+K*C.I*D-M)
    #==================================================#
    S=-C.I*(A*Q+D)
    return P,Q,R,S

