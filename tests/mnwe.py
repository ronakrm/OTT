import numpy as np

np.random.seed(0)

print('')
print('-----------2D TENSOR CASE-----------')

# construct a random matrix using aOTT representation
# using a 3 mode decomp
n = [3,3]
r = 3

cores1 = [np.random.uniform(size=[1,r]) for i in range(0,n[0])]
cores1 = [cores1[i]/np.linalg.norm(cores1[i]) for i in range(0,n[0])]
cores2 = [np.random.uniform(size=[r,1]) for i in range(0,n[1])]
cores2 = [cores2[i]/np.linalg.norm(cores2[i]) for i in range(0,n[1])]

# rebuild full tensor
T_OTT = np.zeros(n)
for i in range(n[0]):
    # print('Norm of core1 (vector): ', np.linalg.norm(cores1[i],'fro'))
    for j in range(n[1]):
        # print('Norm of core2 (matrix): ', np.linalg.norm(cores2[j],'fro'))
        T_OTT[i,j] = np.matmul(cores1[i],cores2[j])

W_OTT = T_OTT # No reshaping needed in matrix 2D case
print(W_OTT)
print('Frob Norm of constructed tensor: ', np.linalg.norm(W_OTT,'fro'))
print('2 Norm of constructed tensor: ', np.linalg.norm(W_OTT,2))
print('W_OTT^T W_OTT should be identity:')
print(W_OTT.T.dot(W_OTT))


print('')
print('-----------3D TENSOR CASE-----------')
def build_ott_cores(n, ri, rip1):
    # build random orthogonal cores using QR of random matrices
    tmp = []
    for j in range(n):
        if ri < rip1:
            tmpA = np.random.uniform(size=[rip1,ri])
            q,_ = np.linalg.qr(tmpA)
            q = np.transpose(q)
        else:
            tmpA = np.random.uniform(size=[ri,rip1])
            q,_ = np.linalg.qr(tmpA)
        q = q/np.linalg.norm(q)
        tmp.append(q)
    return tmp

# construct a random matrix using aOTT representation
# using a 3 mode decomp
n = [2,2,4]
r = [1,4,4,1]

cores1 = build_ott_cores(n[0],r[0],r[1])
cores2 = build_ott_cores(n[1],r[1],r[2])
cores3 = build_ott_cores(n[2],r[2],r[3])

# rebuild full tensor
T_OTT = np.zeros(n)
for i in range(n[0]):
    # print('Norm of core1 (vector): ', np.linalg.norm(cores1[i],'fro'))
    for j in range(n[1]):
        # print('Norm of core2 (matrix): ', np.linalg.norm(cores2[j],'fro'))
        for k in range(n[2]):
            # print('Norm of core3 (vector): ', np.linalg.norm(cores3[k],'fro'))
            T_OTT[i,j,k] = np.matmul(cores1[i],np.matmul(cores2[j],cores3[k]))


##### THIS IS THE BIG ISSUE ####
W_OTT = np.reshape(T_OTT, (4,4))
print(W_OTT)
print('Frob Norm of constructed tensor: ', np.linalg.norm(W_OTT,'fro'))
print('2 Norm of constructed tensor: ', np.linalg.norm(W_OTT,2))
print('W_OTT^T W_OTT should be identity:')
print(W_OTT.T.dot(W_OTT))
print('')