import numpy as np
import cvxpy as cp

def optsi (A,B1,B2,K_Pattern,Q,R) :
    # For a given pattern of K, calculate the optimal feedback gain using sparsity invirance
    # A: system matrix;  B1: distrubance matrix;  B2: control input matrix;

    n = len(A)
    m = len(B2[0]) # number of driver nodes
    epsilon = 1e-5;

    # Sparsity invariance 
    Tp = K_Pattern
    Rp = pattern_invariance(Tp)

    # variables
    W = cp.Variable((m+n,m+n), symmetric=True)
    #                W[0:m,0:m] == Y,
    #                W[0:m, m:] == Z,   - Matrix Z has sparsity pattern in K_Pattern
    #                W[m:,m:] == X,    - block diagonal X

    for i in range(n):
        for j in range(i,n):
            if Rp[i,j] == 0:
                W[m+i,m+j] = 0 
                W[m+j,m+i] = 0

    for i in range(m):
        for j in range(n):
            if Tp[i,j] == 0:
                W[i,m+j] = 0;
  
    # constraint

    objective    = cp.Minimize( cp.trace( Q@W[m:,m:] ) + cp.trace( R@W[0:m,0:m] ) )
    constraints  = [(A@W[m:,m:] - B2@W[0:m,m:]) + (A@W[m:,m:] - B2@W[0:m,m:]).T + B1@(B1).T << 0]
    constraints += [ W[m:,m:] - epsilon*np.identity(n) >> 0]
    constraints += [W >> 0]
    
    # constraints  = [-((A@X - B@Z) + (A@X - B@Z).T + H@(H).T) >> 0]
    # constraints += [X - epsilon*np.identity(n) >> 0]
    # constraints += [np.block([[Y, Z], [Z.T, X]]) >> 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.MOSEK)
    
    X1 = W[m:,m:].value
    Z1 = W[0:m,m:].value

    
    K_Opt = Z1 @ np.linalg.inv(X1)

    # Const = [X-epsilon*eye(n) >= 0, [Y Z; Z' X] >= 0,... 
    #     (A*X-B2*Z)+(A*X-B2*Z)'+B1*B1' <= 0];

    # %% cost function

    # Obj = trace(Q*X)+trace(R*Y);

    # %% solution via mosek
    # ops = sdpsettings('solver','mosek');
    # Info = optimize(Const,Obj,ops);
    # X1 = value(X);
    # Z1 = value(Z);
    # K_Opt = Z1*X1^(-1);
