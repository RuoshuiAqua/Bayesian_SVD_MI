set.seed(3)

## Generate N random integers that sum to M 
rand_vect <- function(N, M, sd = 1, pos.only = TRUE) {
  vec <- rnorm(N, M/N, sd)
  if (abs(sum(vec)) < 0.01) vec <- vec + 1
  vec <- round(vec / sum(vec) * M)
  deviation <- M - sum(vec)
  for (. in seq_len(abs(deviation))) {
    vec[i] <- vec[i <- sample(N, 1)] + sign(deviation)
  }
  if (pos.only) while (any(vec < 0)) {
    negs <- vec < 0
    pos  <- vec > 0
    vec[negs][i] <- vec[negs][i <- sample(sum(negs), 1)] + 1
    vec[pos][i]  <- vec[pos ][i <- sample(sum(pos ), 1)] - 1
  }
  return(vec)
}

## silumale missing indicators under MCAR
## missing completely at random; omega is the missing indicator: 1: missing; omega is a matrix of m x n; this function controls that their is not a whole row/col missing
f.omega.mcar = function(m, n, prob){
  r = round( min(m,n) / 2)
  ind = matrix(rbinom(r^2, size=1, prob),r,r)
  repcol = rand_vect(r,m)
  reprow = rand_vect(r,n)
  omega = ind[rep(1:r,repcol), rep(1:r,reprow)]
  return(omega)
}

#### mcar: silumale missing indicators
## missing completely at random
## omega is the missing indicator: 1: missing
## omega is a matrix of m x n
## when missing prob is hing, it's likely to have a whole row/col missing
f.omega.mcar = function(m, n, mprob){
  sim=TRUE
  while(sim){
    omega=matrix(rbinom(m*n,size=1,mprob),m,n)
    sim = sum(colSums(omega)==m) | sum(rowSums(omega)==n)
  }
  return(omega)
}


## silumale missing indicators under MAR
f.omega.mar = function(Y, prob=0.2, method){
  m_Y= dim(Y)[1]; n_Y = dim(Y)[2]
  ## missing at random
  if(method=="sv"){
    ## missing according to magnatitude of singular values
    ## more likely to miss when the biggest dj contributes more to a point?
    ## omega is the missing indicator: 1: missing
    ## omega is a matrix of m x n
    ## Ysvd = svd(Y)
    Ysvd = svd(Y)
    U = Ysvd$u
    D = Ysvd$d
    V = Ysvd$v
    
    notmaxD = which(D < max(D))
    Dnew = D
    Dnew[notmaxD] = 0
    Ynew = U %*% diag(Dnew) %*% t(V)
    
    beta = 10
    f.alpha = function(alpha, beta, Y){
      m_Y = dim(Y)[1]; n_Y = dim(Y)[2]
      a = sum(apply(Y, c(1,2), function(x){1/(1+exp(-alpha-beta*x))} ))
      return(a - prob * m_Y * n_Y)      
    }
    
    alpha = uniroot(f.alpha, beta=beta, Y=Ynew, c(-100000000, 100000000))$root
    omega=apply(Ynew,c(1,2),function(x){ p=1/(1+exp(-alpha-beta*x)); rbinom(1, size=1, prob=p)})
    
  }else if(method=="corr"){
    ## each column of the data Y missing depends on the column that's most highly correlated with that column
    corY = cor(Y)
    diag(corY) = 0
    corYmax = apply(corY, 2, function(x){ x=abs(x);maxX = which(x == max(x));return(maxX)})
    omega = matrix(NA, m_Y,n_Y)
    for(j in 1:n_Y){
      beta = 10
      f.alpha = function(alpha, beta, Y){
        n_Y = dim(Y)[2]
        a = sum(sapply(Y[,corYmax[j]], function(x){1/(1+exp(-alpha-beta*x))} ))
        return(a - prob * m_Y)      
      }
      alpha = uniroot(f.alpha,beta=beta, Y=Y, c(-100000000, 100000000))$root
      omega[,j]=sapply(Y[,corYmax[j]],function(x){ p=1/(1+exp(-alpha-beta*x)); rbinom(1, size=1, prob=p)})
    }
  }else{print("please specify method as 'sv'(singular value) or 'corr'")}
  return(omega)
}


args = commandArgs(TRUE)
m = as.numeric(args[1])
n = as.numeric(args[2])
K = as.numeric(args[3])

print(m)
print(n)
print(K)

omega = f.omega.mcar(m=m, n=n, mprob=0.15)
dir.create("./output/omega")
write.table(omega, file=paste("./output/omega/omega_mcar", "_m_", m, "_n_", n, "_K_", K, ".csv", sep = ""), row.names = FALSE, sep=",", col.names = FALSE)

M = read.csv(paste("./output/simulated_data/M", "_m_", m, "_n_", n, "_K_", K, ".csv", sep = ""), header = FALSE)
omega = f.omega.mar(Y=M, prob=0.15, method="sv")
write.table(omega, file=paste("./output/omega/omega_sv", "_m_", m, "_n_", n, "_K_", K, ".csv", sep = ""), row.names = FALSE, sep=",", col.names = FALSE)

omega = f.omega.mar(Y=M, prob=0.15, method="corr")
write.table(omega, file=paste("./output/omega/omega_corr", "_m_", m, "_n_", n, "_K_", K, ".csv", sep = ""), row.names = FALSE, sep=",", col.names = FALSE)

q()
